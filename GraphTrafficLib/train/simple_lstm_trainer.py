"""The trainer clases
"""
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import tensorboard_trace_handler
import torch.nn.functional as F

from ..utils.data_utils import create_dataloaders
from ..utils import encode_onehot
from ..utils import val_lstm, train_lstm
from ..utils.losses import torch_nll_gaussian, kl_categorical, cyc_anneal
from ..utils.general_utils import count_parameters
from ..models import SimpleLSTM


class SimpleLSTMTrainer:
    """The trainer class"""

    def __init__(
        self,
        batch_size=25,
        n_epochs=100,
        dropout_p=0,
        shuffle_train=True,
        shuffle_val=False,
        experiment_name="test",
        normalize=True,
        train_frac=0.8,
        burn_in_steps=30,
        split_len=40,
        burn_in=True,  # maybe remove this
        lstm_hid=128,
        lstm_dropout=0,
    ):

        # Training settings
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout_p = dropout_p
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        # Saving settings

        # Set up results folder
        self.experiment_name = experiment_name
        self.experiment_folder_path = f"../models/{self.experiment_name}"
        next_version = 2
        while os.path.exists(self.experiment_folder_path):
            new_experiment_name = f"{self.experiment_name}_v{next_version}"
            self.experiment_folder_path = f"../models/{new_experiment_name}"
            next_version += 1
        print(self.experiment_folder_path)
        os.mkdir(self.experiment_folder_path)
        print(f"Created {self.experiment_folder_path}")

        # Setup logger
        self.experiment_log_path = f"{self.experiment_folder_path}/runs"
        self.writer = SummaryWriter(log_dir=self.experiment_log_path)
        print(f"Logging at {self.experiment_log_path}")

        # Data settings
        self.normalize = normalize
        self.train_frac = train_frac

        # Model settings
        self.burn_in_steps = burn_in_steps
        self.split_len = split_len
        self.pred_steps = self.split_len - self.burn_in_steps
        self.encoder_steps = self.split_len
        assert self.burn_in_steps + self.pred_steps == self.split_len

        self.burn_in = burn_in
        self.lstm_dropout = lstm_dropout

        # Net sizes
        self.n_in = 1  # TODO update this hardcode
        self.lstm_hid = lstm_hid

        # self.model_settings = {
        #     "batch_size": self.batch_size,
        #     "n_epochs": self.n_epochs,
        #     "dropout_p": self.dropout_p,
        #     "shuffle_train": self.shuffle_train,
        #     "shuffle_val": self.shuffle_val,
        #     "experiment_name": self.experiment_name,
        #     "normalize": self.normalize,
        #     "train_frac": self.train_frac,
        #     "burn_in_steps": self.burn_in_steps,
        #     "split_len": self.split_len,
        #     "burn_in": self.burn_in,  # maybe remove this
        #     "lstm_hid": self.lstm_hid,
        #     "lstm_dropout": self.lstm_dropout,
        # }
        # print(self.model_settings)

        self._init_model()
        self.n_model_params = count_parameters(self.model)

        
        self.self_parameters = [x + ": " + str(y) + "\n" for x, y in vars(locals()['self']).items() if not x in (["encoder", "decoder", "model_params"])]
        self.parameters = [x + ": " + str(y) + "\n" for x, y in locals().items()]
        with open(
            os.path.join(self.experiment_folder_path, "parameters.txt"), "w"
        ) as f:
            f.writelines(self.parameters)
        self.writer.add_text("parameters", "\n".join(self.parameters))

        with open(
            os.path.join(self.experiment_folder_path, "self_parameters.txt"), "w"
        ) as f:
            f.writelines(self.self_parameters)
        self.writer.add_text("self_parameters", "\n".join(self.self_parameters))

        print("stop")


    def load_data(self, data_path, weather_data_path, dropoff_data_path=None):

        # Load data
        data = np.load(data_path)

        if dropoff_data_path is not None:
            dropoff_data = np.load(dropoff_data_path)

            # Create data tensor
            pickup_tensor = torch.Tensor(data)
            dropoff_tensor = torch.Tensor(dropoff_data)

            # Stack data tensor
            data_tensor = torch.cat([pickup_tensor, dropoff_tensor], dim=0)
        else:
            data_tensor = torch.Tensor(data)

        # load weather data
        weather_df = pd.read_csv(weather_data_path, parse_dates=[0, 7])
        # temp fix for na temp
        weather_df.loc[weather_df.temperature.isna(), "temperature"] = 0
        sum(weather_df.temperature.isna())
        # Create weather vector
        weather_vector = weather_df.loc[:, ("temperature", "precipDepth")].values
        weather_tensor = torch.Tensor(weather_vector)

        # Create data loader with max min normalization
        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.train_max,
            self.train_min,
        ) = create_dataloaders(
            data=data_tensor,
            weather_data=weather_tensor,
            split_len=self.split_len,
            batch_size=self.batch_size,
            normalize=self.normalize,
            train_frac=self.train_frac,
        )

        min_date = pd.Timestamp(year=2019, month=1, day=1)
        max_date = pd.Timestamp(year=2019 + 1, month=1, day=1)

        # # Note that this misses a bit from the beginning but this will not be a big problem when we index finer
        # bins_dt = pd.date_range(start=min_date, end=max_date, freq="1H")
        # split_bins_dt = bins_dt[: -(self.split_len + 1)]

        # self.test_dates = split_bins_dt[int(self.train_frac * len(split_bins_dt)) :]
        # self.train_dates = split_bins_dt[: int(self.train_frac * len(split_bins_dt))]

        # print(f"train_dates len: {len(self.train_dates)}")
        # print(f"test_dates len: {len(self.test_dates)}")

    def _init_model(self):
        self.model = SimpleLSTM(
            input_dims=self.n_in, hidden_dims=self.lstm_hid, dropout=self.lstm_dropout
        ).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        print("Starting training")
        train_mse_arr = []

        val_mse_arr = []

        for i in tqdm(range(self.n_epochs)):
            t = time.time()

            train_mse = train_lstm(
                model=self.model,
                train_dataloader=self.train_dataloader,
                optimizer=self.optimizer,
                burn_in=self.burn_in,
                burn_in_steps=self.burn_in_steps,
                split_len=self.split_len,
            )
            self.writer.add_scalar("Train_MSE", train_mse, i)

            if i % 10 == 0:
                val_mse = val_lstm(
                    model=self.model,
                    val_dataloader=self.val_dataloader,
                    optimizer=self.optimizer,
                    burn_in=self.burn_in,
                    burn_in_steps=self.burn_in_steps,
                    split_len=self.split_len,
                )
                self.writer.add_scalar("val_MSE", val_mse, i)

                val_mse_arr.append(val_mse)
            train_mse_arr.append(train_mse)
            self.train_dict = {
                "val": {"mse": val_mse_arr},
                "train": {"mse": train_mse_arr},
            }

    def save_model(self):
        model_path = f"{self.experiment_folder_path}/model_dict.pth"

        model_dict = {
            "model": self.model.state_dict(),
            "settings": self.model_settings,
            "train_res": self.train_dict,
        }

        torch.save(model_dict, model_path)
        print(f"Model saved at {model_path}")