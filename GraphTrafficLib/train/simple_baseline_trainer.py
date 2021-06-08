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

from ..utils.data_utils import create_test_train_split_max_min_normalize, create_test_train_split_max_min_normalize_no_split
from ..utils import encode_onehot
from ..utils import test_lstm, train_lstm
from ..utils.losses import torch_nll_gaussian, kl_categorical, cyc_anneal
from ..models import SimpleLSTM


class SimpleBaselineTrainer:
    """The trainer class"""

    def __init__(
        self,
        shuffle_train=True,
        shuffle_test=True,
        experiment_name="Baselines",
        normalize=False,
        train_frac=0.8,
        split_len=40,
        batch_size=16
    ):
        # "test"
        # Training settings
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test


        # Data settings
        self.normalize = normalize
        self.train_frac = train_frac
        self.split_len = split_len
        self.batch_size = batch_size

        self.model_settings = {
            "shuffle_train": self.shuffle_train,
            "shuffle_test": self.shuffle_test,
            "normalize": self.normalize,
            "train_frac": self.train_frac,
            "split_len": self.split_len,
            "batch_size": self.batch_size
        }

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
            self.train_dataset,
            self.test_dataset,
            self.train_max,
            self.train_min,
        ) = create_test_train_split_max_min_normalize_no_split(
            data=data_tensor,
            weather_data=weather_tensor,
            normalize=self.normalize,
            train_frac=self.train_frac,
        )

        min_date = pd.Timestamp(year=2019, month=1, day=1)
        max_date = pd.Timestamp(year=2019 + 1, month=1, day=1)

        # Note that this misses a bit from the beginning but this will not be a big problem when we index finer
        bins_dt = pd.date_range(start=min_date, end=max_date, freq="1H")
        split_bins_dt = bins_dt[: -(self.split_len + 1)]

        self.test_dates = split_bins_dt[int(self.train_frac * len(split_bins_dt)) :]
        self.train_dates = split_bins_dt[: int(self.train_frac * len(split_bins_dt))]

        print(f"train_dates len: {len(self.train_dates)}")
        print(f"test_dates len: {len(self.test_dates)}")

    def _init_model(self):
        raise NotImplementedError

    def train(self):
        print("Starting training")
        train_mse_arr = []

        test_mse_arr = []

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
                test_mse = test_lstm(
                    model=self.model,
                    test_dataloader=self.test_dataloader,
                    optimizer=self.optimizer,
                    burn_in=self.burn_in,
                    burn_in_steps=self.burn_in_steps,
                    split_len=self.split_len,
                )
                self.writer.add_scalar("Test_MSE", test_mse, i)

                test_mse_arr.append(test_mse)
            train_mse_arr.append(train_mse)
            self.train_dict = {
                "test": {"mse": test_mse_arr},
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