"""The trainer clases
"""
import time
import os

import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pandas as pd

from ..utils.data_utils import create_test_train_split_max_min_normalize
from ..utils import encode_onehot
from ..utils import test, train
from ..models.latent_graph import MLPEncoder, GRUDecoder_multistep


class Trainer:
    """The trainer class
    """

    def __init__(
        self,
        batch_size=25,
        n_epochs=100,
        dropout_p=0,
        shuffle_train=True,
        shuffle_test=False,
        encoder_factor=True,
        experiment_name="test",
        normalize=True,
        train_frac=0.8,
        burn_in_steps=30,
        split_len=40,
        burn_in=True,  # maybe remove this
        kl_frac=1,
        enc_n_hid=128,
        enc_n_out=2,
        dec_n_hid=16,
        dec_n_out=1,
        dec_f_in=1,
        dec_msg_hid=8,
        dec_msg_out=8,
        dec_gru_hid=8,
        dec_edge_types=2,
    ):
        # Training settings
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout_p = dropout_p
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test

        # Model settings
        self.encoder_factor = encoder_factor

        # Saving settings
        # Set up results folder
        self.experiment_name = experiment_name
        self.experiment_folder_path = f"../models/{self.experiment_name}"
        next_version = 2
        while os.path.exists(self.experiment_folder_path):
            new_experiment_name = f"{self.experiment_name}_v{next_version}"
            self.experiment_folder_path = f"../models/{new_experiment_name}"
            next_version += 1
        os.mkdir(self.experiment_folder_path)
        print(f"Created {self.experiment_folder_path}")


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
        self.kl_frac = kl_frac

        # Net sizes
        # Encoder
        self.enc_n_in = self.encoder_steps * 1  # TODO update this hardcode
        self.enc_n_hid = enc_n_hid
        self.enc_n_out = enc_n_out

        # Decoder
        self.dec_n_hid = dec_n_hid
        self.dec_n_out = dec_n_out
        self.dec_f_in = dec_f_in
        self.dec_msg_hid = dec_msg_hid
        self.dec_msg_out = dec_msg_out
        self.dec_gru_hid = dec_gru_hid
        self.dec_edge_types = dec_edge_types

        self.model_settings = {
            "enc_n_in": self.enc_n_in,
            "enc_n_hid": self.enc_n_hid,
            "enc_n_out": self.enc_n_out,
            "dec_n_hid": self.dec_n_hid,
            "dec_n_out": self.dec_n_out,
            "dec_f_in": self.dec_f_in,
            "dec_msg_hid": self.dec_msg_hid,
            "dec_msg_out": self.dec_msg_out,
            "dec_gru_hid": self.dec_gru_hid,
            "dec_edge_types": self.dec_edge_types,
        }

        (   self.train_dataloader,
            self.test_dataloader,
            self.train_max,
            self.train_min,
            self.test_dates,
            self.train_dates,
        ) = self._load_data()

        self._init_model()

    def _load_data(self):
        dataset_folder = "../datafolder"
        proc_folder = f"{dataset_folder}/procdata"

        # Load data
        pickup_data_path = f"{proc_folder}/full_year_manhattan_vector_pickup.npy"
        pickup_data = np.load(pickup_data_path)
        dropoff_data_path = f"{proc_folder}/full_year_manhattan_vector_dropoff.npy"
        dropoff_data = np.load(dropoff_data_path)
        weather_data_path = f"{proc_folder}/LGA_weather_full_2019.csv"
        weather_df = pd.read_csv(weather_data_path, parse_dates=[0, 7])

        # temp fix for na temp
        weather_df.loc[weather_df.temperature.isna(), "temperature"] = 0
        sum(weather_df.temperature.isna())

        # Create weather vector
        weather_vector = weather_df.loc[:, ("temperature", "precipDepth")].values

        # Create data tensor
        pickup_tensor = torch.Tensor(pickup_data)
        dropoff_tensor = torch.Tensor(dropoff_data)
        weather_tensor = torch.Tensor(weather_vector)

        # Stack data tensor
        data_tensor = torch.cat([pickup_tensor, dropoff_tensor], dim=0)

        # Create data loader with max min normalization
        (
            train_dataloader,
            test_dataloader,
            train_max,
            train_min,
        ) = create_test_train_split_max_min_normalize(
            data=data_tensor,
            weather_data=weather_tensor,
            split_len=self.split_len,
            batch_size=self.batch_size,
            normalize=self.normalize,
            train_frac=self.train_frac,
        )

        min_date = pd.Timestamp(year=2019, month=1, day=1)
        max_date = pd.Timestamp(year=2019 + 1, month=1, day=1)

        # Note that this misses a bit from the beginning but this will not be a big problem when we index finer
        bins_dt = pd.date_range(start=min_date, end=max_date, freq="1H")
        split_bins_dt = bins_dt[: -(self.split_len + 1)]

        test_dates = split_bins_dt[int(self.train_frac * len(split_bins_dt)) :]
        train_dates = split_bins_dt[: int(self.train_frac * len(split_bins_dt))]

        print(f"train_dates len: {len(train_dates)}")
        print(f"test_dates len: {len(test_dates)}")

        return (
            train_dataloader,
            test_dataloader,
            train_max,
            train_min,
            test_dates,
            train_dates,
        )

    def _init_model(self):
        self.encoder = MLPEncoder(
            n_in=self.enc_n_in,
            n_hid=self.enc_n_hid,
            n_out=self.enc_n_out,
            do_prob=self.dropout_p,
            factor=self.encoder_factor,
        ).cuda()

        self.decoder = GRUDecoder_multistep(
            n_hid=self.dec_n_hid,
            f_in=self.dec_f_in,
            msg_hid=self.dec_msg_hid,
            msg_out=self.dec_msg_out,
            gru_hid=self.dec_gru_hid,
            edge_types=self.dec_edge_types,
        ).cuda()

        self.model_params = list(self.encoder.parameters()) + list(
            self.decoder.parameters()
        )

        self.optimizer = optim.Adam(self.model_params, lr=0.001)

        # Generate off-diagonal interaction graph
        off_diag = np.ones([132, 132]) - np.eye(132)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

        # Set up prior
        prior = np.array([0.99, 0.01])
        print("Using prior")
        print(prior)
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        self.log_prior = Variable(log_prior).cuda()


    def train(self):
        print("Starting training")
        train_mse_arr = []
        train_nll_arr = []
        train_kl_arr = []

        test_mse_arr = []
        test_nll_arr = []
        test_kl_arr = []

        for i in range(self.n_epochs):
            t = time.time()

            train_mse, train_nll, train_kl = train(
                encoder=self.encoder,
                decoder=self.decoder,
                train_dataloader=self.train_dataloader,
                optimizer=self.optimizer,
                rel_rec=self.rel_rec,
                rel_send=self.rel_send,
                burn_in=self.burn_in,
                burn_in_steps=self.burn_in_steps,
                split_len=self.split_len,
                log_prior=self.log_prior,
            )

            print(
                f"EPOCH: {i}, TIME: {time.time() - t}, MSE: {train_mse}, NLL: {train_nll}, KL: {train_kl} "
            )
            if i % 10 == 0:
                test_mse, test_nll, test_kl = test(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    test_dataloader=self.test_dataloader,
                    optimizer=self.optimizer,
                    rel_rec=self.rel_rec,
                    rel_send=self.rel_send,
                    burn_in=self.burn_in,
                    burn_in_steps=self.burn_in_steps,
                    split_len=self.split_len,
                    log_prior=self.log_prior,
                )

                print("::::::::TEST::::::::")
                print(f"EPOCH: {i}, MSE: {test_mse}, NLL: {test_nll}, KL: {test_kl} ")
                print("::::::::::::::::::::")
                test_mse_arr.append(test_mse)
                test_nll_arr.append(test_nll)
                test_kl_arr.append(test_kl)
            train_mse_arr.append(train_mse)
            train_nll_arr.append(train_nll)
            train_kl_arr.append(train_kl)
            self.train_dict = {"test": {"mse": test_mse_arr, "nll": test_nll_arr, "kl": test_kl_arr},
                          "train": {"mse": train_mse_arr, "nll": train_nll_arr, "kl": train_kl_arr}}

    def save_model(self):
        model_path = f"{self.experiment_folder_path}/model_dict.pth"

        gru_dev_1_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "settings": self.model_settings,
            "train_res": self.train_dict
        }

        torch.save(gru_dev_1_dict, model_path)
        print(f"Model saved at {model_path}")
        