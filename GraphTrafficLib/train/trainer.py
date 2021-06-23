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
from ..utils import val, train, dnri_train, dnri_val
from ..utils.losses import torch_nll_gaussian, kl_categorical, cyc_anneal
from ..models.latent_graph import (
    MLPEncoder,
    CNNEncoder,
    GRUDecoder_multistep,
    FixedEncoder,
    RecurrentEncoder,
    DynamicGRUDecoder_multistep,
)


class Trainer:
    """The trainer class"""

    def __init__(
        self,
        batch_size=25,
        n_epochs=100,
        dropout_p=0,
        shuffle_train=True,
        shuffle_val=False,
        encoder_factor=True,
        experiment_name="test",
        normalize=True,
        train_frac=0.8,
        burn_in_steps=30,
        split_len=40,
        burn_in=True,  # maybe remove this
        kl_frac=1,
        kl_cyc=None,
        loss_type=None,
        edge_rate=0.01,
        encoder_type="mlp",
        node_f_dim=1,
        enc_n_hid=128,
        n_edge_types=2,
        dec_n_hid=16,
        dec_msg_hid=8,
        dec_msg_out=8,
        dec_gru_hid=8,
        skip_first=True,
        lr=0.001,
        lr_decay_step=100,
        lr_decay_gamma=0.5,
        fixed_adj_matrix_path=None,
        encoder_lr_frac=1,
        use_bn=True,
    ):

        # Training settings
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout_p = dropout_p
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.lr = lr
        self.encoder_lr_frac = encoder_lr_frac
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma
        self.use_bn = use_bn

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
        self.node_f_dim = node_f_dim

        # Model settings
        self.burn_in_steps = burn_in_steps
        self.split_len = split_len
        self.pred_steps = self.split_len - self.burn_in_steps
        self.encoder_steps = self.split_len
        assert self.burn_in_steps + self.pred_steps == self.split_len

        self.burn_in = burn_in
        self.kl_frac = kl_frac
        self.kl_cyc = kl_cyc
        self.loss_type = loss_type
        self.edge_rate = edge_rate

        # Net sizes
        self.encoder_type = encoder_type

        # Encoder
        if self.encoder_type == "mlp":
            self.enc_n_in = self.encoder_steps * self.node_f_dim
        elif self.encoder_type in ["cnn", "gru", "lstm"]:
            self.enc_n_in = self.node_f_dim  # TODO update these hardcodes?
        elif self.encoder_type == "fixed":
            assert (
                fixed_adj_matrix_path is not None
            ), "fixed encoder need fixed adj matrix"
            self.fixed_adj_matrix = torch.tensor(np.load(fixed_adj_matrix_path))
            self.enc_n_in = self.node_f_dim
        self.enc_n_hid = enc_n_hid
        self.n_edge_types = n_edge_types

        # Decoder
        self.dec_n_hid = dec_n_hid
        self.dec_msg_hid = dec_msg_hid
        self.dec_msg_out = dec_msg_out
        self.dec_gru_hid = dec_gru_hid
        self.skip_first = skip_first

        # init model
        self._init_model()

        # save settings
        self.model_settings = {
            "node_f_dim": self.node_f_dim,
            "encoder_type": self.encoder_type,
            "enc_n_in": self.enc_n_in,
            "enc_n_hid": self.enc_n_hid,
            "enc_n_out": self.n_edge_types,
            "dec_n_hid": self.dec_n_hid,
            "dec_msg_hid": self.dec_msg_hid,
            "dec_msg_out": self.dec_msg_out,
            "dec_gru_hid": self.dec_gru_hid,
            "dec_edge_types": self.n_edge_types,
            "loss_type": self.loss_type,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "lr_decay_step": self.lr_decay_step,
            "lr_decay_gamma": self.lr_decay_gamma,
            "dropout_p": self.dropout_p,
            "shuffle_train": self.shuffle_train,
            "shuffle_val": self.shuffle_val,
            "encoder_factor": self.encoder_factor,
            "normalize": self.normalize,
            "train_frac": self.train_frac,
            "burn_in_steps": self.burn_in_steps,
            "split_len": self.split_len,
            "burn_in": self.burn_in,
            "kl_frac": self.kl_frac,
            "kl_cyc": self.kl_cyc,
            "skip_first": self.skip_first,
            "encoder_lr_frac": self.encoder_lr_frac,
            "use_bn": self.use_bn,
        }

        # Save all parameters to txt file and add to tensorboard
        self.parameters = [x + ": " + str(y) + "\n" for x, y in locals().items()]
        with open(
            os.path.join(self.experiment_folder_path, "parameters.txt"), "w"
        ) as f:
            f.writelines(self.parameters)
        self.writer.add_text("parameters", "\n".join(self.parameters))

        # Init best loss val
        self.best_mse = None

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
            _,
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

        # # self.test_dates = split_bins_dt[int(self.train_frac * len(split_bins_dt)) :]
        # # self.train_dates = split_bins_dt[: int(self.train_frac * len(split_bins_dt))]

        # # print(f"train_dates len: {len(self.train_dates)}")
        # # print(f"test_dates len: {len(self.test_dates)}")

    def _init_model(self):

        if self.encoder_type == "mlp":
            self.encoder = MLPEncoder(
                n_in=self.enc_n_in,
                n_hid=self.enc_n_hid,
                n_out=self.n_edge_types,
                do_prob=self.dropout_p,
                factor=self.encoder_factor,
                use_bn=self.use_bn,  # TODO not checked
            ).cuda()
        elif self.encoder_type == "cnn":
            self.enc_n_hid = 10  # Remember this hardcode
            self.encoder = CNNEncoder(
                n_in=self.enc_n_in,
                n_hid=self.enc_n_hid,
                n_out=self.n_edge_types,
                do_prob=self.dropout_p,
                factor=self.encoder_factor,
                use_bn=self.use_bn,
            ).cuda()
        elif self.encoder_type == "gru" or self.encoder_type == "lstm":
            self.encoder = RecurrentEncoder(
                n_in=self.enc_n_in,
                n_hid=self.enc_n_hid,
                n_out=self.n_edge_types,
                do_prob=self.dropout_p,
                factor=self.encoder_factor,
                rnn_type=self.encoder_type,
                use_bn=self.use_bn,
            ).cuda()
        elif self.encoder_type == "fixed":
            self.encoder = FixedEncoder(adj_matrix=self.fixed_adj_matrix)

        if self.encoder_type in ["gru", "lstm"]:
            self.decoder = DynamicGRUDecoder_multistep(
                n_hid=self.dec_n_hid,
                f_in=self.node_f_dim,
                msg_hid=self.dec_msg_hid,
                msg_out=self.dec_msg_out,
                gru_hid=self.dec_gru_hid,
                edge_types=self.n_edge_types,
                skip_first=self.skip_first,
            ).cuda()
        else:
            self.decoder = GRUDecoder_multistep(
                n_hid=self.dec_n_hid,
                f_in=self.node_f_dim,
                msg_hid=self.dec_msg_hid,
                msg_out=self.dec_msg_out,
                gru_hid=self.dec_gru_hid,
                edge_types=self.n_edge_types,
                skip_first=self.skip_first,
            ).cuda()

        self.model_params = [
            {
                "params": self.encoder.parameters(),
                "lr": self.encoder_lr_frac * self.lr,
            },
            {"params": self.decoder.parameters(), "lr": self.lr},
        ]

        self.optimizer = optim.Adam(self.model_params)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=self.lr_decay_step,
            gamma=self.lr_decay_gamma,
        )

        # Set up prior
        if self.n_edge_types == 2:
            prior = np.array([1 - self.edge_rate, self.edge_rate])
        else:
            prior = np.empty(self.n_edge_types)
            prior[0] = 1 - self.edge_rate
            prior[1:] = self.edge_rate / (self.n_edge_types - 1)

        print(f"Using prior: {prior}")

        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        self.log_prior = Variable(log_prior).cuda()

    def train(self):
        print("Starting training")
        train_mse_arr = []
        train_nll_arr = []
        train_kl_arr = []

        val_mse_arr = []
        val_nll_arr = []
        val_kl_arr = []

        # Generate off-diagonal interaction graph
        n_nodes = self.train_dataloader.dataset[0][0].shape[0]
        off_diag = np.ones([n_nodes, n_nodes]) - np.eye(n_nodes)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

        for i in tqdm(range(self.n_epochs)):

            if self.kl_cyc is not None:
                self.kl_frac = cyc_anneal(i, self.kl_cyc)

            if self.encoder_type in ["gru", "lstm"]:
                train_mse, train_nll, train_kl = dnri_train(
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
                    kl_frac=self.kl_frac,
                    loss_type=self.loss_type,
                    pred_steps=self.pred_steps,
                    skip_first=self.skip_first,
                    n_nodes=n_nodes,
                )
            else:
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
                    kl_frac=self.kl_frac,
                    loss_type=self.loss_type,
                    pred_steps=self.pred_steps,
                    skip_first=self.skip_first,
                    n_nodes=n_nodes,
                )

            self.lr_scheduler.step()
            self.writer.add_scalar("Train_MSE", train_mse, i)
            self.writer.add_scalar("Train_NLL", train_nll, i)
            self.writer.add_scalar("Train_KL", train_kl, i)
            self.writer.add_scalar("KL_frac", self.kl_frac, i)

            if i % 5 == 0:
                if self.encoder_type in ["gru", "lstsm"]:
                    val_mse, val_nll, val_kl = dnri_val(
                        encoder=self.encoder,
                        decoder=self.decoder,
                        val_dataloader=self.val_dataloader,
                        optimizer=self.optimizer,
                        rel_rec=self.rel_rec,
                        rel_send=self.rel_send,
                        burn_in=self.burn_in,
                        burn_in_steps=self.burn_in_steps,
                        split_len=self.split_len,
                        log_prior=self.log_prior,
                        n_nodes=n_nodes,
                    )
                else:
                    val_mse, val_nll, val_kl = val(
                        encoder=self.encoder,
                        decoder=self.decoder,
                        val_dataloader=self.val_dataloader,
                        optimizer=self.optimizer,
                        rel_rec=self.rel_rec,
                        rel_send=self.rel_send,
                        burn_in=self.burn_in,
                        burn_in_steps=self.burn_in_steps,
                        split_len=self.split_len,
                        log_prior=self.log_prior,
                        n_nodes=n_nodes,
                    )
                self.writer.add_scalar("val_MSE", val_mse, i)
                self.writer.add_scalar("val_NLL", val_nll, i)
                self.writer.add_scalar("val_KL", val_kl, i)

                val_mse_arr.append(val_mse)
                val_nll_arr.append(val_nll)
                val_kl_arr.append(val_kl)
            train_mse_arr.append(train_mse)
            train_nll_arr.append(train_nll)
            train_kl_arr.append(train_kl)
            self.train_dict = {
                "val": {"mse": val_mse_arr, "nll": val_nll_arr, "kl": val_kl_arr},
                "train": {
                    "mse": train_mse_arr,
                    "nll": train_nll_arr,
                    "kl": train_kl_arr,
                },
            }

            if self.best_mse is None or self.best_mse > val_mse:
                self.best_mse = val_mse
                self._checkpoint_model(i)

    def _checkpoint_model(self, epoch):
        checkpoint_path = f"{self.experiment_folder_path}/checkpoint_model_dict.pth"
        torch.save(
            {
                "epoch": epoch,
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "settings": self.model_settings,
                "train_res": self.train_dict,
                "optimizer": self.optimizer.state_dict(),
                "params": self.parameters,
            },
            checkpoint_path,
        )

    def save_model(self):
        model_path = f"{self.experiment_folder_path}/model_dict.pth"

        gru_dev_1_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "settings": self.model_settings,
            "train_res": self.train_dict,
        }

        torch.save(gru_dev_1_dict, model_path)
        print(f"Model saved at {model_path}")

    def profile_model(self):
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6),
            on_trace_ready=tensorboard_trace_handler(self.experiment_log_path),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as profiler:
            for step, (data, _) in enumerate(self.val_dataloader, 0):
                print("step:{}".format(step))
                data = data.cuda()

                logits = self.encoder(data, self.rel_rec, self.rel_send)
                edges = F.gumbel_softmax(
                    logits, tau=0.5, hard=True
                )  # RelaxedOneHotCategorical
                edge_probs = F.softmax(logits, dim=-1)

                pred_arr = self.decoder(
                    data.transpose(1, 2),
                    self.rel_rec,
                    self.rel_send,
                    edges,
                    burn_in=self.burn_in,
                    burn_in_steps=self.burn_in_steps,
                    split_len=self.split_len,
                )
                pred = pred_arr.transpose(1, 2).contiguous()
                target = data[:, :, 1:, :]

                loss_nll = torch_nll_gaussian(pred, target, 5e-5)
                loss_kl = kl_categorical(
                    preds=edge_probs, log_prior=self.log_prior, num_atoms=132
                )  # Here I chose theirs since my implementation runs out of RAM :(
                loss = loss_nll + loss_kl

                loss.backward()
                self.optimizer.step()
                profiler.step()
