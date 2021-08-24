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
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from ..utils.data_utils import create_dataloaders
from ..utils import encode_onehot
from ..utils import val, train, dnri_train, dnri_val, gumbel_tau_scheduler
from ..utils.losses import torch_nll_gaussian, kl_categorical, cyc_anneal
from ..utils.visual_utils import visualize_prob_adj
from ..utils.general_utils import count_parameters

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
        rnn_enc_n_hid=None,
        n_edge_types=2,
        dec_n_hid=16,
        dec_msg_hid=8,
        dec_gru_hid=8,
        skip_first=True,
        lr=0.001,
        lr_decay_step=100,
        lr_decay_gamma=0.5,
        fixed_adj_matrix_path=None,
        encoder_lr_frac=1,
        use_bn=True,
        init_weights=False,
        gumbel_tau=0.5,
        gumbel_hard=True,
        gumbel_anneal=None,
        weight_decay=0
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
        self.gumbel_tau = gumbel_tau
        self.gumbel_hard = gumbel_hard
        self.gumbel_anneal = gumbel_anneal
        self.weight_decay = weight_decay

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
            self.rnn_enc_n_hid = rnn_enc_n_hid
        elif self.encoder_type == "fixed":
            assert (
                fixed_adj_matrix_path is not None
            ), "fixed encoder need fixed adj matrix"
            self.fixed_adj_matrix = torch.tensor(np.load(fixed_adj_matrix_path))
            self.enc_n_in = self.node_f_dim
        self.enc_n_hid = enc_n_hid
        self.n_edge_types = n_edge_types
        self.init_weights = init_weights

        # Decoder
        self.dec_n_hid = dec_n_hid
        self.dec_msg_hid = dec_msg_hid
        self.dec_gru_hid = dec_gru_hid
        self.skip_first = skip_first

        # init model
        self._init_model()
        self.n_encoder_params = count_parameters(self.encoder)
        self.n_decoder_params = count_parameters(self.decoder)

        # save settings
        self.model_settings = {
            "node_f_dim": self.node_f_dim,
            "encoder_type": self.encoder_type,
            "enc_n_in": self.enc_n_in,
            "enc_n_hid": self.enc_n_hid,
            "enc_n_out": self.n_edge_types,
            "dec_n_hid": self.dec_n_hid,
            "dec_msg_hid": self.dec_msg_hid,
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
            "gumbel_tau": self.gumbel_tau,
            "gumbel_hard": self.gumbel_hard,
            "gumbel_anneal": self.gumbel_anneal,
            "weight_decay": self.weight_decay
        }

        # Save all parameters to txt file and add to tensorboard
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
            self.encoder = CNNEncoder(
                n_in=self.enc_n_in,
                n_hid=self.enc_n_hid,
                n_out=self.n_edge_types,
                do_prob=self.dropout_p,
                factor=self.encoder_factor,
                use_bn=self.use_bn,
                init_weights=self.init_weights
            ).cuda()
        elif self.encoder_type == "gru" or self.encoder_type == "lstm":
            self.encoder = RecurrentEncoder(
                n_in=self.enc_n_in,
                n_hid=self.enc_n_hid,
                n_out=self.n_edge_types,
                rnn_hid=self.rnn_enc_n_hid,
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
                gru_hid=self.dec_gru_hid,
                edge_types=self.n_edge_types,
                skip_first=self.skip_first,
            ).cuda()
        else:
            self.decoder = GRUDecoder_multistep(
                n_hid=self.dec_n_hid,
                f_in=self.node_f_dim,
                msg_hid=self.dec_msg_hid,
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

        self.optimizer = optim.Adam(self.model_params, weight_decay=self.weight_decay)
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

        print("save init graph")
        # self._save_graph_examples(-1)

        for epoch in range(self.n_epochs):

            if self.kl_cyc is not None:
                self.kl_frac = cyc_anneal(epoch, self.kl_cyc)
            
            if self.gumbel_anneal:
                # TODO Notice hard coded start tau
                self.gumbel_curr_tau = gumbel_tau_scheduler(2, self.gumbel_tau, epoch, self.n_epochs)
            else: 
                self.gumbel_curr_tau = self.gumbel_tau

            if self.encoder_type in ["gru", "lstm"]:
                train_mse, train_nll, train_kl, mean_edge_prob = dnri_train(
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
                    gumbel_tau=self.gumbel_curr_tau,
                    gumbel_hard=self.gumbel_hard,
                )
                mean_edge_prob = np.mean(mean_edge_prob, 0)
            else:
                train_mse, train_nll, train_kl, mean_edge_prob = train(
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
                    gumbel_tau=self.gumbel_curr_tau,
                    gumbel_hard=self.gumbel_hard,
                )

            self.lr_scheduler.step()
            self.writer.add_scalar("Train/MSE", train_mse, epoch)
            self.writer.add_scalar("Train/NLL", train_nll, epoch)
            self.writer.add_scalar("Train/KL", train_kl, epoch)
            self.writer.add_scalar("Train/tau", self.gumbel_curr_tau, epoch)
            self.writer.add_scalar("KL_frac", self.kl_frac, epoch)
            for i, prob in enumerate(mean_edge_prob):
                self.writer.add_scalar(f"Mean_edge_prob/train_{i}", prob, epoch)

            if epoch % 5 == 0:
                if self.encoder_type in ["gru", "lstm"]:
                    val_mse, val_nll, val_kl = dnri_val(
                        encoder=self.encoder,
                        decoder=self.decoder,
                        val_dataloader=self.val_dataloader,
                        rel_rec=self.rel_rec,
                        rel_send=self.rel_send,
                        burn_in=self.burn_in,
                        burn_in_steps=self.burn_in_steps,
                        split_len=self.split_len,
                        log_prior=self.log_prior,
                        n_nodes=n_nodes,
                    )
                    self._save_graph_examples_dnri(epoch) # Double check placement
                else:
                    val_mse, val_nll, val_kl, mean_edge_prob = val(
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
                        pred_steps=self.pred_steps,
                        n_nodes=n_nodes,
                    )
                    self._save_graph_examples(epoch) # Double check placement
                self.writer.add_scalar("Val/MSE", val_mse, epoch)
                self.writer.add_scalar("Val/NLL", val_nll, epoch)
                self.writer.add_scalar("Val/KL", val_kl, epoch)
                for i, prob in enumerate(mean_edge_prob):
                    self.writer.add_scalar(f"Mean_edge_prob/val_{i}", prob, epoch)



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

            if epoch % 5 == 0:
                if self.best_mse is None or self.best_mse >= val_mse:
                    self.best_mse = val_mse
                    self._checkpoint_model(epoch)
                print(
                    f"Epoch {epoch} Train MSE: {train_mse}, Val MSE: {val_mse}, Best MSE: {self.best_mse}, Checkpoint: {self.best_mse == val_mse}"
                )
            else:
                print(f"Epoch {epoch} Train MSE: {train_mse}")

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

    def _save_graph_examples(self, epoch):
        with torch.no_grad():
            # Calc edge probs
            _, (val_batch, _) = next(enumerate(self.val_dataloader))
            batch_subset = val_batch[:10].cuda()
            logits = self.encoder(batch_subset, self.rel_rec, self.rel_send)
            edge_probs = F.softmax(logits, dim=-1)
            
            # Create matrices
            adj_matrices = []
            for i in range(edge_probs.shape[0]):
                adj_matrices.append(
                    visualize_prob_adj(
                        edge_list=edge_probs[i],
                        rel_send=self.rel_send,
                        rel_rec=self.rel_rec,
                    )
                )
            adj_matrices = torch.stack(adj_matrices).unsqueeze(1)
            adj_matrices = torch.nn.functional.interpolate(
                adj_matrices, scale_factor=5, mode="nearest"
            ).squeeze()

            fig, axs = plt.subplots(5, 2, figsize=(10, 20))
            [axi.set_axis_off() for axi in axs.ravel()]
            for i in range(10):
                im = axs[i % 5][i % 2].imshow(adj_matrices[i])
                fig.colorbar(im, ax=axs[i % 5, i % 2])

            self.writer.add_figure(
                "adj_examples_test", fig, global_step=epoch, close=True
            )
        return

    def _save_graph_examples_dnri(self, epoch):
        with torch.no_grad():
            _, (val_batch, _) = next(enumerate(self.val_dataloader))
            batch_subset = val_batch[:2].cuda()
            _, posterior_logits, prior_state = self.encoder(
                batch_subset[:, :, :self.burn_in_steps, :], self.rel_rec, self.rel_send
            )
            burn_in_edges = F.gumbel_softmax(
                posterior_logits, tau=0.5, hard=True
            )  # RelaxedOneHotCategorical
            burn_in_edge_probs = F.softmax(posterior_logits, dim=-1)

            batch_subset = batch_subset.transpose(1,2)
            pred_all = []

            hidden = torch.autograd.Variable(
                torch.zeros(batch_subset.size(0), batch_subset.size(2), self.decoder.gru_hid)
            )
            edges = torch.autograd.Variable(
                torch.zeros(
                    burn_in_edges.size(0),
                    burn_in_edges.size(1),
                    batch_subset.size(1),
                    burn_in_edges.size(3),
                )
            )
            edge_probs = torch.autograd.Variable(
                torch.zeros(
                    burn_in_edges.size(0),
                    burn_in_edges.size(1),
                    batch_subset.size(1),
                    burn_in_edges.size(3),
                )
            )

            if batch_subset.is_cuda:
                hidden = hidden.cuda()
                edges = edges.cuda()
                edge_probs = edge_probs.cuda()

            edges[:, :, :self.burn_in_steps, :] = burn_in_edges
            edge_probs[:, :, :self.burn_in_steps, :] = burn_in_edge_probs

            for step in range(0, batch_subset.shape[1] - 1):
                if self.burn_in:
                    if step <= self.burn_in_steps - 1:
                        ins = batch_subset[
                            :, step, :, :
                        ]  # obs step different here to be time dim
                    else:
                        ins = pred_all[step - 1]
                        prior_logits, prior_state = self.encoder.single_step_forward(
                            ins, self.rel_rec, self.rel_send, prior_state
                        )
                        edges[:, :, step : step + 1, :] = F.gumbel_softmax(
                            prior_logits, tau=0.5, hard=True
                        )  # RelaxedOneHotCategorical
                        edge_probs[:, :, step : step + 1, :] = F.softmax(
                            prior_logits, dim=-1
                        )
                    
                pred, hidden = self.decoder.do_single_step_forward(
                    ins, self.rel_rec, self.rel_send, edges, hidden, step
                )
                pred_all.append(pred)

            # Create matrices
            adj_matrices = []
            for i in range(edge_probs.shape[0]):
                batch_adj_matrices = []
                for j in range(self.burn_in_steps-10, self.burn_in_steps+10):
                    batch_adj_matrices.append(
                        visualize_prob_adj(
                            edge_list=edge_probs[i, :, j, :],
                            rel_send=self.rel_send,
                            rel_rec=self.rel_rec,
                        )
                    )
                adj_matrices.append(torch.stack(batch_adj_matrices))
            adj_matrices = torch.stack(adj_matrices)
            adj_matrices = adj_matrices.reshape(-1, adj_matrices.shape[-2], adj_matrices.shape[-1])
            adj_matrices = adj_matrices.unsqueeze(1)
            
            adj_matrices = torch.nn.functional.interpolate(
                adj_matrices, scale_factor=5, mode="nearest"
            ).squeeze()
            
            adj_matrices = adj_matrices.reshape(2, -1, adj_matrices.shape[-2], adj_matrices.shape[-1])

            fig, axs = plt.subplots(20, 2, figsize=(10, 60))
            [axi.set_axis_off() for axi in axs.ravel()]
            for j in range(2):
                for i in range(20):
                    im = axs[i][j].imshow(adj_matrices[j, i])
                    fig.colorbar(im, ax=axs[i, j])

            self.writer.add_figure(
                "adj_examples_test", fig, global_step=epoch, close=True
            )

        return

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
