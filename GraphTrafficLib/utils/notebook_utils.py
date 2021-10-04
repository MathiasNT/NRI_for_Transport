import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from GraphTrafficLib.models.latent_graph import (
    MLPEncoder,
    GRUDecoder_multistep,
    CNNEncoder,
    FixedEncoder,
    RecurrentEncoder,
    MLPEncoder_weather,
    GRUDecoder_multistep_weather,
    CNNEncoder_weather,
    FixedEncoder_weather,
)
from GraphTrafficLib.models import SimpleLSTM
from GraphTrafficLib.utils import encode_onehot
from GraphTrafficLib.utils.data_utils import (
    create_test_train_split_max_min_normalize,
    create_dataloaders,
    create_dataloaders_bike,
)


def load_model(experiment_path, device, encoder_type, load_checkpoint=False):
    # Load the model settings and weights
    if load_checkpoint:
        model_dict = torch.load(
            f"{experiment_path}/checkpoint_model_dict.pth",
            map_location=torch.device(device),
        )
    else:
        model_dict = torch.load(
            f"{experiment_path}/model_dict.pth", map_location=device
        )
    model_settings = model_dict["settings"]
    train_res = model_dict["train_res"]

    # temp legacy fix TODO fix
    if "node_f_dim" not in model_dict["settings"].keys():
        model_dict["settings"]["node_f_dim"] = model_dict["settings"]["dec_f_in"]

    print(f"Model settings are: {model_settings}")

    # Init model
    if encoder_type == "mlp":
        if model_settings["use_weather"]:
            encoder = MLPEncoder_weather(
                n_in=model_settings["enc_n_in"],
                n_in_weather=model_settings["split_len"]
                * 2,  # hardcoded same as model init TODO fix
                n_hid=model_settings["enc_n_hid"],
                n_out=model_settings["enc_n_out"],
                do_prob=model_settings["dropout_p"],
                factor=model_settings["encoder_factor"],
                use_bn=model_settings["use_bn"],
            ).cuda()
        else:
            encoder = MLPEncoder(
                n_in=model_settings["enc_n_in"],
                n_hid=model_settings["enc_n_hid"],
                n_out=model_settings["enc_n_out"],
                do_prob=model_settings["dropout_p"],
                factor=model_settings["encoder_factor"],
                use_bn=model_settings["use_bn"],
            ).to(device)
    elif encoder_type == "cnn":
        if model_settings["use_weather"]:
            encoder = CNNEncoder_weather(
                n_in=model_settings["enc_n_in"],
                n_hid=model_settings["enc_n_hid"],
                n_out=model_settings["enc_n_out"],
                do_prob=model_settings["dropout_p"],
                factor=model_settings["encoder_factor"],
                use_bn=model_settings["use_bn"],
            ).cuda()
        else:
            encoder = CNNEncoder(
                n_in=model_settings["enc_n_in"],
                n_hid=model_settings["enc_n_hid"],
                n_out=model_settings["enc_n_out"],
                do_prob=model_settings["dropout_p"],
                factor=model_settings["encoder_factor"],
                use_bn=model_settings["use_bn"],
            ).to(device)
    elif encoder_type == "gru":
        if "rnn_n_hid" not in model_settings.keys():
            model_settings["rnn_hid"] = model_settings["enc_n_hid"]
        encoder = RecurrentEncoder(
            n_in=model_settings["enc_n_in"],
            n_hid=model_settings["enc_n_hid"],
            rnn_hid=model_settings["rnn_hid"],
            n_out=model_settings["enc_n_out"],
            do_prob=model_settings["dropout_p"],
            factor=model_settings["encoder_factor"],
        ).to(device)
    elif encoder_type == "fixed":
        if model_settings["use_weather"]:
            encoder = FixedEncoder_weather(
                adj_matrix=model_dict["encoder"]["adj_matrix"]
            ).to(device)
        else:
            encoder = FixedEncoder(adj_matrix=model_dict["encoder"]["adj_matrix"]).to(
                device
            )
    if model_settings["use_weather"]:
        decoder = GRUDecoder_multistep_weather(
            n_hid=model_settings["dec_n_hid"],
            f_in=model_settings["node_f_dim"],
            msg_hid=model_settings["dec_msg_hid"],
            gru_hid=model_settings["dec_gru_hid"],
            edge_types=model_settings["dec_edge_types"],
            skip_first=model_settings["skip_first"],
            do_prob=model_settings["dropout_p"],
        ).to(device)
    else:
        decoder = GRUDecoder_multistep(
            n_hid=model_settings["dec_n_hid"],
            f_in=model_settings["node_f_dim"],
            msg_hid=model_settings["dec_msg_hid"],
            gru_hid=model_settings["dec_gru_hid"],
            edge_types=model_settings["dec_edge_types"],
            skip_first=model_settings["skip_first"],
            do_prob=model_settings["dropout_p"],
        ).to(device)

    # Load trained weights
    encoder.load_state_dict(model_dict["encoder"])
    decoder.load_state_dict(model_dict["decoder"])

    if load_checkpoint:
        model_params = [
            {
                "params": encoder.parameters(),
                "lr": model_settings["encoder_lr_frac"] * model_settings["lr"],
            },
            {"params": decoder.parameters(), "lr": model_settings["lr"]},
        ]
        optimizer = optim.Adam(
            model_params, weight_decay=model_settings["weight_decay"]
        )
        optimizer.load_state_dict(model_dict["optimizer"])

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.2,
            patience=15,
            threshold=0.001,
            min_lr=0.0000001,
            verbose=True,
        )
        # lr_scheduler.load_state_dict(model_dict['lr_scheduler'])

    else:
        optimizer = None
        lr_scheduler = None

    print(f"loaded model at {experiment_path}")
    print(f"Continuing from epoch {model_dict['epoch']}")

    return encoder, decoder, optimizer, lr_scheduler, model_settings, train_res


def load_lstm_model(lstm_path, device):
    model_dict = torch.load(f"{lstm_path}/model_dict.pth", map_location=device)
    model_settings = model_dict["settings"]
    train_res = model_dict["train_res"]

    lstm = SimpleLSTM(
        input_dims=1,
        hidden_dims=model_settings["lstm_hid"],
        dropout=model_settings["lstm_dropout"],
    ).to(device)
    lstm.load_state_dict(model_dict["model"])
    return lstm, model_settings, train_res


def load_data(
    data_path,
    weather_data_path,
    split_len,
    batch_size,
    normalize,
    train_frac,
    dropoff_data_path=None,
):
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
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_max,
        train_min,
    ) = create_dataloaders(
        data=data_tensor,
        weather_data=weather_tensor,
        split_len=split_len,
        batch_size=batch_size,
        normalize=normalize,
        train_frac=train_frac,
    )

    return (
        data_tensor,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_max,
        train_min,
    )


def load_data_bike(
    bike_folder_path,
    split_len,
    batch_size,
    normalize,
):
    x_data = torch.load(f"{bike_folder_path}/nyc_bike_cgc_x_standardised")
    y_data = torch.load(f"{bike_folder_path}/nyc_bike_cgc_y_standardised")
    data_tensor = torch.as_tensor(
        torch.load(f"{bike_folder_path}/standard_preprocessed_NYC_bike")
    ).permute(1, 0, 2)

    # load weather data
    weather_df = pd.read_csv(f"{bike_folder_path}/bike_weather.csv")
    # temp fix for na temp
    weather_df.loc[weather_df.temperature.isna(), "temperature"] = 0
    assert sum(weather_df.temperature.isna()) == 0
    # Create weather vector
    weather_vector = weather_df.loc[:, ("temperature", "precipDepth")].values
    weather_tensor = torch.Tensor(weather_vector)

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        mean,
        std,
    ) = create_dataloaders_bike(
        x_data=x_data,
        y_data=y_data,
        weather_tensor=weather_tensor,
        batch_size=batch_size,
        normalize=normalize,
    )
    return data_tensor, train_dataloader, val_dataloader, test_dataloader, mean, std


# This here is actually my old dataloader currently kept for legacy reasons
def load_data2(
    data_path,
    weather_data_path,
    split_len,
    batch_size,
    normalize,
    train_frac,
    dropoff_data_path=None,
):

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
        train_dataloader,
        test_dataloader,
        train_max,
        train_min,
    ) = create_test_train_split_max_min_normalize(
        data=data_tensor,
        weather_data=weather_tensor,
        split_len=split_len,
        batch_size=batch_size,
        normalize=normalize,
        train_frac=train_frac,
    )

    min_date = pd.Timestamp(year=2019, month=1, day=1)
    max_date = pd.Timestamp(year=2019 + 1, month=1, day=1)

    # Note that this misses a bit from the beginning but this will not be a big problem when we index finer
    bins_dt = pd.date_range(start=min_date, end=max_date, freq="1H")
    split_bins_dt = bins_dt[: -(split_len + 1)]

    test_dates = split_bins_dt[int(train_frac * len(split_bins_dt)) :]
    train_dates = split_bins_dt[: int(train_frac * len(split_bins_dt))]

    print(f"train_dates len: {len(train_dates)}")
    print(f"test_dates len: {len(test_dates)}")

    return data_tensor, train_dataloader, test_dataloader, train_max, train_min


def plot_training(train_res, test_res, graph_model=True):
    n_res = len(train_res["mse"])
    if n_res <= 5:
        return

    test_x = np.arange(0, n_res, n_res // len(test_res["mse"]))

    if graph_model:
        fig, axs = plt.subplots(3, figsize=(25, 13))
        axs[0].plot(train_res["mse"][1:])
        axs[0].plot(test_x, test_res["mse"])
        axs[0].title.set_text("MSE")

        axs[1].plot(train_res["nll"][1:])
        axs[1].plot(test_x, test_res["nll"])
        axs[1].title.set_text("NLL")

        axs[2].plot(train_res["kl"][1:])
        axs[2].plot(test_x, test_res["kl"])
        axs[2].title.set_text("KL")

        # axs[3].plot(train_acc_arr[5:])
        # axs[3].title.set_text("Edge Acc")
    else:
        fig, axs = plt.subplots(1, figsize=(25, 13))
        axs.plot(train_res["mse"][1:])
        axs.plot(test_x, test_res["mse"])
        axs.title.set_text("MSE")
    plt.show()


def create_predictions(
    encoder,
    decoder,
    test_dataloader,
    rel_rec,
    rel_send,
    burn_in,
    burn_in_steps,
    split_len,
    use_weather,
    sample_graph,
    device,
    tau,
):
    y_true = []
    y_pred = []
    graph_list = []
    graph_probs = []
    mse = 0
    pred_steps = split_len - burn_in_steps
    steps = 0
    encoder.eval()
    decoder.eval()
    for _, (data, weather) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            steps += len(data)
            data = data.to(device)

            if use_weather:
                weather = weather.cuda()
                logits = encoder(
                    data[:, :, :burn_in_steps, :], weather, rel_rec, rel_send
                )
            else:
                logits = encoder(data[:, :, :burn_in_steps, :], rel_rec, rel_send)

            edge_probs = F.softmax(logits, dim=-1)
            if sample_graph:
                edges = F.gumbel_softmax(logits, tau=tau, hard=True)
                graph_probs.append(edge_probs.cpu())
            else:
                edges = edge_probs == edge_probs.max(dim=2, keepdims=True).values

            graph_list.append(edges.cpu())

            if use_weather:
                pred_arr = decoder(
                    data.transpose(1, 2),
                    weather,
                    rel_rec,
                    rel_send,
                    edges,
                    burn_in=burn_in,
                    burn_in_steps=burn_in_steps,
                    split_len=split_len,
                )
            else:
                pred_arr = decoder(
                    data.transpose(1, 2),
                    rel_rec,
                    rel_send,
                    edges,
                    burn_in=burn_in,
                    burn_in_steps=burn_in_steps,
                    split_len=split_len,
                )
            pred = pred_arr.transpose(1, 2).contiguous()
            target = data[:, :, 1:, :]

            y_true.append(target)
            y_pred.append(pred)

            rmse_pred = pred[:, :, -pred_steps:, :]
            rmse_target = data[:, :, -pred_steps:, :]
            mse += F.mse_loss(rmse_pred, rmse_target).item() * len(data)

    y_true = torch.cat(y_true).cpu().detach()
    y_pred = torch.cat(y_pred).squeeze().cpu().detach()
    mse = mse / steps
    rmse = mse ** 0.5
    return y_pred, y_true, mse, rmse


def create_predictions_gru(
    encoder,
    decoder,
    test_dataloader,
    rel_rec,
    rel_send,
    burn_in,
    burn_in_steps,
    split_len,
    sample_graph,
    device,
):
    y_true = []
    y_pred = []
    graph_list = []
    graph_probs = []
    encoder.eval()
    decoder.eval()
    print(decoder)
    for _, (data, _) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            data = data.to(device)

            _, posterior_logits, prior_state = encoder(
                data[:, :, :burn_in_steps, :], rel_rec, rel_send
            )
            burn_in_edges = F.gumbel_softmax(
                posterior_logits, tau=0.5, hard=True
            )  # RelaxedOneHotCategorical
            burn_in_edge_probs = F.softmax(posterior_logits, dim=-1)

            data = data.transpose(1, 2)
            pred_all = []

            hidden = torch.autograd.Variable(
                torch.zeros(data.size(0), data.size(2), decoder.gru_hid)
            )
            edges = torch.autograd.Variable(
                torch.zeros(
                    burn_in_edges.size(0),
                    burn_in_edges.size(1),
                    data.size(1),
                    burn_in_edges.size(3),
                )
            )
            edge_probs = torch.autograd.Variable(
                torch.zeros(
                    burn_in_edges.size(0),
                    burn_in_edges.size(1),
                    data.size(1),
                    burn_in_edges.size(3),
                )
            )

            if data.is_cuda:
                hidden = hidden.cuda()
                edges = edges.cuda()
                edge_probs = edge_probs.cuda()

            edges[:, :, :burn_in_steps, :] = burn_in_edges
            edge_probs[:, :, :burn_in_steps, :] = burn_in_edge_probs

            for step in range(0, data.shape[1] - 1):
                if burn_in:
                    if step <= burn_in_steps - 1:
                        ins = data[
                            :, step, :, :
                        ]  # obs step different here to be time dim
                    else:
                        ins = pred_all[step - 1]
                        prior_logits, prior_state = encoder.single_step_forward(
                            ins, rel_rec, rel_send, prior_state
                        )
                        edges[:, :, step : step + 1, :] = F.gumbel_softmax(
                            prior_logits, tau=0.5, hard=True
                        )  # RelaxedOneHotCategorical
                        edge_probs[:, :, step : step + 1, :] = F.softmax(
                            prior_logits, dim=-1
                        )

                pred, hidden = decoder.do_single_step_forward(
                    ins, rel_rec, rel_send, edges, hidden
                )
                pred_all.append(pred)

            pred_arr = torch.stack(pred_all, dim=1)

            pred = pred_arr.transpose(1, 2).contiguous()
            target = data[:, :, 1:, :]

            y_true.append(target[:, :, burn_in_steps - 1, :].cpu().squeeze())
            y_pred.append(pred[:, :, burn_in_steps - 1, :].cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred).squeeze().cpu().detach()
    return y_pred, y_true


def create_lstm_predictions(model, test_dataloader, burn_in_steps, split_len):
    y_true = []
    y_pred = []
    model.eval()
    for i, (data, weather) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            data = data.cuda()
            burn_in_data = data[:, :, :burn_in_steps, :].reshape(-1, burn_in_steps, 1)
            target = data[:, :, (burn_in_steps):, :]
            pred = model(x=burn_in_data, pred_steps=split_len - burn_in_steps).reshape(
                target.shape
            )

            y_true.append(target[:, :, 0, :].cpu().squeeze())
            y_pred.append(pred[:, :, 0, :].cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred).squeeze().cpu().detach()
    return y_pred, y_true


def create_adj_vectors(n_nodes, device):
    # Generate off-diagonal interaction graph
    off_diag = np.ones([n_nodes, n_nodes]) - np.eye(n_nodes)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec).to(device)
    rel_send = torch.FloatTensor(rel_send).to(device)
    return rel_rec, rel_send


def create_lag1_and_ha_predictions(
    test_dataloader, burn_in, burn_in_steps, split_len, ha
):
    y_true = []
    y_lag1 = []
    for i, (data, weather) in tqdm(enumerate(test_dataloader)):
        y_true.append(data[:, :, burn_in_steps, :].squeeze())
        y_lag1.append(data[:, :, burn_in_steps - 1, :].squeeze())

    y_true = torch.cat(y_true)
    y_lag1 = torch.cat(y_lag1).squeeze().cpu().detach()

    return y_lag1, y_true
