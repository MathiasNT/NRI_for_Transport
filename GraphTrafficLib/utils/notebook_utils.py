from statistics import mode
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from GraphTrafficLib.models.latent_graph import (
    MLPEncoder,
    GRUDecoder,
    FixedEncoder,
    LearnedAdjacancy,
    MLPEncoder_global,
    GRUDecoder_global,
    FixedEncoder_global,
    LearnedAdjacancy_global,
)
from GraphTrafficLib.utils.general_utils import encode_onehot
from GraphTrafficLib.utils.dataloader_utils import (
    create_dataloaders_taxi,
    create_dataloaders_bike,
    create_dataloaders_road,
)

# TODO: Go through notebook creation and remove unnecessary code.


def load_model(experiment_path, device, encoder_type, load_checkpoint=False):
    # Load the model settings and weights
    if load_checkpoint:
        model_dict = torch.load(
            f"{experiment_path}",
            map_location=torch.device(device),
        )
    else:
        model_dict = torch.load(f"{experiment_path}/model_dict.pth", map_location=device)
    model_settings = model_dict["settings"]
    train_res = model_dict["train_res"]

    # temp legacy fix TODO: fix
    if "node_f_dim" not in model_dict["settings"].keys():
        model_dict["settings"]["node_f_dim"] = model_dict["settings"]["dec_f_in"]

    if "decoder_f_dim" not in model_dict["settings"].keys():
        model_settings["decoder_f_dim"] = model_dict["settings"]["node_f_dim"]

    print(f"Model settings are: {model_settings}")

    # Init model
    if encoder_type == "mlp":
        if model_settings.get("use_global", False) or model_settings.get(
            "use_weather", False
        ):  # use_weather here for legacy reasons
            encoder = MLPEncoder_global(
                n_in=model_settings["enc_n_in"],
                n_in_global=model_settings["split_len"] * 2,
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
    elif encoder_type == "fixed":
        if model_settings.get("use_global", False) or model_settings.get("use_weather", False):
            encoder = FixedEncoder_global(adj_matrix=model_dict["encoder"]["adj_matrix"]).to(device)
        else:
            encoder = FixedEncoder(adj_matrix=model_dict["encoder"]["adj_matrix"]).to(device)
    elif encoder_type == "learned_adj":
        # hacky way to calc number of nodes - done for legacy reasons
        n_edges = model_dict["encoder"]["logits"].shape[1]
        n_nodes = int(max(np.roots([1, -1, -n_edges])))
        if model_settings.get("use_global", False) or model_settings.get("use_weather", False):
            encoder = LearnedAdjacancy_global(
                n_nodes=n_nodes, n_edge_types=model_settings["dec_edge_types"]
            )
        else:
            encoder = LearnedAdjacancy(
                n_nodes=n_nodes, n_edge_types=model_settings["dec_edge_types"]
            )

    if model_settings.get("use_global", False) or model_settings.get("use_weather", False):
        decoder = GRUDecoder_global(
            n_hid=model_settings["dec_n_hid"],
            f_in=model_settings["decoder_f_dim"],
            msg_hid=model_settings["dec_msg_hid"],
            gru_hid=model_settings["dec_gru_hid"],
            edge_types=model_settings["dec_edge_types"],
            skip_first=model_settings["skip_first"],
            do_prob=model_settings["dropout_p"],
        ).to(device)
    else:
        decoder = GRUDecoder(
            n_hid=model_settings["dec_n_hid"],
            f_in=model_settings["decoder_f_dim"],
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
        optimizer = optim.Adam(model_params, weight_decay=model_settings["weight_decay"])
        optimizer.load_state_dict(model_dict["optimizer"])

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.2,
            patience=15,
            threshold=0.001,
            min_lr=0.0000001,
            verbose=True,
        )
        lr_scheduler.load_state_dict(model_dict["lr_scheduler"])

    else:
        optimizer = None
        lr_scheduler = None

    print(f"loaded model at {experiment_path}")
    if "epoch" in model_dict.keys():
        print(f"Continuing from epoch {model_dict['epoch']}")

    return encoder, decoder, optimizer, lr_scheduler, model_settings, train_res


def load_data_taxi(
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
    weather_df = pd.read_csv(weather_data_path)
    # temp fix for na temp
    weather_df.loc[weather_df.temperature.isna(), "temperature"] = 0
    sum(weather_df.temperature.isna())
    # Create weather vector
    weather_vector = weather_df.loc[:, ("temperature", "precipDepth")].values
    weather_tensor = torch.Tensor(weather_vector)

    # Create time list
    min_date = pd.Timestamp(year=2019, month=1, day=1)
    max_date = min_date + timedelta(hours=data_tensor.shape[1])
    time_list = pd.date_range(start=min_date, end=max_date, freq="1H")[:-1]

    # Create data loader with max min normalization
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_mean,
        train_std,
    ) = create_dataloaders_taxi(
        data=data_tensor,
        weather_data=weather_tensor,
        split_len=split_len,
        batch_size=batch_size,
        normalize=normalize,
        train_frac=train_frac,
        time_list=time_list,
    )

    return (
        data_tensor,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_mean,
        train_std,
        time_list,
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

    (train_dataloader, val_dataloader, test_dataloader, mean, std,) = create_dataloaders_bike(
        x_data=x_data,
        y_data=y_data,
        weather_tensor=weather_tensor,
        batch_size=batch_size,
        normalize=normalize,
    )
    return data_tensor, train_dataloader, val_dataloader, test_dataloader, mean, std


def load_data_road(road_folder, batch_size, normalize, test_subset_size=None):

    train_data = np.load(f"{road_folder}/train_data.npy")
    val_data = np.load(f"{road_folder}/val_data.npy")
    test_data = np.load(f"{road_folder}/test_data.npy")

    if test_subset_size is not None:
        test_data = test_data[:test_subset_size, :, :, :]

    (train_dataloader, val_dataloader, test_dataloader, mean, std,) = create_dataloaders_road(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=batch_size,
        normalize=normalize,
    )

    return (
        train_data,
        val_data,
        test_data,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        mean,
        std,
    )


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
    subset_dim=None,
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
    for _, (data, weather, idxs) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            steps += len(data)
            data = data.to(device)

            if use_weather:
                weather = weather.cuda()
                logits = encoder(data[:, :, :burn_in_steps, :], weather, rel_rec, rel_send)
            else:
                logits = encoder(data[:, :, :burn_in_steps, :], rel_rec, rel_send)

            edge_probs = F.softmax(logits, dim=-1)
            if sample_graph:
                edges = F.gumbel_softmax(logits, tau=tau, hard=True)
                graph_probs.append(edge_probs.cpu())
            else:
                edges = edge_probs == edge_probs.max(dim=2, keepdims=True).values

            graph_list.append(edges.cpu())

            if subset_dim is not None:
                data = data[..., subset_dim].unsqueeze(-1)

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
            target_idxs = idxs[
                :,
            ]

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


def create_predictions_ha(
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
    time_list,
    subset_dim=None,
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
    for _, (data, weather, idxs) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            steps += len(data)
            data = data.to(device)

            if use_weather:
                weather = weather.cuda()
                logits = encoder(data[:, :, :burn_in_steps, :], weather, rel_rec, rel_send)
            else:
                logits = encoder(data[:, :, :burn_in_steps, :], rel_rec, rel_send)

            edge_probs = F.softmax(logits, dim=-1)
            if sample_graph:
                edges = F.gumbel_softmax(logits, tau=tau, hard=True)
                graph_probs.append(edge_probs.cpu())
            else:
                edges = edge_probs == edge_probs.max(dim=2, keepdims=True).values

            graph_list.append(edges.cpu())

            if subset_dim is not None:
                data = data[..., subset_dim].unsqueeze(-1)

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


def create_adj_vectors(n_nodes, device):
    # Generate off-diagonal interaction graph
    off_diag = np.ones([n_nodes, n_nodes]) - np.eye(n_nodes)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec).to(device)
    rel_send = torch.FloatTensor(rel_send).to(device)
    return rel_rec, rel_send


def create_lag1_and_ha_predictions(test_dataloader, burn_in, burn_in_steps, split_len, ha):
    y_true = []
    y_lag1 = []
    for i, (data, weather) in tqdm(enumerate(test_dataloader)):
        y_true.append(data[:, :, burn_in_steps, :].squeeze())
        y_lag1.append(data[:, :, burn_in_steps - 1, :].squeeze())

    y_true = torch.cat(y_true)
    y_lag1 = torch.cat(y_lag1).squeeze().cpu().detach()

    return y_lag1, y_true


def congested_hypothesis_check(cut_off_prob_in, adj_stack, yn_true, lower_mean_time):
    test_sparse_adj_stack = torch.clone(adj_stack)
    test_sparse_adj_stack[adj_stack < cut_off_prob_in] = 0

    n_edges = 0
    n_congested_senders = 0

    for ts in range(yn_true.shape[0]):
        congested_areas = np.where(lower_mean_time[ts])[0]

        for zone in congested_areas:
            zone_msg_senders = np.where(test_sparse_adj_stack[ts][zone] > 0)[0]
            senders_is_congested = np.isin(zone_msg_senders, congested_areas)

            n_edges += len(zone_msg_senders)
            n_congested_senders += senders_is_congested.sum()

    return n_edges, n_congested_senders, n_congested_senders / n_edges


def pems_hypothesis_check2(yn_true, hypothesis_bool, adj_stack):
    congested_probs = []
    uncongested_probs = []

    for ts in tqdm(range(yn_true.shape[0])):
        congested_areas = np.where(hypothesis_bool[ts])[0]
        for zone in congested_areas:
            zone_edge_probs_true = adj_stack[ts, zone, hypothesis_bool[ts]]
            zone_edge_probs_true = zone_edge_probs_true[zone_edge_probs_true.nonzero()]
            congested_probs.append(zone_edge_probs_true)

            zone_edge_probs_false = adj_stack[ts, zone, ~hypothesis_bool[ts]]
            zone_edge_probs_false = zone_edge_probs_false[zone_edge_probs_false.nonzero()]
            uncongested_probs.append(zone_edge_probs_false)

    return torch.cat(congested_probs), torch.cat(uncongested_probs)
