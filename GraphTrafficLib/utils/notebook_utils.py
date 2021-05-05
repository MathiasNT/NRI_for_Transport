import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from GraphTrafficLib.models.latent_graph import MLPEncoder, GRUDecoder_multistep, CNNEncoder
from GraphTrafficLib.models import SimpleLSTM
from GraphTrafficLib.utils import encode_onehot


def load_model(experiment_path, device, encoder_type):
    # Load the model
    model_dict = torch.load(f"{experiment_path}/model_dict.pth", map_location=device)
    model_settings = model_dict["settings"]
    train_res = model_dict["train_res"]
    print(f"Model settings are: {model_settings}")

    dropout_p = 0
    encoder_factor = True
    if encoder_type == "mlp":
        encoder = MLPEncoder(
            n_in=model_settings["enc_n_in"],
            n_hid=model_settings["enc_n_hid"],
            n_out=model_settings["enc_n_out"],
            do_prob=dropout_p,
            factor=encoder_factor,
        ).to(device)
    elif encoder_type == "cnn":
        encoder = CNNEncoder(
            n_in=model_settings["enc_n_in"],
            n_hid=model_settings["enc_n_hid"],
            n_out=model_settings["enc_n_out"],
            do_prob=dropout_p,
            factor=encoder_factor,
        ).to(device)
    
    decoder = GRUDecoder_multistep(
        n_hid=model_settings["dec_n_hid"],
        f_in=model_settings["dec_f_in"],
        msg_hid=model_settings["dec_msg_hid"],
        msg_out=model_settings["dec_msg_out"],
        gru_hid=model_settings["dec_gru_hid"],
        edge_types=model_settings["dec_edge_types"],
    ).to(device)
    encoder.load_state_dict(model_dict["encoder"])
    decoder.load_state_dict(model_dict["decoder"])
    return encoder, decoder, model_settings, train_res


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


def load_data(dataset_folder, zone):
    raw_folder = f"{dataset_folder}/rawdata"
    proc_folder = f"{dataset_folder}/procdata"

    # Load data
    pickup_data_path = f"{proc_folder}/full_year_{zone}_vector_pickup.npy"
    pickup_data = np.load(pickup_data_path)
    dropoff_data_path = f"{proc_folder}/full_year_{zone}_vector_dropoff.npy"
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
    return data_tensor, weather_tensor


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
    sample_graph,
    device,
):
    y_true = []
    y_pred = []
    graph_list = []
    graph_probs = []
    encoder.eval()
    decoder.eval()
    for i, (data, weather) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            data = data.to(device)

            logits = encoder(data, rel_rec, rel_send)
            edge_probs = F.softmax(logits, dim=-1)
            if sample_graph:
                edges = F.gumbel_softmax(logits, tau=1e-10, hard=True)
                graph_probs.append(edge_probs.cpu())
            else:
                edges = edge_probs == edge_probs.max(dim=2, keepdims=True).values

            graph_list.append(edges.cpu())

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