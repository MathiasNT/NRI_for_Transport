import sys
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import imageio



import torch
import torch.nn.functional as F


from IPython.display import clear_output
from IPython.utils import io

from GraphTrafficLib.models.latent_graph import MLPEncoder, GRUDecoder_multistep
from GraphTrafficLib.utils import encode_onehot, plot_training
from GraphTrafficLib.utils import Encoder_Visualizer, visualize_all_graph_adj, visualize_mean_graph_adj, visualize_continous_adj
from GraphTrafficLib.utils.notebook_utils import load_model, load_data, plot_training, create_predictions, create_adj_vectors, load_lstm_model, create_lstm_predictions, load_data_bike
from GraphTrafficLib.utils.data_utils import create_test_train_split_max_min_normalize, renormalize_data, restandardize_data
from GraphTrafficLib.utils.training_utils import val_lstm, val
from GraphTrafficLib.utils.visual_utils import get_rels_from_topk, plot_top_bot_k_rels, plot_adj_w_grid, plot_zone_and_map, plot_adj_and_time, get_rels_from_zone_id, merge_on_and_off_diagonal


if __name__ == "__main__":
    torch.cuda.set_device(1)
    gpu = torch.cuda.current_device()

    
    # # Select experiment
    
    # SELECT PATHS HERE


    #experiment_path = "../models/bike_models/mlp_lr0.01_s42/"
    #experiment_path = "../models/bike_models/mlp_lr0.001_s42/"
    experiment_path = "../models/bike_models/mlp_lr0.005_s42/"

    dataset_folder = "../datafolder"
    dataset_path = f"{dataset_folder}/procdata/bike_data/"
    weather_data_path = f"bike_data/bike_weather.csv"

    
    # ### Load model


    device = torch.device(gpu)
    encoder, decoder, optimizer, lr_scheduler, model_settings, train_res = load_model(experiment_path, device, "mlp")

    
    # ### Load data


    x_data = torch.load(f"{dataset_path}/nyc_bike_cgc_x_standardised")
    y_data = torch.load(f"{dataset_path}/nyc_bike_cgc_y_standardised")




    data_tensor, train_dataloader, val_dataloader, test_dataloader, mean, std = load_data_bike(bike_folder_path=dataset_path,
                                                                                split_len=model_settings['split_len'],
                                                                                batch_size=model_settings['batch_size'],
                                                                                normalize=model_settings['normalize'])

    n_nodes = data_tensor.shape[0]
    print(f"data tensor shape: {data_tensor.shape}")

    #train_dataloader = train_dataloader.to(device)
    #test_dataloader = test_dataloader.to(device)

    rel_rec, rel_send = create_adj_vectors(data_tensor.shape[0], device)



    min_date = pd.Timestamp(year=2016, month=4, day=1)
    max_date = pd.Timestamp(year=2016, month=7, day=1)
        
    # Note that this misses a bit from the beginning but this will not be a big problem when we index finer
    bins_dt = pd.date_range(start=min_date, end=max_date, freq="30min")
    split_bins_dt = bins_dt[12:-12]

    train_dates = split_bins_dt[:3001]
    val_dates = split_bins_dt[3001:-672]
    test_dates = split_bins_dt[-672:]


    print(f"train_dates len: {len(train_dates)}")
    print(f"val_dates len: {len(val_dates)}")
    print(f"test_dates len: {len(test_dates)}")

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    weekday_strs = [weekdays[test_dates[t].weekday()] for t in range(len(test_dates))]

    print(f"{test_dates[17]}, {weekdays[test_dates[17].weekday()]}")
    print(f"{test_dates[17+168]}, {weekdays[test_dates[17+168].weekday()]}")

    
    # ### Load LSTM baseline


    # TODO FIX LSTM STUFF HERE, when I have model
    #lstm, model_settings, train_res_lstm = load_lstm_model(lstm_path, device)

    
    # # Create and analyze predictions
    
    # ### Create predictions from model

    # y_pred, y_true, rmse, multi_step_pred = create_predictions(encoder=encoder,
    #                                         decoder=decoder,
    #                                         test_dataloader=test_dataloader,
    #                                         rel_rec=rel_rec,
    #                                         rel_send=rel_send,
    #                                         burn_in=model_settings['burn_in'],
    #                                         burn_in_steps=model_settings['burn_in_steps'],
    #                                         split_len=model_settings['split_len'],
    #                                         sample_graph=True,
    #                                         device=device,
    #                                         use_weather=model_settings['use_weather'],
    #                                         tau=0.05
    #                                         )



    y_pred_argmax, y_true_argmax, rmse_argmax, multi_step_pred_argmax = create_predictions(encoder=encoder,
                                                                                        decoder=decoder,
                                                                                        test_dataloader=test_dataloader,
                                                                                        rel_rec=rel_rec,
                                                                                        rel_send=rel_send,
                                                                                        burn_in=model_settings['burn_in'],
                                                                                        burn_in_steps=model_settings['burn_in_steps'],
                                                                                        split_len=model_settings['split_len'],
                                                                                        sample_graph=False,
                                                                                        device=device,
                                                                                        use_weather=model_settings['use_weather'],
                                                                                        tau=None,
                                                                                    )

    y_pred_argmax2, y_true_argmax2, rmse_argmax2, multi_step_pred_argmax2 = create_predictions(encoder=encoder,
                                                                                    decoder=decoder,
                                                                                    test_dataloader=test_dataloader,
                                                                                    rel_rec=rel_rec,
                                                                                    rel_send=rel_send,
                                                                                    burn_in=model_settings['burn_in'],
                                                                                    burn_in_steps=model_settings['burn_in_steps'],
                                                                                    split_len=model_settings['split_len'],
                                                                                    sample_graph=False,
                                                                                    device=device,
                                                                                    use_weather=model_settings['use_weather'],
                                                                                    tau=None,
                                                                                )
    # ### Create predictions from baseline


    print("test")



    # TODO create LSTM for comparison
    # y_pred_lstm, y_true_lstm = create_lstm_predictions(model=lstm, test_dataloader=test_dataloader, burn_in_steps=30, split_len=40)

    
    # ### Compare predictions


    yn_true = restandardize_data(data=y_true, data_mean=mean, data_std=std)
    yn_pred = restandardize_data(data=y_pred, data_mean=mean, data_std=std)
    #yn_pred_lstm = renormalize_data(y_pred_lstm, train_min, train_max)
    #yn_true_lstm = renormalize_data(y_true_lstm, train_min, train_max)
    yn_true_argmax = restandardize_data(data=y_true_argmax, data_mean=mean, data_std=std)
    yn_pred_argmax = restandardize_data(data=y_pred_argmax, data_mean=mean, data_std=std)



    # TODO This needs to be updated to use val
    print(f"last GNN test loss:{train_res['val']['mse'][-1]}")
    #print(f"last LSTM test loss:{train_res_lstm['val']['mse'][-1]}")



    print(f"sample: {rmse*std}")
    print(f"argmax: {rmse_argmax*std}")



    rmse



    y_pred.shape



    torch.sqrt(F.mse_loss(y_pred, y_true))



    print('\nGNN model')
    print(f"normalized")
    print(f"MSE: {F.mse_loss(y_pred, y_true)}")
    print(f"L1: {F.l1_loss(y_pred, y_true)}")
    print(f"rescaled")
    print(f"MSE: {F.mse_loss(yn_pred, yn_true)}")
    print(f"L1: {F.l1_loss(yn_pred, yn_true)}")
    print('\nGNN argmax model')
    print(f"normalized")
    print(f"MSE: {F.mse_loss(y_pred_argmax, y_true)}")
    print(f"L1: {F.l1_loss(y_pred_argmax, y_true)}")
    print(f"rescaled")
    print(f"MSE: {F.mse_loss(yn_pred_argmax, yn_true)}")
    print(f"L1: {F.l1_loss(yn_pred_argmax, yn_true)}")
    # print('\nLSTM model')
    # print(f"normalized")
    # print(f"MSE: {F.mse_loss(y_pred_lstm, y_true_lstm)}")
    # print(f"L1: {F.l1_loss(y_pred_lstm, y_true_lstm)}")
    # print(f"rescaled")
    # print(f"MSE: {F.mse_loss(yn_pred_lstm, yn_true_lstm)}")
    # print(f"L1: {F.l1_loss(yn_pred_lstm, yn_true_lstm)}")

    
    # ### Visual plot of results


    time_slice = slice(0,1*168)
    fig, axs = plt.subplots(20, figsize=(25,100))
    for i in range(20):
        axs[i].plot(yn_true.squeeze()[time_slice,i, 0], color='tab:green')
        axs[i].plot(yn_pred[time_slice,i, 0], color='tab:blue')
        #axs[i].plot(yn_pred_lstm[time_slice,i, 0], color='tab:orange')
        
        axs[i].plot(yn_true.squeeze()[time_slice,i, 1], '--', color='tab:green')
        axs[i].plot(yn_pred[time_slice,i, 1], '--', color='tab:blue')
        #axs[i].plot(yn_pred_lstm[time_slice,i, 1], '--', color='tab:orange')
        
        gnn_mse = F.mse_loss(yn_pred[time_slice,i], yn_true.squeeze()[time_slice,i])
        #lstm_mse = F.mse_loss(yn_pred_lstm[time_slice,i], yn_true.squeeze()[time_slice,i])
        
        axs[i].legend([f'True {i}', f'GNN {gnn_mse:.2f}'])

    
    # # Analyze learned graphs
    
    # ### Edge prob timeseries


    encoder_visualizer = Encoder_Visualizer(encoder=encoder,
                                            rel_rec=rel_rec,
                                            rel_send=rel_send,
                                            burn_in=model_settings['burn_in'],
                                            burn_in_steps=model_settings['burn_in_steps'],
                                            split_len=model_settings['split_len'],
                                            use_weather=model_settings['use_weather'])

    temp = test_dataloader.dataset[:]
    full_data, full_weather = temp[0].contiguous(), temp[1]
    zipped_data = zip(full_data, full_weather)



    full_graph_list, full_graph_probs = encoder_visualizer.infer_graphs(data=zipped_data, gumbel_temp=1e-10)



    full_probs_stack = torch.stack(full_graph_probs).squeeze()



    off_diag = full_probs_stack[:,:,1].reshape(len(full_probs_stack), 250, 249)
    on_diag = torch.zeros(len(full_probs_stack), 250)



    full_adj_stack = merge_on_and_off_diagonal(on_diag, off_diag)



    plt.figure(figsize=(30,15))
    plt.plot(full_data[:,10,12,0])
    for i in range(12):
        time = 45 + i * 48
        plt.axvline(x=time, color="red", alpha=0.5)
        print(f"{test_dates[time]}")



    # From samples

    plt.figure(figsize=(30,15))
    plt.plot(torch.stack(full_graph_list).mean(dim=2).mean(dim=1)[:,1])
    for i in range(12):
        time = 45 + i * 48
        plt.axvline(x=time, color="red", alpha=0.5)
        print(f"{test_dates[time]}")



    fig, ax = plt.subplots(1,figsize=(10,10))
    ax.imshow(full_adj_stack[1])
    ax.set_title("test")



    get_ipython().run_line_magic('matplotlib', 'notebook')
    gif_name = f'{experiment_path}{"adj_ts.gif"}'
    if not os.path.exists(gif_name):
        im_arr = []
        start_date_idx = 0

        for t in tqdm(range(672)):
            with io.capture_output() as captured:
                cur_fig = plot_adj_and_time(full_adj_stack[start_date_idx + t], f"{test_dates[start_date_idx + t]}, a {weekday_strs[start_date_idx + t]}")
            cur_fig.canvas.draw()
            cur_im = cur_fig.canvas.buffer_rgba()
            im_arr.append(cur_im)
            #plt.close()

        imageio.mimsave(gif_name, im_arr, fps=2)


