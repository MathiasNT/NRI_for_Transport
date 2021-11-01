"""
Dataloader stuff
"""
from torch.utils.data import DataLoader
import torch
from .Nyc_w_Weather import Nyc_w_Weather, Nyc_w_Weather2, Nyc_no_Weather2
from torch.utils.data.dataset import TensorDataset, Dataset
import numpy as np
from .data_preprocess import get_ha_normalization_matrices, ha_batch_renormalization, ha_normalization, ha_renormalization

def create_dataloaders(
    data,
    weather_data,
    split_len,
    batch_size,
    time_list,
    normalize=False,
    train_frac=0.8,
    fixed_max=None,
    fixed_min=None,
):
    if len(data.shape) == 2:
        demand_tensor = torch.Tensor(data).permute(1, 0).unsqueeze(-1)
    else:
        demand_tensor = torch.Tensor(data).permute(1, 0, 2)

    weather_tensor = torch.Tensor(weather_data)

    # Create an index list to save which indexes the split covers.
    index_tensor = torch.arange(len(demand_tensor))

    # Normalize data
    if normalize == "z":
        train_mean = demand_tensor[: int(train_frac * len(demand_tensor))].mean()
        train_std = demand_tensor[: int(train_frac * len(demand_tensor))].std()
         
        weather_train_mean = weather_tensor[: int(train_frac * len(weather_tensor))].mean()
        weather_train_std = weather_tensor[: int(train_frac * len(weather_tensor))].std()

        demand_tensor = (demand_tensor - train_mean) / train_std
        weather_tensor = (weather_tensor - weather_train_mean) / weather_train_std
        
    elif normalize == "ha":
        train_index = int(train_frac * len(demand_tensor))
        train_mean, train_std = get_ha_normalization_matrices(demand_tensor[:train_index], time_list[:train_index])
        # Fix for zone/day/hour combinations with only zero observation in train but observations in the val/test
        train_std[train_std == 0] = 1
        demand_tensor = ha_normalization(demand_tensor, time_list, mean_matrix=train_mean, std_matrix=train_std)

        weather_train_mean = weather_tensor[: int(train_frac * len(weather_tensor))].mean()
        weather_train_std = weather_tensor[: int(train_frac * len(weather_tensor))].std()
        weather_tensor = (weather_tensor - weather_train_mean) / weather_train_std


    splits = []
    weather_splits = []
    index_splits = []
    for i in range(demand_tensor.shape[0] - split_len):
        splits.append(demand_tensor[i : i + split_len])
        weather_splits.append(weather_tensor[i : i + split_len])
        index_splits.append(index_tensor[i: i + split_len])
    splits = torch.stack(splits).transpose(1, 2)
    weather_splits = torch.stack(weather_splits)
    index_splits = torch.stack(index_splits)

    train_end_id = int(train_frac * len(splits))
    val_end_id = int((train_frac + (1 - train_frac) / 2) * len(splits))

    train_splits = splits[:train_end_id]
    val_splits = splits[train_end_id:val_end_id]
    test_splits = splits[val_end_id:]

    print(f"train splits shape: {train_splits.shape}")
    print(f"val splits shape: {val_splits.shape}")
    print(f"test splits shape: {test_splits.shape}")

    train_weather = weather_splits[:train_end_id]
    val_weather = weather_splits[train_end_id:val_end_id]
    test_weather = weather_splits[val_end_id:]

    train_idxs = index_splits[:train_end_id]
    val_idxs = index_splits[train_end_id:val_end_id]
    test_idxs = index_splits[val_end_id:]


    train_dataset = TensorDataset(train_splits, train_weather, train_idxs)
    val_dataset = TensorDataset(val_splits, val_weather, val_idxs)
    test_dataset = TensorDataset(test_splits, test_weather, test_idxs)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=2,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
    )

    return (train_dataloader, val_dataloader, test_dataloader, train_mean, train_std)

def create_dataloaders_bike(x_data, y_data, weather_tensor, batch_size, normalize):
    full_data = torch.cat([x_data, y_data], dim=1)
    full_data = full_data.permute(0, 2, 1, 3)
    train_data = full_data[:3001, :, :]
    val_data = full_data[3001:-672, :, :]
    test_data = full_data[-672:, :, :]

    weather_train_max = weather_tensor[:3001].amax(dim=0)

    weather_train_min = weather_tensor[:3001].amin(dim=0)

    if normalize:
        weather_tensor = (weather_tensor - weather_train_min) / (
            weather_train_max - weather_train_min
        )

    # Code from https://github.com/Essaim/CGCDemandPrediction to make sure it matches their data

        # TODO ADD IDXS SPLITS HERE

    X_list = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    Y_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    X_, Y_ = list(), list()
    index_X_, index_Y_ = list(), list()
    weather_tensor = weather_tensor.numpy()
    index_tensor = torch.arange(weather_tensor.shape[0])
    for i in range(max(X_list), weather_tensor.shape[0] - max(Y_list)):
        X_.append([weather_tensor[i - j] for j in X_list])
        Y_.append([weather_tensor[i + j] for j in Y_list])
        index_X_.append([index_tensor[i - j] for j in X_list])
        index_Y_.append([index_tensor[i + j] for j in Y_list])
    X_ = torch.from_numpy(np.asarray(X_)).float()
    Y_ = torch.from_numpy(np.asarray(Y_)).float()
    index_X_ = torch.from_numpy(np.asarray(index_X_)).float()
    index_Y_ = torch.from_numpy(np.asarray(index_Y_)).float()

    weather_tensor = torch.cat([X_, Y_], dim=1)
    train_weather = weather_tensor[:3001, :, :]
    val_weather = weather_tensor[3001:-672, :, :]
    test_weather = weather_tensor[-672:, :, :]

    index_tensor = torch.cat([index_X_, index_Y_], dim=1)
    train_index = index_tensor[:3001, :]
    val_index = index_tensor[3001:-672, :]
    test_index = index_tensor[-672:, :]

    # packing on dummy weather to make the data fit rest of code
    train_dataset = TensorDataset(train_data, train_weather, train_index)
    val_dataset = TensorDataset(val_data, val_weather, val_index)
    test_dataset = TensorDataset(test_data, test_weather, test_index)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # mean and std values are grabbed from https://github.com/Essaim/CGCDemandPrediction
    mean = 2.7760608974358973
    std = 4.010208024889097
    return (train_dataloader, val_dataloader, test_dataloader, mean, std)


def create_dataloaders_road(train_data, val_data, test_data, batch_size, normalize):
    # TODO think about this
    # Currently I ignore the timestamp data
    train_data = torch.Tensor(train_data[..., :1]).permute(0,2,1,3)
    val_data = torch.Tensor(val_data[..., :1]).permute(0,2,1,3)
    test_data = torch.Tensor(test_data[..., :1]).permute(0,2,1,3)

    if normalize:
         train_mean = train_data.mean()
         train_std = train_data.std()

         train_data = (train_data - train_mean) / train_std
         val_data = (val_data - train_mean) / train_std
         test_data = (test_data - train_mean) / train_std

    train_weather = torch.zeros_like(train_data)
    val_weather = torch.zeros_like(val_data)
    test_weather = torch.zeros_like(test_data)

    train_index = torch.stack([torch.arange(i, i+24) for i in torch.arange(train_weather.shape[0])], 0)
    val_index = torch.stack([torch.arange(i, i+24) for i in torch.arange(val_weather.shape[0])], 0)
    test_index = torch.stack([torch.arange(i, i+24) for i in torch.arange(test_weather.shape[0])], 0)

    # packing on dummy weather to make the data fit rest of code
    train_dataset = TensorDataset(train_data, train_weather, train_index)
    val_dataset = TensorDataset(val_data, val_weather, val_index)
    test_dataset = TensorDataset(test_data, test_weather, test_index)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # mean and std values are grabbed from https://github.com/Essaim/CGCDemandPrediction
    return (train_dataloader, val_dataloader, test_dataloader, train_mean, train_std)

