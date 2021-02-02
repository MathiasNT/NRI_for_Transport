"""
Dataloader stuff
"""
from torch.utils.data import DataLoader
import torch
from .Nyc_w_Weather import Nyc_w_Weather, Nyc_w_Weather2, Nyc_no_Weather2


def create_test_train_split(
    data, weather_data, split_len, batch_size, normalize=False, train_frac=0.8
):
    demand_tensor = torch.Tensor(data).permute(1, 0)
    weather_tensor = torch.Tensor(weather_data)

    # Calculate mean of the training part of the data
    mean = demand_tensor[: int(train_frac * len(demand_tensor))].mean()
    std = demand_tensor[: int(train_frac * len(demand_tensor))].std()
    if normalize:
        demand_tensor = (demand_tensor - mean) / std

    splits = []
    weather_splits = []
    for i in range(demand_tensor.shape[0] - split_len):
        splits.append(demand_tensor[i : i + split_len])
        weather_splits.append(weather_tensor[i : i + split_len])

    train_splits = splits[: int(train_frac * len(splits))]
    test_splits = splits[int(train_frac * len(splits)) :]

    train_weather = weather_splits[: int(train_frac * len(splits))]
    test_weather = weather_splits[int(train_frac * len(splits)) :]

    list_ids = range(len(train_splits))
    train_data = Nyc_w_Weather(train_splits, train_weather, list_ids)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, pin_memory=True, shuffle=True
    )

    test_list_ids = range(len(test_splits))
    test_data = Nyc_w_Weather(test_splits, test_weather, test_list_ids)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, pin_memory=True, shuffle=False
    )
    return (train_dataloader, test_dataloader, mean, std)


def create_test_train_split2(
    data, weather_data, split_len, batch_size, normalize=False, train_frac=0.8
):
    demand_tensor = torch.Tensor(data).permute(1, 0)
    weather_tensor = torch.Tensor(weather_data)

    # Calculate mean of the training part of the data
    mean = demand_tensor[: int(train_frac * len(demand_tensor))].mean()
    std = demand_tensor[: int(train_frac * len(demand_tensor))].std()
    if normalize:
        demand_tensor = (demand_tensor - mean) / std

    splits = []
    weather_splits = []
    for i in range(demand_tensor.shape[0] - split_len):
        splits.append(demand_tensor[i : i + split_len])
        weather_splits.append(weather_tensor[i : i + split_len])

    train_splits = splits[: int(train_frac * len(splits))]
    test_splits = splits[int(train_frac * len(splits)) :]

    train_weather = weather_splits[: int(train_frac * len(splits))]
    test_weather = weather_splits[int(train_frac * len(splits)) :]

    list_ids = range(len(train_splits))
    train_data = Nyc_w_Weather2(train_splits, train_weather, list_ids)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, pin_memory=True, shuffle=True
    )

    test_list_ids = range(len(test_splits))
    test_data = Nyc_w_Weather2(test_splits, test_weather, test_list_ids)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, pin_memory=True, shuffle=False
    )
    return (train_dataloader, test_dataloader, mean, std)


def create_test_train_split2_no_weather(
    data, split_len, batch_size, normalize=False, train_frac=0.8
):
    demand_tensor = torch.Tensor(data).permute(1, 0)

    # Calculate mean of the training part of the data
    mean = demand_tensor[: int(train_frac * len(demand_tensor))].mean()
    std = demand_tensor[: int(train_frac * len(demand_tensor))].std()
    if normalize:
        demand_tensor = (demand_tensor - mean) / std

    splits = []
    for i in range(demand_tensor.shape[0] - split_len):
        splits.append(demand_tensor[i : i + split_len])

    train_splits = splits[: int(train_frac * len(splits))]
    test_splits = splits[int(train_frac * len(splits)) :]

    list_ids = range(len(train_splits))
    train_data = Nyc_no_Weather2(train_splits, list_ids)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, pin_memory=True, shuffle=True
    )

    test_list_ids = range(len(test_splits))
    test_data = Nyc_no_Weather2(test_splits, test_list_ids)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, pin_memory=True, shuffle=False
    )
    return (train_dataloader, test_dataloader, mean, std)
