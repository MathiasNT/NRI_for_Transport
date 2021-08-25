"""
Dataloader stuff
"""
from torch.utils.data import DataLoader
import torch
from .Nyc_w_Weather import Nyc_w_Weather, Nyc_w_Weather2, Nyc_no_Weather2
from torch.utils.data.dataset import TensorDataset


# def create_test_train_split(
#     data, weather_data, split_len, batch_size, normalize=False, train_frac=0.8
# ):
#     demand_tensor = torch.Tensor(data).permute(1, 0)
#     weather_tensor = torch.Tensor(weather_data)

#     # Calculate mean of the training part of the data
#     mean = demand_tensor[: int(train_frac * len(demand_tensor))].mean()
#     std = demand_tensor[: int(train_frac * len(demand_tensor))].std()
#     if normalize:
#         demand_tensor = (demand_tensor - mean) / std

#     splits = []
#     weather_splits = []
#     for i in range(demand_tensor.shape[0] - split_len):
#         splits.append(demand_tensor[i : i + split_len])
#         weather_splits.append(weather_tensor[i : i + split_len])

#     train_splits = splits[: int(train_frac * len(splits))]
#     test_splits = splits[int(train_frac * len(splits)) :]

#     train_weather = weather_splits[: int(train_frac * len(splits))]
#     test_weather = weather_splits[int(train_frac * len(splits)) :]

#     list_ids = range(len(train_splits))
#     train_data = Nyc_w_Weather(train_splits, train_weather, list_ids)
#     train_dataloader = DataLoader(
#         train_data, batch_size=batch_size, pin_memory=True, shuffle=True
#     )

#     test_list_ids = range(len(test_splits))
#     test_data = Nyc_w_Weather(test_splits, test_weather, test_list_ids)
#     test_dataloader = DataLoader(
#         test_data, batch_size=batch_size, pin_memory=True, shuffle=False
#     )
#     return (train_dataloader, test_dataloader, mean, std)


# def create_test_train_split2(
#     data, weather_data, split_len, batch_size, normalize=False, train_frac=0.8
# ):
#     demand_tensor = torch.Tensor(data).permute(1, 0)
#     weather_tensor = torch.Tensor(weather_data)

#     # Calculate mean of the training part of the data
#     mean = demand_tensor[: int(train_frac * len(demand_tensor))].mean()
#     std = demand_tensor[: int(train_frac * len(demand_tensor))].std()
#     if normalize:
#         demand_tensor = (demand_tensor - mean) / std

#     splits = []
#     weather_splits = []
#     for i in range(demand_tensor.shape[0] - split_len):
#         splits.append(demand_tensor[i : i + split_len])
#         weather_splits.append(weather_tensor[i : i + split_len])

#     train_splits = splits[: int(train_frac * len(splits))]
#     test_splits = splits[int(train_frac * len(splits)) :]

#     train_weather = weather_splits[: int(train_frac * len(splits))]
#     test_weather = weather_splits[int(train_frac * len(splits)) :]

#     list_ids = range(len(train_splits))
#     train_data = Nyc_w_Weather2(train_splits, train_weather, list_ids)
#     train_dataloader = DataLoader(
#         train_data, batch_size=batch_size, pin_memory=True, shuffle=True
#     )

#     test_list_ids = range(len(test_splits))
#     test_data = Nyc_w_Weather2(test_splits, test_weather, test_list_ids)
#     test_dataloader = DataLoader(
#         test_data, batch_size=batch_size, pin_memory=True, shuffle=False
#     )
#     return (train_dataloader, test_dataloader, mean, std)


# def create_test_train_split2_no_weather(
#     data,
#     split_len,
#     batch_size,
#     normalize=False,
#     train_frac=0.8,
#     shuffle_train=True,
#     shuffle_test=False,
# ):
#     demand_tensor = torch.Tensor(data).permute(1, 0)

#     # Calculate mean of the training part of the data
#     mean = demand_tensor[: int(train_frac * len(demand_tensor))].mean()
#     std = demand_tensor[: int(train_frac * len(demand_tensor))].std()
#     if normalize:
#         demand_tensor = (demand_tensor - mean) / std

#     splits = []
#     for i in range(demand_tensor.shape[0] - split_len):
#         splits.append(demand_tensor[i : i + split_len])

#     train_splits = splits[: int(train_frac * len(splits))]
#     test_splits = splits[int(train_frac * len(splits)) :]

#     list_ids = range(len(train_splits))
#     train_data = Nyc_no_Weather2(train_splits, list_ids)
#     train_dataloader = DataLoader(
#         train_data, batch_size=batch_size, pin_memory=True, shuffle=shuffle_train
#     )

#     test_list_ids = range(len(test_splits))
#     test_data = Nyc_no_Weather2(test_splits, test_list_ids)
#     test_dataloader = DataLoader(
#         test_data, batch_size=batch_size, pin_memory=True, shuffle=shuffle_test
#     )
#     return (train_dataloader, test_dataloader, mean, std)


def create_test_train_split_max_min_normalize(
    data,
    weather_data,
    split_len,
    batch_size,
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

    # do max-min normal of data
    if fixed_max is not None:
        train_max = fixed_max
    else:
        train_max = demand_tensor[: int(train_frac * len(demand_tensor))].amax(
            dim=(0, 1)
        )

    if fixed_min is not None:
        train_min = fixed_min
    else:
        train_min = demand_tensor[: int(train_frac * len(demand_tensor))].amin(
            dim=(0, 1)
        )

    if normalize:
        # demand_tensor = (demand_tensor - train_min) * 2 / (train_max - train_min) - 1  // between -1 and 1
        demand_tensor = (demand_tensor - train_min) / (train_max - train_min)

    splits = []
    weather_splits = []
    for i in range(demand_tensor.shape[0] - split_len):
        splits.append(demand_tensor[i : i + split_len])
        weather_splits.append(weather_tensor[i : i + split_len])
    splits = torch.stack(splits).transpose(1, 2)
    weather_splits = torch.stack(weather_splits)
    train_splits = splits[: int(train_frac * len(splits))]
    test_splits = splits[int(train_frac * len(splits)) :]

    print(f"train splits shape: {train_splits.shape}")
    print(f"test splits shape: {test_splits.shape}")

    train_weather = weather_splits[: int(train_frac * len(splits))]
    test_weather = weather_splits[int(train_frac * len(splits)) :]

    train_dataset = TensorDataset(train_splits, train_weather)
    test_dataset = TensorDataset(test_splits, test_weather)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=2,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=2,
    )

    return (train_dataloader, test_dataloader, train_max, train_min)


def create_dataloaders(
    data,
    weather_data,
    split_len,
    batch_size,
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

    # do max-min normal of data
    if fixed_max is not None:
        train_max = fixed_max
    else:
        train_max = demand_tensor[: int(train_frac * len(demand_tensor))].amax(
            dim=(0, 1)
        )
        weather_train_max = weather_tensor[: int(train_frac * len(weather_tensor))].amax(dim=0)

    if fixed_min is not None:
        train_min = fixed_min
    else:
        train_min = demand_tensor[: int(train_frac * len(demand_tensor))].amin(
            dim=(0, 1)
        )
        weather_train_min = weather_tensor[: int(train_frac * len(weather_tensor))].amin(dim=0)

    if normalize:
        # demand_tensor = (demand_tensor - train_min) * 2 / (train_max - train_min) - 1  // between -1 and 1
        demand_tensor = (demand_tensor - train_min) / (train_max - train_min)
        weather_tensor = (weather_tensor - weather_train_min) / (weather_train_max - weather_train_min)

    splits = []
    weather_splits = []
    for i in range(demand_tensor.shape[0] - split_len):
        splits.append(demand_tensor[i : i + split_len])
        weather_splits.append(weather_tensor[i : i + split_len])
    splits = torch.stack(splits).transpose(1, 2)
    weather_splits = torch.stack(weather_splits)

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

    train_dataset = TensorDataset(train_splits, train_weather)
    val_dataset = TensorDataset(val_splits, val_weather)
    test_dataset = TensorDataset(test_splits, test_weather)

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

    return (train_dataloader, val_dataloader, test_dataloader, train_max, train_min)


def create_test_train_split_max_min_normalize_no_split(
    data,
    weather_data,
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

    # do max-min normal of data
    if fixed_max is not None:
        train_max = fixed_max
    else:
        train_max = demand_tensor[: int(train_frac * len(demand_tensor))].amax(
            dim=(0, 1)
        )

    if fixed_min is not None:
        train_min = fixed_min
    else:
        train_min = demand_tensor[: int(train_frac * len(demand_tensor))].amin(
            dim=(0, 1)
        )

    if normalize:
        # demand_tensor = (demand_tensor - train_min) * 2 / (train_max - train_min) - 1  // between -1 and 1
        demand_tensor = (demand_tensor - train_min) / (train_max - train_min)

    train_weather = weather_tensor[: int(train_frac * len(weather_tensor))]
    test_weather = weather_tensor[int(train_frac * len(weather_tensor)) :]

    train_demand = demand_tensor[: int(train_frac * len(demand_tensor))]
    test_demand = demand_tensor[int(train_frac * len(demand_tensor)) :]

    train_dataset = TensorDataset(train_demand, train_weather)
    test_dataset = TensorDataset(test_demand, test_weather)

    return (train_dataset, test_dataset, train_max, train_min)


def renormalize_data(data, data_min, data_max, new_way=True):
    if new_way:
        return data * (data_max - data_min) + data_min
    else:
        return (data + 1) * (data_max - data_min) / 2 + data_min
