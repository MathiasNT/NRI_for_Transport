import torch
import numpy as np


def get_ha_normalization_matrices(data, datetime_list):
    """
    Creates normalization matrix for fine-grained HA normalization
    """
    weekday_idx = [np.where(datetime_list.weekday == i)[0] for i in range(7)]
    hour_idx = [np.where(datetime_list.hour == i)[0] for i in range(24)]
    weekday_idx_set = [set(x) for x in weekday_idx]
    hour_idx_set = [set(x) for x in hour_idx]

    mean_matrix = torch.zeros(7, 24, 66, 2)
    std_matrix = torch.zeros(7, 24, 66, 2)
    for i in range(len(weekday_idx_set)):
        for j in range(len(hour_idx)):
            day_hour_idxs = weekday_idx_set[i].intersection(hour_idx_set[j])
            mean_matrix[i, j] = data[list(day_hour_idxs)].mean(0)
            std_matrix[i, j] = data[list(day_hour_idxs)].std(0)

    device = torch.device(torch.cuda.current_device())
    mean_matrix = mean_matrix.to(device)
    std_matrix = std_matrix.to(device)

    return mean_matrix, std_matrix


def ha_normalization(data, datetime_list, mean_matrix, std_matrix):
    """
    Do fine-grained HA normalization
    """
    days = np.array(datetime_list.weekday)
    hours = np.array(datetime_list.hour)
    normalized_data = ((data.cuda() - mean_matrix[days, hours]) / std_matrix[days, hours]).cpu()
    normalized_data = torch.nan_to_num(normalized_data)
    return normalized_data


def ha_renormalization(data, datetime_list, mean_matrix, std_matrix):
    """
    Function to remove fine-grained HA normalization
    """
    days = datetime_list.weekday
    hours = datetime_list.hour
    renormalized_data = data * std_matrix[days, hours] + mean_matrix[days, hours]
    return renormalized_data


def ha_batch_renormalization(batch, batch_idxs, datetime_list, mean_matrix, std_matrix):
    """
    Batched version of the fine-grained HA normalization
    """
    days = datetime_list.weekday[batch_idxs]
    hours = datetime_list.hour[batch_idxs]
    renormalized_batch = (
        batch.permute(0, 2, 1, 3) * std_matrix[days, hours] + mean_matrix[days, hours]
    ).permute(0, 2, 1, 3)
    return renormalized_batch


def renormalize_data(data, data_min, data_max, new_way=True):
    if new_way:
        return data * (data_max - data_min) + data_min
    else:
        return (data + 1) * (data_max - data_min) / 2 + data_min


def restandardize_data(data, data_mean, data_std):
    return data * data_std + data_mean
