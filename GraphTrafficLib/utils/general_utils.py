import numpy as np
import torch
from prettytable import PrettyTable


def encode_onehot(labels):
    """This function creates a onehot encoding.
    copied from https://github.com/ethanfetaya/NRI
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def RMSE(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def MAE(pred, target):
    return torch.abs(pred - target).mean()


def MAPE(pred, target):
    return torch.abs((target - pred) / target).mean()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    table.title = model._get_name()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
