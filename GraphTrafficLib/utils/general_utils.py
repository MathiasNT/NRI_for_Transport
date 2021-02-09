import numpy as np
import torch


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

