import torch
import numpy as np


def torch_nll_gaussian(preds, target, variance):
    pred_dists = torch.distributions.Normal(target, scale=np.sqrt(variance))
    return -pred_dists.log_prob(preds).sum() / (
        target.shape[0] * target.shape[1]
    )  # Mean over batch and nodes


def kl_categorical_uniform_direct(preds, num_atoms, num_edge_types, eps=1e-16):
    kl_div = preds * torch.log(preds + eps) - np.log(1 / num_edge_types)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


# Taken from https://github.com/Essaim/CGCDemandPrediction
def pcc(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return np.corrcoef(x, y)[0][1]


def mape(pred, y):
    return (y - pred).abs() / (y.abs() + 1e-8)


def masked_mape(pred, y):
    abs_error = (y - pred).abs()
    mask = y == 0
    abs_error[mask] = 0
    return abs_error / (y.abs() + 1e-8)
