import torch
import numpy as np

# My loss functions - these have not seemed to work earlier
def torch_nll_gaussian(preds, target, variance):
    pred_dists = torch.distributions.Normal(target, scale=np.sqrt(variance))
    return -pred_dists.log_prob(preds).sum() / (target.shape[0] * target.shape[1]) # Mean over batch and nodes


def kl_categorical_uniform_direct(preds, num_atoms, num_edge_types, eps=1e-16):
    kl_div = preds * torch.log(preds + eps) - np.log(1 / num_edge_types)
    # kl_div = preds * torch.log((preds + eps) /(1/num_edge_types))
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def cyc_anneal(epoch, cyclic):
    """
    returns anneal weight to multiply KLD with in elbo
    takes half a cycle to get to 1. for the rest of the cycle it will remain 1
    so it resembles https://github.com/haofuml/cyclical_annealing
    Function assumes epoch starts at 0
    """

    cycle = (epoch) % cyclic
    anneal = min(1, 2 / cyclic * cycle + 0.1)

    return anneal
