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

def get_prior_from_adj(adj_matrix, adj_prior, rel_send, rel_rec):
    edge_prior = torch.zeros(rel_send.shape[0], 2)

    for i in range(len(edge_prior)):
        send = rel_send[i].argmax().item()
        rec = rel_rec[i].argmax().item()
        edge_prior[i, int(adj_matrix[send, rec])] += adj_prior
    zero_idxs = edge_prior == 0
    edge_prior[zero_idxs] = 1 - adj_prior
    log_prior = np.log(edge_prior)
    return log_prior

def get_simple_prior(n_edge_types, edge_rate):
    # Set up prior
    if n_edge_types == 2:
        prior = np.array([1 - edge_rate, edge_rate])
    else:
        prior = np.empty(n_edge_types)
        prior[0] = 1 - edge_rate
        prior[1:] = edge_rate / (n_edge_types - 1)

    print(f"Using simple prior: {prior}")

    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    return log_prior
    

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

def cyc_anneal_delayed(epoch, cyclic, delay):
    """
    returns anneal weight to multiply KLD with in elbo
    takes half a cycle to get to 1. for the rest of the cycle it will remain 1
    so it resembles https://github.com/haofuml/cyclical_annealing
    Function assumes epoch starts at 0
    """
    if epoch < delay:
         return 1

    cycle = (epoch) % cyclic
    anneal = min(1, 2 / cyclic * cycle + 0.1)

    return anneal

# Taken from https://github.com/Essaim/CGCDemandPrediction
def pcc(x, y):
    x,y = x.reshape(-1),y.reshape(-1)
    return np.corrcoef(x,y)[0][1]

def mape(pred, y):
    return ((y-pred).abs() / (y.abs() + 1e-8))