import torch
import numpy as np


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


def get_prior_from_adj(adj_matrix, adj_prior, rel_send, rel_rec):
    edge_prior = torch.ones(adj_matrix.shape[0] * adj_matrix[0] - 1, 2)
    edge_prior *= 1 - adj_prior
    for i in range(len(edge_prior)):
        send = rel_send[i].argmax().item()
        rec = rel_rec[i].argmax().item()
        edge_prior[i, adj_matrix[send, rec]] += adj_prior
    return edge_prior
