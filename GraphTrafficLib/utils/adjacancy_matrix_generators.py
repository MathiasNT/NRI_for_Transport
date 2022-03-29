import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
import torch


def dtw_adj_generator(demand_vector, end_index, coordinate_version=False, **kwargs):
    # note that if coordinate version kwargs n_lat_bins and n_lon_bins is needed
    if coordinate_version:
        n_nodes = kwargs["n_lat_bins"] * kwargs["n_lon_bins"]
    else:
        n_nodes = demand_vector.shape[0]

    adjacancy_matrix = np.zeros((n_nodes, n_nodes))
    for i in tqdm(range(n_nodes)):
        dists, _ = zip(
            *Parallel(n_jobs=-1)(
                delayed(fastdtw)(
                    demand_vector[i][:end_index],
                    demand_vector[j][:end_index],
                    dist=euclidean,
                    radius=1,
                )
                for j in range(n_nodes)
            )
        )
        adjacancy_matrix[i, :] = dists
    return adjacancy_matrix


def get_local_adj_matrix(shp):
    adj_matrix = torch.zeros([len(shp), len(shp)])

    for zone_idx, zone in shp.iterrows():
        neighbours = shp[~shp.geometry.disjoint(zone.geometry)].index
        for neighbour_idx in neighbours:
            adj_matrix[zone_idx, neighbour_idx] = 1

    adj_matrix = adj_matrix - torch.eye(len(shp))

    return adj_matrix
