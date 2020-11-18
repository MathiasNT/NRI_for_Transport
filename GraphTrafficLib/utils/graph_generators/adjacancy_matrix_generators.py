import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtw_adj_generator(demand_vector, n_lat_bins, n_lon_bins, end_index):
    n_nodes = n_lat_bins * n_lon_bins
    adjacancy_matrix = np.zeros((n_nodes, n_nodes))
    # Might need to double check the direction on this at some point + I've done a simplification
    # of only looking at the first 500 places
    for i in tqdm(range(n_nodes)):
        for j in range(n_nodes):
            distance, _ = fastdtw(
                demand_vector[i][:end_index],
                demand_vector[j][:end_index],
                dist=euclidean,
                radius=1,
            )
            adjacancy_matrix[i, j] = distance

    return adjacancy_matrix
