import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import torch

import sys

sys.path.append("../")
from GraphTrafficLib.utils.visual_utils import PEMS_folium_plot

raw_data_folder = "../datafolder/rawdata/pems"
proc_data_folder = "../datafolder/procdata/pems_data"

# The processed data can be found in the numpy zip files
pems_train = np.load(f"{raw_data_folder}/train.npz")
pems_val = np.load(f"{raw_data_folder}/val.npz")
pems_test = np.load(f"{raw_data_folder}/test.npz")

x_train = pems_train["x"]
y_train = pems_train["y"]
x_val = pems_val["x"]
y_val = pems_val["y"]
x_test = pems_test["x"]
y_test = pems_test["y"]

# We can collect the data back to match my format
train_data = np.concatenate([x_train, y_train], axis=1)
val_data = np.concatenate([x_val, y_val], axis=1)
test_data = np.concatenate([x_test, y_test], axis=1)

np.save(f"{proc_data_folder}/train_data.npy", train_data)
np.save(f"{proc_data_folder}/val_data.npy", val_data)
np.save(f"{proc_data_folder}/test_data.npy", test_data)

# The distances can be found in another csv
distances_df = pd.read_csv(
    f"{raw_data_folder}/distances_bay_2017.csv", header=None, names=["from", "to", "dist"]
)

# Remove self loops
distances_df = distances_df[distances_df.dist != 0]

# I load their index to sensor id lookup and their adjacancy matrix
with open(f"{raw_data_folder}/adj_mx_bay.pkl", "rb") as f:
    sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

adj_mx_no_loop = adj_mx - np.eye(325)

# we also make the reverse lookup
sensor_ind_to_id = {v: k for k, v in sensor_id_to_ind.items()}

# The locations of the sensors can be found in this csv
location_df = pd.read_csv(
    f"{raw_data_folder}/graph_sensor_locations_bay.csv",
    header=None,
    names=["id", "lat", "lon"],
)
location_df = location_df.set_index("id")

gdf = gpd.GeoDataFrame(
    location_df, geometry=gpd.points_from_xy(location_df.lon, location_df.lat), crs="EPSG:4326"
)

# Based on the distances I make a sparse spatial adjacancy matrix
spatial_adj_matrix = np.zeros_like(adj_mx_no_loop)
problem_list = []
for sender in distances_df["from"].unique():
    sender_ind = sensor_id_to_ind[str(sender)]
    receivers = distances_df.loc[(distances_df["from"] == sender)].sort_values("dist")["to"]

    # Try closest one
    for receiver in receivers:
        receiver_ind = sensor_id_to_ind[str(receiver)]
        if spatial_adj_matrix[receiver_ind, sender_ind] != 1:
            spatial_adj_matrix[sender_ind, receiver_ind] = 1
            break
        else:  # if already have connection other direction go to second closest
            problem_list.append(sender)

spatial_adj_matrix += spatial_adj_matrix.T
spatial_adj_matrix[spatial_adj_matrix != 0] = 1

spatial_adj_tensor = torch.Tensor(spatial_adj_matrix)

# Create empty adj
empty_adj = np.zeros_like(spatial_adj_matrix)

# Create full adj
full_adj = np.ones_like(spatial_adj_matrix) - np.eye(len(spatial_adj_matrix))

# We save the adj matrices
np.save(f"{proc_data_folder}/approx_local_adj.npy", adj_mx_no_loop)
np.save(f"{proc_data_folder}/sparse_local_adj.npy", spatial_adj_matrix)
np.save(f"{proc_data_folder}/pems_full_adj.npy", full_adj)
np.save(f"{proc_data_folder}/pems_empty_adj.npy", empty_adj)
