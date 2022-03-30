import pandas as pd
import geopandas as gpd
import numpy as np
import meteostat
from datetime import datetime, timedelta

import sys

dataset_folder = f"../datafolder/rawdata/taxi"
shapefile_folder = f"{dataset_folder}/shapefiles"
proc_data_folder = f"../datafolder/procdata/taxi_data/full_manhattan"  # Full manhattan because previous experiments used a subset of zones.
sys.path.append("../")  # Change this for the path to your folder, however this should work

from GraphTrafficLib.utils.data_preprocess import (
    preprocess_NYC_borough_dropoff,
    preprocess_NYC_borough_pickup,
)
from GraphTrafficLib.utils.adjacancy_matrix_generators import (
    dtw_adj_generator,
    get_local_adj_matrix,
)

### Load metadata ###
NYC_shapefile = f"{shapefile_folder}/taxi_zones.shp"
shp = gpd.read_file(NYC_shapefile)
manhattan_ids = shp.loc[shp.borough == "Manhattan"].LocationID
EWR_ids = shp.loc[shp.borough == "EWR"].LocationID  # Small airport area with 1 zone
queens_ids = shp.loc[shp.borough == "Queens"].LocationID
bronx_ids = shp.loc[shp.borough == "Bronx"].LocationID
brooklyn_ids = shp.loc[shp.borough == "Brooklyn"].LocationID
staten_ids = shp.loc[shp.borough == "Staten Island"].LocationID

# Some hardcoded clean up of weird zones
manhattan_ids = manhattan_ids.loc[manhattan_ids != 103]
queens_ids = queens_ids.loc[(queens_ids != 30) & (queens_ids != 2) & (queens_ids != 27)].unique()
bronx_ids = bronx_ids.loc[(bronx_ids != 199)].unique()

# Data file strings and dtype settings
# 2018 data files
NYC_2018_filenames = [f"yellow_tripdata_2018-0{i}.csv" for i in range(1, 10)]
NYC_2018_filenames2 = [f"yellow_tripdata_2018-{i}.csv" for i in range(10, 13)]
NYC_2018_filenames = NYC_2018_filenames + NYC_2018_filenames2
file_paths_2018 = [f"{dataset_folder}/2018/{filename}" for filename in NYC_2018_filenames]

# 2019 data files
NYC_filenames = [f"yellow_tripdata_2019-0{i}.csv" for i in range(1, 10)]
NYC_filenames2 = [f"yellow_tripdata_2019-{i}.csv" for i in range(10, 13)]
NYC_filenames = NYC_filenames + NYC_filenames2
file_paths_2019 = [f"{dataset_folder}/2019/{filename}" for filename in NYC_filenames]

file_paths = file_paths_2018 + file_paths_2019

# As we do not need the precision of 64 bits we change the precision to 32 bits for all numerical columns
important_cols = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
]
df_test = pd.read_csv(file_paths[0], nrows=100, usecols=important_cols)
float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
float32_cols = {c: np.float32 for c in float_cols}
int_cols = [c for c in df_test if df_test[c].dtype == "int64"]
int32_cols = {c: np.int32 for c in int_cols}
dtype_cols = {**int32_cols, **float32_cols}

### The actual preprocessing ###

# Start preprocessing 2019
print("Preprocessing pickups 2019")
pickup_2019, full_time_list_2019 = preprocess_NYC_borough_pickup(
    file_paths=file_paths_2019, location_ids=manhattan_ids
)

print("Preprocessing dropoffs 2019")
dropoff_2019, full_time_list_2019 = preprocess_NYC_borough_dropoff(
    file_paths=file_paths_2019, location_ids=manhattan_ids
)
full_data_2019 = np.stack([pickup_2019, dropoff_2019], axis=-1)

# Start preprocessing 2018
print("Preprocessing pickups 2018")
pickup_2018, full_time_list_2018 = preprocess_NYC_borough_pickup(
    file_paths=file_paths_2018, location_ids=manhattan_ids, year=2018
)
print("Preprocessing dropoffs 2018")
dropoff_2018, full_time_list_2018 = preprocess_NYC_borough_dropoff(
    file_paths=file_paths_2018, location_ids=manhattan_ids, year=2018
)
full_data_2018 = np.stack([pickup_2018, dropoff_2018], axis=-1)

# collect the data and save
full_data = np.concatenate([full_data_2018, full_data_2019], 1)
full_time_list = full_time_list_2018.union(full_time_list_2019)

# Save the data
print("Saving 2 year data")
np.save(f"{proc_data_folder}/twoyear_full_manhattan_2d.npy", full_data)
np.save(f"{proc_data_folder}/twoyear_time_list.npy", full_time_list)

# Preprocessing of short year
short_2019 = file_paths_2019[:10]

print("Preprocessing short year prickups")
pickup_short_2019, short_time_list_2019 = preprocess_NYC_borough_pickup(
    file_paths=short_2019, location_ids=manhattan_ids
)
print("Preprocessing short year dropoffs")
dropoff_short_2019, short_time_list_2019 = preprocess_NYC_borough_dropoff(
    file_paths=short_2019, location_ids=manhattan_ids
)
short_data_2019 = np.stack([pickup_short_2019, dropoff_short_2019], axis=-1)
# Save the data
print("Saving short year data")
np.save(f"{proc_data_folder}/short_year_full_manhattan_2d.npy", short_data_2019)

### Generate adjs ###
# Make DTW adj for prior
print("Generate adj matrices")
train_steps = int(0.8 * short_data_2019.shape[1])

# Generating ADJ matrix
dtw_adj = dtw_adj_generator(short_data_2019, train_steps)
pickle_path_full_manhattan_short_year_dtw_adj = (
    f"{proc_data_folder}/short_year_train_full_manhattan_dtw_adj.npy"
)
np.save(pickle_path_full_manhattan_short_year_dtw_adj, dtw_adj)
print("DTW adj matrix done")

# Discretize the DTW adj for fixed adj graph
dtw_adj_bin = dtw_adj > np.quantile(dtw_adj, 0.9)
pickle_path_full_manhattan_short_year_dtw_adj_bin = (
    f"{proc_data_folder}/short_year_train_full_manhattan_dtw_adj_bin.npy"
)
np.save(pickle_path_full_manhattan_short_year_dtw_adj_bin, dtw_adj_bin)
print("Discrete DTW adj matrix done")

# Create local adj matrix based on shapefile
manhattan_shp = shp.loc[shp.LocationID.isin(manhattan_ids)]
local_adj = get_local_adj_matrix(manhattan_shp.reset_index(drop=True))
pickle_path_full_manhattan_full_year_local_adj = (
    f"{proc_data_folder}/full_year_full_manhattan_local_adj"
)
np.save(pickle_path_full_manhattan_full_year_local_adj, local_adj)
print("Local adj matrix done")

# Create empty adj
n_nodes = len(manhattan_shp)
empty_adj = np.zeros([n_nodes, n_nodes])
np.save(f"{proc_data_folder}/empty_adj.npy", empty_adj)
print("Empty adj matrix done")

# Create full adj
full_adj = np.ones([n_nodes, n_nodes]) - np.eye(n_nodes)
np.save(f"{proc_data_folder}/full_adj.npy", full_adj)
print("Full adj matrix done")

### Preprocessing weather data ###
print("Preprocess weather data")
two_year_data = np.load(f"{proc_data_folder}/twoyear_full_manhattan_2d.npy")
start = datetime(2018, 1, 1)
end = start + timedelta(hours=two_year_data.shape[1])
# IDs of the airports around NYC
LaGuardia = 72503
NewArk = 72502
JFK = 74486

# Get LGA data
LaGuardiaData = meteostat.Hourly(LaGuardia, start, end)
LaGuardiaData = LaGuardiaData.fetch()[:-1]
LaGuardiaData.prcp.interpolate(inplace=True)
LaGuardiaData.prcp.iloc[0] = 0

# Get New Ark Data
NewArkData = meteostat.Hourly(NewArk, start, end)
NewArkData = NewArkData.fetch()[:-1]
NewArkData.prcp.interpolate(inplace=True)
NewArkData.prcp.iloc[0] = 0

# Get JFK Data
JFKData = meteostat.Hourly(JFK, start, end)
JFKData = JFKData.fetch()[:-1]
JFKData.prcp.interpolate(inplace=True)
JFKData.prcp.iloc[0] = 0

# Calculate mean over airports
airport_mean_prcp = (LaGuardiaData.prcp.values + NewArkData.prcp.values + JFKData.prcp.values) / 3
airport_mean_temp = (LaGuardiaData.temp.values + NewArkData.temp.values + JFKData.temp.values) / 3
mean_weather_df = pd.DataFrame(
    np.array([airport_mean_prcp, airport_mean_temp]).T, columns=["precipDepth", "temperature"]
)
mean_weather_df = mean_weather_df.set_index(JFKData.index)

# Save weather data
mean_weather_df.to_csv(f"{proc_data_folder}/../mean_airport_weather.csv")
mean_weather_2019 = mean_weather_df.loc[mean_weather_df.index > datetime(2018, 12, 31, 23)]
mean_weather_2019.to_csv(f"{proc_data_folder}/../mean_airport_weather_2019.csv")
print("Weather data saved")
