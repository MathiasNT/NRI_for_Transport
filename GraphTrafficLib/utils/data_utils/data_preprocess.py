import pandas as pd
import numpy as np


def add_spatial_bins(df, n_lat_bins, n_lon_bins):
    # This functions add latitude and longitude bins to a dataframe
    # with geo coordinates
    # Note that the function requires "pickup_x" for it to work

    # Add the longitude bins
    df.loc[:, "longitude_bins"] = pd.cut(
        df["pickup_longitude"], n_lon_bins, labels=range(n_lon_bins)
    )

    # Add the latitude bins
    df.loc[:, "latitude_bins"] = pd.cut(
        df["pickup_latitude"], n_lat_bins, labels=range(n_lat_bins - 1, -1, -1)
    )  # We reverse the latitude bins as latitude increases up, not down
    return df


def add_temporal_bins(df, dt):
    # This function adds temporal bins to a dataframe with timestamps

    # Get the min and max date
    min_date = df.pickup_time.min()
    max_date = df.pickup_time.max()

    # Create the bins. Note that this misses a bit from the
    # beginning but this will not be a big problem when we index finer
    bins_dt = pd.date_range(start=min_date, end=max_date, freq=dt)
    n_bins_dt = len(bins_dt) - 1
    print(f"from {min_date} to {max_date}")

    # Create the temporal bins
    df.loc[:, "time_bins"] = pd.cut(
        df["pickup_time"], bins=bins_dt, labels=range(n_bins_dt)
    )

    # due to the time indexing we have some values that gets outside the bins - here is a temporary fix
    df = df.dropna()
    return df, n_bins_dt


def create_binned_matrix(df, n_lat_bins, n_lon_bins, n_bins_dt):
    # Group the data based on the spatial bins. Note that the temporal order is in the data already
    location_groups = [x for _, x in df.groupby(["longitude_bins", "latitude_bins"])]
    group_idx = [idx for idx, _ in df.groupby(["longitude_bins", "latitude_bins"])]

    # Create matrix with data of sie [lat, lon, time]
    binned_matrix = np.zeros((n_lat_bins, n_lon_bins, n_bins_dt))
    for i in range(len(location_groups)):
        location_time_series = location_groups[i].time_bins.value_counts().sort_index()
        lon, lat = group_idx[i]
        binned_matrix[lat, lon, :] = location_time_series.values

    return binned_matrix
