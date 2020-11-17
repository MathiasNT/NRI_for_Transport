import pandas as pd


def add_spatial_bins(df, n_lat_bins, n_lon_bins):
    """This functions add latitude and longitude bins to a dataframe with geo coordinates
    Note that the function requires "pickup_x" for it to work

    Args:
        df (pd.Dataframe): The dataframe
        n_lat_bins (int): number of latitude bins
        n_lon_bins (int): number of longitude bins

    Returns:
        pd.DataFrame: Dataframe with added bins
    """

    # Add the longitude bins
    df.loc[:, "longitude_bins"] = pd.cut(
        df["pickup_longitude"], n_lon_bins, labels=range(n_lon_bins)
    )

    # Add the latitude bins
    longitude_bin_coord = pd.cut(df["pickup_longitude"], n_lon_bins).cat.categories
    df.loc[:, "latitude_bins"] = pd.cut(
        df["pickup_latitude"], n_lat_bins, labels=range(n_lat_bins - 1, -1, -1)
    )  # We reverse the latitude bins as latitude increases up, not down
    return df


def add_temporal_bins(df, dt):
    """This function adds temporal bins to a dataframe with timestamps


    Args:
        df (pd.DataFrame): The dataframe
        dt (string): The frequency to bin into. The string must match a possible pd.date_range freq string

    Returns:
        pd.DataFrame: The dataframe with added temporal bins
    """
    min_date = demand_df.pickup_time.min()

    # Get the max date
    max_date = demand_df.pickup_time.max()

    # Create the bins. Note that this misses a bit from the beginning but this will not be a big problem when we index finer
    bins_dt = pd.date_range(start=min_date, end=max_date, freq=dt)
    bins_str = bins_dt.astype(str).values
    n_bins_dt = len(bins_dt) - 1
    print(f"from {min_date} to {max_date}")

    # Create the temporal bins
    demand_df.loc[:, "time_bins"] = pd.cut(
        df["pickup_time"], bins=bins_dt, labels=range(n_bins_dt)
    )

    # due to the time indexing we have some values that gets outside the bins - here is a temporary fix
    demand_df = demand_df.dropna()
    return demand_df

