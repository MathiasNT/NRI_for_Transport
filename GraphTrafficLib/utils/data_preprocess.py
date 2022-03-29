import pandas as pd
import numpy as np
import datetime as dt


def add_spatial_bins(df, n_lat_bins, n_lon_bins):
    # This functions add latitude and longitude bins to a dataframe
    # with geo coordinates
    # Note that the function requires "pickup_x" for it to work

    df.loc[:, "longitude_bins"] = pd.cut(
        df["pickup_longitude"], n_lon_bins, labels=range(n_lon_bins)
    )

    df.loc[:, "latitude_bins"] = pd.cut(
        df["pickup_latitude"], n_lat_bins, labels=range(n_lat_bins - 1, -1, -1)
    )  # We reverse the latitude bins as latitude increases up, not down
    return df


def add_temporal_bins(df, time_col_name, dt_freq, year, month):
    # This function adds temporal bins to a dataframe with timestamps

    min_date = pd.Timestamp(year=year, month=month, day=1)
    if month + 1 > 12:
        max_date = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        max_date = pd.Timestamp(year=year, month=month + 1, day=1)

    # Note that this misses a bit from the beginning but this will not be a big problem when we index finer
    bins_dt = pd.date_range(start=min_date, end=max_date, freq=dt_freq)
    n_bins_dt = len(bins_dt) - 1
    df.loc[:, "time_bins"] = pd.cut(df[time_col_name], bins=bins_dt, labels=range(n_bins_dt))

    # due to the time indexing we have some values that gets outside the bins - here is a temporary fix
    df = df.dropna()
    return df, n_bins_dt, bins_dt


def create_binned_matrix(df, n_lat_bins, n_lon_bins, n_bins_dt):
    # Group the data based on the spatial bins. Note that the temporal order is in the data already
    location_groups = [x for _, x in df.groupby(["longitude_bins", "latitude_bins"])]
    group_idx = [idx for idx, _ in df.groupby(["longitude_bins", "latitude_bins"])]

    # Create matrix with data of size [lat, lon, time]
    binned_matrix = np.zeros((n_lat_bins, n_lon_bins, n_bins_dt))
    for i in range(len(location_groups)):
        location_time_series = location_groups[i].time_bins.value_counts().sort_index()
        lon, lat = group_idx[i]
        binned_matrix[lat, lon, :] = location_time_series.values

    return binned_matrix


def create_binned_vector(df, n_spatial_bins, n_bins_dt, spatial_bins_name, temporal_bins_name):
    # Group the data based on the spatial bins. Note that the temporal order is in the data already
    location_groups = [x for _, x in df.groupby(spatial_bins_name)]
    group_idx = [idx for idx, _ in df.groupby(spatial_bins_name)]

    # Create matrix with data of size [n_locations, n_timesteps]
    output_vector = np.zeros((n_spatial_bins, n_bins_dt))
    for i in range(len(location_groups)):
        location_index = location_groups[i][spatial_bins_name].iloc[0]
        vector_index = group_idx.index(location_index)
        location_time_series = location_groups[i][temporal_bins_name].value_counts().sort_index()
        output_vector[vector_index, :] = location_time_series.values

    return output_vector, group_idx


def create_OD_matrix_ts(
    df, n_spatial_bins, n_bins_dt, pu_bins_name, do_bins_name, temporal_bins_name
):

    pu_location_groups = [x for _, x in df.groupby(pu_bins_name)]
    group_idx = [idx for idx, _ in df.groupby(pu_bins_name)]

    output_vector = np.zeros((n_bins_dt, n_spatial_bins, n_spatial_bins))
    for pu_idx, pu_location_df in enumerate(pu_location_groups):
        pu_location = pu_location_df[pu_bins_name].iloc[0]
        do_location_groups = [x for _, x in pu_location_df.groupby(do_bins_name)]
        for do_location_df in do_location_groups:
            do_location = do_location_df[do_bins_name].iloc[0]
            do_idx = group_idx.index(do_location)
            OD_timeseries = do_location_df[temporal_bins_name].value_counts().sort_index()
            output_vector[:, do_idx, pu_idx] = OD_timeseries.values
    return output_vector, group_idx


def preprocess_NYC_borough_dropoff(file_paths, location_ids, year=2019):

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

    data_list = []
    time_list = []
    for idx, path in enumerate(file_paths):

        # Load data
        df = pd.read_csv(path, parse_dates=[0, 1], dtype=dtype_cols, usecols=important_cols)

        # Extract area
        trip_idxs = df.PULocationID.isin(location_ids) & df.DOLocationID.isin(location_ids)
        df = df.loc[trip_idxs]

        print(f"{len(df)} eligble trips")

        # Remove too short trips
        df_len = len(df)
        df = df.loc[df.trip_distance > 0.1]
        print(f"removed {df_len - len(df)} too spatially short trips")
        df_len = len(df)

        # Remove to short trips temporally
        min_duration = dt.timedelta(minutes=1)
        df["trip_duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df = df.loc[df.trip_duration > min_duration]
        print(f"removed {df_len - len(df)} too temporally short trips")
        df_len = len(df)

        # Remove free trips and negative trips
        df = df.loc[df.fare_amount > 0]
        print(f"removed {df_len - len(df)} negative trips")
        df_len = len(df)

        # Remove observations with wrong dates and sort by time at the same time
        df = df.loc[
            (df.tpep_dropoff_datetime.dt.year == year)
            & (df.tpep_dropoff_datetime.dt.month == (idx + 1))
        ].sort_values("tpep_dropoff_datetime")
        print(f"removed {df_len - len(df)} trips with wrong datetime")
        df_len = len(df)

        # Add temporal bins on dopoff time
        df, n_bins_dt, bins_dt = add_temporal_bins(
            df, "tpep_dropoff_datetime", dt_freq="1H", year=year, month=(idx + 1)
        )
        print(f"Data from {df.tpep_dropoff_datetime.min()} to {df.tpep_dropoff_datetime.max()}")

        # Add spatial bins on dropoff zone
        n_spatial_bins = len(df.DOLocationID.unique())

        # Create binned vector
        binned_vector, group_idx = create_binned_vector(
            df=df,
            n_spatial_bins=n_spatial_bins,
            n_bins_dt=n_bins_dt,
            spatial_bins_name="DOLocationID",
            temporal_bins_name="time_bins",
        )

        if len(group_idx) != len(location_ids):
            print(f"Missing {set(location_ids) - set(group_idx)}")
            print(f"location_ids {len(location_ids)}")
            print(location_ids)
            print(f"group_idx {len(group_idx)}")
            print(group_idx)
            raise NameError("Not all groups represented!")

        if group_idx != sorted(group_idx):
            raise NameError("Group index is not sorted!")
        data_list.append(binned_vector)
        time_list.append(bins_dt[:-1])

        print(f"added {binned_vector.shape}")

        print(f"{path} done")

    full_year_vector = np.concatenate(data_list, axis=1)
    full_time_list = time_list[0].union_many(time_list[1:])

    return full_year_vector, full_time_list


def preprocess_NYC_borough_pickup(file_paths, location_ids, year=2019):

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

    data_list = []
    time_list = []
    for idx, path in enumerate(file_paths):

        # Load data
        df = pd.read_csv(path, parse_dates=[0, 1], dtype=dtype_cols, usecols=important_cols)

        # Extract area
        trip_idxs = df.PULocationID.isin(location_ids) & df.DOLocationID.isin(location_ids)
        df = df.loc[trip_idxs]

        print(f"{len(df)} eligble trips")

        # Remove too short trips
        df_len = len(df)
        df = df.loc[df.trip_distance > 0.1]
        print(f"removed {df_len - len(df)} too spatially short trips")
        df_len = len(df)

        # Remove to short trips temporally
        min_duration = dt.timedelta(minutes=1)
        df["trip_duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df = df.loc[df.trip_duration > min_duration]
        print(f"removed {df_len - len(df)} too temporally short trips")
        df_len = len(df)

        # Remove free trips and negative trips
        df = df.loc[df.fare_amount > 0]
        print(f"removed {df_len - len(df)} negative trips")
        df_len = len(df)

        # Remove observations with wrong dates and sort by time at the same time
        df = df.loc[
            (df.tpep_pickup_datetime.dt.year == year)
            & (df.tpep_pickup_datetime.dt.month == (idx + 1))
        ].sort_values("tpep_pickup_datetime")
        print(f"removed {df_len - len(df)} trips with wrong datetime")
        df_len = len(df)

        # Add temporal bins
        df, n_bins_dt, bins_dt = add_temporal_bins(
            df, "tpep_pickup_datetime", dt_freq="1H", year=year, month=(idx + 1)
        )
        print(f"Data from {df.tpep_pickup_datetime.min()} to {df.tpep_pickup_datetime.max()}")

        n_spatial_bins = len(df.PULocationID.unique())

        # Create binned vector
        binned_vector, group_idx = create_binned_vector(
            df=df,
            n_spatial_bins=n_spatial_bins,
            n_bins_dt=n_bins_dt,
            spatial_bins_name="PULocationID",
            temporal_bins_name="time_bins",
        )

        if len(group_idx) != len(location_ids):
            print(f"Missing {set(location_ids) - set(group_idx)}")
            print(f"location_ids {len(location_ids)}")
            print(location_ids)
            print(f"group_idx {len(group_idx)}")
            print(group_idx)
            raise NameError("Not all groups represented!")

        if group_idx != sorted(group_idx):
            raise NameError("Group index is not sorted!")
        data_list.append(binned_vector)
        time_list.append(bins_dt[:-1])

        print(f"added {binned_vector.shape}")

        print(f"{path} done")

    full_year_vector = np.concatenate(data_list, axis=1)
    full_time_list = time_list[0].union_many(time_list[1:])
    return full_year_vector, full_time_list
