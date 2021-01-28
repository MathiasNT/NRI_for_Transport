from torch.utils.data import Dataset


class Nyc_w_Weather(Dataset):
    def __init__(self, timeseries, weather, list_IDs):
        super(Nyc_w_Weather, self).__init__()
        self.timeseries = timeseries
        self.weatherseries = weather
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Get the id of the sample (here just the same number but hey we want general)
        ID = self.list_IDs[index]

        # Load data
        time_data = self.timeseries[ID]
        X = time_data[:-1, :]
        y = time_data[-1:, :]
        weather = self.weatherseries[ID]

        return X, y, weather[:-1, :]


class Nyc_w_Weather2(Dataset):
    def __init__(self, timeseries, weather, list_IDs):
        super(Nyc_w_Weather2, self).__init__()
        self.timeseries = timeseries
        self.weatherseries = weather
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Get the id of the sample (here just the same number but hey we want general)
        ID = self.list_IDs[index]

        # Load data
        time_data = self.timeseries[ID]
        weather = self.weatherseries[ID]

        return time_data, weather
