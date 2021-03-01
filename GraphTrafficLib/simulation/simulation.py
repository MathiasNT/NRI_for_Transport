import numpy as np


class Zone:
    """ Class for the different zones.
        Has a position and a raw timeseries
    """

    def __init__(self, n_days, noise_mean=0, noise_std=0):
        self.pos = np.random.uniform(low=0, high=10, size=2)
        hours = n_days * 24

        self.ts_bias = np.random.uniform(low=0, high=60, size=1)
        self.ts_amplitude = np.random.uniform(low=20, high=60, size=1)
        self.demand_timeseries = np.squeeze(
            np.array(
                [
                    np.floor(
                        self.ts_bias
                        + self.ts_amplitude * np.cos((hour - 2) * np.pi / 12)
                        + np.random.normal(noise_mean, noise_std)
                    )
                    for hour in range(hours)
                ]
            )
        ).clip(min=0)
        self.dropoff_timeseries = np.zeros(hours + 1)

    def set_dropoff_timeseries(self, dropoff_timeseries):
        self.dropoff_timeseries = dropoff_timeseries

    def add_to_dropoff_timeseries(self, dropoff_contrib):
        self.dropoff_timeseries += dropoff_contrib

    def calc_dropoff_contrib(self, contrib_fracs):
        contribs_arr = []
        for frac in contrib_fracs:
            contribs_arr.append(np.floor(self.demand_timeseries * frac))

        contribs_arr = np.array(contribs_arr)
        contribs_arr = np.concatenate(
            (np.zeros((len(contrib_fracs), 1)), contribs_arr), axis=1
        )
        return contribs_arr


class Simulation:
    """The simulation class that uses the zone class
    """

    def __init__(self, n_zones, n_days, n_neighbours):
        self.n_zones = n_zones
        self.n_days = n_days
        self.zones = [Zone(n_days) for _ in range(n_zones)]
        self.n_neighbours = n_neighbours
        self.adj = 0
        self.dist_matrix = 0

    def create_distance_graph(self):
        # Calculate the distance between zones
        dist_matrix = np.zeros((self.n_zones, self.n_zones))
        for i, zone_1 in enumerate(self.zones):
            for j, zone_2 in enumerate(self.zones):
                dist_matrix[i, j] = np.linalg.norm(zone_1.pos - zone_2.pos, ord=2)

        # Create adjacancy matrix with n_neighbours neighbours
        adj = dist_matrix + np.eye(self.n_zones) * 1000
        np.put_along_axis(
            adj,
            np.argpartition(adj, self.n_neighbours, axis=1)[:, self.n_neighbours :],
            0,
            axis=1,
        )
        adj = adj != 0
        self.adj = adj
        self.dist_matrix = dist_matrix
        return adj, dist_matrix

    def simulate_dropoffs(self):
        for i, zone in enumerate(self.zones):
            contrib_fracs = [1 / self.n_neighbours for _ in range(self.n_neighbours)]
            contrib_arrs = zone.calc_dropoff_contrib(contrib_fracs)
            for j, neighbour in enumerate(self.adj[i].nonzero()[0]):
                self.zones[neighbour].add_to_dropoff_timeseries(contrib_arrs[j])

    def get_demand_matrix(self):
        # Note that cut some ends of for easier handling afterwards
        timeseries_arr = []
        for zone in self.zones:
            timeseries_arr.append(zone.demand_timeseries[1:-1])
        return np.array(timeseries_arr)

    def get_dropoff_matrix(self):
        # Note that i cut some ends of for easier handling afterwards
        timeseries_arr = []
        for zone in self.zones:
            timeseries_arr.append(zone.dropoff_timeseries[1:-2])
        return np.array(timeseries_arr)

