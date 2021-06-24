import warnings
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import folium
from folium.plugins import PolyLineTextPath


class Encoder_Visualizer(object):
    def __init__(
        self,
        encoder,
        rel_rec,
        rel_send,
        burn_in,
        burn_in_steps,
        split_len,
    ):
        super().__init__()
        self.rel_rec = rel_rec
        self.rel_send = rel_send
        self.encoder = encoder
        self.burn_in = burn_in
        self.burn_in_steps = burn_in_steps
        self.split_len = split_len

    def infer_graphs(self, data, gumbel_temp):
        graph_list = []
        graph_probs = []
        self.encoder.eval()
        for _, data in enumerate(data):
            with torch.no_grad():
                data = data.unsqueeze(dim=0).cuda()
                logits = self.encoder(data, self.rel_rec, self.rel_send)
                edges = F.gumbel_softmax(logits, tau=gumbel_temp, hard=True)
                edge_probs = F.softmax(logits, dim=-1)
                graph_list.append(edges.cpu())
                graph_probs.append(edge_probs.cpu())
        return graph_list, graph_probs

    def infer_max_graphs(self, data):
        graph_list = []
        self.encoder.evel()
        for _, data in enumerate(data):
            with torch.no_grad():
                data.unsqueeze(dim=0).cuda()
                logits = self.encoder(data, self.rel_rec, self.rel_send)


def visualize_all_graph_adj(graph_list, rel_send, rel_rec, dates):
    _, axs = plt.subplots(len(graph_list), 1, figsize=(50, 50))
    for k, graph in enumerate(graph_list):
        for j in range(1):
            adj_matrix = torch.zeros(rel_rec.shape[1], rel_rec.shape[1])
            for i, row in enumerate(graph[j]):
                if row.argmax().item():
                    send = rel_send[i].argmax().item()
                    rec = rel_rec[i].argmax().item()
                    adj_matrix[rec, send] = 1
            axs[k].imshow(adj_matrix)
            axs[k].set_title(f"{k} - {dates[k]}")


def visualize_prob_adj(edge_list, rel_send, rel_rec):
    adj_matrix = torch.zeros(rel_rec.shape[1], rel_rec.shape[1])
    for i, row in enumerate(edge_list):
        send = rel_send[i].argmax().item()
        rec = rel_rec[i].argmax().item()
        adj_matrix[send, rec] = row[1]
    return adj_matrix


def visualize_mean_graph_adj(graph_list, rel_send, rel_rec):
    all_graphs = torch.stack(graph_list[:-1])
    mean_graph = all_graphs.mean(dim=(0, 1))
    mean_adj_matrix = torch.zeros(rel_rec.shape[1], rel_rec.shape[1])
    for i, row in enumerate(mean_graph):
        if row.argmax().item():
            send = rel_send[i].argmax().item()
            rec = rel_rec[i].argmax().item()
            mean_adj_matrix[rec, send] = 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mean_adj_matrix)
    return mean_adj_matrix


def plot_adj_on_map(adj_matrix, map_shp):
    # plot base map
    m = folium.Map(
        location=[40.8, -73.8],  # center of the folium map
        tiles="cartodbpositron",  # type of map
        zoom_start=12,
    )  # initial zoom

    # plot chorpleth over the base map
    folium.Choropleth(
        map_shp,  # geo data
        data=map_shp,  # data
        key_on="feature.properties.borough",  # feature.properties.key
        columns=["zone", "Shape_Area"],  # [key, value]
        fill_color="RdPu",  # cmap
        line_weight=1,
        line_opacity=1,
        fill_opacity=0.2,
    ).add_to(m)

    for i, row in map_shp.iterrows():
        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=f"{row.LocationID}{row.zone}",
            radius=1,
        ).add_to(m)

    for idx, row in enumerate(adj_matrix):
        neighbour_idxs = torch.nonzero(row)
        zone_centroid = [
            map_shp.iloc[idx].geometry.centroid.y,
            map_shp.iloc[idx].geometry.centroid.x,
        ]
        for neighbour_idx in neighbour_idxs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                neighbour_centroid = [
                    map_shp.iloc[neighbour_idx].geometry.centroid.y,
                    map_shp.iloc[neighbour_idx].geometry.centroid.x,
                ]
                folium.PolyLine([zone_centroid, neighbour_centroid]).add_to(m)
    # add layer controls
    folium.LayerControl().add_to(m)

    return m


def plot_directed_adj_on_map(adj_matrix, map_shp):
    # plot base map
    m = folium.Map(
        location=[40.8, -73.8],  # center of the folium map
        tiles="cartodbpositron",  # type of map
        zoom_start=12,
    )  # initial zoom

    # plot chorpleth over the base map
    folium.Choropleth(
        map_shp,  # geo data
        data=map_shp,  # data
        key_on="feature.properties.borough",  # feature.properties.key
        columns=["zone", "Shape_Area"],  # [key, value]
        fill_color="RdPu",  # cmap
        line_weight=1,
        line_opacity=1,
        fill_opacity=0.2,
    ).add_to(m)

    for i, row in map_shp.iterrows():
        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=f"{row.LocationID}{row.zone}",
            radius=1,
        ).add_to(m)

    for idx, row in enumerate(adj_matrix):
        sender_idxs = torch.nonzero(row)
        zone_centroid = [
            map_shp.iloc[idx].geometry.centroid.y,
            map_shp.iloc[idx].geometry.centroid.x,
        ]
        for sender_idx in sender_idxs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sender_centroid = [
                    map_shp.iloc[sender_idx].geometry.centroid.y,
                    map_shp.iloc[sender_idx].geometry.centroid.x,
                ]
                line = folium.PolyLine([zone_centroid, sender_centroid])
                line.add_to(m)
                PolyLineTextPath(
                    line,
                    " < ",
                    repeat=True,
                    center=True,
                    offset=0,
                    attributes={"font-size": "12"},
                ).add_to(m)

    # add layer controls
    folium.LayerControl().add_to(m)

    return m
