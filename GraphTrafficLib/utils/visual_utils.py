import warnings
import numpy as np
import matplotlib
import matplotlib.dates as mdates
from matplotlib import cm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import folium
from folium.plugins import PolyLineTextPath
import contextily as cx
from pyproj import Geod


# TODO: remove all of the code that is not used for final plots.


class Encoder_Visualizer(object):
    def __init__(
        self,
        encoder,
        rel_rec,
        rel_send,
        burn_in,
        burn_in_steps,
        split_len,
        use_weather,
    ):
        super().__init__()
        self.rel_rec = rel_rec
        self.rel_send = rel_send
        self.encoder = encoder
        self.burn_in = burn_in
        self.burn_in_steps = burn_in_steps
        self.split_len = split_len
        self.use_weather = use_weather

    def infer_graphs(self, data, gumbel_temp):
        graph_list = []
        graph_probs = []
        self.encoder.eval()
        for _, (data, weather) in enumerate(data):
            with torch.no_grad():
                data = data.unsqueeze(dim=0).cuda()
                if self.use_weather:
                    weather = weather.unsqueeze(dim=0).cuda()
                    logits = self.encoder(
                        data[:, :, : self.burn_in_steps, :],
                        weather,
                        self.rel_rec,
                        self.rel_send,
                    )
                else:
                    logits = self.encoder(
                        data[:, :, : self.burn_in_steps, :], self.rel_rec, self.rel_send
                    )
                edges = F.gumbel_softmax(logits, tau=gumbel_temp, hard=True)
                edge_probs = F.softmax(logits, dim=-1)
                graph_list.append(edges.cpu())
                graph_probs.append(edge_probs.cpu())
        return graph_list, graph_probs


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

    for i, row in map_shp.reset_index().iterrows():
        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=f"{i}: {row.LocationID} {row.zone}",
            radius=2,
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


def plot_top_bot_k_rels(adj, topk_idxs, botk_idxs=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(adj)

    for i, topk_idx in enumerate(topk_idxs):
        plt.axvline(x=topk_idx, color="red", alpha=1, linewidth=(len(topk_idxs) - i))
        plt.axhline(y=topk_idx, color="red", alpha=1, linewidth=(len(topk_idxs) - i))

    if botk_idxs is not None:
        for i, botk_idx in enumerate(botk_idxs):
            plt.axvline(x=botk_idx, color="green", alpha=1, linewidth=(len(botk_idxs) - i))
            plt.axhline(y=botk_idx, color="green", alpha=1, linewidth=(len(botk_idxs) - i))


def get_rels_from_topk(topk_idxs, adj):
    rels = []
    for i, topk_idx in enumerate(topk_idxs):
        temp_adj = torch.zeros_like(adj)
        temp_adj[topk_idx, :] = adj[topk_idx, :]
        temp_adj[:, topk_idx] = adj[:, topk_idx]
        rels.append(temp_adj)
    return rels


def get_rels_from_zone_id(zone_idx, adj):
    temp_adj = torch.zeros_like(adj)
    temp_adj[zone_idx, :] = adj[zone_idx, :]
    temp_adj[:, zone_idx] = adj[:, zone_idx]
    return temp_adj


def plot_adj_w_grid(adj):
    plt.imshow(adj)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_yticks([i for i in range(len(adj))], minor="True")
    ax.set_xticks([i for i in range(len(adj))], minor="True")
    ax.xaxis.grid(True, which="both", alpha=0.25)
    ax.yaxis.grid(True, which="both", alpha=0.25)
    return ax


def plot_zone_and_map(adj, zone_idx, map_shp, text=None, timestep=None, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.axis("off")
        fig.tight_layout(pad=0)
        ax.margins(0)
        if text is not None:
            ax.set_title(text, y=0.9)

        gs = plt.GridSpec(2, 3, figure=fig, width_ratios=[2, 2, 1])
        ax0 = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[:, 1])
        ax20 = fig.add_subplot(gs[0, 2])
        ax21 = fig.add_subplot(gs[1, 2])

    if timestep is not None:
        map_shp.plot(column=f"ts_{timestep}", ax=ax0, alpha=0.75)
        map_shp.plot(column=f"ts_{timestep}", ax=ax1, alpha=0.75)
    else:
        map_shp.plot(column="mean_activity", ax=ax0, alpha=0.75)
        map_shp.plot(column="mean_activity", ax=ax1, alpha=0.75)
    ax0.axis("off")
    ax0.set_title("Receiving edges", x=0.3, y=0.8)
    ax1.axis("off")
    ax1.set_title("Sending edges", x=0.3, y=0.8)

    for idx, row in enumerate(adj):
        sender_idxs = torch.nonzero(row).numpy()
        zone_centroid = [
            map_shp.iloc[idx].geometry.centroid.y,
            map_shp.iloc[idx].geometry.centroid.x,
        ]
        for sender_idx in sender_idxs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sender_centroid = [
                    map_shp.iloc[sender_idx.squeeze()].geometry.centroid.y,
                    map_shp.iloc[sender_idx.squeeze()].geometry.centroid.x,
                ]
                x_values = np.array([zone_centroid[1], sender_centroid[1]])
                y_values = np.array([zone_centroid[0], sender_centroid[0]])
                val = row[sender_idx].numpy()
                if idx == zone_idx.item():  # if they match it is a receiving edge
                    color = cm.Reds(val)[0]
                    ax0.plot(x_values, y_values, color=color, linewidth=1.5)
                else:
                    color = cm.Greens(val)[0]
                    ax1.plot(x_values, y_values, color=color, linewidth=1.5)

    ax20.imshow(adj)
    ax20.set_aspect("equal")
    ax20.set_yticks(range(len(adj)), minor="True")
    ax20.set_xticks(range(len(adj)), minor="True")
    ax20.xaxis.grid(True, which="both", alpha=0.25)
    ax20.yaxis.grid(True, which="both", alpha=0.25)
    ax20.set_title("Adjacancy matrix")

    ax21.axis("off")
    return fig


def visualize_continous_adj(edge_list, rel_send, rel_rec):
    adj_matrix = torch.zeros(rel_rec.shape[1], rel_rec.shape[1])
    for i, row in enumerate(edge_list):
        send = rel_send[i].argmax().item()
        rec = rel_rec[i].argmax().item()
        adj_matrix[rec, send] = row[1]
    return adj_matrix


# Function taken from
# https://discuss.pytorch.org/t/how-can-i-merge-diagonal-and-off-diagonal-matrix-elements-into-a-single-matrix/128074/3
def merge_on_and_off_diagonal(on_diag, off_diag):
    # store output shape, to remove and unsqueeze'd dims
    output_shape = (*off_diag.shape[:-1], off_diag.shape[-2])
    if len(on_diag.shape) == 1 and len(off_diag.shape) == 2:
        on_diag = on_diag.unsqueeze(0).unsqueeze(1)
        off_diag = off_diag.unsqueeze(0).unsqueeze(1)
    elif len(on_diag.shape) == 2 and len(off_diag.shape) == 3:
        on_diag = on_diag.unsqueeze(1)
        off_diag = off_diag.unsqueeze(1)
    # reform input on_diag and off_diag to shape
    # B = batch, D = number of vector for given batch, A = dimension of vector
    # on_diag shape:  [B, D, A]
    # off_diag shape: [B, D, A, A-1]
    if on_diag.shape[-1] != off_diag.shape[-2]:
        raise ValueError("index on_diag.shape[-1] must match off_diag.shape[-2]")
    dim = len(on_diag.shape)
    tmp = torch.cat(
        (
            on_diag[:, :, :-1].unsqueeze(dim),
            off_diag.view((*off_diag.shape[0 : (dim - 1)], off_diag.shape[-1], off_diag.shape[-2])),
        ),
        dim=dim,
    )
    res = torch.cat(
        (tmp.view(*off_diag.shape[0 : (dim - 1)], -1), on_diag[:, :, -1].unsqueeze(2)),
        dim=dim - 1,
    ).view(*off_diag.shape[0 : (dim - 1)], on_diag.shape[-1], on_diag.shape[-1])

    return res.view(output_shape)


def plot_adj_and_time(adj, time_str):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(adj, vmin=0, vmax=1)
    ax.set_title(time_str)
    return fig


def plot_diff_adj_and_time(adj, time_str):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(adj, cmap="bwr")
    ax.set_title(time_str)
    return fig


def plot_pems_adj_connection_map(
    zone_idx,
    map_shp,
    adj,
    text=None,
    timestep=None,
    vmin=None,
    vmax=None,
    xlim=None,
    ylim=None,
    cmap="bwr",
):
    fig, ax = plt.subplots(2, figsize=(5, 10))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    # plot map and loop detectors
    if timestep is not None:
        map_shp.plot(
            column=f"ts_{timestep}",
            ax=ax[0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.75,
            markersize=16,
            zorder=2,
            legend=True,
            legend_kwds={"pad": 0.04, "fraction": 0.043, "label": "Traffic speed (mph)"},
        )
    else:
        map_shp.plot(ax=ax[0])
    cx.add_basemap(
        ax[0],
        source=cx.providers.Stamen.TonerLines,
        attribution=False,
        zorder=1,
        alpha=0.5,
        zoom=12,
    )
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    # cx.add_basemap(ax[0], source=cx.providers.OpenStreetMap.HOT, alpha=0.5)

    if text is not None:
        ax[0].set_title(text)

    important_ids = []
    for idx, row in enumerate(adj):
        sender_idxs = torch.nonzero(row)
        zone_centroid = [
            map_shp.iloc[idx].geometry.centroid.y,
            map_shp.iloc[idx].geometry.centroid.x,
        ]
        if row.sum() != 0:
            for sender_idx in sender_idxs:
                sender_centroid = [
                    map_shp.iloc[sender_idx.squeeze().numpy()].geometry.centroid.y,
                    map_shp.iloc[sender_idx.squeeze().numpy()].geometry.centroid.x,
                ]
                x_values = np.array([zone_centroid[1], sender_centroid[1]])
                y_values = np.array([zone_centroid[0], sender_centroid[0]])
                val = row[sender_idx]
                if idx == zone_idx:  # if they match it is a receiving edge
                    map_shp.iloc[[sender_idx]].plot(
                        column=f"ts_{timestep}",
                        ax=ax[0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=3,
                        markersize=60,
                    )
                    color = cm.Blues(val.numpy())[0]
                    ax[0].plot(x_values, y_values, color=color, linewidth=3, linestyle="--")
                    important_ids.append(sender_idx)
                    # ax.plot(x_values, y_values, color="blue", linewidth=1.5)
                else:
                    map_shp.iloc[[idx]].plot(
                        column=f"ts_{timestep}",
                        ax=ax[0],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        zorder=3,
                        markersize=60,
                    )
                    color = cm.Reds(val.numpy())[0]
                    ax[0].plot(x_values, y_values, color=color, linewidth=3, linestyle="--")
                    important_ids.append(idx)
                    # ax.plot(x_values, y_values, color="red", linewidth=1.5)

    map_shp.iloc[[zone_idx]].plot(
        column=f"ts_{timestep}",
        ax=ax[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
        markersize=100,
    )
    important_ids.append(zone_idx)

    map_shp.iloc[important_ids].plot(
        ax=ax[1], column="id", legend=True, alpha=1, zorder=2, markersize=80
    )
    map_shp.plot(ax=ax[1], alpha=0, zorder=2)
    cx.add_basemap(
        ax[1],
        source=cx.providers.Stamen.TonerLines,
        attribution=False,
        zorder=1,
        alpha=0.5,
        zoom=12,
    )
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)

    ax[1].set_title("Loop sensor ids")

    fig.tight_layout(h_pad=-8)

    return fig


def plot_pems_adj_on_map(
    zone_idx, map_shp, adj, text=None, timestep=None, vmin=None, vmax=None, xlim=None, ylim=None
):
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.set_xticks([])
    ax.set_yticks([])
    # plot map and loop detectors
    if timestep is not None:
        map_shp.plot(
            column=f"ts_{timestep}",
            ax=ax,
            cmap="bwr",
            vmin=vmin,
            vmax=vmax,
            alpha=0.75,
            markersize=16,
            zorder=2,
        )
    else:
        map_shp.plot(ax=ax)
    cx.add_basemap(
        ax, source=cx.providers.Stamen.TonerLines, attribution=False, zorder=1, alpha=0.5, zoom=12
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # cx.add_basemap(ax, source=cx.providers.OpenStreetMap.HOT, alpha=0.5)

    if text is not None:
        ax.set_title(text)

    for idx, row in enumerate(adj):
        sender_idxs = torch.nonzero(row)
        zone_centroid = [
            map_shp.iloc[idx].geometry.centroid.y,
            map_shp.iloc[idx].geometry.centroid.x,
        ]
        if row.sum() != 0:
            for sender_idx in sender_idxs:
                sender_centroid = [
                    map_shp.iloc[sender_idx.squeeze().numpy()].geometry.centroid.y,
                    map_shp.iloc[sender_idx.squeeze().numpy()].geometry.centroid.x,
                ]
                x_values = np.array([zone_centroid[1], sender_centroid[1]])
                y_values = np.array([zone_centroid[0], sender_centroid[0]])
                val = row[sender_idx]
                if idx == zone_idx:  # if they match it is a receiving edge
                    map_shp.iloc[[sender_idx]].plot(
                        column=f"ts_{timestep}",
                        ax=ax,
                        cmap="bwr",
                        vmin=vmin,
                        vmax=vmax,
                        zorder=3,
                        markersize=60,
                    )
                    color = cm.Blues(val.numpy())[0]
                    ax.plot(x_values, y_values, color=color, linewidth=3, linestyle="--")
                    # ax.plot(x_values, y_values, color="blue", linewidth=1.5)
                else:
                    map_shp.iloc[[idx]].plot(
                        column=f"ts_{timestep}",
                        ax=ax,
                        cmap="bwr",
                        vmin=vmin,
                        vmax=vmax,
                        zorder=3,
                        markersize=60,
                    )
                    color = cm.Reds(val.numpy())[0]
                    ax.plot(x_values, y_values, color=color, linewidth=3, linestyle="--")
                    # ax.plot(x_values, y_values, color="red", linewidth=1.5)

    map_shp.iloc[[zone_idx]].plot(
        column=f"ts_{timestep}",
        ax=ax,
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        zorder=3,
        markersize=100,
    )

    return fig


def PEMS_folium_plot(gdf, adj_matrix, sensor_ind_to_id):
    map = folium.Map(control_scale=True)

    # add sensors to map
    loop_sensor_layer = folium.FeatureGroup(name="loop sensors")
    geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in gdf.geometry]
    for i, coordinates in enumerate(geo_df_list):
        loop_sensor_layer.add_child(
            folium.CircleMarker(
                location=coordinates,
                popup=f"ID: {gdf.index[i]}",
                color="red",
                radius=3,
            )
        )
    loop_sensor_layer.add_to(map)

    # add connections to map
    geodesic = Geod(ellps="WGS84")
    arrow_layer = folium.FeatureGroup("arrows")
    for sender, row in enumerate(adj_matrix):
        receivers = np.nonzero(row)[0]
        sender_id = int(sensor_ind_to_id[sender])
        sender_point = list(gdf.loc[sender_id].values)
        for receiver in receivers:
            receiver_id = int(sensor_ind_to_id[receiver])
            receiver_point = list(gdf.loc[receiver_id].values)
            line = folium.PolyLine(
                [sender_point[:2], receiver_point[:2]], tooltip=f"{sender_id} to {receiver_id}"
            )
            line.add_to(map)
            rot = (
                geodesic.inv(
                    receiver_point[1], receiver_point[0], sender_point[1], sender_point[0]
                )[0]
                + 90
            )
            # create your arrow
            folium.RegularPolygonMarker(
                location=list((np.array(receiver_point[:2]) + np.array(sender_point[:2])) / 2),
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=1,
                number_of_sides=3,
                rotation=rot,
                radius=5,
            ).add_to(arrow_layer)
    arrow_layer.add_to(map)

    map.fit_bounds(
        [gdf[["lat", "lon"]].min().values.tolist(), gdf[["lat", "lon"]].max().values.tolist()]
    )
    folium.LayerControl().add_to(map)

    return map


def update_pos(temp):
    """
    Used to update the positions to unclutter pems plot
    """
    offset = temp[0] - temp[1]
    new_1 = temp[0] + 5 * offset
    new_2 = temp[1] - 5 * offset
    return np.array([new_1, new_2])


def plot_pems_timeseries_and_map_two_col(
    zone_idxs,
    test_dates,
    in_sum_ts,
    out_sum_ts,
    yn_true,
    first_pred_step,
    gdf,
    df,
    xtick_hour_interval=6,
    time_slice=None,
    time_emp=None,
    fontsize=18,
):

    matplotlib.rcParams.update({"font.size": fontsize})

    if time_slice is None:
        time_slice = slice(0, len(test_dates))

    myFmt = mdates.DateFormatter("%A %H:%M")
    n_rows = np.int(np.ceil(len(zone_idxs) // 2))

    fig, ax = plt.subplots(n_rows, 2, figsize=(15 * 2, 5 * n_rows))
    for i, zone_idx in enumerate(zone_idxs):
        row = int(np.ceil(i // 2))
        col = i % 2
        ax[row, col].set_title(f"Sensor ID: {df.T.index[zone_idx]}", fontsize=fontsize + 1)
        lns1 = ax[row, col].plot(
            test_dates[time_slice],
            in_sum_ts[zone_idx, time_slice],
            color="tab:blue",
            label="Mean ingoing edge probability",
        )
        lns2 = ax[row, col].plot(
            test_dates[time_slice],
            out_sum_ts[zone_idx, time_slice],
            color="tab:red",
            label="Mean outgoing edge probability",
        )
        ax2 = ax[row, col].twinx()
        lns3 = ax2.plot(
            test_dates[time_slice],
            yn_true[time_slice, zone_idx, first_pred_step - 1],
            color="tab:green",
            alpha=1,
            label="Traffic speed",
        )
        lns4 = ax2.plot(
            test_dates[time_slice],
            yn_true[time_slice, :, first_pred_step - 1].mean(1),
            color="tab:green",
            alpha=0.5,
            linestyle="--",
            label="Mean Traffic speed",
        )

        if time_emp is not None:
            ax[row, col].axvline(time_emp, linestyle="--", color="black")

        ax[row, col].set_ylim(0, 1)
        ax[row, col].xaxis.set_major_locator(mdates.HourLocator(interval=xtick_hour_interval))
        ax[row, col].xaxis.set_major_formatter(myFmt)
        ax[row, col].set_ylabel("Mean edge probability")
        ax2.set_ylim(0, 80)
        ax2.set_ylabel("Traffic speed (mph)")

        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        # ax[i].legend(lns, labs, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(lns, labs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(h_pad=1)


def plot_pems_timeseries_and_map(
    zone_idxs,
    test_dates,
    in_sum_ts,
    out_sum_ts,
    yn_true,
    first_pred_step,
    gdf,
    df,
    xtick_hour_interval=6,
    time_slice=None,
    time_emp=None,
    fontsize=18,
):

    matplotlib.rcParams.update({"font.size": fontsize})

    if time_slice is None:
        time_slice = slice(0, len(test_dates))

    myFmt = mdates.DateFormatter("%A %H:%M")
    n_zones = len(zone_idxs)

    fig, ax = plt.subplots(n_zones, 1, figsize=(15, 5 * n_zones))
    for i, zone_idx in enumerate(zone_idxs):
        ax[i].set_title(f"Sensor ID: {df.T.index[zone_idx]}", fontsize=fontsize + 1)
        lns1 = ax[i].plot(
            test_dates[time_slice],
            in_sum_ts[zone_idx, time_slice],
            color="tab:blue",
            label="Mean ingoing edge probability",
        )
        lns2 = ax[i].plot(
            test_dates[time_slice],
            out_sum_ts[zone_idx, time_slice],
            color="tab:red",
            label="Mean outgoing edge probability",
        )
        ax2 = ax[i].twinx()
        lns3 = ax2.plot(
            test_dates[time_slice],
            yn_true[time_slice, zone_idx, first_pred_step - 1],
            color="tab:green",
            alpha=1,
            label="Traffic speed",
        )
        lns4 = ax2.plot(
            test_dates[time_slice],
            yn_true[time_slice, :, first_pred_step - 1].mean(1),
            color="tab:green",
            alpha=0.5,
            linestyle="--",
            label="Mean Traffic speed",
        )

        if time_emp is not None:
            ax[i].axvline(time_emp, linestyle="--", color="black")

        ax[i].set_ylim(0, 1)
        ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=xtick_hour_interval))
        ax[i].xaxis.set_major_formatter(myFmt)
        ax[i].set_ylabel("Mean edge probability")
        ax2.set_ylim(0, 80)
        ax2.set_ylabel("Traffic speed (mph)")

        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        # ax[i].legend(lns, labs, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(lns, labs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(h_pad=1)


def plot_pems_timeseries_and_map_old(
    zone_idxs,
    test_dates,
    in_sum_ts,
    out_sum_ts,
    yn_true,
    first_pred_step,
    gdf,
    map_xlim,
    map_ylim,
    df,
    xtick_hour_interval=6,
    time_slice=None,
    time_emp=None,
):

    matplotlib.rcParams.update({"font.size": 18})

    if time_slice is None:
        time_slice = slice(0, len(test_dates))

    myFmt = mdates.DateFormatter("%A %H:%M")
    n_zones = len(zone_idxs)

    fig, ax = plt.subplots(
        n_zones, 2, figsize=(25, 7 * n_zones), gridspec_kw={"width_ratios": [5, 1]}
    )
    for i, zone_idx in enumerate(zone_idxs):
        ax[i, 0].set_title(f"Sensor ID: {df.T.index[zone_idx]}")
        lns1 = ax[i, 0].plot(
            test_dates[time_slice],
            in_sum_ts[zone_idx, time_slice],
            color="tab:blue",
            label="Mean ingoing edge probability",
        )
        lns2 = ax[i, 0].plot(
            test_dates[time_slice],
            out_sum_ts[zone_idx, time_slice],
            color="tab:red",
            label="Mean outgoing edge probability",
        )
        ax2 = ax[i, 0].twinx()
        lns3 = ax2.plot(
            test_dates[time_slice],
            yn_true[time_slice, zone_idx, first_pred_step - 1],
            color="tab:green",
            alpha=1,
            label="Traffic speed",
        )
        lns4 = ax2.plot(
            test_dates[time_slice],
            yn_true[time_slice, :, first_pred_step - 1].mean(1),
            color="tab:green",
            alpha=0.5,
            linestyle="--",
            label="Mean Traffic speed",
        )

        if time_emp is not None:
            ax[i, 0].axvline(time_emp, linestyle="--", color="black")

        ax[i, 0].set_ylim(0, 1)
        ax[i, 0].xaxis.set_major_locator(mdates.HourLocator(interval=xtick_hour_interval))
        ax[i, 0].xaxis.set_major_formatter(myFmt)
        ax[i, 0].set_ylabel("Mean edge probability")
        ax2.set_ylim(0, 80)
        ax2.set_ylabel("Traffic speed (mph)")

        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        ax[i, 0].legend(lns, labs, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)

        sensor = gdf.iloc[[zone_idx]]
        ax[i, 1].set_title("Sensor location")
        ax[i, 1].set_xlim(map_xlim)
        ax[i, 1].set_ylim(map_ylim)
        sensor.plot(ax=ax[i, 1], legend=True, alpha=1, zorder=2, color="red")
        cx.add_basemap(
            ax[i, 1],
            source=cx.providers.Stamen.TonerLines,
            zoom=12,
            zorder=1,
            alpha=0.5,
            attribution=False,
        )
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])

    fig.tight_layout(h_pad=3)
