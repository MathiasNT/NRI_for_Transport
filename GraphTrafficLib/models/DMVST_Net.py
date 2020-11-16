import torch
from torch_geometric.nn import GCNConv


class DMVST_Net(torch.nn.Module):
    """
    This model is based on Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction [Yao et al]
    """

    def __init__(
        self,
        input_dims,
        feature_dims,
        batch_size,
        lstm_dims,
        graph_features,
        gnn_dim,
        gnn_out_features,
        gnn_emb_dim,
        conv_channels,
        spatial_emb_size,
    ):
        super(DMVST_Net, self).__init__()

        # General Params
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.feature_dims = feature_dims

        # LSTM params
        self.lstm_dims = lstm_dims

        # Graph embed params
        self.graph_features = graph_features
        self.gnn_dim = gnn_dim
        self.gnn_out_features = gnn_out_features
        self.gnn_emb_dim = gnn_emb_dim

        # Spatial embedding params
        self.conv_channels = conv_channels
        self.spatial_emb_size = spatial_emb_size

        # Convolution layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.spatial_fc = torch.nn.Linear(
            in_features=self.input_dims * self.conv_channels,
            out_features=self.spatial_emb_size,
        )

        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=self.spatial_emb_size + self.feature_dims,
            hidden_size=self.lstm_dims,
            batch_first=True,
        )
        # self.register_parameter(name='h0', param=torch.nn.Parameter(torch.randn(1, self.batch_size, self.lstm_dims).type(torch.FloatTensor)))
        # self.register_parameter(name='c0', param=torch.nn.Parameter(torch.randn(1, self.batch_size, self.lstm_dims).type(torch.FloatTensor)))

        # GNN layers
        self.gnn_conv1 = GCNConv(self.graph_features, self.gnn_dim)
        self.gnn_conv2 = GCNConv(self.gnn_dim, self.gnn_dim)
        self.gnn_conv3 = GCNConv(
            self.gnn_dim, self.gnn_out_features
        )  # Outpt features could in principle be anything
        self.gnn_emb_fc = torch.nn.Linear(
            self.input_dims * self.gnn_out_features, self.gnn_emb_dim
        )

        # Output MLP Layer
        self.out_fc = torch.nn.Linear(
            in_features=self.lstm_dims + self.gnn_emb_dim, out_features=self.input_dims
        )

        self.relu = torch.nn.ReLU()

    def embed_graph(self, node_vals, edge_idx):
        h = self.gnn_conv1(node_vals, edge_idx)
        h = self.relu(h)
        h = self.gnn_conv2(h, edge_idx)
        h = self.relu(h)
        h = self.gnn_conv3(h, edge_idx)
        h = self.relu(h)
        gnn_emb = self.gnn_emb_fc(h.squeeze())
        return gnn_emb

    def embed_spatial(self, x):
        # Do convolution-spatial-embedding for each timestep
        spatial_emb_local = []
        for t in range(x.shape[1]):
            h1 = self.conv1(x[:, t, :, :].unsqueeze(1))
            h1 = self.relu(h1)

            h2 = self.conv2(h1)
            h2 = self.relu(h2)

            h3 = self.conv3(h2)
            h3 = self.relu(h3)

            h3 = h3.view(x.shape[0], -1)
            h4 = self.spatial_fc(h3)
            h4 = self.relu(h4)

            spatial_emb_local.append(h4.unsqueeze(1))

        return torch.cat(spatial_emb_local, 1)

    def forward(self, x, features, node_vals, edge_idx):
        # Get spatial embedding
        spatial_emb = self.embed_spatial(x)
        input = torch.cat((spatial_emb, features), 2)

        # Send through lstm
        _, (hn, cn) = self.lstm(input)
        hn = self.relu(hn)

        # Send through gnn
        gnn_emb = self.embed_graph(node_vals, edge_idx)
        gnn_emb = gnn_emb.repeat(x.shape[0], 1).unsqueeze(0)

        # Concat embeddings
        full_emb = torch.cat((hn, gnn_emb), 2)

        # Compute output
        out = self.out_fc(full_emb)
        return out.permute(1, 0, 2)
