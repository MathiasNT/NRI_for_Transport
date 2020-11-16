import torch
from torch_geometric.nn import GCNConv


class LSTM_GNN_feature_model(torch.nn.Module):
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
    ):
        super(LSTM_GNN_feature_model, self).__init__()

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

        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=self.input_dims + self.feature_dims,
            hidden_size=self.lstm_dims,
            batch_first=True,
        )
        # self.register_parameter(name='h0', param=torch.nn.Parameter(torch.randn(1, self.batch_size, self.lstm_dims).type(torch.FloatTensor)))
        # self.register_parameter(name='c0', param=torch.nn.Parameter(torch.randn(1, self.batch_size, self.lstm_dims).type(torch.FloatTensor)))

        # GNN layers
        self.conv1 = GCNConv(self.graph_features, self.gnn_dim)
        self.conv2 = GCNConv(self.gnn_dim, self.gnn_dim)
        self.conv3 = GCNConv(
            self.gnn_dim, self.gnn_out_features
        )  # Outpt features could in principle be anything
        self.gnn_emb_fc = torch.nn.Linear(
            self.input_dims * self.gnn_out_features, self.gnn_emb_dim
        )

        # Output MLP Layer
        self.out_fc = torch.nn.Linear(
            in_features=self.lstm_dims + self.gnn_emb_dim, out_features=self.input_dims
        )

    def embed_graph(self, node_vals, edge_idx):
        h = self.conv1(node_vals, edge_idx)
        h = h.relu()
        h = self.conv2(h, edge_idx)
        h = h.relu()
        h = self.conv3(h, edge_idx)
        h = h.relu()
        gnn_emb = self.gnn_emb_fc(h.squeeze())
        return gnn_emb

    def forward(self, x, features, node_vals, edge_idx):
        # Send through lstm
        input = torch.cat((x, features), 2)
        _, (hn, cn) = self.lstm(input)
        hn = hn.relu()

        # Send through gnn
        gnn_emb = self.embed_graph(node_vals, edge_idx)
        gnn_emb = gnn_emb.repeat(x.shape[0], 1).unsqueeze(0)

        # Concat embeddings
        full_emb = torch.cat((hn, gnn_emb), 2)

        # Compute output
        out = self.out_fc(full_emb)
        return out.permute(1, 0, 2)
