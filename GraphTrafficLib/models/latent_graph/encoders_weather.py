import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from .modules import MLP, CNN
from GraphTrafficLib.utils import encode_onehot


# TODO add in weight initialization to see if it improves the performance
# TODO add in batchnormalization to see it if improves the performance
class MLPEncoder_weather(nn.Module):
    """[summary]m

    Parameters
    ----------
    nn : [type]
        [description]
    """

    def __init__(self, n_in, n_hid, n_out, do_prob, factor, use_bn=True):
        super().__init__()

        self.factor = factor

        # We need 4 MLPs to implement the model from NRI
        # MLP for embedding the input, hence dimensions are straight forward
        self.mlp1 = MLP(
            n_in=n_in, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn
        )
        self.weather_mlp = MLP(n_in=n_in, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn)
        # MLP for v->e, hence input is double size
        self.mlp2 = MLP(
            n_in=n_hid * 3,
            n_hid=n_hid,
            n_out=n_hid,
            dropout_prob=do_prob,
            use_bn=use_bn,
        )
        # MLP for e->v, so dimensions should be straight forward
        self.mlp3 = MLP(
            n_in=n_hid * 2, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn
        )
        # MLP for second v->e, so dimensions should be straight foward

        if self.factor:
            # If we do factor graph we need to increase the input size
            self.mlp4 = MLP(
                n_in=n_hid * 3,
                n_hid=n_hid,
                n_out=n_hid,
                dropout_prob=do_prob,
                use_bn=use_bn,
            )
        else:
            self.mlp4 = MLP(
                n_in=n_hid * 2,
                n_hid=n_hid,
                n_out=n_hid,
                dropout_prob=do_prob,
                use_bn=use_bn,
            )

        # FC layer for going from the edge embeddings to the edge mean in the latent code
        self.fc = nn.Linear(in_features=n_hid, out_features=n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    # TODO add in normalization and see how it improve
    def edge2node(self, x, rel_rec):
        """This function makes the aggregation over the incomming edge embeddings"""
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings"""
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat(
            [senders, receivers], dim=-1
        )  # TODO double check dim - pretty sure it is right, could do -1 instead
        return edges

    def forward(self, inputs, weather, rel_rec, rel_send):
        """This is the forward pass"""
        # permute to match the wanted [B, N, T * F] (which is a bit weird)
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        # The weather is permuted to get embeddings along batch dim
        weather = weather.view(weather.size(0), -1)

        # Embed nodes
        x = self.mlp1(inputs)

        #  Embed weather
        w = self.weather_mlp(weather).unsqueeze(1)
        # Create, cat weather and embed messages
        x = self.node2edge(x, rel_rec, rel_send)
        w_edge = w.repeat(1, x.size(1), 1)
        x = torch.cat([x,w_edge], dim=-1)
        x = self.mlp2(x)
        
        if self.factor:
            x_skip = x
            x = self.edge2node(x, rel_rec)
            w_node = w.repeat(1, x.size(1), 1)            
            x = torch.cat([x,w_node], dim=-1)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=-1)
            x = self.mlp4(x)
        else:
            # Note that my no factor differs from the paper as they value the
            # skip connection over the graph for some reason
            x = self.edge2node(x, rel_rec)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = self.mlp4(x)

        x = self.fc(x)

        return x

class FixedEncoder_weather(nn.Module):
    def __init__(self, adj_matrix):
        super().__init__()
        self.adj_matrix = torch.nn.Parameter(adj_matrix, requires_grad=False)

    def forward(self, inputs, weather, rel_rec, rel_send):
        edge_types = torch.zeros(rel_rec.shape[0], device=inputs.device)
        for edge_idx in range(rel_rec.shape[0]):
            rec_idx = torch.where(rel_rec[edge_idx])
            send_idx = torch.where(rel_send[edge_idx])
            if self.adj_matrix[send_idx, rec_idx]:
                edge_types[edge_idx] = 1
        edge_types = F.one_hot(edge_types.long())
        edge_types = edge_types.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
        return edge_types.float()