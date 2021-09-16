import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from .modules import MLP, CNN
from GraphTrafficLib.utils import encode_onehot


# TODO add in weight initialization to see if it improves the performance
# TODO add in batchnormalization to see it if improves the performance
class MLPEncoder(nn.Module):
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
        # MLP for v->e, hence input is double size
        self.mlp2 = MLP(
            n_in=n_hid * 2,
            n_hid=n_hid,
            n_out=n_hid,
            dropout_prob=do_prob,
            use_bn=use_bn,
        )
        # MLP for e->v, so dimensions should be straight forward
        self.mlp3 = MLP(
            n_in=n_hid, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn
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

    def forward(self, inputs, rel_rec, rel_send):
        """This is the forward pass"""
        # permute to match the wanted [B, N, T * F] (which is a bit weird)
        # TODO fix the normal data such that this here might be unnecessary
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        x = self.mlp1(inputs)

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        if self.factor:
            x_skip = x
            x = self.edge2node(x, rel_rec)
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


class CNNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.0, factor=True, use_bn=True, init_weights=False):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob

        self.factor = factor

        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob, init_weights)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob, use_bn=use_bn)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob, use_bn=use_bn)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob, use_bn=use_bn)
        self.fc_out = nn.Linear(n_hid, n_out)

        if self.factor:
            print("Using factor graph CNN encoder")
        else:
            print("Using CNN encoder")

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        """This function makes a matrix of "stacked timeseries" of the sender and receiver nodes"""
        # TODO look into if there is a way to not assume the same graph across all samples
        # Input has shape [B, N, T, F]

        x = inputs.view(inputs.size(0), inputs.size(1), -1)  # [B, N, T*F]

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(
            inputs.size(0) * receivers.size(1), inputs.size(2), inputs.size(3)
        )
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(
            inputs.size(0) * senders.size(1), inputs.size(2), inputs.size(3)
        )
        senders = senders.transpose(2, 1)

        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec):
        """This function makes the aggregation over the incomming edge embeddings"""
        incoming = torch.matmul(rel_rec.t(), x)  # Corresponds to a sum aggregation
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings"""
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat(
            [senders, receivers], dim=-1
        )  # TODO double check dim - pretty sure it is right, could do -1 instead
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input [B, N, T, F]

        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)  # [B*E, 2F, T]
        x = self.cnn(edges)  # [B*E, F']
        x = x.view(
            inputs.size(0), inputs.size(1) * (inputs.size(1) - 1), -1
        )  # [B, E, F']
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec)  # [B, N, F']
            x = self.mlp2(x)

            x = self.node2edge(x, rel_rec, rel_send)  # [B, E, 2F']
            x = torch.cat((x, x_skip), dim=2)  # [B, E, 3F']
            x = self.mlp3(x)  # [B, E, F']

        return self.fc_out(x)


class FixedEncoder(nn.Module):
    def __init__(self, adj_matrix):
        super().__init__()
        self.adj_matrix = torch.nn.Parameter(adj_matrix, requires_grad=False)
        self.edge_types = None
    def forward(self, inputs, rel_rec, rel_send):
        if self.edge_types is None:
            edge_types = torch.zeros(rel_rec.shape[0], device=inputs.device)
            for edge_idx in range(rel_rec.shape[0]):
                rec_idx = torch.where(rel_rec[edge_idx])
                send_idx = torch.where(rel_send[edge_idx])
                if self.adj_matrix[send_idx, rec_idx]:
                    edge_types[edge_idx] = 1
            if self.adj_matrix.sum() == 0: # hack to fix for empty graph
                edge_types = F.one_hot(edge_types.long(), num_classes=2)
            else:    
                edge_types = F.one_hot(edge_types.long())
            self.edge_types = edge_types
        
        temp = self.edge_types.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
        logits = torch.log(temp.float())
        return logits


class RecurrentEncoder(nn.Module):
    """[summary]m

    Parameters
    ----------
    nn : [type]
        [description]
    """

    def __init__(
        self,
        n_in,
        n_hid,
        rnn_hid,
        n_out,
        do_prob,
        factor,
        rnn_type="gru",
        use_bn=True,
    ):
        super().__init__()

        self.factor = factor

        # We need 4 MLPs to implement the model from NRI
        # MLP for embedding the input, hence dimensions are straight forward
        self.mlp1 = MLP(
            n_in=n_in, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn
        )
        # MLP for v->e, hence input is double size
        self.mlp2 = MLP(
            n_in=n_hid * 2,
            n_hid=n_hid,
            n_out=n_hid,
            dropout_prob=do_prob,
            use_bn=use_bn,
        )
        # MLP for e->v, so dimensions should be straight forward
        self.mlp3 = MLP(
            n_in=n_hid, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn
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

        if rnn_hid is None:
            rnn_hid = n_hid

        if rnn_type == "gru":
            self.forward_rnn = nn.GRU(n_hid, rnn_hid, batch_first=True)
            self.reverse_rnn = nn.GRU(n_hid, rnn_hid, batch_first=True)
        else:
            raise NotImplementedError

        # FC layer for going from the edge embeddings to the edge mean in the latent code
        self.prior_fc = nn.Linear(in_features=n_hid, out_features=n_out)
        self.encoder_fc = nn.Linear(in_features=2 * n_hid, out_features=n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    # TODO add in normalization and see how it improve
    def edge2node(self, x, rel_rec):
        """This function makes the aggregation over the incomming edge embeddings"""
        x = x.transpose(1, 2)
        incoming = torch.matmul(rel_rec.t(), x)
        incoming = incoming.transpose(1, 2)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings"""
        x = x.transpose(1, 2)
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat(
            [senders, receivers], dim=-1
        )  # TODO double check dim - pretty sure it is right, could do -1 instead
        return edges.transpose(1, 2)

    def forward(self, inputs, rel_rec, rel_send):
        """This is the forward pass"""
        # permute to match the wanted [B, N, T * F] (which is a bit weird)
        # TODO fix the normal data such that this here might be unnecessary

        x = self.mlp1(inputs)
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        if self.factor:
            x_skip = x
            x = self.edge2node(x, rel_rec)
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

        old_shape = x.shape
        # Reshape to 3d by merging batch and edge dim
        x = x.view(-1, old_shape[2], old_shape[3])
        forward_x, prior_state = self.forward_rnn(x)
        reverse_x = x.flip(
            1
        )  # TODO Maybe this reverse here is a bit weird pretty sure it should be fine with the flip back
        reverse_x, _ = self.reverse_rnn(reverse_x)
        reverse_x = reverse_x.flip(1)

        prior_result = self.prior_fc(forward_x).view(
            [old_shape[0], old_shape[1], old_shape[2], -1]
        )
        combined_x = torch.cat([forward_x, reverse_x], dim=-1)
        encoder_result = self.encoder_fc(combined_x).view(
            [old_shape[0], old_shape[1], old_shape[2], -1]
        )
        return prior_result, encoder_result, prior_state

    def single_step_forward(self, inputs, rel_rec, rel_send, prior_state):
        """This does a single step forward"""
        # permute to match the wanted [B, N, T * F] (which is a bit weird)
        # TODO fix the normal data such that this here might be unnecessary

        x = self.mlp1(inputs)
        x = self.node2edge(x.unsqueeze(2), rel_rec, rel_send)
        x = self.mlp2(x)
        if self.factor:
            x_skip = x
            x = self.edge2node(x, rel_rec)
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

        old_shape = x.shape
        # Reshape to 3d by merging batch and edge dim
        x = x.view(-1, old_shape[2], old_shape[3])
        forward_x, prior_state = self.forward_rnn(x, prior_state)

        prior_result = self.prior_fc(forward_x).view(
            [old_shape[0], old_shape[1], old_shape[2], -1]
        )
        return prior_result, prior_state
