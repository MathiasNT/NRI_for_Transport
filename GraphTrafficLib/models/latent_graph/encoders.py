import torch
import torch.nn.functional as F
from torch import nn

from .modules import MLP


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob, factor, use_bn=True):
        super().__init__()

        self.factor = factor

        # We need 4 MLPs to implement the model from NRI
        # MLP for embedding the input, hence dimensions are straight forward
        self.mlp1 = MLP(n_in=n_in, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn)
        # MLP for v->e, hence input is double size
        self.mlp2 = MLP(
            n_in=n_hid * 2,
            n_hid=n_hid,
            n_out=n_hid,
            dropout_prob=do_prob,
            use_bn=use_bn,
        )
        # MLP for e->v, so dimensions should be straight forward
        self.mlp3 = MLP(n_in=n_hid, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn)
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

    def edge2node(self, x, rel_rec):
        """This function makes the aggregation over the incomming edge embeddings"""
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings"""
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=-1)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        """This is the forward pass"""
        # permute to match the wanted [B, N, T * F]
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
            x = self.edge2node(x, rel_rec)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = self.mlp4(x)

        x = self.fc(x)

        return x


class MLPEncoder_global(nn.Module):
    def __init__(self, n_in, n_in_global, n_hid, n_out, do_prob, factor, use_bn=True):
        super().__init__()

        self.factor = factor

        # We need 4 MLPs to implement the model from NRI
        # MLP for embedding the input, hence dimensions are straight forward
        self.mlp1 = MLP(n_in=n_in, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob, use_bn=use_bn)
        self.global_mlp = MLP(
            n_in=n_in_global,
            n_hid=n_hid,
            n_out=n_hid,
            dropout_prob=do_prob,
            use_bn=False,
        )
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
            n_in=n_hid * 2,
            n_hid=n_hid,
            n_out=n_hid,
            dropout_prob=do_prob,
            use_bn=use_bn,
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

    def edge2node(self, x, rel_rec):
        """This function makes the aggregation over the incomming edge embeddings"""
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings"""
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=-1)
        return edges

    def forward(self, inputs, global_inputs, rel_rec, rel_send):
        """This is the forward pass"""
        # permute to match the wanted [B, N, T * F] (which is a bit weird)
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        # The global is permuted to get embeddings along batch dim
        global_inputs = global_inputs.view(global_inputs.size(0), -1)

        # Embed nodes
        x = self.mlp1(inputs)

        #  Embed global
        w = self.global_mlp(global_inputs).unsqueeze(1)
        # Create, cat global and embed messages
        x = self.node2edge(x, rel_rec, rel_send)
        w_edge = w.repeat(1, x.size(1), 1)
        x = torch.cat([x, w_edge], dim=-1)
        x = self.mlp2(x)

        if self.factor:
            x_skip = x
            x = self.edge2node(x, rel_rec)
            w_node = w.repeat(1, x.size(1), 1)
            x = torch.cat([x, w_node], dim=-1)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=-1)
            x = self.mlp4(x)
        else:
            x = self.edge2node(x, rel_rec)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = self.mlp4(x)

        x = self.fc(x)

        return x


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
            if self.adj_matrix.sum() == 0:  # hack to fix for empty graph
                edge_types = F.one_hot(edge_types.long(), num_classes=2)
            else:
                edge_types = F.one_hot(edge_types.long())
            self.edge_types = edge_types

        temp = self.edge_types.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
        logits = torch.log(temp.float())
        return logits


class FixedEncoder_global(nn.Module):
    def __init__(self, adj_matrix):
        super().__init__()
        self.adj_matrix = torch.nn.Parameter(adj_matrix, requires_grad=False)
        self.edge_types = None

    def forward(self, inputs, global_inputs, rel_rec, rel_send):
        if self.edge_types is None:
            edge_types = torch.zeros(rel_rec.shape[0], device=inputs.device)
            for edge_idx in range(rel_rec.shape[0]):
                rec_idx = torch.where(rel_rec[edge_idx])
                send_idx = torch.where(rel_send[edge_idx])
                if self.adj_matrix[send_idx, rec_idx]:
                    edge_types[edge_idx] = 1
            if self.adj_matrix.sum() == 0:  # hack to fix for empty graph
                edge_types = F.one_hot(edge_types.long(), num_classes=2)
            else:
                edge_types = F.one_hot(edge_types.long())
            self.edge_types = edge_types

        temp = self.edge_types.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
        logits = torch.log(temp.float())
        return logits


class LearnedAdjacancy(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.log(torch.ones((1, n_nodes * (n_nodes - 1), n_edge_types)) * 0.5),
            requires_grad=True,
        )

    def forward(self, inputs, rel_rec, rel_send):
        batch_size = inputs.shape[0]
        repeated_logits = self.logits.repeat(batch_size, 1, 1)
        return repeated_logits.to(device=inputs.device)


class LearnedAdjacancy_global(nn.Module):
    def __init__(self, n_nodes, n_edge_types):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.log(torch.ones((1, n_nodes * (n_nodes - 1), n_edge_types)) * 0.5),
            requires_grad=True,
        )

    def forward(self, inputs, global_inputs, rel_rec, rel_send):
        batch_size = inputs.shape[0]
        repeated_logits = self.logits.repeat(batch_size, 1, 1)
        return repeated_logits.to(device=inputs.device)
