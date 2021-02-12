import torch
from torch import nn
from .modules import MLP


# TODO add in weight initialization to see if it improves the performance
# TODO add in batchnormalization to see it if improves the performance
class MLPEncoder(nn.Module):
    """[summary]m

    Parameters
    ----------
    nn : [type]
        [description]
    """

    def __init__(self, n_in, n_hid, n_out, do_prob, factor):
        super().__init__()

        self.factor = factor

        # We need 4 MLPs to implement the model from NRI
        # MLP for embedding the input, hence dimensions are straight forward
        self.mlp1 = MLP(n_in=n_in, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob)
        # MLP for v->e, hence input is double size
        self.mlp2 = MLP(n_in=n_hid * 2, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob)
        # MLP for e->v, so dimensions should be straight forward
        self.mlp3 = MLP(n_in=n_hid, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob)
        # MLP for second v->e, so dimensions should be straight foward

        if self.factor:
            # If we do factor graph we need to increase the input size
            self.mlp4 = MLP(
                n_in=n_hid * 3, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob
            )
        else:
            self.mlp4 = MLP(
                n_in=n_hid * 2, n_hid=n_hid, n_out=n_hid, dropout_prob=do_prob
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
        """This function makes the aggregation over the incomming edge embeddings
        """
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings
        """
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat(
            [senders, receivers], dim=-1
        )  # TODO double check dim - pretty sure it is right, could do -1 instead
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        """This is the forward pass
        """
        # permute to match the wanted [B, N, T * F] (which is a bit weird)
        # TODO fix the normal data such that this here might be unnecessary
        if len(inputs.shape) == 3:
            inputs = inputs.permute(0, 2, 1)
        else:
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
