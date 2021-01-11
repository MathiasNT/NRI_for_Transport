import torch
from torch import nn
from .modules import MLP


# TODO add in weight initialization to see if it improves the performance
# TODO add in batchnormalization to see it if improves the performance
class MLPEncoder(nn.Module):
    """[summary]

    Parameters
    ----------
    nn : [type]
        [description]
    """

    def __init__(self, n_in, n_hid, n_out):
        super().__init__()

        # We need 4 MLPs to implement the model from NRI

        # MLP for embedding the input, hence dimensions are straight forward
        self.mlp1 = MLP(n_in=n_in, n_hid=n_hid, n_out=n_hid)

        # MLP for v->e, hence input is double size
        self.mlp2 = MLP(n_in=n_in * 2, n_hid=n_hid, n_out=n_hid)

        # MLP for e->v, so dimensions should be straight forward
        self.mlp3 = MLP(n_in=n_hid, n_hid=n_hid, n_out=n_hid)

        # MLP for second v->e, so dimensions should be straight foward
        # TODO look into the factor graph stuff - I think it is somekind of residual link
        self.mlp4 = MLP(n_in=n_hid * 2, n_hid=n_hid, n_out=n_hid)

        # FC layer for going from the edge embeddings to the edge mean in the latent code
        self.fc = nn.Linear(in_features=n_hid, out_features=n_out)

    # TODO add in normalization and see how it improve
    def edge2node(self, x, rel_rec):
        """This function makes the aggregation over the incomming edge embeddings
        """
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming

    def node2edge(self, x, rel_rec, rel_send):
        """This function makes a matrix of [node_i, node_j] rows for the edge embeddings
        """
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)  # TODO double check dim
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        """This is the forward pass
        """
        # TODO maybe add dim permutation / view

        x = self.mlp1(inputs)

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)

        x = self.edge2node(x, rel_rec)
        x = self.mlp3(x)

        # TODO add in skip connection to see if that improves the performance

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp4(x)

        x = self.fc(x)
        return x
