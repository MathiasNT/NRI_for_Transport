from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


# TODO add in dropout
# TODO Add in skip connections
# TODO Figure out how to enforce no edge - might have to look at the amortized paper - maybe the latent code can take
#  care of it like it is here
# TODO Add in doing multiple steps ahead - right now I implement it as a single step in the forward but that could lead
# to the model ignoring the latent code (??)


class MLPDecoder(nn.Module):
    """ empty
    """

    def __init__(self, n_in, n_hid, n_out, msg_hid, msg_out):
        super().__init__()

        # FC layers to compute messages
        # TODO Do not know why they do not use the MLP module here
        # TODO Fix this for multiple types of edges.
        self.msg_fc1 = nn.Linear(in_features=n_in * 2, out_features=msg_hid)
        self.msg_fc2 = nn.Linear(in_features=msg_hid, out_features=msg_out)

        # FC for generating the output
        self.out_fc1 = nn.Linear(in_features=msg_out, out_features=n_hid)
        self.out_fc2 = nn.Linear(in_features=n_hid, out_features=n_hid)
        self.out_fc3 = nn.Linear(in_features=n_hid, out_features=n_out)

        # TODO figure out what determines the shape that we want
        self.msg_out_shape = msg_out

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

    def forward(self, inputs, rel_rec, rel_send, rel_types):

        # TODO fix the dimension to match what we want
        # So according to their implementation we want
        # input shape [batch_size, num_timesteps, num_atoms, num_dims]
        # rel_types [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
        pre_msg = self.node2edge(inputs, rel_rec, rel_send)

        # Create variable to aggregate the messages in
        # TODO double check why we need this torch.autograd variable
        all_msgs = Variable(
            torch.zeros(
                pre_msg.size(0), pre_msg.size(1), pre_msg.size(2), self.msg_out_shape
            )
        )

        # Go over the different edge types and compute their contribution to the overall messages
        # TODO change to be able to handle multiple edge types
        for i in range(0, 2):
            msg = F.relu(self.msg_fc1(pre_msg))
            msg = F.relu(self.msg_fc2(msg))
            msg = (
                msg * rel_types[:, :, :, i : i + 1]
            )  # This is the magic line that enforces 0 to be no edge
            all_msgs += msg

        # Aggregate all msgs to receiver
        # TODO doulbe check the dimensions of the messages
        agg_msgs = (
            all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        )  # This is simillar to the edge2node and could potentially be moved there
        agg_msgs = agg_msgs.contigous()

        # Output MLP
        pred = F.relu(self.out_fc1(agg_msgs))
        pred = F.relu(self.out_fc2(pred))
        pred = self.out_fc3(pred)

        # TODO fix the output dimensions
        return pred
