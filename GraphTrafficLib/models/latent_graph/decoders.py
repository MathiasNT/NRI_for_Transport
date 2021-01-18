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
# TODO double check the dimensions for the forward pass - it seems like it is wrong but still works
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
        """[summary]
        """

        # So according to their implementation we want
        # input shape [batch_size, num_timesteps, num_atoms, num_dims]
        # rel_types [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
        # Since we do a simplified version timesteps here are considered dims so permute to get [B, N, T/F]
        inputs = inputs.permute(0, 1, 2)

        pre_msg = self.node2edge(inputs, rel_rec, rel_send)
        # print(f"pre_msg: {pre_msg.shape}")
        # Create variable to aggregate the messages in
        # TODO double check why we need this torch.autograd variable
        all_msgs = Variable(
            torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
        )
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()
        # print(f"all_msgs: {all_msgs.shape}")

        # Go over the different edge types and compute their contribution to the overall messages
        # TODO change to be able to handle multiple edge types
        for i in range(0, 2):
            msg = F.relu(self.msg_fc1(pre_msg))
            msg = F.relu(self.msg_fc2(msg))
            msg = (
                msg * rel_types[:, :, i : i + 1]
            )  # This is the magic line that enforces 0 to be no edge
            # print(f"msg: {msg.shape}")
            all_msgs += msg

        # Aggregate all msgs to receiver
        # TODO doulbe check the dimensions of the messages
        agg_msgs = (
            all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        )  # This is simillar to the edge2node and could potentially be moved there
        agg_msgs = agg_msgs.contiguous()

        # Output MLP
        pred = F.relu(self.out_fc1(agg_msgs))
        pred = F.relu(self.out_fc2(pred))
        pred = self.out_fc3(pred)

        # TODO fix the output dimensions
        return pred


class GRUDecoder(nn.Module):
    """summary
    """

    def __init__(self, n_in, n_hid, n_out, msg_hid, msg_out, gru_hid, gru_out):
        super().__init__()

        # FC layers to compute messages
        # TODO Do not know why they do not use the MLP module here
        # TODO Fix this for multiple types of edges.
        self.msg_fc1 = nn.Linear(in_features=n_hid * 2, out_features=msg_hid)
        self.msg_fc2 = nn.Linear(in_features=msg_hid, out_features=msg_out)

        # TODO figure out what determines the shape that we want
        self.msg_out_shape = msg_out

        # GRU network
        # TODO consider whether it makes sense to try the torch implementation - would need a bit of massaging probably
        self.gru_ir = nn.Linear(in_features=n_in, out_features=gru_hid)
        self.gru_hr = nn.Linear(in_features=msg_out, out_features=gru_hid)
        self.gru_ii = nn.Linear(in_features=n_in, out_features=gru_hid)
        self.gru_hi = nn.Linear(in_features=msg_out, out_features=gru_hid)
        self.gru_in = nn.Linear(in_features=n_in, out_features=gru_hid)
        self.gru_hn = nn.Linear(in_features=msg_out, out_features=gru_hid)

        # FC for generating the output
        self.out_fc1 = nn.Linear(in_features=gru_hid, out_features=n_hid)
        self.out_fc2 = nn.Linear(in_features=n_hid, out_features=n_hid)
        self.out_fc3 = nn.Linear(in_features=n_hid, out_features=n_out)

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

    def do_single_step_forward(self, inputs, rel_rec, rel_send, rel_types, hidden):
        # TODO fix the dimension to match what we want
        # So according to their implementation we want
        # input shape [batch_size, num_timesteps, num_atoms, num_dims]
        # rel_types [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
        # print(f"inputs: {inputs.shape}")
        pre_msg = self.node2edge(hidden, rel_rec, rel_send)
        # print(f"pre_msg: {pre_msg.shape}")
        # Create variable to aggregate the messages in
        # TODO double check why we need this torch.autograd variable
        all_msgs = Variable(
            torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
        )
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()
        # print(f"all_msgs: {all_msgs.shape}")

        # Go over the different edge types and compute their contribution to the overall messages
        # TODO change to be able to handle multiple edge types
        for i in range(0, 2):
            msg = F.relu(self.msg_fc1(pre_msg))
            msg = F.relu(self.msg_fc2(msg))
            msg = (
                msg * rel_types[:, :, i : i + 1]
            )  # This is the magic line that enforces 0 to be no edge
            # print(f"msg: {msg.shape}")
            all_msgs += msg
            # TODO test with normalization like they do here - note that they only do it in the GRU implementation

        # Aggregate all msgs to receiver
        # TODO doulbe check the dimensions of the messages
        agg_msgs = (
            all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        )  # This is simillar to the edge2node and could potentially be moved there
        agg_msgs = agg_msgs.contiguous()

        # Send through GRU network
        # TODO check if this could be done with the torch implementation or maybe at least move it out as a module
        r = F.sigmoid(self.gru_ir(inputs) + self.gru_hr(agg_msgs))
        i = F.sigmoid(self.gru_ii(inputs) + self.gru_hi(agg_msgs))
        n = F.tanh(self.gru_in(inputs) + r * self.gru_hn(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.relu(self.out_fc1(hidden))
        pred = F.relu(self.out_fc2(pred))
        pred = self.out_fc3(pred)

        # TODO fix the output dimensions and test with skip connection
        return pred, hidden

    def forward(self, inputs, rel_rec, rel_send, rel_types):
        pred_all = []

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)
        )

        for step in range(0, inputs.size(1) - 1):
            ins = inputs[:, step, :, :]
            pred, hidden = self.do_single_step_forward(
                ins, rel_rec, rel_send, rel_types, hidden
            )
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds
