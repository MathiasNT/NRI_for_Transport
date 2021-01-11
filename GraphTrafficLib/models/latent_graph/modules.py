from torch import nn
import torch.nn.functional as F


# TODO add in weight initialization to see if it improves the performance
# TODO add in batchnormalization to see it if improves the performance
class MLP(nn.Module):
    """[summary]

    Parameters
    ----------
    nn : [type]
        [description]
    """

    def __init__(self, n_in, n_hid, n_out, dropout_prob=0):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = dropout_prob

    def forward(self, inputs):
        # input shape [num_sims / batches, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.elu(self.fc2(x))
        return x
