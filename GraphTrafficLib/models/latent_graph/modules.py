from torch import nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """The standard MLP module w. batchnorm, initializationa and dropout"""

    def __init__(self, n_in, n_hid, n_out, dropout_prob=0, use_bn=True):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = dropout_prob
        self.use_bn = use_bn

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        """We do batch norm over batches and things so we reshape first"""
        # x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        # input shape [num_sims / batches, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        if self.use_bn:
            x = self.batch_norm(x)
        return x


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0, init_weights=False):
        super(CNN, self).__init__()

        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_out = nn.Conv1d(n_hid, n_out, kernel_size=5)
        self.conv_att = nn.Conv1d(n_hid, 1, kernel_size=5)

        if init_weights:
            self.init_weights()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        val = self.conv_out(x)
        attention = F.softmax(self.conv_att(x), dim=2)
        out = (val * attention).mean(dim=2)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()