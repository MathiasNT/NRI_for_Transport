import torch


class LSTM_w_Conv(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        feature_dims,
        hidden_dims,
        batch_size,
        conv_channels,
        spatial_emb_size,
    ):
        super(LSTM_w_Conv, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims
        self.feature_dims = feature_dims
        self.batch_size = batch_size
        self.conv_channels = conv_channels
        self.spatial_emb_size = spatial_emb_size

        # Add batch norm
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.spatial_fc = torch.nn.Linear(
            in_features=self.input_dims * self.conv_channels,
            out_features=self.spatial_emb_size,
        )

        self.lstm = torch.nn.LSTM(
            input_size=self.spatial_emb_size + self.feature_dims,
            hidden_size=self.hidden_dims,
            batch_first=True,
        )
        # self.register_parameter(name='h0', param=torch.nn.Parameter(torch.randn(1, self.batch_size, self.hidden_dims).type(torch.FloatTensor)))
        # self.register_parameter(name='c0', param=torch.nn.Parameter(torch.randn(1, self.batch_size, self.hidden_dims).type(torch.FloatTensor)))

        self.fc = torch.nn.Linear(
            in_features=self.hidden_dims, out_features=self.input_dims
        )

        self.relu = torch.nn.ReLU()

    def forward(self, x, features):

        # Do convolution-spatial-embedding for each timestep
        spatial_emb_local = []
        for t in range(x.shape[1]):
            h1 = self.conv1(x[:, t, :, :].unsqueeze(1))
            h1 = self.relu(h1)

            h2 = self.conv2(h1)
            h2 = self.relu(h2)

            h3 = self.conv3(h2)
            h3 = self.relu(h3)

            h3 = h3.view(x.shape[0], -1)
            h4 = self.spatial_fc(h3)
            h4 = self.relu(h4)
            spatial_emb_local.append(h4.unsqueeze(1))
        spatial_emb = torch.cat(spatial_emb_local, 1)

        # Concatenate spatial embedding with features
        input = torch.cat((spatial_emb, features), 2)

        # Send through LSTM
        out, (hn, cn) = self.lstm(input)
        out = self.relu(hn)
        out = self.fc(hn)
        return out.permute(1, 0, 2)

