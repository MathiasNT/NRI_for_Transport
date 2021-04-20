import torch


class SimpleLSTM(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        feature_dims,
        hidden_dims,
        batch_size,
        conv_channels,
        spatial_emb_size,
    ):
        super(SimpleLSTM, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims
        self.batch_size = batch_size

        self.lstm = torch.nn.LSTM(
            input_size=self.feature_dims,
            hidden_size=self.hidden_dims,
            batch_first=True,
            dropout=0.5,
        )

        self.register_parameter(
            name="h0",
            param=torch.nn.Parameter(
                torch.randn(1, self.batch_size, self.hidden_dims).type(
                    torch.FloatTensor
                )
            ),
        )
        self.register_parameter(
            name="c0",
            param=torch.nn.Parameter(
                torch.randn(1, self.batch_size, self.hidden_dims).type(
                    torch.FloatTensor
                )
            ),
        )

        self.fc = torch.nn.Linear(
            in_features=self.hidden_dims, out_features=self.input_dims
        )

        self.relu = torch.nn.ReLU()

    def forward(self, x, features):

        # Send through LSTM
        out, (hn, cn) = self.lstm(x)
        out = self.relu(hn)
        out = self.fc(hn)
        return out.permute(1, 0, 2)

