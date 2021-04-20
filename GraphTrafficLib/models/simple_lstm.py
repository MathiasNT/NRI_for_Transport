import torch


class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, dropout):
        super(SimpleLSTM, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims
        self.dropout = dropout

        self.lstm = torch.nn.LSTM(
            input_size=self.input_dims,
            hidden_size=self.hidden_dims,
            batch_first=True,
            dropout=self.dropout,
        )

        self.fc = torch.nn.Linear(
            in_features=self.hidden_dims, out_features=self.input_dims
        )

        self.relu = torch.nn.ReLU()

    def forward(self, x, pred_steps):

        # Burn in
        _, (h_t, c_t) = self.lstm(x)

        # Pred steps
        preds = []
        for _ in range(pred_steps):
            out = self.relu(h_t)
            out = self.fc(out).permute(1, 0, 2)
            preds.append(out)
            _, (h_t, c_t) = self.lstm(out, (h_t, c_t))

        preds = torch.stack(preds, dim=1).squeeze(dim=-1)

        return preds
