import torch

class LSTM_model(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, batch_size):
        super(LSTM_model, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims
        self.batch_size = batch_size

        self.lstm = torch.nn.LSTM(input_size=self.input_dims, hidden_size=self.hidden_dims, batch_first=True)
        #self.register_parameter(name='h0', param=torch.nn.Parameter(torch.randn(1, self.batch_size, self.hidden_dims).type(torch.FloatTensor)))
        #self.register_parameter(name='c0', param=torch.nn.Parameter(torch.randn(1, self.batch_size, self.hidden_dims).type(torch.FloatTensor)))
    
        self.fc = torch.nn.Linear(in_features=self.hidden_dims, out_features=self.input_dims)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = torch.nn.ReLU(hn)
        out = self.fc(hn)



        return out.permute(1,0,2)