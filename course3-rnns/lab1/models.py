# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch import nn

# Simple RNN
class Model(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()
        self.h_dim = n_hidden
        self.input_size = 1
        self.rnn1 = nn.RNN(input_size=1, num_layers=n_layers, \
                           hidden_size=n_hidden, batch_first=True,\
                           nonlinearity='tanh')
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x, h=None):
        if h == None:
            x, h = self.rnn1(x)
        else:
            x, h = self.rnn1(x, h)
        x = self.fc(x)
        out = torch.tanh(x)
        return out, h

# LSTM
class LSTM(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()
        self.h_dim = n_hidden
        self.input_size = 1
        self.rnn1 = nn.LSTM(input_size=1, num_layers=n_layers, \
                            hidden_size=n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x, h_c=None):
        if h_c == None:
            x, (h, c) = self.rnn1(x)
        else:
            x, (h, c) = self.rnn1(x, h_c)
        x = self.fc(x)
        out = torch.tanh(x)
        return out, (h, c)

# GRU
class GRU(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()
        self.h_dim = n_hidden
        self.input_size = 1
        self.rnn1 = nn.GRU(input_size=1, num_layers=n_layers, \
                           hidden_size=n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x, h=None):
        if h == None:
            x, h = self.rnn1(x)
        else:
            x, h = self.rnn1(x, h)
        x = self.fc(x)
        out = torch.tanh(x)
        return out, h

if __name__ == '__main__':
    x = torch.zeros((100,2,1)).float()
    model = Model(25)
    print(model(x)[0].shape)