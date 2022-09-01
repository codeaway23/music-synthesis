from torch import nn

class RecurrentGeneratorModel(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, activation='relu', dropout=0):
        super(RecurrentGeneratorModel, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers)
        self.relu = nn.ReLU() if activation == 'relu' else nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, in_size)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        # _, (x, _) = self.rnn(x)
        x, _ = self.rnn(x)
        x = self.relu(x[-1, :])
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class RecurrentDiscriminatorModel(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, activation='relu', dropout=0.2):
        super(RecurrentDiscriminatorModel, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers)
        self.relu = nn.ReLU() if activation == 'relu' else nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # _, (x, _) = self.rnn(x)
        x, _ = self.rnn(x)
        x = self.relu(x[-1, :])
        x= self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


