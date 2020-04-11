import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Input layer
        self.hidden_size = 128
        self.input_layer = nn.Linear(2, self.hidden_size)
        # Hidden layers
        self.hidden1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden2 = nn.Linear(self.hidden_size, self.hidden_size)
        # Output layer
        self.output_layer = nn.Linear(self.hidden_size, 1)

        # Randomly initialize weights
        nn.init.xavier_uniform_(self.input_layer.weight, gain=2)
        nn.init.xavier_uniform_(self.hidden1.weight, gain=2)
        nn.init.xavier_uniform_(self.hidden2.weight, gain=2)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=2)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
