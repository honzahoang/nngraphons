import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size):
        super(MLP, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        # Input layer
        self.input_layer = nn.Linear(2, self.hidden_size)
        # Hidden layers
        self.hidden_layers = []
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden_activation = nn.LeakyReLU()
        # Output layer
        self.output_layer = nn.Linear(self.hidden_size, 1)

        # Randomly initialize weights
        nn.init.xavier_uniform_(self.input_layer.weight, gain=1)
        for h in self.hidden_layers:
            nn.init.xavier_uniform_(h.weight, gain=1)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=1)

    def forward(self, x):
        x = self.input_layer(x)
        for h in self.hidden_layers:
            x = self.hidden_activation(h(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
