import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List


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

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output_layer(x)
        return x


def backprop(net: torch.nn.Module, g: List, t_g: List[float], n: int):
    """
    Parameters
    ----------
    net : torch.nn.Module
        PyTorch neural netowrk to represent graphon
    g : List
        List of small graphs for which to match homomorphism densities of the
        network and training graph
    t_g : List
        List of homomorphism densities of small graphs g into the training
        graph, must be the same length as g
    n : int
        The number of times to sample the network for each small graph in g to
        calculate the homomorphism density of that graph
    """
    # Create optimizer
    optimizer = optim.Adam(net.parameters())

    # Training loop
    while True:
        # Zero out old accumulated gradients
        optimizer.zero_grad()

        errors = []
        for i in range(len(g)):
            # Unpack small graph structure
            V, E = g[i]
            # Sample unit hyper-cube n times
            S_n = np.random.random_sample(size=(n, len(V)))
            # Map vertices to real numbers from <0,1>
            mapped_edges = np.empty(shape=(len(E)*n, 2))
            for i in range(n):
                mapped_edges[i*len(E):(i+1)*len(E)] = S_n[i][E]
            # Sort floats in each edge in ascending order to train the upper
            # triangle of the graphon (network)
            mapped_edges.sort()
            mapped_edges = torch.from_numpy(mapped_edges)
            # Sample the network for different edge probabilites for the small
            # graph g[i]
            net_output = net(mapped_edges)
            # Calculate homomorphism density for the small graph g[i]
            reshaped_for_prod = net_output.view(len(E), n)
            edge_probs_prods = torch.prod(reshaped_for_prod, dim=1)
            hom_density = torch.mean(edge_probs_prods)
            # Calculate square error of the homomorphism density w.r.t.
            # training graph homomorphism density for the graph g[i]
            err = t_g[i] - hom_density
            sqerr = err * err
            errors.append(sqerr)
        # Calculate final loss value
        loss = torch.mean(torch.stack(errors))

        # Progress print
        print(f'Current loss: {loss.item()}')

        # Backprop gradients
        loss.backward()

        # Update weights
        optimizer.step()
