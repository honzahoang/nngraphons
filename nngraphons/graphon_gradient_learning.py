import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
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


# https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
class RBFLayer(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        return self.basis_func(distances)


# RBFs
def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi


# class RBFNet(nn.Module):


def t_nn(g, net, n, track_computation=True):
    """Calculates the homomorphism density of finite graph g into the graphon net."""
    torch.set_grad_enabled(track_computation)

    # Unpack small graph structure
    V, E = g

    # Uniformly sample unit hyper-cube n times
    S_n = np.random.random_sample(size=(n, len(V)))

    # Map vertices to real numbers from <0,1> n times
    mapped_edges = np.empty(shape=(len(E)*n, 2))
    for j in range(n):
        mapped_edges[j*len(E):(j+1)*len(E)] = S_n[j][E]

    # Sort floats in each edge in ascending order to use upper triangle of graphon
    mapped_edges.sort()
    mapped_edges = torch.from_numpy(mapped_edges).float()

    # Sample the network for different edge probabilites for the small graph g
    net_output = net(mapped_edges)

    # Calculate homomorphism density for the small graph g
    reshaped_for_prod = net_output.view(len(E), n)
    edge_probs_prods = torch.prod(reshaped_for_prod, dim=0)
    hom_density = torch.sum(edge_probs_prods) / n

    torch.set_grad_enabled(True)

    return hom_density


def backprop(net: torch.nn.Module, g: List, t_g: List[float], epsilon: float, gamma: float):
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
    epsilon : float
    gamma : float
    """
    # Sample size
    n = math.ceil((math.log(2) - math.log(1 - gamma)) / (2*epsilon**2))

    # Create optimizer
    optimizer = optim.Adam(net.parameters())

    # Training loop
    while True:
        # Zero out old accumulated gradients
        optimizer.zero_grad()

        errors = []
        for i in range(len(g)):
            # Calculate square error of the homomorphism density w.r.t.
            # training graph homomorphism density for the graph g[i]
            err = t_g[i] - t_nn(g[i], net, n, True)
            sqerr = err * err
            errors.append(sqerr)
        # Calculate final loss value
        loss = torch.max(torch.stack(errors))

        # Progress print
        print(f'Current loss: {loss.item()}')

        # Visualize progress
        visualize_pytorch_net_graphon(net)

        # Stopping criterion
        if loss.item() < 0.000001:
            break

        # Backprop gradients
        loss.backward()

        # Update weights
        optimizer.step()


def visualize_pytorch_net_graphon(net, resolution=300):
    """Plots the neural network as a greyscale image. The darker the closer to 1."""
    uniform_args = np.linspace(start=0, stop=1, num=resolution)
    cartesian_product = np.transpose(
        [np.tile(uniform_args, len(uniform_args)),
         np.repeat(uniform_args, len(uniform_args))]
    )
    cartesian_product.sort()
    with torch.no_grad():
        img_mat = (
            net(torch.from_numpy(cartesian_product).float())
            .numpy()
            .reshape(resolution, resolution)
        )
    # Make visualization symmetric
    tril = np.tril_indices(len(img_mat))
    img_mat[tril] = img_mat[tril[1],tril[0]]
    plt.figure()
    plt.imshow(
        X=img_mat,
        origin='lower',
        extent=[0,1,0,1],
        cmap='plasma',
        vmin=0,
        vmax=1
    )
    plt.colorbar()
    plt.show()
