import random
import os

import torch
import torch.nn as nn
import numpy as np


class RBFMixture(nn.Module):
    def __init__(self, n_centers, basis_func, sigma_range=(5, 10)):
        super(RBFMixture, self).__init__()
        # RBF layer
        self.rbf_layer = RBFLayer(
            in_features=2,
            out_features=n_centers,
            basis_func=basis_func,
            sigma_range=sigma_range
        )
        # Output layer
        self.output_layer = nn.Linear(
            in_features=n_centers,
            out_features=1,
            bias=False
        )
        # Initialize output layer weights
        nn.init.uniform_(self.output_layer.weight, 0.1, 1)

    def forward(self, x):
        # RBF scores (RBF part)
        # Keep centers in unit square
        x = self.rbf_layer(x)
        # FC layer
        x = self.output_layer(x)
        # Sigmoid activation to ensure probabilities
        x = torch.sigmoid(x)
        return x

    def escape_pareto_stationary_point(self, spawn_prob=0.8):
        if random.random() > spawn_prob:
            self.rbf_layer.spawn_community()
        else:
            self.rbf_layer.despawn_community()

    def mutate(self, step=None, lr=1):
        if step is None:
            # Generate new random mutation
            weight_steps = []
            rbf_mutation = self.rbf_layer.mutate(step=step, lr=lr)
            for rbf_step in rbf_mutation:
                rbf_i = rbf_step[0]
                weight_steps.append((rbf_i, 2*random.random()-1))
        else:
            weight_steps, rbf_mutation = step

        # Apply mutation
        for i, shift in weight_steps:
            self.output_layer.weight.data[0, i] += shift

        return (weight_steps, rbf_mutation)


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

    def __init__(self, in_features, out_features, basis_func, sigma_range):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.sigma_range = sigma_range
        nn.init.uniform_(self.centres, 0, 1)
        nn.init.uniform_(self.sigmas, self.sigma_range[0], self.sigma_range[1])

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        return self.basis_func(distances)

    def spawn_community(self):
        self.out_features += 1
        new_center = torch.Tensor(1, self.in_features)
        new_sigma = torch.Tensor(1)
        nn.init.uniform_(new_center, 0, 1)
        nn.init.uniform_(new_sigma, self.sigma_range[0], self.sigma_range[1])
        new_centers = torch.cat(self.centres.data.clone(), new_center)
        new_sigmas = torch.cat(self.sigmas.data.clone(), new_sigma)
        self.centres = nn.Parameter(new_centers.to(os.environ['COMPUTATION_DEVICE']))
        self.sigmas = nn.Parameter(new_sigmas.to(os.environ['COMPUTATION_DEVICE']))

    def despawn_community(self):
        self.out_features -= 1
        random_community = random.randint(0, self.out_features)
        selection_indices = [i for i in range(self.out_features) if i != random_community]
        selection_indices = np.array(selection_indices)
        self.centres = nn.Parameter(
            self.centres
            .data
            .clone()
            [selection_indices, :]
            .to(os.environ['COMPUTATION_DEVICE'])
        )
        self.sigmas = nn.Parameter(
            self.sigmas
            .data
            .clone()
            [selection_indices]
            .to(os.environ['COMPUTATION_DEVICE'])
        )

    def mutate(self, step=None, lr=1):
        if step is None:
            # Generate new random mutaiton
            step = []
            rbfs_to_mutate = random.choices(
                population=range(self.out_features),
                k=random.randint(1, self.out_features+1)
            )
            for i in rbfs_to_mutate:
                loc = torch.from_numpy(np.random.uniform(-1, 1, self.in_features))
                scale = np.random.uniform(-1, 1)
                step.append((i, loc, scale))

        # Apply mutation
        for i, loc, scale in step:
            self.centres.data[i, :] += lr * loc
            self.sigmas.data[i] += lr * scale

        # Keep centers in unit square
        self.centres.data = self.centres.data.clamp(0, 1)

        return step


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
    phi = (
        ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha))
        * alpha * torch.exp(-alpha)
    )
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi
