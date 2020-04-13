import torch
import torch.nn as nn


class RBFMixture(nn.Module):
    def __init__(self, n_centers, basis_func):
        super(RBFMixture, self).__init__()
        # RBF layer
        self.rbf_layer = RBFLayer(
            in_features=2,
            out_features=n_centers,
            basis_func=basis_func
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

        # Weighted average (linear part)
        # Normalize output linear weights to have a normalized weighted average of RBF outputs
        with torch.no_grad():
            # Make weights non-negative
            self.output_layer.weight.clamp_(0)
            # Calculate normalization constant to make weights <= 1
            norm = torch.sum(self.output_layer.weight)
            # Check for zero division
            if norm.item() != 0:
                self.output_layer.weight.div_(norm)
        x = self.output_layer(x)
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
        nn.init.uniform_(self.centres, 0, 1)
        nn.init.uniform_(self.sigmas, 0.05, 1)

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


# class RBFNet(nn.Module):
