from typing import List, Tuple, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output

from nngraphons.visualization.graphon_visualization import visualize_pytorch_net_graphon
from nngraphons.learning.homomorphism_density import t_nn


def gradient_descent(
    g: List[Tuple],
    net: nn.Module,
    n: int,
    t_g: List[float],
    L: Callable[[List[float], List[torch.Tensor]], float],
    stopping_criterion: Callable[[float], bool],
    optimizer: Optional[torch.optim.Optimizer] = None
) -> None:
    """
    Iteratively updates net's parameters in the gradient direction w.r.t L

    Parameters
    ----------
    g : List[Tuple]
        List of small graphs for which to match homomorphism densities of the
        network and training graph
    net : torch.nn.Module
        PyTorch neural netowrk to represent graphon
    n : int
        Number of samples to use for density approximation at each training iteration
    t_g : List[float]
        List of ground truth homomorphism densities of small graphs g
    L : Callable[[List[float], List[torch.Tensor]], float]
        Loss function accepting list of ground truth homomorphism densities and netowrk densities
    stopping_criterion : Callable[[float], bool]
        Training stopping criterion function accepting loss function value
    optimizer : torch.optim.Optimizer, optional
        Optimizer to use, Adam is used by default
    """
    # Create optimizer
    if optimizer is None:
        optimizer = optim.Adam(net.parameters())

    # Training loop
    while True:
        # Zero out old accumulated gradients
        optimizer.zero_grad()

        # Calculate homomorphism densities w.r.t network
        t_net = [t_nn(F, net, n, True) for F in g]

        # Calculate loss value
        loss = L(t_g, t_net)

        # Progress print
        clear_output(wait=True)
        print(f'Current loss: {loss.item()}')

        # Visualize progress
        visualize_pytorch_net_graphon(net)

        # Stopping criterion
        if stopping_criterion(loss.item()):
            break

        # Backprop gradients
        loss.backward()

        # Update weights
        optimizer.step()
