import os
from typing import List, Tuple

import math
import torch
import torch.nn as nn
import numpy as np


def t_discrete(g, G, epsilon=0.01, gamma=0.95):
    """Homomorphism density for finite graphs estimation with naive Monte-Carlo"""
    # Unpack graph structs
    V_g, E_g = g
    V_G, E_G = G

    # Create adjacency matrix
    A_G = np.zeros((len(V_G), len(V_G)), dtype=int)
    A_G[E_G[:, 0], E_G[:, 1]] = 1
    A_G[E_G[:, 1], E_G[:, 0]] = 1

    # Sample size
    N = math.ceil((math.log(2) - math.log(1 - gamma)) / (2*epsilon**2))

    # Sample vertex mappings V_g -> V_G
    mappings = np.random.randint(low=0, high=len(V_G)-1, size=N*len(V_g))

    # Create mapped vertex edges to later check in adjacency matrix
    mapping_indices = (
        np.tile(E_g.T, N).T
        + len(V_g)
        * np.tile(
            np.repeat(
                np.arange(N).reshape(-1, 1),
                len(E_g),
                axis=0
            ),
            2
        )
    )
    mapped_edges = mappings[mapping_indices]

    # Get adjacency for each edge of the random mappings
    adjacency_indicators = A_G[mapped_edges[:, 0], mapped_edges[:, 1]]

    # Multiply adjacencies to check if the mappings represent homomorphisms then average
    hom_density = (
        adjacency_indicators
        .reshape(N, len(E_g))
        .prod(axis=1)
        .sum()
        / N
    )

    return hom_density


def t_nn(
    g: List[Tuple],
    net: nn.Module,
    n: int,
    track_computation: bool = True
) -> float:
    """
    Calculates the homomorphism density of finite graph g w.r.t the network net.

    Parameters
    ----------
    g : Tuple
        Finite graph for which to calculate homomorphism density w.r.t. the neural network net
    net : nn.Module
        Pytorch neural network
    n : int
        Number of samples to use for density approximation
    track_computation : bool, optional
        Whether to build computation graph for Pytorch's autograd, by default True

    Returns
    -------
    hom_density : float
        Homomorphism density approximation
    """
    torch.set_grad_enabled(track_computation)

    # Unpack graph structure
    V, E = g

    # Scale number of samples from graphon to not bias the gradients with high number of edges
    n = int(math.ceil(n/len(E)))
    # Uniformly sample unit hyper-cube n times
    S_n = np.random.uniform(size=n*len(V))

    # Map vertices to real numbers from <0,1> n times
    mapping_indices = (
        np.tile(E.T, n).T
        + len(V)
        * np.tile(
            np.repeat(
                np.arange(n).reshape(-1, 1),
                len(E),
                axis=0
            ),
            2
        )
    )
    mapped_edges = S_n[mapping_indices]
    # Sort floats in each edge in ascending order to use upper triangle of graphon
    mapped_edges.sort()
    mapped_edges = (
        torch
        .from_numpy(mapped_edges)
        .float()
        .to(os.environ['COMPUTATION_DEVICE'])
    )

    # Sample the network for different edge probabilites for the small graph g
    net_output = net(mapped_edges)

    # Calculate homomorphism density for the small graph g
    reshaped_for_prod = net_output.view(n, len(E))
    edge_probs_prods = torch.prod(reshaped_for_prod, dim=1)
    hom_density = torch.mean(edge_probs_prods)

    torch.set_grad_enabled(True)

    return hom_density
