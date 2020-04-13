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
    mappings = np.random.randint(low=0, high=len(V_G)-1, size=(N, len(V_g)))

    # Create mapped vertex edges to later check in adjacency matrix
    mapped_edges = np.empty(shape=(len(E_g)*N, 2), dtype=np.uint16)
    for i in range(N):
        mapped_edges[i*len(E_g):(i+1)*len(E_g)] = mappings[i][E_g]

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
    reshaped_for_prod = net_output.view(n, len(E))
    edge_probs_prods = torch.prod(reshaped_for_prod, dim=1)
    hom_density = torch.mean(edge_probs_prods)

    torch.set_grad_enabled(True)

    return hom_density
