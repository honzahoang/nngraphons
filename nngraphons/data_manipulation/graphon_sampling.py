import os
import random
import numpy as np
import torch


def sample_synthetic_graphon(W, size=None, max_size=None):
    """Returns a random graph sampled from graphon W."""
    # Number of vertices
    n_V = random.randint(2, max_size) if size is None else size
    # Randomly select nodes in graphon
    V = np.random.uniform(size=n_V)
    # All possible edges
    edge_indices = np.transpose(np.vstack(np.tril_indices(len(V))))
    edges = V[edge_indices]
    # Edge probabilites (graphon values)
    edge_probs = W(edges[:, 0], edges[:, 1])

    # Convert float values of vertices to integer indexes
    V = np.arange(len(V))

    # Threshold probabilites to decide which pairs become edges
    E = np.random.uniform(size=len(edge_probs)) <= edge_probs
    E = edge_indices[E]

    return V, E


def sample_neural_network_graphon(net, size=None, max_size=None):
    """Returns a random graph sampled from neural network net."""
    # Number of vertices
    n_V = random.randint(2, max_size) if size is None else size
    # Randomly select nodes in graphon
    V = np.random.uniform(size=n_V)
    # All possible edges
    edge_indices = np.transpose(np.vstack(np.tril_indices(len(V))))
    edges = V[edge_indices]
    edges.sort()
    # Edge probabilites (graphon values)
    with torch.no_grad():
        edge_probs = (
            net(
                torch
                .from_numpy(edges)
                .float()
                .to(os.environ['COMPUTATION_DEVICE'])
            )
            .cpu()
            .numpy()
            .squeeze()
        )

    # Convert float values of vertices to integer indexes
    V = np.arange(len(V))

    # Threshold probabilites to decide which pairs become edges
    E = np.random.uniform(size=len(edge_probs)) <= edge_probs
    E = edge_indices[E]

    return V, E
