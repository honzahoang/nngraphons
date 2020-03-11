import random
import numpy as np
import matplotlib.pyplot as plt


def graphon_grow_unit_attach(x, y):
    """Growing uniform attachment graphon"""
    return 1 - np.maximum(x, y)


def graphon_inv_grow_unit_attach(x, y):
    """Inverse growing uniform attachment graphon"""
    return np.maximum(x, y)


def graphon_constant(x, y):
    """Constant 0.5 graphon"""
    return np.full(len(x), 0.5)


def graphon_complete_bipartite(x, y):
    """Complete bipartite graphon"""
    return (
        ((x <= 0.5) & (y <= 0.5))
        | ((x > 0.5) & (y > 0.5))
    ).astype(float)


def sample_graphon(W, size=None, max_size=None):
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
    E = np.random.uniform(size=len(edge_probs)) >= 0.5
    E = edge_indices[E]

    return V, E


def visualize_graphon(W, resolution=300, ax=plt):
    """Plots the graphon W as a greyscale image. The darker the closer to 1."""
    uniform_args = np.linspace(start=0, stop=1, num=resolution)
    cartesian_product = np.transpose(
        [np.tile(uniform_args, len(uniform_args)),
         np.repeat(uniform_args, len(uniform_args))]
    )
    img_mat = (
        W(cartesian_product[:, 0], cartesian_product[:, 1])
        .reshape(resolution, resolution)
    )
    plt.gray()
    ax.imshow(X=1-img_mat, origin='lower', extent=[0, 1, 0, 1])
