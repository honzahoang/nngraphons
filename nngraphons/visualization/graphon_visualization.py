import os
from typing import Callable, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def visualize_synthetic_graphon(
    W: Callable[[np.ndarray, np.ndarray], np.ndarray],
    resolution: int = 300
) -> Tuple:
    """Plots the graphon W as a unit square function."""
    uniform_args = np.linspace(start=0, stop=1, num=resolution)
    cartesian_product = np.transpose(
        [np.tile(uniform_args, len(uniform_args)),
         np.repeat(uniform_args, len(uniform_args))]
    )
    img_mat = (
        W(cartesian_product[:, 0], cartesian_product[:, 1])
        .reshape(resolution, resolution)
    )
    fig = plt.figure()
    ax = plt.imshow(
        X=img_mat,
        origin='lower',
        extent=[0, 1, 0, 1],
        cmap='plasma',
        vmin=0,
        vmax=1
    )
    plt.show()

    return fig, ax


def visualize_neural_network_graphon(net: nn.Module, resolution: int = 300) -> Tuple:
    """Plots the neural network net as a unit square function."""
    uniform_args = np.linspace(start=0, stop=1, num=resolution)
    cartesian_product = np.transpose(
        [np.tile(uniform_args, len(uniform_args)),
         np.repeat(uniform_args, len(uniform_args))]
    )
    cartesian_product.sort()
    with torch.no_grad():
        img_mat = (
            net(
                torch
                .from_numpy(cartesian_product)
                .float()
                .to(os.environ['COMPUTATION_DEVICE'])
            )
            .cpu()
            .numpy()
            .reshape(resolution, resolution)
        )
    # Make visualization symmetric
    tril = np.tril_indices(len(img_mat))
    img_mat[tril] = img_mat[tril[1], tril[0]]
    fig = plt.figure()
    ax = plt.imshow(
        X=img_mat,
        origin='lower',
        extent=[0, 1, 0, 1],
        cmap='plasma',
        vmin=0,
        vmax=1
    )
    plt.colorbar()
    plt.show()

    return fig, ax
