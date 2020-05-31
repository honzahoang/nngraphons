from typing import List
import torch


def loss_MSE(t_g: List[float], t_net: List[torch.Tensor]) -> torch.Tensor:
    """
    Mean squared error of homomorphism densities

    Parameters
    ----------
    t_g : List[float]
        Ground truth densities
    t_net : List[torch.Tensor]
        Network densities

    Returns
    -------
    torch.Tensor
        Calculated loss
    """
    SE = [(t_g[i] - t_net[i])**2 for i in range(len(t_g))]
    MSE = torch.mean(torch.stack(SE))

    return MSE


def loss_maxSE(t_g: List[float], t_net: List[torch.Tensor]) -> torch.Tensor:
    """
    Maximum squared error of homomorphism densities

    Parameters
    ----------
    t_g : List[float]
        Ground truth densities
    t_net : List[torch.Tensor]
        Network densities

    Returns
    -------
    torch.Tensor
        Calculated loss
    """
    SE = [(t_g[i] - t_net[i])**2 for i in range(len(t_g))]
    maxSE = torch.max(torch.stack(SE))

    return maxSE
