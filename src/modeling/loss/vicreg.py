# From solo-learn development team.
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ...utils.modeling import gather


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and projected features z2
    from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor, z2: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and projected
    features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and projected
    features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = (
        cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    )
    return cov_loss


def vicreg_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
    gamma_param: float = 1.0,
    return_dict=False,
) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Computes VICReg's loss given batch of projected features z1 from view 1 and projected
    features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.

    Returns:
        torch.Tensor: VICReg loss.
    """

    sim_loss = invariance_loss(z1, z2)

    # vicreg's official code gathers the tensors here
    # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    z1, z2 = gather(z1), gather(z2)

    var_loss = variance_loss(z1, z2, gamma_param)
    cov_loss = covariance_loss(z1, z2)

    loss = (
        sim_loss_weight * sim_loss
        + var_loss_weight * var_loss
        + cov_loss_weight * cov_loss
    )

    if return_dict:
        return {
            "loss": loss,
            "sim_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }

    return loss, (sim_loss, var_loss, cov_loss)
    # return loss # for old online eval
