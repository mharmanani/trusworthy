from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils.metrics import (
    expected_calibration_error,
    brier_score,
    reliability_diagram,
)
import torch 

import sys
sys.path.append("/home/harmanan/TRUSnet")


def show_reliability_diagram(df):
    probs = df.prob_1.values.squeeze()
    targets = df.y.values.squeeze()
    preds = (probs > 0.5).astype(int)
    conf = np.max(np.stack([probs, 1 - probs], axis=1), axis=1).squeeze()

    ece, _ = expected_calibration_error(preds, conf, targets, n_bins=20)
    brier = brier_score(probs, targets)

    plt.figure()
    reliability_diagram(preds, conf, targets, n_bins=20)
    # put floating caption on plot saying ECE
    plt.text(
        0.1,
        0.9,
        f"ECE: {ece:.3f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.text(
        0.1,
        0.8,
        f"Brier: {brier:.3f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )


def show_prob_histogram(df):
    plt.figure()
    df.query("y == 1").prob_1.hist(bins=20, alpha=0.5, label="cancer", density=True)
    df.query("y == 0").prob_1.hist(bins=20, alpha=0.5, label="benign", density=True)
    plt.axvline(
        0.5,
        color="k",
        linestyle="--",
    )
    plt.legend()
    plt.xlabel("Probability of cancer")
    plt.ylabel("Density")


def convert_patchwise_to_corewise_dataframe(df):
    corewise_df = df.groupby(["core_specifier"]).prob_1.mean().reset_index()
    corewise_df["y"] = (
        df.groupby(["core_specifier"]).y.first().reset_index().y.astype(int)
    )
    return corewise_df


def apply_temperature_calibration(val_df, test_df, lr=1e-2, mode="ce"):
    val_probs = torch.tensor(val_df.prob_1.values).view(-1, 1)
    val_y = torch.tensor(val_df.y.values).view(-1, 1)

    from src.utils.calibration import (
        compute_temperature_and_bias_for_calibration,
        apply_temperature_and_bias,
    )

    temp, bias = compute_temperature_and_bias_for_calibration(
        val_probs, val_y, lr=lr, mode=mode
    )

    val_df["prob_1"] = apply_temperature_and_bias(val_df.prob_1.values, temp, bias)
    test_df["prob_1"] = apply_temperature_and_bias(test_df.prob_1.values, temp, bias)

    return val_df, test_df

import torch
import torch.distributed as dist
from torch.optim import Optimizer
import torch.nn as nn


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)