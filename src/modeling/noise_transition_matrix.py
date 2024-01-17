"""
An additional network layer that models the noise transition matrix for noisy labels, as described in the paper
Training deep neural-networks using a noise adaptation layer
https://openreview.net/forum?id=H12GRgcxg
"""


import torch
from torch.nn.functional import linear


class NoiseTransitionMatrix(torch.nn.Module):
    def __init__(self, num_classes, initial_temperature=1e-1):
        super().__init__()
        data = torch.eye(num_classes)
        data = data + torch.rand_like(data) * 1e-1
        data = data / initial_temperature

        self.noise_transition_matrix = torch.nn.Parameter(data, requires_grad=True)

    def forward(self, logits):
        probs = logits.softmax(1)
        if self.training:
            noise_transition_matrix = self.noise_transition_matrix.softmax(1)
            return linear(probs, noise_transition_matrix)
        else:
            return probs
