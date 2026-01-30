"""Base building blocks for Skat ML models."""

import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual Block for Deep Networks.

    Structure: Input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> + Input
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)
