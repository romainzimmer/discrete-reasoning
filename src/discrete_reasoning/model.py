from __future__ import annotations

import torch
from torch import nn

GRID_SIZE = 9
NUM_DIGITS = 10
FLAT_DIM = GRID_SIZE * GRID_SIZE * NUM_DIGITS


class NextStateModel(nn.Module):
    """Predict solved grid logits from current one-hot state."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FLAT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, FLAT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, 9, 10) -> logits (B, 9, 9, 10)."""
        b = x.size(0)
        logits = self.net(x.reshape(b, FLAT_DIM))
        return logits.reshape(b, GRID_SIZE, GRID_SIZE, NUM_DIGITS)
