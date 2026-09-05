from __future__ import annotations

import torch
from torch import nn

GRID_SIZE = 9
NUM_CLASSES = 9
INPUT_DIM = GRID_SIZE * GRID_SIZE * NUM_CLASSES
OUTPUT_DIM = GRID_SIZE * GRID_SIZE * NUM_CLASSES


class NextStateModel(nn.Module):
    """Predict solved grid logits from current one-hot state."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, 9, 9) -> logits (B, 9, 9, 9)."""
        b = x.size(0)
        logits = self.net(x.reshape(b, INPUT_DIM))
        return logits.reshape(b, GRID_SIZE, GRID_SIZE, NUM_CLASSES)
