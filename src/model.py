from __future__ import annotations

import torch
from torch import nn

from encoding import NUM_STATE_CHANNELS

GRID_SIZE = 9
NUM_CLASSES = 9
INPUT_DIM = GRID_SIZE * GRID_SIZE * NUM_STATE_CHANNELS
OUTPUT_DIM = GRID_SIZE * GRID_SIZE * NUM_CLASSES


class NextStateModel(nn.Module):
    """Predict solved grid logits from current one-hot state + clue mask."""

    def __init__(self, hidden_sizes: list[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 512]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer width")

        layers: list[nn.Module] = []
        in_dim = INPUT_DIM
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)
        self.hidden_sizes = list(hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, 9, 10) -> logits (B, 9, 9, 9). Last channel is clue mask."""
        b = x.size(0)
        logits = self.net(x.reshape(b, INPUT_DIM))
        return logits.reshape(b, GRID_SIZE, GRID_SIZE, NUM_CLASSES)
