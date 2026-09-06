from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from encoding import NUM_STATE_CHANNELS

GRID_SIZE = 9
NUM_CLASSES = 9
INPUT_DIM = GRID_SIZE * GRID_SIZE * NUM_STATE_CHANNELS
OUTPUT_DIM = GRID_SIZE * GRID_SIZE * NUM_CLASSES


def _ffn_intermediate_dim(width: int) -> int:
    hidden = int(2 * (4 * width) / 3)
    return ((hidden + 7) // 8) * 8


class FFNBlock(nn.Module):
    """Pre-norm SwiGLU block: x + down(silu(gate(norm(x))) * up(norm(x)))."""

    def __init__(self, width: int, intermediate_dim: int | None = None):
        super().__init__()
        hidden = intermediate_dim if intermediate_dim is not None else _ffn_intermediate_dim(width)
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class OutputHead(nn.Module):
    def __init__(self, width: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class NextStateModel(nn.Module):
    """Predict solved grid logits from current one-hot state + clue mask."""

    def __init__(self, *, width: int = 512, num_blocks: int = 2):
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        layers: list[nn.Module] = [nn.Linear(INPUT_DIM, width)]
        layers.extend(FFNBlock(width) for _ in range(num_blocks))
        layers.append(OutputHead(width, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)
        self.width = width
        self.num_blocks = num_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9, 9, 10) -> logits (B, 9, 9, 9). Last channel is clue mask."""
        b = x.size(0)
        logits = self.net(x.reshape(b, INPUT_DIM))
        return logits.reshape(b, GRID_SIZE, GRID_SIZE, NUM_CLASSES)
