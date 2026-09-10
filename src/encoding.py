from __future__ import annotations

import torch

NUM_VOCAB = 10
SEQ_LEN = 81
GRID_SIZE = 9


def decode_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode per-cell logits into digit ids 0-9."""
    return logits.argmax(dim=-1)


def target_mask(answer: torch.Tensor, clue_pin: torch.Tensor) -> torch.Tensor:
    """Loss mask: non-clue cells that are filled in the target."""
    return (answer > 0) & (clue_pin == 0)


def cell_acc_mask(clues: torch.Tensor) -> torch.Tensor:
    """Accuracy mask: non-clue cells (matches predict_grid clue pinning)."""
    return clues == 0
