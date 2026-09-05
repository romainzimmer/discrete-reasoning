from __future__ import annotations

import torch


def grid_to_onehot(grid: torch.Tensor) -> torch.Tensor:
    """Encode a grid as one-hot per cell.

    Args:
        grid: (..., 9, 9) int tensor, 0 = empty, 1-9 = digit.

    Returns:
        (..., 9, 9, 10) float tensor. Empty cells are all zeros; digit d has a 1 at index d.
    """
    *batch, h, w = grid.shape
    onehot = torch.zeros(*batch, h, w, 10, dtype=torch.float32, device=grid.device)
    mask = grid != 0
    if mask.any():
        digits = grid[mask]
        idx = torch.arange(mask.sum(), device=grid.device)
        onehot.view(-1, 10)[idx, digits] = 1.0
    return onehot


def decode_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode per-cell logits into a grid.

    Takes argmax per cell; if max logit > 0, fill with that index, else empty (0).
    """
    max_val, max_idx = logits.max(dim=-1)
    return torch.where(max_val > 0, max_idx, torch.zeros_like(max_idx))
