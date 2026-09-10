from __future__ import annotations

import torch

from amp import STORAGE_DTYPE, to_storage_dtype


def ema_combine(
    h: torch.Tensor,
    ema: torch.Tensor,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """Convex blend: α·h + (1−α)·ema."""
    return alpha * h + (1.0 - alpha) * ema


def ema_update(ema: torch.Tensor, value: torch.Tensor, alpha: float) -> torch.Tensor:
    return ema_combine(to_storage_dtype(value.detach()), ema, alpha).detach()


def memory_init(value: torch.Tensor) -> torch.Tensor:
    """Detached memory carry for the next outer step (no grad)."""
    return to_storage_dtype(value.detach()).detach()


def zero_ema(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch, 9, 9, dim, device=device, dtype=STORAGE_DTYPE)
