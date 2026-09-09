from __future__ import annotations

import torch

from amp import STORAGE_DTYPE, to_storage_dtype

DEFAULT_EMA_ALPHA = 0.05


def validate_ema_alpha(alpha: float) -> None:
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("ema_alpha must be in (0, 1]")


def uses_ema(alpha: float) -> bool:
    return alpha < 1.0


def ema_combine(h: torch.Tensor, ema: torch.Tensor, alpha: float) -> torch.Tensor:
    """Convex blend: α·h + (1−α)·ema."""
    return alpha * h + (1.0 - alpha) * ema


def ema_update(ema: torch.Tensor, value: torch.Tensor, alpha: float) -> torch.Tensor:
    return ema_combine(to_storage_dtype(value.detach()), ema, alpha).detach()


def memory_init(value: torch.Tensor) -> torch.Tensor:
    """Detached memory carry for the next outer step (no grad)."""
    return to_storage_dtype(value.detach()).detach()


def zero_ema(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch, 9, 9, dim, device=device, dtype=STORAGE_DTYPE)
