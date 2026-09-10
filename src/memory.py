from __future__ import annotations

import torch

from amp import STORAGE_DTYPE, to_storage_dtype


def memory_init(value: torch.Tensor) -> torch.Tensor:
    """Detached memory carry for the next outer step (no grad)."""
    return to_storage_dtype(value.detach()).detach()


def zero_memory(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch, 9, 9, dim, device=device, dtype=STORAGE_DTYPE)
