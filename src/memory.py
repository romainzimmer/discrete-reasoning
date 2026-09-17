from __future__ import annotations

import torch

from amp import STORAGE_DTYPE, to_storage_dtype


def memory_init(value: torch.Tensor) -> torch.Tensor:
    """Detached memory carry for the next outer step (no grad)."""
    return to_storage_dtype(value.detach()).detach()


def zero_memory(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch, 9, 9, dim, device=device, dtype=STORAGE_DTYPE)


def inner_halpern_alphas(inner_iters: int) -> list[float]:
    """Anchor weight α_t before forward t (pre-forward blend)."""
    if inner_iters <= 0:
        raise ValueError("inner_iters must be positive")
    n = inner_iters
    if n == 1:
        return [0.0]
    return [(n - 1 - t) / (n - 1) for t in range(n)]


def inner_halpern_input(
    *,
    anchor: torch.Tensor | None,
    carry: torch.Tensor | None,
    alpha: float,
) -> torch.Tensor | None:
    """Blend outer-step anchor with raw carry before the next forward."""
    if alpha == 0.0:
        return carry
    if alpha == 1.0:
        return anchor
    if anchor is None:
        assert carry is not None
        return carry.mul(1.0 - alpha)
    assert carry is not None
    return torch.lerp(carry.float(), anchor.float(), alpha)
