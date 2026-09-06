from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AugmentConfig:
    p_digit: float = 0.5
    p_rot: float = 0.5
    p_band: float = 0.3


def _sample(probability: float, generator: torch.Generator | None) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    value = torch.rand((), generator=generator)
    return bool(value.item() < probability)


def _permute_digits(grid: Tensor, perm: Tensor) -> Tensor:
    out = grid.clone()
    mask = grid > 0
    out[mask] = perm[grid[mask] - 1]
    return out


def _rotate_grid(grid: Tensor, k: int) -> Tensor:
    return torch.rot90(grid, k, dims=(0, 1))


def _row_indices(band_perm: Tensor, within: Tensor) -> Tensor:
    rows: list[int] = []
    for b in band_perm.tolist():
        base = int(b) * 3
        rows.extend(base + int(r) for r in within[b].tolist())
    return torch.tensor(rows, dtype=torch.long)


def _band_stack_perm(
    grid: Tensor,
    band_perm: Tensor,
    within: Tensor,
    stack_perm: Tensor,
    col_within: Tensor,
) -> Tensor:
    rows = _row_indices(band_perm, within)
    grid = grid[rows, :]
    cols = _row_indices(stack_perm, col_within)
    return grid[:, cols]


def apply_augment(
    clues: Tensor,
    answer: Tensor,
    config: AugmentConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    clues = clues.clone()
    answer = answer.clone()

    if _sample(config.p_band, generator):
        band_perm = torch.randperm(3, generator=generator)
        within = torch.stack([torch.randperm(3, generator=generator) for _ in range(3)])
        stack_perm = torch.randperm(3, generator=generator)
        col_within = torch.stack([torch.randperm(3, generator=generator) for _ in range(3)])
        clues = _band_stack_perm(clues, band_perm, within, stack_perm, col_within)
        answer = _band_stack_perm(answer, band_perm, within, stack_perm, col_within)

    if _sample(config.p_rot, generator):
        k = int(torch.randint(1, 4, (1,), generator=generator).item())
        clues = _rotate_grid(clues, k)
        answer = _rotate_grid(answer, k)

    if _sample(config.p_digit, generator):
        perm = torch.randperm(9, generator=generator) + 1
        clues = _permute_digits(clues, perm)
        answer = _permute_digits(answer, perm)

    return clues, answer
