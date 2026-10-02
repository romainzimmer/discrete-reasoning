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


def constraint_violation_mask(digit_id: torch.Tensor) -> torch.Tensor:
    """Per-cell Sudoku uniqueness violations for the current grid.

    digit_id: (B, 9, 9) or (9, 9), values 0-9. Empty cells (0) are never violations.
    Returns bool tensor of the same spatial shape: True if the cell's digit duplicates
    another filled cell in its row, column, or 3×3 block.
    """
    if digit_id.dim() == 2:
        digit_id = digit_id.unsqueeze(0)
        squeeze = True
    elif digit_id.dim() == 3:
        squeeze = False
    else:
        raise ValueError("digit_id must be (9, 9) or (B, 9, 9)")

    g = digit_id
    filled = g > 0

    row_same = (g.unsqueeze(-1) == g.unsqueeze(-2)) & filled.unsqueeze(-1)
    row_viol = row_same.sum(dim=-1) > 1

    col_same = (g.unsqueeze(2) == g.unsqueeze(1)) & filled.unsqueeze(2)
    col_viol = col_same.sum(dim=1) > 1

    blocks = g.view(g.size(0), 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(g.size(0), 3, 3, 9)
    block_same = (blocks.unsqueeze(-1) == blocks.unsqueeze(-2)) & (blocks.unsqueeze(-1) > 0)
    block_viol_local = block_same.sum(dim=-1) > 1
    block_viol = (
        block_viol_local.view(g.size(0), 3, 3, 3, 3)
        .permute(0, 1, 3, 2, 4)
        .reshape(g.size(0), GRID_SIZE, GRID_SIZE)
    )

    out = row_viol | col_viol | block_viol
    return out[0] if squeeze else out
