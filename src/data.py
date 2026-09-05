from __future__ import annotations

import csv
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"


def parse_grid(s: str) -> list[list[int]]:
    """Parse an 81-char row-major string into a 9x9 grid (0 = empty)."""
    if len(s) != 81:
        raise ValueError(f"Expected 81 characters, got {len(s)}")
    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            c = s[i * 9 + j]
            row.append(0 if c == "." else int(c))
        grid.append(row)
    return grid


def grid_to_tensor(grid: list[list[int]]) -> torch.Tensor:
    """Return shape (9, 9) int64 tensor."""
    return torch.tensor(grid, dtype=torch.long)


def puzzle_to_tensor(question: str) -> torch.Tensor:
    return grid_to_tensor(parse_grid(question))


def answer_to_tensor(answer: str) -> torch.Tensor:
    return grid_to_tensor(parse_grid(answer))


def tensor_to_string(grid: torch.Tensor) -> str:
    """Encode a (9, 9) grid as an 81-char string."""
    chars: list[str] = []
    for r in range(9):
        for c in range(9):
            value = int(grid[r, c].item())
            chars.append("." if value == 0 else str(value))
    return "".join(chars)


def load_split(split: str = "train") -> list[dict[str, str | int]]:
    path = TRAIN_CSV if split == "train" else TEST_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run download-dataset"
        )
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["rating"] = int(row["rating"])
    return rows
