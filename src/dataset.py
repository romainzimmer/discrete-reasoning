from __future__ import annotations

import torch
from torch.utils.data import Dataset

from augment import AugmentConfig, apply_augment
from data import answer_to_tensor, load_split, puzzle_to_tensor
from encoding import grid_to_onehot


def filter_rows(
    split: str,
    *,
    min_rating: int | None = None,
    max_rating: int | None = None,
    max_samples: int | None = None,
) -> list[dict]:
    rows = load_split(split)
    if min_rating is not None:
        rows = [r for r in rows if r["rating"] >= min_rating]
    if max_rating is not None:
        rows = [r for r in rows if r["rating"] <= max_rating]
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


class PuzzleDataset(Dataset):
    """Clues and answer pairs."""

    def __init__(
        self,
        split: str = "train",
        *,
        rows: list[dict] | None = None,
        min_rating: int | None = None,
        max_rating: int | None = None,
        max_samples: int | None = None,
        augment: bool = False,
        aug_config: AugmentConfig | None = None,
        aug_seed: int | None = None,
    ):
        if rows is not None:
            self.rows = rows
        else:
            self.rows = filter_rows(
                split,
                min_rating=min_rating,
                max_rating=max_rating,
                max_samples=max_samples,
            )
        self.augment = augment
        self.aug_config = aug_config
        self.aug_seed = aug_seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        clues = puzzle_to_tensor(row["question"])
        answer = answer_to_tensor(row["answer"])
        if self.augment and self.aug_config is not None:
            generator = None
            if self.aug_seed is not None:
                generator = torch.Generator().manual_seed(self.aug_seed + idx)
            clues, answer = apply_augment(
                clues,
                answer,
                self.aug_config,
                generator=generator,
            )
        return {
            "clues": clues,
            "clues_onehot": grid_to_onehot(clues),
            "answer": answer,
        }


def collate_puzzles(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "clues": torch.stack([item["clues"] for item in batch]),
        "clues_onehot": torch.stack([item["clues_onehot"] for item in batch]),
        "answer": torch.stack([item["answer"] for item in batch]),
    }

