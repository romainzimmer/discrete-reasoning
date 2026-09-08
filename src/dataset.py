from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from augment import AugmentConfig, apply_augment
from data import answer_to_tensor, load_split, puzzle_to_tensor


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
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _aug_generator(self, idx: int) -> torch.Generator | None:
        if self.aug_seed is None:
            return None
        # Vary augmentations across epochs while keeping them deterministic per (epoch, idx).
        seed = self.aug_seed + self._epoch * len(self.rows) + idx
        return torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        clues = puzzle_to_tensor(row["question"])
        answer = answer_to_tensor(row["answer"])
        if self.augment and self.aug_config is not None:
            clues, answer = apply_augment(
                clues,
                answer,
                self.aug_config,
                generator=self._aug_generator(idx),
            )
        return {
            "clues": clues,
            "answer": answer,
        }


def collate_puzzles(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "clues": torch.stack([item["clues"] for item in batch]),
        "answer": torch.stack([item["answer"] for item in batch]),
    }


@dataclass
class PuzzleTensorCache:
    clues: torch.Tensor
    answers: torch.Tensor

    @classmethod
    def build(cls, dataset: PuzzleDataset, *, pin_memory: bool = False) -> PuzzleTensorCache:
        clues_list: list[torch.Tensor] = []
        answers_list: list[torch.Tensor] = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            clues_list.append(item["clues"])
            answers_list.append(item["answer"])
        clues = torch.stack(clues_list)
        answers = torch.stack(answers_list)
        if pin_memory:
            clues = clues.pin_memory()
            answers = answers.pin_memory()
        return cls(clues=clues, answers=answers)

