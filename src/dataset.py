from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from augment import AugmentConfig, apply_augment
from rating_groups import rating_group
from data import answer_to_tensor, load_split, puzzle_to_tensor


def sample_rows(
    rows: list[dict],
    *,
    max_samples: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    if max_samples is None or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_samples)


def filter_rows(
    split: str,
    *,
    min_rating: int | None = None,
    max_rating: int | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    rows = load_split(split)
    if min_rating is not None:
        rows = [r for r in rows if r["rating"] >= min_rating]
    if max_rating is not None:
        rows = [r for r in rows if r["rating"] <= max_rating]
    return sample_rows(rows, max_samples=max_samples, seed=seed)


class PuzzleDataset(Dataset):
    """Clues and answer pairs with cached base tensors for random rollout sampling."""

    def __init__(
        self,
        split: str = "train",
        *,
        rows: list[dict] | None = None,
        min_rating: int | None = None,
        max_rating: int | None = None,
        max_samples: int | None = None,
        sample_seed: int | None = None,
        augment: bool = False,
        aug_config: AugmentConfig | None = None,
        aug_seed: int | None = None,
        pin_memory: bool = False,
    ):
        if rows is not None:
            self.rows = rows
        else:
            self.rows = filter_rows(
                split,
                min_rating=min_rating,
                max_rating=max_rating,
                max_samples=max_samples,
                seed=sample_seed,
            )
        self.augment = augment
        self.aug_config = aug_config
        self.aug_seed = aug_seed
        self._epoch = 0
        self._sample_counter = 0
        self._materialize_base_tensors(pin_memory=pin_memory)

    @classmethod
    def from_tensors(
        cls,
        clues: torch.Tensor,
        answers: torch.Tensor,
        *,
        augment: bool = False,
        aug_config: AugmentConfig | None = None,
        aug_seed: int | None = None,
    ) -> PuzzleDataset:
        """Test helper: dataset backed by pre-built tensors instead of row strings."""
        if clues.dim() == 2:
            clues = clues.unsqueeze(0)
            answers = answers.unsqueeze(0)
        n = clues.size(0)
        placeholder = {"question": "0" * 81, "answer": "1" * 81, "source": "test", "rating": 0}
        ds = cls.__new__(cls)
        ds.rows = [placeholder] * n
        ds.augment = augment
        ds.aug_config = aug_config
        ds.aug_seed = aug_seed
        ds._epoch = 0
        ds._sample_counter = 0
        ds._base_clues = clues
        ds._base_answers = answers
        ds._base_rating_groups = torch.zeros(n, dtype=torch.long)
        return ds

    def _materialize_base_tensors(self, *, pin_memory: bool = False) -> None:
        clues = torch.stack([puzzle_to_tensor(row["question"]) for row in self.rows])
        answers = torch.stack([answer_to_tensor(row["answer"]) for row in self.rows])
        rating_groups = torch.tensor(
            [rating_group(int(row["rating"])) for row in self.rows],
            dtype=torch.long,
        )
        # Only for main-process sampling (train refill). Unsafe with DataLoader workers.
        if pin_memory:
            clues = clues.pin_memory()
            answers = answers.pin_memory()
        self._base_clues = clues
        self._base_answers = answers
        self._base_rating_groups = rating_groups

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._sample_counter = 0

    def _aug_generator(self, idx: int) -> torch.Generator | None:
        if self.aug_seed is None:
            return None
        # Fixed per (epoch, idx) for DataLoader __getitem__ paths.
        seed = self.aug_seed + self._epoch * len(self.rows) + idx
        return torch.Generator().manual_seed(seed)

    def _sample_aug_generator(self, idx: int) -> torch.Generator | None:
        if self.aug_seed is None:
            return None
        n = len(self.rows)
        seed = self.aug_seed + self._epoch * n + idx + self._sample_counter * (n + 1)
        self._sample_counter += 1
        return torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def augment_pair(
        self,
        idx: int,
        clues: torch.Tensor,
        answer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment and self.aug_config is not None:
            return apply_augment(
                clues,
                answer,
                self.aug_config,
                generator=self._aug_generator(idx),
            )
        return clues, answer

    def _sample_augment_pair(
        self,
        idx: int,
        clues: torch.Tensor,
        answer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment and self.aug_config is not None:
            return apply_augment(
                clues,
                answer,
                self.aug_config,
                generator=self._sample_aug_generator(idx),
            )
        return clues, answer

    def sample(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clues = self._base_clues[indices]
        answers = self._base_answers[indices]
        rating_groups = self._base_rating_groups[indices]
        if not self.augment or self.aug_config is None:
            return clues, answers, rating_groups
        aug_clues: list[torch.Tensor] = []
        aug_answers: list[torch.Tensor] = []
        for offset, idx in enumerate(indices.tolist()):
            c, a = self._sample_augment_pair(idx, clues[offset], answers[offset])
            aug_clues.append(c)
            aug_answers.append(a)
        return torch.stack(aug_clues), torch.stack(aug_answers), rating_groups

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clues, answer = self.augment_pair(idx, self._base_clues[idx], self._base_answers[idx])
        return {
            "clues": clues,
            "answer": answer,
        }


def collate_puzzles(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "clues": torch.stack([item["clues"] for item in batch]),
        "answer": torch.stack([item["answer"] for item in batch]),
    }
