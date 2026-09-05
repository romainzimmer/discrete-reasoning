from __future__ import annotations

import torch
from torch.utils.data import Dataset

from discrete_reasoning.data import answer_to_tensor, load_split, puzzle_to_tensor
from discrete_reasoning.encoding import grid_to_onehot
from discrete_reasoning.trajectory import Trajectory, demo_trajectory


class PuzzleDataset(Dataset):
    """Clues and answer pairs."""

    def __init__(
        self,
        split: str = "train",
        *,
        min_rating: int | None = None,
        max_rating: int | None = None,
        max_samples: int | None = None,
    ):
        rows = load_split(split)
        if min_rating is not None:
            rows = [r for r in rows if r["rating"] >= min_rating]
        if max_rating is not None:
            rows = [r for r in rows if r["rating"] <= max_rating]
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        clues = puzzle_to_tensor(row["question"])
        return {
            "clues": clues,
            "clues_onehot": grid_to_onehot(clues),
            "answer": answer_to_tensor(row["answer"]),
        }


def collate_puzzles(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "clues": torch.stack([item["clues"] for item in batch]),
        "clues_onehot": torch.stack([item["clues_onehot"] for item in batch]),
        "answer": torch.stack([item["answer"] for item in batch]),
    }


def build_transitions(traj: Trajectory) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Step-wise (current, next) pairs for debugging."""
    pairs = []
    for t in range(len(traj.steps)):
        current = grid_to_onehot(traj.grid_tensor_at(t))
        nxt = grid_to_onehot(traj.grid_tensor_at(t + 1))
        pairs.append((current, nxt))
    return pairs


def demo_transitions(question: str, answer: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return build_transitions(demo_trajectory(question, answer))
