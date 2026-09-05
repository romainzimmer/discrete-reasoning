from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from data import parse_grid


@dataclass
class Trajectory:
    question: str
    answer: str
    steps: list[tuple[int, int, int]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add_fill(self, row: int, col: int, value: int) -> None:
        self.steps.append((row, col, value))

    def grid_at(self, t: int) -> list[list[int]]:
        """Return 9x9 grid after t fill steps (t=0 is clues only)."""
        grid = [row[:] for row in parse_grid(self.question)]
        for row, col, value in self.steps[:t]:
            grid[row][col] = value
        return grid

    def grid_tensor_at(self, t: int) -> torch.Tensor:
        return torch.tensor(self.grid_at(t), dtype=torch.long)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "steps": [list(step) for step in self.steps],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Trajectory:
        return cls(
            question=data["question"],
            answer=data["answer"],
            steps=[tuple(step) for step in data["steps"]],
            meta=data.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        return cls.from_dict(json.loads(Path(path).read_text()))


def demo_trajectory(question: str, answer: str, **meta) -> Trajectory:
    """Build a demo trajectory by revealing the answer one empty cell at a time."""
    traj = Trajectory(question=question, answer=answer, meta=meta)
    clues = parse_grid(question)
    solution = parse_grid(answer)
    for r in range(9):
        for c in range(9):
            if clues[r][c] == 0:
                traj.add_fill(r, c, solution[r][c])
    return traj


def demo_states(clues: torch.Tensor, answer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build demo trajectory grids from tensor clues/answer. Returns (states, T)."""
    b = clues.size(0)
    t = (clues == 0).sum(dim=(1, 2))
    max_t = int(t.max().item())
    states = torch.zeros(b, max_t + 1, 9, 9, dtype=clues.dtype, device=clues.device)
    states[:, 0] = clues
    for i in range(b):
        grid = clues[i].clone()
        step = 1
        for r in range(9):
            for c in range(9):
                if clues[i, r, c] == 0:
                    grid = grid.clone()
                    grid[r, c] = answer[i, r, c]
                    states[i, step] = grid
                    step += 1
    return states, t
