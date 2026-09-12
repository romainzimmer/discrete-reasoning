from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

NUM_RATING_GROUPS = 5

# Quintile bins from sapientinc/sudoku-extreme train.csv (~3.83M puzzles).
RATING_GROUP_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 11),
    (12, 23),
    (24, 38),
    (39, 10_000),
)

CURRICULUM_P_GT_TARGET_ACC = 0.5
CURRICULUM_P_GT_LOGIT_STEP = 0.1
DEFAULT_CURRICULUM_P_GT = 0.5
DEFAULT_CURRICULUM_P_GT_LOGIT = 0.0

GROUP_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")


def rating_group(rating: int) -> int:
    for group, (lo, hi) in enumerate(RATING_GROUP_BOUNDS):
        if lo <= rating <= hi:
            return group
    return NUM_RATING_GROUPS - 1


def rating_group_label(group: int) -> str:
    lo, hi = RATING_GROUP_BOUNDS[group]
    return f"G{group} [{lo},{hi}]"


def curriculum_p_gt_from_logit(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def curriculum_p_gt_logit_from_p_gt(p_gt: float) -> float:
    p = min(max(p_gt, 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def update_curriculum_p_gt_logit(
    logit: float,
    puzzle_acc: float,
    *,
    target_acc: float = CURRICULUM_P_GT_TARGET_ACC,
) -> float:
    """Adjust curriculum p_gt in logit space so done puzzle accuracy stays near target_acc."""
    return logit + CURRICULUM_P_GT_LOGIT_STEP * (target_acc - puzzle_acc)


@dataclass
class CurriculumState:
    logits: list[float]
    _p_gt_tensors: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    @classmethod
    def default(cls) -> CurriculumState:
        return cls(logits=[DEFAULT_CURRICULUM_P_GT_LOGIT] * NUM_RATING_GROUPS)

    @classmethod
    def from_legacy_logit(cls, logit: float) -> CurriculumState:
        return cls(logits=[logit] * NUM_RATING_GROUPS)

    @classmethod
    def from_legacy_p_gt(cls, p_gt: float) -> CurriculumState:
        return cls.from_legacy_logit(curriculum_p_gt_logit_from_p_gt(p_gt))

    def p_gt_by_group(self) -> tuple[float, ...]:
        return tuple(curriculum_p_gt_from_logit(logit) for logit in self.logits)

    def mean_p_gt(self) -> float:
        values = self.p_gt_by_group()
        return sum(values) / len(values)

    def update_from_group_accs(self, group_puzzle_accs: list[float | None]) -> None:
        if len(group_puzzle_accs) != NUM_RATING_GROUPS:
            raise ValueError(
                f"expected {NUM_RATING_GROUPS} group accuracies, got {len(group_puzzle_accs)}"
            )
        for group, acc in enumerate(group_puzzle_accs):
            if acc is None:
                continue
            self.logits[group] = update_curriculum_p_gt_logit(self.logits[group], acc)
        self._p_gt_tensors.clear()

    def p_gt_tensor(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._p_gt_tensors.get(key)
        if cached is not None:
            return cached
        tensor = torch.tensor(self.p_gt_by_group(), device=device, dtype=torch.float32)
        self._p_gt_tensors[key] = tensor
        return tensor

    def format_p_gt_log(self) -> str:
        return ",".join(f"{p:.4f}" for p in self.p_gt_by_group())

    def history_fields(self) -> dict[str, float]:
        return {
            f"curriculum_p_gt_g{group}": p
            for group, p in enumerate(self.p_gt_by_group())
        }
