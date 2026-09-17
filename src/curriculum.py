from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from rating_groups import NUM_RATING_GROUPS

CURRICULUM_P_GT_TARGET_ACC = 0.5
CURRICULUM_P_GT_LOGIT_DECAY = 0.99
CURRICULUM_P_GT_LOGIT_STEP = 1.0
DEFAULT_CURRICULUM_P_GT_CAP = 0.5


def curriculum_p_gt_cap_from_logit(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def curriculum_p_gt_cap_logit_from_cap(p_gt_cap: float) -> float:
    p = min(max(p_gt_cap, 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def update_curriculum_p_gt_cap_logit(
    logit: float,
    puzzle_acc: float,
    *,
    target_acc: float = CURRICULUM_P_GT_TARGET_ACC,
    logit_step: float = CURRICULUM_P_GT_LOGIT_STEP,
    logit_decay: float = CURRICULUM_P_GT_LOGIT_DECAY,
) -> float:
    """Adjust p_gt cap in logit space so done puzzle accuracy stays near target_acc."""
    return logit * logit_decay + logit_step * (target_acc - puzzle_acc)


@dataclass
class CurriculumState:
    logits: list[float]
    logit_step: float = CURRICULUM_P_GT_LOGIT_STEP
    logit_decay: float = CURRICULUM_P_GT_LOGIT_DECAY
    _p_gt_cap_tensors: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    @classmethod
    def default(
        cls,
        *,
        logit_step: float = CURRICULUM_P_GT_LOGIT_STEP,
        logit_decay: float = CURRICULUM_P_GT_LOGIT_DECAY,
    ) -> CurriculumState:
        return cls(
            logits=[curriculum_p_gt_cap_logit_from_cap(DEFAULT_CURRICULUM_P_GT_CAP)]
            * NUM_RATING_GROUPS,
            logit_step=logit_step,
            logit_decay=logit_decay,
        )

    def p_gt_cap_by_group(self) -> tuple[float, ...]:
        return tuple(curriculum_p_gt_cap_from_logit(logit) for logit in self.logits)

    def mean_p_gt_cap(self) -> float:
        values = self.p_gt_cap_by_group()
        return sum(values) / len(values)

    def update_from_group_accs(self, group_puzzle_accs: list[float | None]) -> None:
        if len(group_puzzle_accs) != NUM_RATING_GROUPS:
            raise ValueError(
                f"expected {NUM_RATING_GROUPS} group accuracies, got {len(group_puzzle_accs)}"
            )
        for group, acc in enumerate(group_puzzle_accs):
            if acc is None:
                continue
            self.logits[group] = update_curriculum_p_gt_cap_logit(
                self.logits[group],
                acc,
                logit_step=self.logit_step,
                logit_decay=self.logit_decay,
            )
        self._p_gt_cap_tensors.clear()

    def p_gt_cap_tensor(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._p_gt_cap_tensors.get(key)
        if cached is not None:
            return cached
        tensor = torch.tensor(self.p_gt_cap_by_group(), device=device, dtype=torch.float32)
        self._p_gt_cap_tensors[key] = tensor
        return tensor

    def format_p_gt_cap_log(self) -> str:
        return ",".join(f"{p:.4f}" for p in self.p_gt_cap_by_group())

    def history_fields(self) -> dict[str, float]:
        return {
            f"curriculum_p_gt_cap_g{group}": p
            for group, p in enumerate(self.p_gt_cap_by_group())
        }
