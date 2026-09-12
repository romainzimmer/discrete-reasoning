from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from rollout import BatchSlotState
from curriculum import (
    CURRICULUM_P_GT_LOGIT_STEP,
    CURRICULUM_P_GT_TARGET_ACC,
    DEFAULT_CURRICULUM_P_GT_LOGIT,
    CurriculumState,
    curriculum_p_gt_from_logit,
    curriculum_p_gt_logit_from_p_gt,
    rating_group,
    update_curriculum_p_gt_logit,
)
from train import (
    TrainEpochStats,
    EpochStats,
    TrainMetricsAccumulator,
    save_epoch_metrics,
)


def test_accumulate_step_metrics_from_training_logits() -> None:
    from rollout import RolloutResult

    pred = torch.tensor(
            [
                [
                    [5, 3, 0, 0, 7, 0, 0, 0, 0],
                    [6, 0, 0, 1, 9, 5, 0, 0, 0],
                    [0, 9, 8, 0, 0, 0, 0, 6, 0],
                    [8, 0, 0, 0, 6, 0, 0, 0, 3],
                    [4, 0, 0, 8, 0, 3, 0, 0, 1],
                    [7, 0, 0, 0, 2, 0, 0, 0, 6],
                    [0, 6, 0, 0, 0, 0, 2, 8, 0],
                    [0, 0, 0, 4, 1, 9, 0, 0, 5],
                    [0, 0, 0, 0, 8, 0, 0, 7, 9],
                ]
            ]
        )
    result = RolloutResult(
        loss=torch.tensor(1.0),
        pred=pred,
        done=torch.tensor([True]),
        halted=torch.tensor([False]),
        halt_target=torch.tensor([0.0]),
        halt_logit=torch.tensor([0.0]),
    )
    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 5
    answer = torch.full((9, 9), 4)
    clues_b = clues.unsqueeze(0)
    answer_b = answer.unsqueeze(0)
    clue_pin = clues_b > 0
    state = BatchSlotState(
        digit_id=clues_b,
        clues=clues_b,
        answer=answer_b,
        clue_pin=clue_pin,
        rating_group=torch.zeros(1, dtype=torch.long),
        outer_count=torch.tensor([1]),
    )
    acc = TrainMetricsAccumulator.empty(torch.device("cpu"))
    acc.add_step(result, state)
    stats = acc.finalize()
    assert int(acc.total_cells.item()) == 81 - 1
    assert int(acc.correct_cells.item()) == int(
        (result.pred[0][clues == 0] == answer[clues == 0]).sum()
    )
    assert stats.completions_per_epoch == 1
    assert stats.avg_steps_per_puzzle == 1.0
    assert int(acc.correct_puzzles_done.item()) == int((result.pred[0] == answer).all())


def test_accumulate_step_uses_pred_with_clue_mask() -> None:
    from rollout import RolloutResult

    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 5
    answer = torch.full((9, 9), 4)
    clues_b = clues.unsqueeze(0)
    answer_b = answer.unsqueeze(0)
    clue_pin = clues_b > 0
    pred = torch.full((1, 9, 9), 2)
    pred[0, 0, 0] = 5
    result = RolloutResult(
        loss=torch.tensor(1.0),
        pred=pred,
        done=torch.tensor([True]),
        halted=torch.tensor([False]),
        halt_target=torch.tensor([0.0]),
        halt_logit=torch.tensor([0.0]),
    )
    state = BatchSlotState(
        digit_id=clues_b,
        clues=clues_b,
        answer=answer_b,
        clue_pin=clue_pin,
        rating_group=torch.zeros(1, dtype=torch.long),
        outer_count=torch.tensor([1]),
    )
    acc = TrainMetricsAccumulator.empty(torch.device("cpu"))
    acc.add_step(result, state)
    mask = clues == 0
    expected_correct = int((pred[0][mask] == answer[mask]).sum())
    assert int(acc.correct_cells.item()) == expected_correct
    assert int(acc.total_cells.item()) == int(mask.sum())
    assert int(acc.correct_puzzles_done.item()) == 0


def test_save_epoch_metrics_includes_train_acc(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = Namespace(epochs=1)
    save_epoch_metrics(
        run_dir,
        epoch=1,
        train=TrainEpochStats(loss=1.25, cell_acc=0.9, puzzle_acc=0.4, halt_acc=0.5),
        val=EpochStats(loss=2.0, cell_acc=0.5, puzzle_acc=0.1, halt_acc=0.6),
        args=args,
        curriculum_state=CurriculumState.from_legacy_p_gt(0.55),
    )
    epoch = json.loads((run_dir / "history.json").read_text())["epochs"][0]
    assert epoch["train_cell_acc"] == 0.9
    assert epoch["train_puzzle_acc"] == 0.4
    assert epoch["curriculum_p_gt"] == pytest.approx(0.55)
    assert epoch["curriculum_p_gt_g0"] == pytest.approx(0.55)


def test_update_curriculum_p_gt_logit_increases_when_acc_below_target() -> None:
    logit = 0.0
    updated = update_curriculum_p_gt_logit(logit, puzzle_acc=0.3)
    assert updated > logit
    assert curriculum_p_gt_from_logit(updated) > curriculum_p_gt_from_logit(logit)


def test_update_curriculum_p_gt_logit_decreases_when_acc_above_target() -> None:
    logit = 0.0
    updated = update_curriculum_p_gt_logit(logit, puzzle_acc=0.7)
    assert updated < logit
    assert curriculum_p_gt_from_logit(updated) < curriculum_p_gt_from_logit(logit)


def test_update_curriculum_p_gt_logit_is_unchanged_at_target_acc() -> None:
    logit = 0.2
    assert update_curriculum_p_gt_logit(logit, puzzle_acc=0.5) == pytest.approx(logit)


def test_default_curriculum_p_gt_logit_is_zero_and_maps_to_half() -> None:
    assert DEFAULT_CURRICULUM_P_GT_LOGIT == pytest.approx(0.0)
    assert curriculum_p_gt_from_logit(DEFAULT_CURRICULUM_P_GT_LOGIT) == pytest.approx(0.5)


def test_update_curriculum_p_gt_logit_uses_target_minus_acc() -> None:
    logit = 0.1
    puzzle_acc = 0.35
    expected = logit + CURRICULUM_P_GT_LOGIT_STEP * (
        CURRICULUM_P_GT_TARGET_ACC - puzzle_acc
    )
    assert update_curriculum_p_gt_logit(logit, puzzle_acc) == pytest.approx(expected)


def test_curriculum_p_gt_logit_from_p_gt_roundtrip() -> None:
    for p_gt in (0.1, 0.5, 0.9):
        logit = curriculum_p_gt_logit_from_p_gt(p_gt)
        assert curriculum_p_gt_from_logit(logit) == pytest.approx(p_gt)


def test_curriculum_logit_drifts_up_when_acc_stays_below_target() -> None:
    logit = DEFAULT_CURRICULUM_P_GT_LOGIT
    for _ in range(5):
        logit = update_curriculum_p_gt_logit(logit, puzzle_acc=0.3)
    assert logit == pytest.approx(5 * CURRICULUM_P_GT_LOGIT_STEP * 0.2)
    assert curriculum_p_gt_from_logit(logit) > 0.5


def test_curriculum_logit_stays_fixed_when_acc_matches_target() -> None:
    logit = DEFAULT_CURRICULUM_P_GT_LOGIT
    for _ in range(5):
        logit = update_curriculum_p_gt_logit(logit, puzzle_acc=CURRICULUM_P_GT_TARGET_ACC)
    assert logit == pytest.approx(DEFAULT_CURRICULUM_P_GT_LOGIT)


def test_rating_group_bins_are_quintiles() -> None:
    assert rating_group(0) == 0
    assert rating_group(1) == 0
    assert rating_group(2) == 1
    assert rating_group(15) == 2
    assert rating_group(30) == 3
    assert rating_group(100) == 4


def test_curriculum_state_updates_groups_independently() -> None:
    state = CurriculumState.default()
    state.update_from_group_accs([0.3, 0.5, None, 0.7, None])
    assert state.p_gt_by_group()[0] > 0.5
    assert state.p_gt_by_group()[1] == pytest.approx(0.5)
    assert state.p_gt_by_group()[2] == pytest.approx(0.5)
    assert state.p_gt_by_group()[3] < 0.5
