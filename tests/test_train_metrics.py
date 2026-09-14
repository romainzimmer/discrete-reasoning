from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from rollout import BatchSlotState
from rating_groups import rating_group
from train import (
    TrainEpochStats,
    EpochStats,
    TrainMetricsAccumulator,
    build_rollout_config,
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


def test_build_rollout_config_gt_reveal_default() -> None:
    config = build_rollout_config(inner_iters=2, max_outer_iters=3)
    assert config.gt_reveal is True


def test_save_epoch_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = Namespace(epochs=1)
    save_epoch_metrics(
        run_dir,
        epoch=1,
        train=TrainEpochStats(loss=1.25, cell_acc=0.9, puzzle_acc=0.4, halt_acc=0.5),
        val=EpochStats(loss=2.0, cell_acc=0.5, puzzle_acc=0.1, halt_acc=0.6),
        args=args,
    )
    epoch = json.loads((run_dir / "history.json").read_text())["epochs"][0]
    assert epoch["train_cell_acc"] == 0.9
    assert epoch["train_puzzle_acc"] == 0.4


def test_rating_group_bins_are_quintiles() -> None:
    assert rating_group(0) == 0
    assert rating_group(1) == 0
    assert rating_group(2) == 1
    assert rating_group(15) == 2
    assert rating_group(30) == 3
    assert rating_group(100) == 4
