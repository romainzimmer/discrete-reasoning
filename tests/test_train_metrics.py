from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from rollout import RolloutResult
from train import (
    EpochStats,
    TrainEpochStats,
    _accumulate_pred_stats,
    save_epoch_metrics,
)


def test_accumulate_pred_stats_from_training_logits() -> None:
    result = RolloutResult(
        loss=torch.tensor(1.0),
        pred=torch.tensor(
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
        ),
    )
    answer = torch.full((9, 9), 4)
    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 5
    correct_cells, total_cells, correct_puzzles = _accumulate_pred_stats(
        result,
        answer,
        clues,
        correct_cells=0,
        total_cells=0,
        correct_puzzles=0,
    )
    assert total_cells == 81 - 1
    assert correct_cells == int((result.pred[clues == 0] == answer[clues == 0]).sum())
    assert correct_puzzles == int((result.pred == answer).all())


def test_save_epoch_metrics_omits_missing_train_acc(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = Namespace(epochs=1)
    save_epoch_metrics(
        run_dir,
        epoch=1,
        train=TrainEpochStats(loss=1.25),
        val=EpochStats(loss=2.0, cell_acc=0.5, puzzle_acc=0.1),
        args=args,
    )
    epoch = json.loads((run_dir / "history.json").read_text())["epochs"][0]
    assert epoch["train_loss"] == 1.25
    assert "train_cell_acc" not in epoch
    assert "train_puzzle_acc" not in epoch
    assert epoch["val_cell_acc"] == 0.5


def test_save_epoch_metrics_includes_train_acc_when_computed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = Namespace(epochs=1, compute_train_acc=True)
    save_epoch_metrics(
        run_dir,
        epoch=1,
        train=TrainEpochStats(loss=1.25, cell_acc=0.9, puzzle_acc=0.4),
        val=EpochStats(loss=2.0, cell_acc=0.5, puzzle_acc=0.1),
        args=args,
    )
    epoch = json.loads((run_dir / "history.json").read_text())["epochs"][0]
    assert epoch["train_cell_acc"] == 0.9
    assert epoch["train_puzzle_acc"] == 0.4
