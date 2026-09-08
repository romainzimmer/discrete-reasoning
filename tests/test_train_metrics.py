from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from train import EpochStats, TrainEpochStats, save_epoch_metrics


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
