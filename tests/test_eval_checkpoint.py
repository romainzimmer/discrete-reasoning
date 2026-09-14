from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from eval import (
    checkpoint_epoch,
    checkpoint_relative_path,
    load_run_args,
    resolve_eval_target,
    save_test_metrics,
)
from model import MixerNextStateModel
from train import EpochStats, save_epoch_checkpoint, save_run_config


def test_resolve_eval_target_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.pt").write_bytes(b"")
    resolved_run, checkpoint = resolve_eval_target(run_dir)
    assert resolved_run == run_dir.resolve()
    assert checkpoint == run_dir / "best.pt"


def test_resolve_eval_target_missing_best(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="best.pt"):
        resolve_eval_target(run_dir)


def test_resolve_eval_target_checkpoint_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    last = run_dir / "last.pt"
    last.write_bytes(b"")
    epoch_ckpt = run_dir / "epochs" / "0003.pt"
    epoch_ckpt.parent.mkdir()
    epoch_ckpt.write_bytes(b"")

    assert resolve_eval_target(last) == (run_dir.resolve(), last.resolve())
    assert resolve_eval_target(epoch_ckpt) == (run_dir.resolve(), epoch_ckpt.resolve())


def test_load_run_args_from_history_for_epoch_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_run_config(
        run_dir,
        Namespace(
            model="looped-mixer",
            dim=32,
            num_blocks=1,
            inner_iters=3,
            train_max_outer_iters=10,
            eval_max_outer_iters=10,
            train_batch_size=8,
            batches_per_epoch=10,
            halt_loss_weight=1.0,
            num_workers=0,
            min_rating=0,
            max_rating=0,
            epochs=5,
        ),
    )
    model = MixerNextStateModel(dim=32, num_blocks=1)
    save_epoch_checkpoint(run_dir, 2, model)
    ckpt = torch.load(run_dir / "epochs" / "0002.pt", weights_only=False)
    run_args = load_run_args(run_dir, ckpt, checkpoint_path=run_dir / "epochs" / "0002.pt")
    assert run_args["dim"] == 32
    assert run_args["inner_iters"] == 3


def test_checkpoint_epoch_from_filename(tmp_path: Path) -> None:
    path = tmp_path / "epochs" / "0007.pt"
    assert checkpoint_epoch({"model": {}}, path) == 7


def test_checkpoint_relative_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "my-run"
    run_dir.mkdir()
    checkpoint = run_dir / "epochs" / "0003.pt"
    checkpoint.parent.mkdir()
    assert checkpoint_relative_path(run_dir, checkpoint) == "epochs/0003.pt"


def test_save_test_metrics_logs_run_and_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "my-run"
    run_dir.mkdir()
    save_test_metrics(
        run_dir,
        checkpoint="last.pt",
        test=EpochStats(loss=1.0, cell_acc=0.5, puzzle_acc=0.1),
        test_samples=10,
        min_rating=None,
        max_rating=None,
        inner_iters=3,
        max_outer_iters=30,
        max_tries=1,
        batch_size=8,
        seed=0,
    )
    test = json.loads((run_dir / "history.json").read_text())["test"]
    assert test["run_id"] == "my-run"
    assert test["checkpoint"] == "last.pt"
    assert "checkpoint_epoch" not in test
    assert "best_epoch" not in test
