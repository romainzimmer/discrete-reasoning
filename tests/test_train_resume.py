from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from curriculum import CurriculumState, curriculum_p_gt_from_logit
from model import MixerNextStateModel
from train import (
    TrainEpochStats,
    EpochStats,
    best_val_cell_acc_for_resume,
    best_val_cell_acc_from_history,
    curriculum_state_for_resume,
    curriculum_state_from_history,
    save_checkpoint,
    save_run_config,
    update_history_args,
    validate_resume_epochs,
)


def test_validate_resume_epochs_rejects_completed_or_lower() -> None:
    validate_resume_epochs(5, 10)
    with pytest.raises(ValueError, match="must be greater than completed epoch 5"):
        validate_resume_epochs(5, 5)
    with pytest.raises(ValueError, match="must be greater than completed epoch 5"):
        validate_resume_epochs(5, 3)


def test_best_val_cell_acc_from_history(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    history = {
        "run_id": run_dir.name,
        "args": {"epochs": 3},
        "epochs": [
            {"epoch": 1, "val_cell_acc": 0.4},
            {"epoch": 2, "val_cell_acc": 0.7},
        ],
    }
    (run_dir / "history.json").write_text(json.dumps(history))
    assert best_val_cell_acc_from_history(run_dir) == pytest.approx(0.7)


def test_curriculum_state_from_history(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    history = {
        "run_id": run_dir.name,
        "args": {"epochs": 2},
        "epochs": [
            {
                "epoch": 1,
                "train_group_0_puzzle_acc": 0.3,
                "train_group_0_puzzles_done": 10,
            },
            {
                "epoch": 2,
                "train_group_0_puzzle_acc": 0.7,
                "train_group_0_puzzles_done": 10,
            },
        ],
    }
    (run_dir / "history.json").write_text(json.dumps(history))
    state = curriculum_state_from_history(run_dir)
    assert state.p_gt_by_group()[0] == pytest.approx(0.5)


def test_curriculum_state_for_resume_prefers_group_logits(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = {"curriculum_p_gt_logits": [0.1, 0.2, 0.0, -0.1, 0.3]}
    state = curriculum_state_for_resume(ckpt, run_dir)
    assert state.logits == pytest.approx([0.1, 0.2, 0.0, -0.1, 0.3])


def test_curriculum_state_for_resume_legacy_p_gt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = {"curriculum_p_gt": 0.55}
    state = curriculum_state_for_resume(ckpt, run_dir)
    assert state.p_gt_by_group()[0] == pytest.approx(0.55)


def test_best_val_cell_acc_for_resume_prefers_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = {"best_val_cell_acc": 0.82}
    assert best_val_cell_acc_for_resume(ckpt, run_dir) == pytest.approx(0.82)


def test_best_val_cell_acc_for_resume_falls_back_to_best_pt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = MixerNextStateModel(dim=32, num_blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    args = Namespace(epochs=5, dim=32, num_blocks=1, lr=1e-4, weight_decay=0.01, model="looped-mixer")
    save_checkpoint(
        run_dir / "best.pt",
        model=model,
        optimizer=optimizer,
        epoch=2,
        train=TrainEpochStats(loss=1.0),
        val=EpochStats(loss=2.0, cell_acc=0.71),
        args=args,
        best_val_cell_acc=0.71,
    )
    ckpt = {"epoch": 2, "val_cell_acc": 0.5}
    assert best_val_cell_acc_for_resume(ckpt, run_dir) == pytest.approx(0.71)


def test_update_history_args_overwrites_epochs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_run_config(run_dir, Namespace(epochs=5, model="looped-mixer"))
    update_history_args(run_dir, Namespace(epochs=10, model="looped-mixer"))
    history = json.loads((run_dir / "history.json").read_text())
    assert history["args"]["epochs"] == 10


def test_last_checkpoint_roundtrip_for_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = MixerNextStateModel(dim=32, num_blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    args = Namespace(
        epochs=5,
        dim=32,
        num_blocks=1,
        lr=1e-4,
        weight_decay=0.01,
        model="looped-mixer",
    )
    curriculum_state = CurriculumState(logits=[0.1, 0.0, -0.1, 0.2, 0.05])
    save_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        train=TrainEpochStats(loss=1.0),
        val=EpochStats(loss=2.0, cell_acc=0.5),
        args=args,
        curriculum_state=curriculum_state,
        best_val_cell_acc=0.5,
    )
    ckpt = torch.load(run_dir / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 3
    assert ckpt["args"]["epochs"] == 5
    assert ckpt["curriculum_p_gt_logits"] == pytest.approx([0.1, 0.0, -0.1, 0.2, 0.05])
    assert ckpt["curriculum_p_gt"] == pytest.approx(curriculum_state.mean_p_gt())
    assert ckpt["best_val_cell_acc"] == pytest.approx(0.5)
