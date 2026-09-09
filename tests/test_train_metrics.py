from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from rollout import BatchSlotState
from train import (
    TrainEpochStats,
    EpochStats,
    TrainMetricsAccumulator,
    save_epoch_metrics,
)


def test_accumulate_step_metrics_from_training_logits() -> None:
    from rollout import RolloutResult

    result = RolloutResult(
        loss=torch.tensor(1.0),
        pred=torch.tensor(
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
        ),
        done=torch.tensor([True]),
        halted=torch.tensor([False]),
        halt_target=torch.tensor([0.0]),
        halt_logit=torch.tensor([0.0]),
    )
    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 5
    answer = torch.full((9, 9), 4)
    state = BatchSlotState(
        digit_id=clues.unsqueeze(0),
        clues=clues.unsqueeze(0),
        answer=answer.unsqueeze(0),
        clue_pin=clues.unsqueeze(0) > 0,
        outer_count=torch.tensor([1]),
    )
    acc = TrainMetricsAccumulator.empty(torch.device("cpu"))
    acc.add_step(result, state)
    stats = acc.finalize()
    assert int(acc.total_cells.item()) == 81 - 1
    assert int(acc.correct_cells.item()) == int((result.pred[0][clues == 0] == answer[clues == 0]).sum())
    assert stats.completions_per_epoch == 1
    assert int(acc.correct_puzzles_done.item()) == int((result.pred[0] == answer).all())


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
    )
    epoch = json.loads((run_dir / "history.json").read_text())["epochs"][0]
    assert epoch["train_cell_acc"] == 0.9
    assert epoch["train_puzzle_acc"] == 0.4
