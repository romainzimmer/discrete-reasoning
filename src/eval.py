from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import PuzzleDataset, collate_puzzles, filter_rows
from model import NextStateModel
from rollout import RolloutConfig
from train import EpochStats, measure_split, require_run_args


def save_test_metrics(run_dir: Path, *, epoch: int, test: EpochStats, test_samples: int) -> None:
    history_path = run_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = {"run_id": run_dir.name, "epochs": []}
    history["test"] = {
        "best_epoch": epoch,
        "test_samples_count": test_samples,
        **asdict(test),
    }
    history_path.write_text(json.dumps(history, indent=2))


def load_best_checkpoint(run_dir: Path, device: torch.device) -> dict:
    path = run_dir / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return torch.load(path, map_location=device, weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best.pt on the test split")
    parser.add_argument("run_dir", type=Path, help="Run directory containing best.pt")
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Max puzzles from test.csv after filters (default: all)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    device = torch.device(args.device)
    ckpt = load_best_checkpoint(run_dir, device)
    run_args = require_run_args(ckpt, source=str(run_dir / "best.pt"))

    test_rows = filter_rows(
        "test",
        min_rating=run_args["min_rating"],
        max_rating=run_args["max_rating"],
        max_samples=args.max_test_samples,
    )
    if not test_rows:
        raise ValueError("No test puzzles after filters")

    use_cuda = device.type == "cuda"
    test_loader = DataLoader(
        PuzzleDataset(rows=test_rows),
        batch_size=1,
        collate_fn=collate_puzzles,
        pin_memory=use_cuda,
        num_workers=run_args["num_workers"],
    )

    rollout_mode = run_args["rollout_mode"]
    model = NextStateModel(
        width=run_args["width"],
        num_blocks=run_args["num_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    loss_fn = (
        nn.BCEWithLogitsLoss()
        if rollout_mode == "threshold"
        else nn.CrossEntropyLoss()
    )
    rollout_config = RolloutConfig(mode=rollout_mode, train_init="clues")

    epoch = int(ckpt["epoch"])
    test = measure_split(
        model,
        test_loader,
        loss_fn,
        device,
        epoch=epoch,
        epochs=epoch,
        phase="test",
        max_rollout_iters=run_args["eval_max_rollout_iter"],
        rollout_config=rollout_config,
        use_cuda=use_cuda,
    )
    save_test_metrics(run_dir, epoch=epoch, test=test, test_samples=len(test_rows))
    print(
        f"test (epoch {epoch}, n={len(test_rows)}): "
        f"loss={test.loss:.4f} cell_acc={test.cell_acc:.4f} "
        f"puzzle_acc={test.puzzle_acc:.4f} steps={test.avg_rollout_steps:.1f} "
        f"cycle={test.avg_cycle_length:.1f} max_iter={test.max_iter_pct:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
