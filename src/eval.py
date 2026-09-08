from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PuzzleDataset, collate_puzzles, filter_rows
from model import NextStateModel
from rollout import DEFAULT_INNER_ITERS, DEFAULT_OUTER_COMMIT_PROB, DEFAULT_OUTER_ITERS, RolloutConfig
from train import EpochStats, build_rollout_config, measure_split, require_run_args


def eval_rollout_iters_from_run_args(run_args: dict) -> tuple[int, int]:
    if "eval_inner_iters" in run_args and "eval_outer_iters" in run_args:
        return int(run_args["eval_inner_iters"]), int(run_args["eval_outer_iters"])
    if "eval_rollout_iter" in run_args:
        return 1, int(run_args["eval_rollout_iter"])
    return DEFAULT_INNER_ITERS, DEFAULT_OUTER_ITERS


def save_test_metrics(
    run_dir: Path,
    *,
    epoch: int,
    test: EpochStats,
    test_samples: int,
    min_rating: int | None,
    max_rating: int | None,
    inner_iters: int,
    outer_iters: int,
) -> None:
    history_path = run_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = {"run_id": run_dir.name, "epochs": []}
    history["test"] = {
        "best_epoch": epoch,
        "test_samples_count": test_samples,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "inner_iters": inner_iters,
        "outer_iters": outer_iters,
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
    parser.add_argument(
        "--min-rating",
        type=int,
        default=None,
        help="Min puzzle rating on test split (default: no filter)",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=None,
        help="Max puzzle rating on test split (default: no filter)",
    )
    parser.add_argument(
        "--inner-iters",
        type=int,
        default=None,
        help="Inner steps per outer loop (default: from checkpoint)",
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=None,
        help="Outer argmax commits (default: from checkpoint)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible test metrics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    device = torch.device(args.device)
    ckpt = load_best_checkpoint(run_dir, device)
    run_args = require_run_args(ckpt, source=str(run_dir / "best.pt"))

    test_rows = filter_rows(
        "test",
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        max_samples=args.max_test_samples,
    )
    if not test_rows:
        raise ValueError("No test puzzles after filters")

    use_cuda = device.type == "cuda"
    batch_size = int(run_args.get("batch_size", 1))
    test_loader = DataLoader(
        PuzzleDataset(rows=test_rows),
        batch_size=batch_size,
        collate_fn=collate_puzzles,
        pin_memory=use_cuda,
        num_workers=run_args["num_workers"],
    )

    model = NextStateModel(
        width=run_args["width"],
        num_blocks=run_args["num_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    default_inner, default_outer = eval_rollout_iters_from_run_args(run_args)
    inner_iters = args.inner_iters if args.inner_iters is not None else default_inner
    outer_iters = args.outer_iters if args.outer_iters is not None else default_outer
    rollout_config = build_rollout_config(
        train_init="clues",
        inner_iters=inner_iters,
        outer_iters=outer_iters,
        outer_commit_prob=float(run_args.get("outer_commit_prob", DEFAULT_OUTER_COMMIT_PROB)),
        fixed_point=bool(run_args.get("fixed_point", True)),
    )

    epoch = int(ckpt["epoch"])
    test = measure_split(
        model,
        test_loader,
        device,
        epoch=epoch,
        epochs=epoch,
        phase="test",
        rollout_config=rollout_config,
        use_cuda=use_cuda,
        seed=args.seed,
    )
    save_test_metrics(
        run_dir,
        epoch=epoch,
        test=test,
        test_samples=len(test_rows),
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        inner_iters=inner_iters,
        outer_iters=outer_iters,
    )
    print(
        f"test (epoch {epoch}, n={len(test_rows)}, inner={inner_iters}, outer={outer_iters}): "
        f"loss={test.loss:.4f} cell_acc={test.cell_acc:.4f} "
        f"puzzle_acc={test.puzzle_acc:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
