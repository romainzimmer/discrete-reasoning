from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from dataset import PuzzleDataset, filter_rows
from model import MixerNextStateModel
from amp import resolve_amp
from train import EpochStats, build_rollout_config, measure_split, require_run_args


SWEEP_INNER_START = 1
SWEEP_INNER_STEP = 1
SWEEP_OUTER_START = 10
SWEEP_OUTER_STEP = 10
SWEEP_TRIES_START = 10
SWEEP_TRIES_STEP = 10
SWEEP_DEFAULT_INNER = 3
SWEEP_DEFAULT_OUTER = 30
SWEEP_DEFAULT_TRIES = 10


def _sweep_values(start: int, stop: int, step: int) -> list[int]:
    if stop < start:
        return []
    return list(range(start, stop + 1, step))


def _test_point(inner_iters: int, max_outer_iters: int, max_tries: int, stats: EpochStats) -> dict:
    return {
        "inner_iters": inner_iters,
        "max_outer_iters": max_outer_iters,
        "max_tries": max_tries,
        **asdict(stats),
    }


def checkpoint_relative_path(run_dir: Path, checkpoint_path: Path) -> str:
    return checkpoint_path.resolve().relative_to(run_dir.resolve()).as_posix()


def save_test_metrics(
    run_dir: Path,
    *,
    checkpoint: str,
    test: EpochStats,
    test_samples: int,
    min_rating: int | None,
    max_rating: int | None,
    inner_iters: int,
    max_outer_iters: int,
    max_tries: int,
    batch_size: int,
    seed: int,
    inner_sweep: list[dict] | None = None,
    outer_sweep: list[dict] | None = None,
    tries_sweep: list[dict] | None = None,
) -> None:
    history_path = run_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = {"run_id": run_dir.name, "epochs": []}
    payload: dict = {
        "run_id": run_dir.name,
        "checkpoint": checkpoint,
        "test_samples_count": test_samples,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "inner_iters": inner_iters,
        "max_outer_iters": max_outer_iters,
        "max_tries": max_tries,
        "batch_size": batch_size,
        "seed": seed,
        **asdict(test),
    }
    if inner_sweep is not None:
        payload["inner_sweep"] = {
            "max_outer_iters": SWEEP_DEFAULT_OUTER,
            "max_tries": SWEEP_DEFAULT_TRIES,
            "points": inner_sweep,
        }
    if outer_sweep is not None:
        payload["outer_sweep"] = {
            "inner_iters": SWEEP_DEFAULT_INNER,
            "max_tries": SWEEP_DEFAULT_TRIES,
            "points": outer_sweep,
        }
    if tries_sweep is not None:
        payload["tries_sweep"] = {
            "inner_iters": SWEEP_DEFAULT_INNER,
            "max_outer_iters": SWEEP_DEFAULT_OUTER,
            "points": tries_sweep,
        }
    history["test"] = payload
    history_path.write_text(json.dumps(history, indent=2))


def resolve_eval_target(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.is_dir():
        checkpoint = path / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{checkpoint} not found")
        return path, checkpoint
    if path.suffix != ".pt":
        raise ValueError(f"expected run directory or .pt checkpoint, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    if path.parent.name == "epochs":
        return path.parent.parent, path
    return path.parent, path


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_run_args(run_dir: Path, ckpt: dict, *, checkpoint_path: Path) -> dict:
    if "args" in ckpt:
        return require_run_args(ckpt, source=str(checkpoint_path))
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint_path} has no 'args' and {history_path} not found"
        )
    history = json.loads(history_path.read_text())
    return require_run_args(history, source=str(history_path))


def checkpoint_epoch(ckpt: dict, checkpoint_path: Path) -> int:
    if "epoch" in ckpt:
        return int(ckpt["epoch"])
    if checkpoint_path.stem.isdigit():
        return int(checkpoint_path.stem)
    raise KeyError(f"{checkpoint_path} has no 'epoch'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split")
    parser.add_argument(
        "target",
        type=Path,
        help="Run directory (uses best.pt) or path to a .pt checkpoint (best.pt, last.pt, epochs/XXXX.pt)",
    )
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
        help="Inner steps per outer round (default: from checkpoint)",
    )
    parser.add_argument(
        "--max-outer-iters",
        type=int,
        default=None,
        help="Max outer commits per puzzle (default: from checkpoint)",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=1,
        help="Max random inits per puzzle; stop at first halt, else keep last try (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Eval batch size (default: val batch size from checkpoint, else train batch size)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible test metrics")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Also sweep one param at a time up to the given values "
            "(inner 1..N step 1, outer 10..N step 10, max-tries 10..N step 10); "
            "non-swept params stay at 3 inner / 30 outer / 10 tries"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir, checkpoint_path = resolve_eval_target(args.target)
    device = torch.device(args.device)
    ckpt = load_checkpoint(checkpoint_path, device)
    run_args = load_run_args(run_dir, ckpt, checkpoint_path=checkpoint_path)
    print(f"eval checkpoint: {checkpoint_path}", flush=True)

    test_rows = filter_rows(
        "test",
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        max_samples=args.max_test_samples,
    )
    if not test_rows:
        raise ValueError("No test puzzles after filters")

    use_cuda = device.type == "cuda"
    default_batch_size = int(
        run_args.get("val_batch_size")
        or run_args.get("train_batch_size")
        or run_args.get("batch_size", 1)
    )
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    test_ds = PuzzleDataset(rows=test_rows)
    print(
        f"test eval: {len(test_rows)} puzzles, slot_batch_size={batch_size}",
        flush=True,
    )

    model = MixerNextStateModel(
        dim=run_args["dim"],
        num_blocks=run_args["num_blocks"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    default_inner = int(run_args["inner_iters"])
    default_outer = int(run_args["eval_max_outer_iters"])
    inner_iters = args.inner_iters if args.inner_iters is not None else default_inner
    max_outer_iters = args.max_outer_iters if args.max_outer_iters is not None else default_outer
    max_tries = args.max_tries
    halt_loss_weight = float(run_args.get("halt_loss_weight", 1.0))

    amp_enabled = bool(run_args.get("amp", True))
    amp = resolve_amp(device, enabled=amp_enabled)

    checkpoint_epoch_num = checkpoint_epoch(ckpt, checkpoint_path)
    checkpoint_rel = checkpoint_relative_path(run_dir, checkpoint_path)

    def run_eval(inner: int, outer: int, tries: int) -> EpochStats:
        rollout_config = build_rollout_config(inner_iters=inner, max_outer_iters=outer)
        return measure_split(
            model,
            test_ds._base_clues,
            test_ds._base_answers,
            device,
            slot_batch_size=batch_size,
            epoch=checkpoint_epoch_num,
            epochs=checkpoint_epoch_num,
            phase="test",
            progress_desc="test",
            rollout_config=rollout_config,
            halt_loss_weight=halt_loss_weight,
            use_cuda=use_cuda,
            seed=args.seed,
            amp=amp,
            max_tries=tries,
        )

    test = run_eval(inner_iters, max_outer_iters, max_tries)

    inner_sweep: list[dict] | None = None
    outer_sweep: list[dict] | None = None
    tries_sweep: list[dict] | None = None
    if args.sweep:
        inner_sweep = []
        for inner in _sweep_values(SWEEP_INNER_START, inner_iters, SWEEP_INNER_STEP):
            stats = run_eval(inner, SWEEP_DEFAULT_OUTER, SWEEP_DEFAULT_TRIES)
            inner_sweep.append(
                _test_point(inner, SWEEP_DEFAULT_OUTER, SWEEP_DEFAULT_TRIES, stats)
            )
            print(
                f"test sweep inner={inner} max_outer={SWEEP_DEFAULT_OUTER} "
                f"max_tries={SWEEP_DEFAULT_TRIES}: "
                f"puzzle_acc={stats.puzzle_acc:.4f}",
                flush=True,
            )
        outer_sweep = []
        for outer in _sweep_values(SWEEP_OUTER_START, max_outer_iters, SWEEP_OUTER_STEP):
            stats = run_eval(SWEEP_DEFAULT_INNER, outer, SWEEP_DEFAULT_TRIES)
            outer_sweep.append(
                _test_point(SWEEP_DEFAULT_INNER, outer, SWEEP_DEFAULT_TRIES, stats)
            )
            print(
                f"test sweep inner={SWEEP_DEFAULT_INNER} max_outer={outer} "
                f"max_tries={SWEEP_DEFAULT_TRIES}: "
                f"puzzle_acc={stats.puzzle_acc:.4f}",
                flush=True,
            )
        tries_sweep = []
        for tries in _sweep_values(SWEEP_TRIES_START, max_tries, SWEEP_TRIES_STEP):
            stats = run_eval(SWEEP_DEFAULT_INNER, SWEEP_DEFAULT_OUTER, tries)
            tries_sweep.append(
                _test_point(SWEEP_DEFAULT_INNER, SWEEP_DEFAULT_OUTER, tries, stats)
            )
            print(
                f"test sweep inner={SWEEP_DEFAULT_INNER} max_outer={SWEEP_DEFAULT_OUTER} "
                f"max_tries={tries}: "
                f"puzzle_acc={stats.puzzle_acc:.4f}",
                flush=True,
            )

    save_test_metrics(
        run_dir,
        checkpoint=checkpoint_rel,
        test=test,
        test_samples=len(test_rows),
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        inner_iters=inner_iters,
        max_outer_iters=max_outer_iters,
        max_tries=max_tries,
        batch_size=batch_size,
        seed=args.seed,
        inner_sweep=inner_sweep,
        outer_sweep=outer_sweep,
        tries_sweep=tries_sweep,
    )
    print(
        f"test ({checkpoint_rel}, n={len(test_rows)}, "
        f"inner={inner_iters}, max_outer={max_outer_iters}, "
        f"max_tries={max_tries}): "
        f"loss={test.loss:.4f} cell_acc={test.cell_acc:.4f} "
        f"cell_loss={test.cell_loss:.4f} halt_loss={test.halt_loss:.4f} "
        f"puzzle_acc={test.puzzle_acc:.4f} halt_rate={test.halt_rate:.4f} "
        f"avg_tries={test.avg_tries:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
