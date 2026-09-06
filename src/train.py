from __future__ import annotations

import argparse
import json
import secrets
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PuzzleDataset, collate_puzzles, filter_rows
from model import NextStateModel
from rollout import DEFAULT_MAX_ROLLOUT_ITER, RolloutConfig, RolloutResult, rollout_train_batch
from viz_data import (
    load_manifest,
    save_epoch_trajectories,
    save_manifest,
    update_manifest_split,
)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


def _epoch_desc(epoch: int, epochs: int, phase: str) -> str:
    return f"epoch {epoch}/{epochs} {phase}"


@dataclass
class EpochStats:
    loss: float
    avg_rollout_steps: float
    max_iter_pct: float
    avg_cycle_length: float = 0.0
    cell_acc: float = 0.0
    puzzle_acc: float = 0.0


def make_run_dir(runs_dir: Path) -> Path:
    run_id = secrets.token_hex(4)
    name = datetime.now().strftime(f"%Y%m%d-%H%M%S-{run_id}")
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def split_train_val(
    rows: list[dict],
    *,
    val_samples: int | None,
    val_fraction: float,
    max_samples: int | None,
) -> tuple[list[dict], list[dict]]:
    pool = rows[:max_samples] if max_samples is not None else rows
    if not pool:
        return [], []
    n_val = val_samples if val_samples is not None else max(1, int(len(pool) * val_fraction))
    n_val = min(n_val, len(pool) - 1) if len(pool) > 1 else 1
    val_rows = pool[:n_val]
    train_rows = pool[n_val:]
    return train_rows, val_rows


def save_epoch_metrics(
    run_dir: Path,
    *,
    epoch: int,
    train: EpochStats,
    val: EpochStats,
    epoch_seconds: float,
    args: argparse.Namespace,
) -> None:
    history_path = run_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = {"run_id": run_dir.name, "args": vars(args), "epochs": []}

    history["args"] = json_safe(vars(args))
    history["epochs"] = [e for e in history["epochs"] if e["epoch"] != epoch]
    history["epochs"].append(
        {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            **{f"train_{k}": v for k, v in asdict(train).items()},
            **{f"val_{k}": v for k, v in asdict(val).items()},
        }
    )
    history["epochs"].sort(key=lambda row: row["epoch"])
    history["total_seconds"] = sum(e.get("epoch_seconds", 0) for e in history["epochs"])
    history_path.write_text(json.dumps(history, indent=2))


def save_best_test_metrics(run_dir: Path, *, epoch: int, test: EpochStats) -> None:
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text())
    history["test"] = {"best_epoch": epoch, **asdict(test)}
    history_path.write_text(json.dumps(history, indent=2))


def save_epoch_checkpoint(run_dir: Path, epoch: int, model: NextStateModel) -> None:
    epoch_dir = run_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict()}, epoch_dir / f"{epoch:04d}.pt")


def save_checkpoint(
    path: Path,
    *,
    model: NextStateModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train: EpochStats,
    val: EpochStats,
    test: EpochStats | None = None,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train.loss,
        "train_avg_rollout_steps": train.avg_rollout_steps,
        "train_max_iter_pct": train.max_iter_pct,
        "train_avg_cycle_length": train.avg_cycle_length,
        "train_cell_acc": train.cell_acc,
        "train_puzzle_acc": train.puzzle_acc,
        "val_loss": val.loss,
        "val_avg_rollout_steps": val.avg_rollout_steps,
        "val_max_iter_pct": val.max_iter_pct,
        "val_avg_cycle_length": val.avg_cycle_length,
        "val_cell_acc": val.cell_acc,
        "val_puzzle_acc": val.puzzle_acc,
        "args": vars(args),
    }
    if test is not None:
        payload.update(
            {
                "test_loss": test.loss,
                "test_avg_rollout_steps": test.avg_rollout_steps,
                "test_max_iter_pct": test.max_iter_pct,
                "test_avg_cycle_length": test.avg_cycle_length,
                "test_cell_acc": test.cell_acc,
                "test_puzzle_acc": test.puzzle_acc,
            }
        )
    torch.save(payload, path)


def _update_accuracy(
    pred: torch.Tensor,
    answer: torch.Tensor,
    *,
    correct_cells: int,
    total_cells: int,
    correct_puzzles: int,
) -> tuple[int, int, int]:
    correct_cells += (pred == answer).sum().item()
    total_cells += answer.numel()
    if torch.equal(pred, answer):
        correct_puzzles += 1
    return correct_cells, total_cells, correct_puzzles


def _accumulate_rollout_stats(
    result: RolloutResult,
    answer: torch.Tensor,
    *,
    total_loss: float,
    total_steps: int,
    max_iter_count: int,
    total_cycle_length: float,
    cycle_count: int,
    correct_cells: int,
    total_cells: int,
    correct_puzzles: int,
    n: int,
) -> tuple[float, int, int, float, int, int, int, int, int]:
    batch_size = answer.size(0) if answer.dim() == 3 else 1
    total_loss += result.loss.item() * batch_size
    total_steps += result.steps * batch_size
    max_iter_count += int(result.hit_max_iter) * batch_size
    if result.cycle_length is not None:
        total_cycle_length += result.cycle_length * batch_size
        cycle_count += batch_size
    n += batch_size
    if result.pred is not None:
        preds = result.pred.unsqueeze(0) if result.pred.dim() == 2 else result.pred
        answers = answer.unsqueeze(0) if answer.dim() == 2 else answer
        for i in range(preds.size(0)):
            correct_cells, total_cells, correct_puzzles = _update_accuracy(
                preds[i],
                answers[i],
                correct_cells=correct_cells,
                total_cells=total_cells,
                correct_puzzles=correct_puzzles,
            )
    return (
        total_loss,
        total_steps,
        max_iter_count,
        total_cycle_length,
        cycle_count,
        correct_cells,
        total_cells,
        correct_puzzles,
        n,
    )


def _stats_from_accumulators(
    *,
    total_loss: float,
    total_steps: int,
    max_iter_count: int,
    total_cycle_length: float,
    cycle_count: int,
    correct_cells: int,
    total_cells: int,
    correct_puzzles: int,
    n: int,
) -> EpochStats:
    if n == 0:
        return EpochStats(loss=0.0, avg_rollout_steps=0.0, max_iter_pct=0.0)
    return EpochStats(
        loss=total_loss / n,
        avg_rollout_steps=total_steps / n,
        max_iter_pct=max_iter_count / n,
        avg_cycle_length=total_cycle_length / cycle_count if cycle_count else 0.0,
        cell_acc=correct_cells / total_cells if total_cells else 0.0,
        puzzle_acc=correct_puzzles / n,
    )


def train_epoch(
    model: NextStateModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    max_rollout_iter: int,
    rollout_config: RolloutConfig,
    batch_size: int,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_steps = 0
    max_iter_count = 0
    total_cycle_length = 0.0
    cycle_count = 0
    correct_cells = 0
    total_cells = 0
    correct_puzzles = 0
    n = 0
    progress = tqdm(
        loader,
        desc=_epoch_desc(epoch, epochs, "train"),
        leave=False,
        unit="batch" if batch_size > 1 else "puzzle",
    )
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = rollout_train_batch(
            model,
            batch["clues"],
            batch["clues_onehot"],
            batch["answer"],
            loss_fn,
            max_rollout_iter=max_rollout_iter,
            config=rollout_config,
            fixed_steps=batch["clues"].size(0) > 1,
        )
        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()
        (
            total_loss,
            total_steps,
            max_iter_count,
            total_cycle_length,
            cycle_count,
            correct_cells,
            total_cells,
            correct_puzzles,
            n,
        ) = _accumulate_rollout_stats(
            result,
            batch["answer"],
            total_loss=total_loss,
            total_steps=total_steps,
            max_iter_count=max_iter_count,
            total_cycle_length=total_cycle_length,
            cycle_count=cycle_count,
            correct_cells=correct_cells,
            total_cells=total_cells,
            correct_puzzles=correct_puzzles,
            n=n,
        )
        progress.set_postfix(
            loss=f"{total_loss / n:.4f}",
            steps=f"{total_steps / n:.1f}",
            cell_acc=f"{correct_cells / total_cells:.4f}",
            cycle=f"{total_cycle_length / cycle_count if cycle_count else 0.0:.1f}",
            refresh=False,
        )
    progress.close()
    return _stats_from_accumulators(
        total_loss=total_loss,
        total_steps=total_steps,
        max_iter_count=max_iter_count,
        total_cycle_length=total_cycle_length,
        cycle_count=cycle_count,
        correct_cells=correct_cells,
        total_cells=total_cells,
        correct_puzzles=correct_puzzles,
        n=n,
    )


@torch.no_grad()
def measure_split(
    model: NextStateModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    phase: str,
    max_rollout_iter: int,
    rollout_config: RolloutConfig,
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    max_iter_count = 0
    total_cycle_length = 0.0
    cycle_count = 0
    correct_cells = 0
    total_cells = 0
    correct_puzzles = 0
    n = 0
    progress = tqdm(
        loader,
        desc=_epoch_desc(epoch, epochs, phase),
        leave=False,
        unit="puzzle",
    )
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = rollout_train_batch(
            model,
            batch["clues"],
            batch["clues_onehot"],
            batch["answer"],
            loss_fn,
            max_rollout_iter=max_rollout_iter,
            config=rollout_config,
        )
        (
            total_loss,
            total_steps,
            max_iter_count,
            total_cycle_length,
            cycle_count,
            correct_cells,
            total_cells,
            correct_puzzles,
            n,
        ) = _accumulate_rollout_stats(
            result,
            batch["answer"],
            total_loss=total_loss,
            total_steps=total_steps,
            max_iter_count=max_iter_count,
            total_cycle_length=total_cycle_length,
            cycle_count=cycle_count,
            correct_cells=correct_cells,
            total_cells=total_cells,
            correct_puzzles=correct_puzzles,
            n=n,
        )
        progress.set_postfix(
            loss=f"{total_loss / n:.4f}",
            cell_acc=f"{correct_cells / total_cells:.4f}",
            refresh=False,
        )
    progress.close()
    return _stats_from_accumulators(
        total_loss=total_loss,
        total_steps=total_steps,
        max_iter_count=max_iter_count,
        total_cycle_length=total_cycle_length,
        cycle_count=cycle_count,
        correct_cells=correct_cells,
        total_cells=total_cells,
        correct_puzzles=correct_puzzles,
        n=n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rollout sudoku model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization on weights only (not bias)")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[512],
        metavar="N",
        help="Hidden layer widths (e.g. 512 for one layer, 512 512 for two)",
    )
    parser.add_argument(
        "--train-max-rollout-iter",
        type=int,
        default=DEFAULT_MAX_ROLLOUT_ITER,
        help="Max rollout iterations per puzzle during training",
    )
    parser.add_argument(
        "--eval-max-rollout-iter",
        type=int,
        default=DEFAULT_MAX_ROLLOUT_ITER,
        help="Max rollout iterations per puzzle during val/test/viz",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size (>1 uses fixed-length parallel rollout)")
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--max-rating", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Max puzzles from train.csv before train/val split")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction from train.csv pool")
    parser.add_argument("--val-samples", type=int, default=None, help="Validation puzzles (overrides val-fraction)")
    parser.add_argument("--test-samples", type=int, default=1000, help="Test puzzles cap from test.csv")
    parser.add_argument("--viz-samples", type=int, default=10, help="Puzzles per split to save for viz")
    parser.add_argument(
        "--rollout-mode",
        choices=["threshold", "categorical"],
        default="threshold",
        help="threshold: BCE + threshold rollout; categorical: CE + argmax rollout; acc/viz use argmax final",
    )
    parser.add_argument(
        "--train-init",
        choices=["clues", "noisy-gt", "zero-gt"],
        default="clues",
        help="clues: clues only, empty elsewhere (same as val/test); noisy-gt: random GT bit noise on non-clue cells; zero-gt: randomly zero non-clue GT cells",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_dir = make_run_dir(args.runs_dir)
    print(f"Run dir: {run_dir}")
    ds_kwargs = {
        "min_rating": args.min_rating,
        "max_rating": args.max_rating,
    }
    train_pool = filter_rows("train", **ds_kwargs)
    train_rows, val_rows = split_train_val(
        train_pool,
        val_samples=args.val_samples,
        val_fraction=args.val_fraction,
        max_samples=args.max_samples,
    )
    test_rows = filter_rows("test", **ds_kwargs, max_samples=args.test_samples)

    train_ds = PuzzleDataset(rows=train_rows)
    val_ds = PuzzleDataset(rows=val_rows)
    test_ds = PuzzleDataset(rows=test_rows)
    args.train_samples = len(train_rows)
    args.val_samples_count = len(val_rows)
    args.test_samples_count = len(test_rows)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_puzzles
    )
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_puzzles)
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_puzzles)

    model = NextStateModel(hidden_sizes=args.hidden_sizes).to(device)
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    loss_fn = (
        nn.BCEWithLogitsLoss()
        if args.rollout_mode == "threshold"
        else nn.CrossEntropyLoss()
    )
    train_init = {
        "clues": "clues",
        "noisy-gt": "noisy_gt",
        "zero-gt": "zero_gt",
    }[args.train_init]
    rollout_config = RolloutConfig(mode=args.rollout_mode, train_init=train_init)
    best_val_loss = float("inf")
    manifest = load_manifest(run_dir)
    viz_rows = {
        "train": train_ds.rows[: args.viz_samples],
        "validation": val_ds.rows[: args.viz_samples],
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch=epoch,
            epochs=args.epochs,
            max_rollout_iter=args.train_max_rollout_iter,
            rollout_config=rollout_config,
            batch_size=args.batch_size,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        val = measure_split(
            model,
            val_loader,
            loss_fn,
            device,
            epoch=epoch,
            epochs=args.epochs,
            phase="val",
            max_rollout_iter=args.eval_max_rollout_iter,
            rollout_config=rollout_config,
        )
        save_epoch_checkpoint(run_dir, epoch, model)
        model.eval()
        for split, rows in viz_rows.items():
            if not rows:
                continue
            puzzle_indices = save_epoch_trajectories(
                model,
                rows,
                split=split,
                epoch=epoch,
                run_dir=run_dir,
                device=device,
                max_rollout_iter=args.eval_max_rollout_iter,
                rollout_config=rollout_config,
            )
            update_manifest_split(manifest, split, epoch, puzzle_indices)
        save_manifest(run_dir, manifest)
        ckpt_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "train": train,
            "val": val,
            "args": args,
        }
        save_checkpoint(run_dir / "last.pt", **ckpt_kwargs)
        test = None
        if val.loss < best_val_loss:
            best_val_loss = val.loss
            test = measure_split(
                model,
                test_loader,
                loss_fn,
                device,
                epoch=epoch,
                epochs=args.epochs,
                phase="test",
                max_rollout_iter=args.eval_max_rollout_iter,
                rollout_config=rollout_config,
            )
            save_checkpoint(run_dir / "best.pt", **ckpt_kwargs, test=test)
        save_epoch_metrics(
            run_dir,
            epoch=epoch,
            train=train,
            val=val,
            epoch_seconds=epoch_seconds,
            args=args,
        )
        if test is not None:
            save_best_test_metrics(run_dir, epoch=epoch, test=test)
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"loss={train.loss:.4f}/{val.loss:.4f} "
            f"cell_acc={train.cell_acc:.4f}/{val.cell_acc:.4f} "
            f"puzzle_acc={train.puzzle_acc:.4f}/{val.puzzle_acc:.4f} "
            f"cycle_len={train.avg_cycle_length:.1f}/{val.avg_cycle_length:.1f} "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
