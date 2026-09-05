from __future__ import annotations

import argparse
import json
import secrets
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import PuzzleDataset, collate_puzzles
from model import NextStateModel
from rollout import rollout_solve, rollout_train_batch
from viz_data import (
    load_manifest,
    register_run,
    save_epoch_trajectories,
    save_manifest,
    update_manifest_split,
)

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


@dataclass
class EpochStats:
    loss: float
    avg_rollout_steps: float
    max_iter_pct: float


@dataclass
class EvalStats(EpochStats):
    cell_acc: float
    puzzle_acc: float


def make_run_dir(checkpoint_dir: Path) -> Path:
    run_id = secrets.token_hex(4)
    name = datetime.now().strftime(f"%Y%m%d-%H%M%S-{run_id}")
    run_dir = checkpoint_dir / name
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


def save_epoch_metrics(
    run_dir: Path,
    *,
    epoch: int,
    train: EpochStats,
    eval_: EvalStats,
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
            **{f"train_{k}": v for k, v in asdict(train).items()},
            **{f"eval_{k}": v for k, v in asdict(eval_).items()},
        }
    )
    history["epochs"].sort(key=lambda row: row["epoch"])
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
    eval_: EvalStats,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train.loss,
            "train_avg_rollout_steps": train.avg_rollout_steps,
            "train_max_iter_pct": train.max_iter_pct,
            "test_loss": eval_.loss,
            "eval_avg_rollout_steps": eval_.avg_rollout_steps,
            "eval_max_iter_pct": eval_.max_iter_pct,
            "cell_acc": eval_.cell_acc,
            "puzzle_acc": eval_.puzzle_acc,
            "args": vars(args),
        },
        path,
    )


def train_epoch(
    model: NextStateModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_steps = 0
    max_iter_count = 0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = rollout_train_batch(
            model,
            batch["clues"],
            batch["clues_onehot"],
            batch["answer"],
            loss_fn,
        )
        optimizer.zero_grad()
        result.loss.backward()
        optimizer.step()
        total_loss += result.loss.item()
        total_steps += result.steps
        max_iter_count += int(result.hit_max_iter)
        n += 1
    return EpochStats(
        loss=total_loss / n,
        avg_rollout_steps=total_steps / n,
        max_iter_pct=max_iter_count / n,
    )


@torch.no_grad()
def eval_epoch(
    model: NextStateModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    max_iter_count = 0
    correct_cells = 0
    total_cells = 0
    correct_puzzles = 0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = rollout_train_batch(
            model,
            batch["clues"],
            batch["clues_onehot"],
            batch["answer"],
            loss_fn,
        )
        total_loss += result.loss.item()
        total_steps += result.steps
        max_iter_count += int(result.hit_max_iter)
        n += 1

        pred = rollout_solve(
            model,
            batch["clues_onehot"],
            batch["clues"],
        )[0]
        answer = batch["answer"][0]
        correct_cells += (pred == answer).sum().item()
        total_cells += answer.numel()
        if torch.equal(pred, answer):
            correct_puzzles += 1

    return EvalStats(
        loss=total_loss / n,
        avg_rollout_steps=total_steps / n,
        max_iter_pct=max_iter_count / n,
        cell_acc=correct_cells / total_cells,
        puzzle_acc=correct_puzzles / n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rollout sudoku model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--max-rating", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Max puzzles per split")
    parser.add_argument("--eval-split", default="test", choices=["train", "test"])
    parser.add_argument("--viz-samples", type=int, default=5, help="Puzzles per split to save for viz")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_dir = make_run_dir(args.checkpoint_dir)
    register_run(args.checkpoint_dir, run_dir.name, json_safe(vars(args)))
    print(f"Run dir: {run_dir}")
    ds_kwargs = {
        "min_rating": args.min_rating,
        "max_rating": args.max_rating,
        "max_samples": args.max_samples,
    }
    train_ds = PuzzleDataset("train", **ds_kwargs)
    eval_ds = PuzzleDataset(args.eval_split, **ds_kwargs)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_puzzles,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        collate_fn=collate_puzzles,
    )

    model = NextStateModel(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_test_loss = float("inf")
    manifest = load_manifest(run_dir)
    viz_rows = {
        "train": train_ds.rows[: args.viz_samples],
        args.eval_split: eval_ds.rows[: args.viz_samples],
    }

    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, train_loader, optimizer, loss_fn, device)
        eval_ = eval_epoch(model, eval_loader, loss_fn, device)
        print(
            f"epoch {epoch}: train_loss={train.loss:.4f} "
            f"eval_loss={eval_.loss:.4f} cell_acc={eval_.cell_acc:.4f} "
            f"puzzle_acc={eval_.puzzle_acc:.4f} "
            f"rollout={train.avg_rollout_steps:.1f}/{eval_.avg_rollout_steps:.1f} "
            f"max_iter={train.max_iter_pct:.2%}/{eval_.max_iter_pct:.2%} ({args.eval_split})",
            flush=True,
        )

        save_epoch_metrics(run_dir, epoch=epoch, train=train, eval_=eval_, args=args)
        save_epoch_checkpoint(run_dir, epoch, model)
        model.eval()
        for split, rows in viz_rows.items():
            puzzle_indices = save_epoch_trajectories(
                model,
                rows,
                split=split,
                epoch=epoch,
                run_dir=run_dir,
                device=device,
            )
            update_manifest_split(manifest, split, epoch, puzzle_indices)
        save_manifest(run_dir, manifest)
        ckpt_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "train": train,
            "eval_": eval_,
            "args": args,
        }
        save_checkpoint(run_dir / "last.pt", **ckpt_kwargs)
        if eval_.loss < best_test_loss:
            best_test_loss = eval_.loss
            save_checkpoint(run_dir / "best.pt", **ckpt_kwargs)


if __name__ == "__main__":
    main()
