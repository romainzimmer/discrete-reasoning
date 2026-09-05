from __future__ import annotations

import argparse
import secrets
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from discrete_reasoning.dataset import PuzzleDataset, collate_puzzles
from discrete_reasoning.model import NextStateModel
from discrete_reasoning.rollout import rollout_solve, rollout_train_batch

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def make_run_dir(checkpoint_dir: Path) -> Path:
    run_id = secrets.token_hex(4)
    name = datetime.now().strftime(f"%Y%m%d-%H%M%S-{run_id}")
    run_dir = checkpoint_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_checkpoint(
    path: Path,
    *,
    model: NextStateModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    test_loss: float,
    cell_acc: float,
    puzzle_acc: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
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
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = rollout_train_batch(
            model,
            batch["clues"],
            batch["clues_onehot"],
            batch["answer"],
            loss_fn,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["clues"].size(0)
        n += batch["clues"].size(0)
    return total_loss / n


@torch.no_grad()
def eval_epoch(
    model: NextStateModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct_cells = 0
    total_cells = 0
    correct_puzzles = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total_loss += rollout_train_batch(
            model,
            batch["clues"],
            batch["clues_onehot"],
            batch["answer"],
            loss_fn,
        ).item() * batch["clues"].size(0)

        for i in range(batch["clues"].size(0)):
            pred = rollout_solve(
                model,
                batch["clues_onehot"][i : i + 1],
                batch["clues"][i : i + 1],
            )[0]
            answer = batch["answer"][i]
            correct_cells += (pred == answer).sum().item()
            total_cells += answer.numel()
            if torch.equal(pred, answer):
                correct_puzzles += 1

    n = len(loader.dataset)
    return total_loss / n, correct_cells / total_cells, correct_puzzles / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rollout sudoku model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--max-rating", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Max puzzles per split")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_dir = make_run_dir(args.checkpoint_dir)
    print(f"Run dir: {run_dir}")
    ds_kwargs = {
        "min_rating": args.min_rating,
        "max_rating": args.max_rating,
        "max_samples": args.max_samples,
    }
    train_ds = PuzzleDataset("train", **ds_kwargs)
    test_ds = PuzzleDataset("test", **ds_kwargs)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_puzzles,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=collate_puzzles,
    )

    model = NextStateModel(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_test_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, cell_acc, puzzle_acc = eval_epoch(model, test_loader, loss_fn, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} cell_acc={cell_acc:.4f} puzzle_acc={puzzle_acc:.4f}"
        )

        ckpt_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "args": args,
        }
        save_checkpoint(run_dir / "last.pt", **ckpt_kwargs)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_checkpoint(run_dir / "best.pt", **ckpt_kwargs)


if __name__ == "__main__":
    main()
