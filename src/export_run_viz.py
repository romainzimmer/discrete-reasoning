from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import PuzzleDataset
from model import NextStateModel
from viz_data import (
    json_safe,
    load_manifest,
    register_run,
    save_epoch_trajectories,
    save_manifest,
    update_manifest_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def load_history(run_dir: Path) -> dict:
    history_path = run_dir / "history.json"
    if history_path.exists():
        return json.loads(history_path.read_text())

    history = {"run_id": run_dir.name, "args": {}, "epochs": []}
    for ckpt_name in ("last.pt", "best.pt"):
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        history["args"] = json_safe(ckpt.get("args", {}))
        history["epochs"].append(
            {
                "epoch": ckpt["epoch"],
                "train_loss": ckpt.get("train_loss"),
                "train_avg_rollout_steps": ckpt.get("train_avg_rollout_steps"),
                "train_max_iter_pct": ckpt.get("train_max_iter_pct"),
                "eval_loss": ckpt.get("test_loss"),
                "eval_avg_rollout_steps": ckpt.get("eval_avg_rollout_steps"),
                "eval_max_iter_pct": ckpt.get("eval_max_iter_pct"),
                "eval_cell_acc": ckpt.get("cell_acc"),
                "eval_puzzle_acc": ckpt.get("puzzle_acc"),
            }
        )
    history["epochs"] = sorted(history["epochs"], key=lambda row: row["epoch"])
    return history


def list_epoch_checkpoints(run_dir: Path) -> dict[int, Path]:
    epoch_dir = run_dir / "epochs"
    if not epoch_dir.exists():
        return {}
    return {int(path.stem): path for path in sorted(epoch_dir.glob("*.pt"))}


def resolve_epochs(requested: str, available: list[int]) -> list[int]:
    if not available:
        raise ValueError("No epoch checkpoints found in run dir")
    if requested == "all":
        return available
    epochs = [int(part.strip()) for part in requested.split(",") if part.strip()]
    missing = [epoch for epoch in epochs if epoch not in available]
    if missing:
        raise ValueError(f"Missing epoch checkpoints: {missing}")
    return epochs


def load_model(checkpoint_path: Path, hidden: int, device: torch.device) -> NextStateModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = NextStateModel(hidden=hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill viz data for an existing run")
    parser.add_argument("--run", type=Path, required=True, help="Run dir under checkpoints/")
    parser.add_argument("--epochs", default="all", help="Comma-separated epochs or 'all'")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--n", type=int, default=5, help="Puzzles per split")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = args.run if args.run.is_absolute() else CHECKPOINT_DIR / args.run
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    history = load_history(run_dir)
    if not (run_dir / "history.json").exists():
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    register_run(CHECKPOINT_DIR, history["run_id"], history.get("args", {}))

    epoch_paths = list_epoch_checkpoints(run_dir)
    if not epoch_paths:
        fallback = run_dir / "last.pt"
        if fallback.exists():
            ckpt = torch.load(fallback, map_location="cpu", weights_only=False)
            epoch_paths = {ckpt["epoch"]: fallback}
    epochs = resolve_epochs(args.epochs, sorted(epoch_paths))
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    train_args = history.get("args", {})
    hidden = int(train_args.get("hidden", 512))
    device = torch.device(args.device)
    manifest = load_manifest(run_dir)

    for split in splits:
        ds = PuzzleDataset(
            split,
            min_rating=train_args.get("min_rating"),
            max_rating=train_args.get("max_rating"),
            max_samples=train_args.get("max_samples"),
        )
        rows = ds.rows[: args.n]
        for epoch in epochs:
            model = load_model(epoch_paths[epoch], hidden, device)
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
    print(f"Backfilled viz data in {run_dir}")


if __name__ == "__main__":
    main()
