from __future__ import annotations

import argparse
import json
import random
import secrets
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from augment import AugmentConfig
from dataset import PuzzleDataset, collate_puzzles, filter_rows
from model import MixerNextStateModel
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_OUTER_ITERS,
    RolloutConfig,
    RolloutResult,
    rollout_train_batch,
)
from viz_data import (
    load_manifest,
    save_epoch_trajectories,
    save_manifest,
    update_manifest_split,
)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

REQUIRED_RUN_ARGS = (
    "model",
    "dim",
    "num_blocks",
    "train_inner_iters",
    "train_outer_iters",
    "eval_inner_iters",
    "eval_outer_iters",
    "num_workers",
    "min_rating",
    "max_rating",
)


def build_rollout_config(
    *,
    train_init: str,
    inner_iters: int,
    outer_iters: int,
) -> RolloutConfig:
    return RolloutConfig(
        train_init=train_init,
        inner_iters=inner_iters,
        outer_iters=outer_iters,
    )


def _epoch_desc(epoch: int, epochs: int, phase: str) -> str:
    return f"epoch {epoch}/{epochs} {phase}"


@dataclass
class TrainEpochStats:
    loss: float
    cell_acc: float | None = None
    puzzle_acc: float | None = None


@dataclass
class EpochStats:
    loss: float
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


def require_run_args(raw: dict, *, source: str) -> dict:
    args = raw.get("args")
    if args is None:
        raise KeyError(f"{source} has no 'args'")
    missing = [key for key in REQUIRED_RUN_ARGS if key not in args]
    if missing:
        raise KeyError(
            f"{source} args missing required keys: {', '.join(missing)}. "
            "Re-run training with the current train script so defaults are saved explicitly."
        )
    return args


def save_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    history_path = run_dir / "history.json"
    history = {"run_id": run_dir.name, "args": json_safe(vars(args)), "epochs": []}
    history_path.write_text(json.dumps(history, indent=2))


def split_train_val(
    rows: list[dict],
    *,
    val_samples: int,
    max_samples: int | None,
    seed: int | None = None,
) -> tuple[list[dict], list[dict]]:
    pool = rows[:max_samples] if max_samples is not None else rows
    if not pool:
        return [], []
    n_val = min(val_samples, len(pool) - 1) if len(pool) > 1 else 1
    indices = list(range(len(pool)))
    random.Random(seed).shuffle(indices)
    val_rows = [pool[i] for i in indices[:n_val]]
    train_rows = [pool[i] for i in indices[n_val:]]
    return train_rows, val_rows


def save_epoch_metrics(
    run_dir: Path,
    *,
    epoch: int,
    train: TrainEpochStats,
    val: EpochStats,
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
            **{f"train_{k}": v for k, v in asdict(train).items() if v is not None},
            **{f"val_{k}": v for k, v in asdict(val).items()},
        }
    )
    history["epochs"].sort(key=lambda row: row["epoch"])
    history_path.write_text(json.dumps(history, indent=2))


def save_epoch_checkpoint(run_dir: Path, epoch: int, model: MixerNextStateModel) -> None:
    epoch_dir = run_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict()}, epoch_dir / f"{epoch:04d}.pt")


def save_checkpoint(
    path: Path,
    *,
    model: MixerNextStateModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train: TrainEpochStats,
    val: EpochStats,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train.loss,
        "val_loss": val.loss,
        "val_cell_acc": val.cell_acc,
        "val_puzzle_acc": val.puzzle_acc,
        "args": vars(args),
    }
    torch.save(payload, path)


def _accumulate_loss(
    result: RolloutResult,
    answer: torch.Tensor,
    *,
    total_loss: float,
    n: int,
) -> tuple[float, int]:
    batch_size = answer.size(0) if answer.dim() == 3 else 1
    total_loss += result.loss.item() * batch_size
    n += batch_size
    return total_loss, n


def _accumulate_eval_stats(
    result: RolloutResult,
    answer: torch.Tensor,
    clues: torch.Tensor,
    *,
    total_loss: float,
    correct_cells: int,
    total_cells: int,
    correct_puzzles: int,
    n: int,
) -> tuple[float, int, int, int, int]:
    batch_size = answer.size(0) if answer.dim() == 3 else 1
    total_loss += result.loss.item() * batch_size
    n += batch_size
    if result.pred is not None:
        preds = result.pred.unsqueeze(0) if result.pred.dim() == 2 else result.pred
        answers = answer.unsqueeze(0) if answer.dim() == 2 else answer
        clue_rows = clues.unsqueeze(0) if clues.dim() == 2 else clues
        mask = clue_rows == 0
        correct_cells += int((preds[mask] == answers[mask]).sum().item())
        total_cells += int(mask.sum().item())
        correct_puzzles += int((preds == answers).all(dim=(-2, -1)).sum().item())
    return total_loss, correct_cells, total_cells, correct_puzzles, n


def _train_stats_from_accumulators(
    *,
    total_loss: float,
    n: int,
) -> TrainEpochStats:
    if n == 0:
        return TrainEpochStats(loss=0.0)
    return TrainEpochStats(loss=total_loss / n)


def _stats_from_accumulators(
    *,
    total_loss: float,
    correct_cells: int,
    total_cells: int,
    correct_puzzles: int,
    n: int,
) -> EpochStats:
    if n == 0:
        return EpochStats(loss=0.0)
    return EpochStats(
        loss=total_loss / n,
        cell_acc=correct_cells / total_cells if total_cells else 0.0,
        puzzle_acc=correct_puzzles / n,
    )


def train_epoch(
    model: MixerNextStateModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    rollout_config: RolloutConfig,
    batch_size: int,
    use_cuda: bool,
    max_grad_norm: float,
) -> TrainEpochStats:
    model.train()
    total_loss = 0.0
    n = 0
    progress = tqdm(
        loader,
        desc=_epoch_desc(epoch, epochs, "train"),
        leave=False,
        unit="batch" if batch_size > 1 else "puzzle",
    )
    for batch in progress:
        batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        result = rollout_train_batch(
            model,
            batch["clues"],
            batch["answer"],
            config=rollout_config,
            compute_pred=False,
            accumulate_grad=True,
        )
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss, n = _accumulate_loss(
            result,
            batch["answer"],
            total_loss=total_loss,
            n=n,
        )
        progress.set_postfix(
            loss=f"{total_loss / n:.4f}",
            refresh=False,
        )
    progress.close()
    return _train_stats_from_accumulators(
        total_loss=total_loss,
        n=n,
    )


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def measure_split(
    model: MixerNextStateModel,
    loader: DataLoader,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    phase: str,
    rollout_config: RolloutConfig,
    use_cuda: bool,
    seed: int | None = None,
) -> EpochStats:
    if seed is not None:
        _seed_all(seed)
    model.eval()
    total_loss = 0.0
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
        batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
        result = rollout_train_batch(
            model,
            batch["clues"],
            batch["answer"],
            config=rollout_config,
        )
        total_loss, correct_cells, total_cells, correct_puzzles, n = _accumulate_eval_stats(
            result,
            batch["answer"],
            batch["clues"],
            total_loss=total_loss,
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
        correct_cells=correct_cells,
        total_cells=total_cells,
        correct_puzzles=correct_puzzles,
        n=n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rollout sudoku model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="L2 regularization on weights only (not bias)")
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="Clip gradient global norm (0 disables)")
    parser.add_argument("--dim", type=int, default=512, help="Embedding / mixer hidden dimension D")
    parser.add_argument("--num-blocks", type=int, default=2, help="Mixer blocks per inner step (layers in M)")
    parser.add_argument(
        "--train-inner-iters",
        type=int,
        default=DEFAULT_INNER_ITERS,
        help="Inner steps per outer loop during training (grad only on the last outer loop)",
    )
    parser.add_argument(
        "--train-outer-iters",
        type=int,
        default=DEFAULT_OUTER_ITERS,
        help="Argmax commits per puzzle during training",
    )
    parser.add_argument(
        "--eval-inner-iters",
        type=int,
        default=DEFAULT_INNER_ITERS,
        help="Inner steps per outer loop during val/viz/test",
    )
    parser.add_argument(
        "--eval-outer-iters",
        type=int,
        default=DEFAULT_OUTER_ITERS,
        help="Argmax commits per puzzle during val/viz/test",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 recommended on Jetson)",
    )
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--max-rating", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Max puzzles from train.csv before train/val split")
    parser.add_argument("--val-samples", type=int, default=100, help="Validation puzzles from train.csv pool")
    parser.add_argument("--viz-samples", type=int, default=10, help="Puzzles per split to save for viz")
    parser.add_argument(
        "--viz-batch-size",
        type=int,
        default=None,
        help="Batch size for viz trajectory rollouts (default: training batch size)",
    )
    parser.add_argument(
        "--train-init",
        choices=["clues", "noisy-gt", "zero-gt", "curriculum"],
        default="noisy-gt",
        help="clues: clues only; noisy-gt: flip non-clue cells to random 0-9; zero-gt: randomly zero non-clue GT cells; curriculum: reveal GT as extra clues with prob (1-p), p~U[0,1], random 0-9 elsewhere",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for augment RNG and training")
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentations")
    parser.add_argument(
        "--compute-train-acc",
        action="store_true",
        help="After each epoch, run eval-style rollouts on the train split for train cell/puzzle accuracy",
    )
    parser.add_argument("--aug-digit-proba", type=float, default=0.5)
    parser.add_argument("--aug-rot-proba", type=float, default=0.5)
    parser.add_argument("--aug-band-proba", type=float, default=0.3)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

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
        max_samples=args.max_samples,
        seed=args.seed,
    )
    aug_config = AugmentConfig(
        p_digit=args.aug_digit_proba,
        p_rot=args.aug_rot_proba,
        p_band=args.aug_band_proba,
    )
    train_ds = PuzzleDataset(
        rows=train_rows,
        augment=not args.no_augment,
        aug_config=aug_config,
        aug_seed=args.seed,
    )
    val_ds = PuzzleDataset(rows=val_rows)
    args.train_samples = len(train_rows)
    args.val_samples_count = len(val_rows)
    save_run_config(run_dir, args)

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "collate_fn": collate_puzzles,
        "pin_memory": use_cuda,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_kwargs)
    train_acc_loader: DataLoader | None = None
    if args.compute_train_acc:
        train_acc_ds = PuzzleDataset(rows=train_rows, augment=False)
        train_acc_loader = DataLoader(
            train_acc_ds,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    args.model = "mixer-looped"
    model = MixerNextStateModel(dim=args.dim, num_blocks=args.num_blocks).to(device)
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
    train_init = {
        "clues": "clues",
        "noisy-gt": "noisy_gt",
        "zero-gt": "zero_gt",
        "curriculum": "curriculum",
    }[args.train_init]
    rollout_config = build_rollout_config(
        train_init=train_init,
        inner_iters=args.train_inner_iters,
        outer_iters=args.train_outer_iters,
    )
    eval_rollout_config = build_rollout_config(
        train_init="clues",
        inner_iters=args.eval_inner_iters,
        outer_iters=args.eval_outer_iters,
    )
    best_val_cell_acc = -1.0
    manifest = load_manifest(run_dir)
    viz_rows = {
        "train": train_ds.rows[: args.viz_samples],
        "validation": val_ds.rows[: args.viz_samples],
    }

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        train = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            epochs=args.epochs,
            rollout_config=rollout_config,
            batch_size=args.batch_size,
            use_cuda=use_cuda,
            max_grad_norm=args.max_grad_norm,
        )
        if args.compute_train_acc:
            assert train_acc_loader is not None
            train_acc = measure_split(
                model,
                train_acc_loader,
                device,
                epoch=epoch,
                epochs=args.epochs,
                phase="train acc",
                rollout_config=eval_rollout_config,
                use_cuda=use_cuda,
                seed=args.seed,
            )
            train = TrainEpochStats(
                loss=train.loss,
                cell_acc=train_acc.cell_acc,
                puzzle_acc=train_acc.puzzle_acc,
            )
        val = measure_split(
            model,
            val_loader,
            device,
            epoch=epoch,
            epochs=args.epochs,
            phase="val",
            rollout_config=eval_rollout_config,
            use_cuda=use_cuda,
            seed=args.seed,
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
                rollout_config=eval_rollout_config,
                batch_size=args.viz_batch_size or args.batch_size,
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
        if val.cell_acc > best_val_cell_acc:
            best_val_cell_acc = val.cell_acc
            save_checkpoint(run_dir / "best.pt", **ckpt_kwargs)
        save_epoch_metrics(
            run_dir,
            epoch=epoch,
            train=train,
            val=val,
            args=args,
        )
        train_msg = f"train_loss={train.loss:.4f}"
        if train.cell_acc is not None and train.puzzle_acc is not None:
            train_msg += f" train_cell_acc={train.cell_acc:.4f} train_puzzle_acc={train.puzzle_acc:.4f}"
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"{train_msg} val_loss={val.loss:.4f} "
            f"val_cell_acc={val.cell_acc:.4f} val_puzzle_acc={val.puzzle_acc:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
