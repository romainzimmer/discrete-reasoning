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
from dataset import PuzzleDataset, PuzzleTensorCache, collate_puzzles, filter_rows
from model import MixerNextStateModel
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_MAX_OUTER_ITERS,
    BatchSlotState,
    RolloutConfig,
    RolloutResult,
    refill_done_slots,
    rollout_eval_batch,
    rollout_train_step,
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
    "inner_iters",
    "train_max_outer_iters",
    "eval_max_outer_iters",
    "train_batch_size",
    "batches_per_epoch",
    "halt_loss_weight",
    "num_workers",
    "min_rating",
    "max_rating",
)


def build_rollout_config(
    *,
    inner_iters: int,
    max_outer_iters: int,
    halt_threshold: float = 0.5,
    rollout_mask_prob: float = 0.0,
    rollout_noise_prob: float = 0.0,
) -> RolloutConfig:
    return RolloutConfig(
        inner_iters=inner_iters,
        max_outer_iters=max_outer_iters,
        halt_threshold=halt_threshold,
        rollout_mask_prob=rollout_mask_prob,
        rollout_noise_prob=rollout_noise_prob,
    )


def _epoch_desc(epoch: int, epochs: int, phase: str) -> str:
    return f"epoch {epoch}/{epochs} {phase}"


@dataclass
class TrainEpochStats:
    loss: float
    cell_loss: float = 0.0
    halt_loss: float = 0.0
    cell_acc: float = 0.0
    puzzle_acc: float = 0.0
    halt_acc: float = 0.0
    avg_outer_iters: float = 0.0
    halt_rate: float = 0.0
    refills_per_step: float = 0.0
    completions_per_epoch: int = 0


@dataclass
class EpochStats:
    loss: float
    cell_loss: float = 0.0
    halt_loss: float = 0.0
    cell_acc: float = 0.0
    puzzle_acc: float = 0.0
    halt_acc: float = 0.0
    avg_outer_iters: float = 0.0
    halt_rate: float = 0.0


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
            **{f"train_{k}": v for k, v in asdict(train).items()},
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


def _accumulate_step_metrics(
    result: RolloutResult,
    state: BatchSlotState,
    *,
    total_loss: float,
    total_cell_loss: float,
    total_halt_loss: float,
    halt_correct: int,
    halt_total: int,
    correct_cells: int,
    total_cells: int,
    correct_puzzles_done: int,
    puzzles_done: int,
    outer_iters_done: float,
    halted_done: int,
    refills: int,
    n_steps: int,
) -> tuple[float, float, float, int, int, int, int, int, int, int, float, int, int, int]:
    b = state.digit_id.size(0)
    total_loss += result.loss.item()
    if result.cell_loss is not None:
        total_cell_loss += result.cell_loss.item()
    if result.halt_loss is not None:
        total_halt_loss += result.halt_loss.item()
    n_steps += 1
    refills += int(result.done.sum().item())

    assert result.pred is not None
    assert result.halt_target is not None
    assert result.halted is not None
    assert result.done is not None

    predict_halt = result.halted
    halt_correct += int((predict_halt == (result.halt_target > 0.5)).sum().item())
    halt_total += b

    mask = state.clues == 0
    correct_cells += int((result.pred[mask] == state.answer[mask]).sum().item())
    total_cells += int(mask.sum().item())

    if result.done.any():
        done = result.done
        correct_puzzles_done += int((result.pred[done] == state.answer[done]).all(dim=(-2, -1)).sum().item())
        puzzles_done += int(done.sum().item())
        outer_iters_done += float(state.outer_count[done].sum().item())
        halted_done += int((result.halted[done]).sum().item())

    return (
        total_loss,
        total_cell_loss,
        total_halt_loss,
        halt_correct,
        halt_total,
        correct_cells,
        total_cells,
        correct_puzzles_done,
        puzzles_done,
        outer_iters_done,
        halted_done,
        refills,
        n_steps,
    )


def _train_stats_from_accumulators(
    *,
    total_loss: float,
    total_cell_loss: float,
    total_halt_loss: float,
    n_steps: int,
    halt_correct: int,
    halt_total: int,
    correct_cells: int,
    total_cells: int,
    correct_puzzles_done: int,
    puzzles_done: int,
    outer_iters_done: float,
    halted_done: int,
    refills: int,
) -> TrainEpochStats:
    if n_steps == 0:
        return TrainEpochStats(loss=0.0)
    return TrainEpochStats(
        loss=total_loss / n_steps,
        cell_loss=total_cell_loss / n_steps,
        halt_loss=total_halt_loss / n_steps,
        cell_acc=correct_cells / total_cells if total_cells else 0.0,
        puzzle_acc=correct_puzzles_done / puzzles_done if puzzles_done else 0.0,
        halt_acc=halt_correct / halt_total if halt_total else 0.0,
        avg_outer_iters=outer_iters_done / puzzles_done if puzzles_done else 0.0,
        halt_rate=halted_done / puzzles_done if puzzles_done else 0.0,
        refills_per_step=refills / n_steps,
        completions_per_epoch=puzzles_done,
    )


def _accumulate_eval_stats(
    result,
    answer: torch.Tensor,
    clues: torch.Tensor,
    *,
    total_loss: float,
    total_cell_loss: float,
    total_halt_loss: float,
    halt_correct: int,
    halt_total: int,
    correct_cells: int,
    total_cells: int,
    correct_puzzles: int,
    outer_iters_sum: float,
    halted_count: int,
    n: int,
) -> tuple[float, float, float, int, int, int, int, int, int, float, int, int]:
    batch_size = answer.size(0) if answer.dim() == 3 else 1
    total_loss += result.loss.item() * batch_size
    total_cell_loss += result.cell_loss.item() * batch_size
    total_halt_loss += result.halt_loss.item() * batch_size
    n += batch_size

    preds = result.pred.unsqueeze(0) if result.pred.dim() == 2 else result.pred
    answers = answer.unsqueeze(0) if answer.dim() == 2 else answer
    clue_rows = clues.unsqueeze(0) if clues.dim() == 2 else clues
    mask = clue_rows == 0
    correct_cells += int((preds[mask] == answers[mask]).sum().item())
    total_cells += int(mask.sum().item())
    correct_puzzles += int((preds == answers).all(dim=(-2, -1)).sum().item())

    halt_correct += result.halt_correct_rounds
    halt_total += result.halt_total_rounds

    outer_steps = result.outer_steps.unsqueeze(0) if result.outer_steps.dim() == 0 else result.outer_steps
    outer_iters_sum += float(outer_steps.sum().item())
    halted = result.halted.unsqueeze(0) if result.halted.dim() == 0 else result.halted
    halted_count += int(halted.sum().item())

    return (
        total_loss,
        total_cell_loss,
        total_halt_loss,
        halt_correct,
        halt_total,
        correct_cells,
        total_cells,
        correct_puzzles,
        outer_iters_sum,
        halted_count,
        n,
    )


def _stats_from_accumulators(
    *,
    total_loss: float,
    total_cell_loss: float,
    total_halt_loss: float,
    halt_correct: int,
    halt_total: int,
    correct_cells: int,
    total_cells: int,
    correct_puzzles: int,
    outer_iters_sum: float,
    halted_count: int,
    n: int,
) -> EpochStats:
    if n == 0:
        return EpochStats(loss=0.0)
    return EpochStats(
        loss=total_loss / n,
        cell_loss=total_cell_loss / n,
        halt_loss=total_halt_loss / n,
        cell_acc=correct_cells / total_cells if total_cells else 0.0,
        puzzle_acc=correct_puzzles / n,
        halt_acc=halt_correct / halt_total if halt_total else 0.0,
        avg_outer_iters=outer_iters_sum / n,
        halt_rate=halted_count / n,
    )


def train_epoch(
    model: MixerNextStateModel,
    state: BatchSlotState,
    cache: PuzzleTensorCache,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    epochs: int,
    rollout_config: RolloutConfig,
    batches_per_epoch: int,
    halt_loss_weight: float,
    refill_generator: torch.Generator,
) -> TrainEpochStats:
    model.train()
    total_loss = 0.0
    total_cell_loss = 0.0
    total_halt_loss = 0.0
    halt_correct = 0
    halt_total = 0
    correct_cells = 0
    total_cells = 0
    correct_puzzles_done = 0
    puzzles_done = 0
    outer_iters_done = 0.0
    halted_done = 0
    refills = 0
    n_steps = 0

    progress = tqdm(
        range(batches_per_epoch),
        desc=_epoch_desc(epoch, epochs, "train"),
        leave=False,
        unit="step",
    )
    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        result = rollout_train_step(
            model,
            state,
            rollout_config,
            halt_loss_weight=halt_loss_weight,
        )
        optimizer.step()
        (
            total_loss,
            total_cell_loss,
            total_halt_loss,
            halt_correct,
            halt_total,
            correct_cells,
            total_cells,
            correct_puzzles_done,
            puzzles_done,
            outer_iters_done,
            halted_done,
            refills,
            n_steps,
        ) = _accumulate_step_metrics(
            result,
            state,
            total_loss=total_loss,
            total_cell_loss=total_cell_loss,
            total_halt_loss=total_halt_loss,
            halt_correct=halt_correct,
            halt_total=halt_total,
            correct_cells=correct_cells,
            total_cells=total_cells,
            correct_puzzles_done=correct_puzzles_done,
            puzzles_done=puzzles_done,
            outer_iters_done=outer_iters_done,
            halted_done=halted_done,
            refills=refills,
            n_steps=n_steps,
        )
        assert result.done is not None
        refill_done_slots(state, result.done, cache, generator=refill_generator)
        postfix = {
            "loss": f"{total_loss / n_steps:.4f}",
            "cell_loss": f"{total_cell_loss / n_steps:.4f}",
            "halt_loss": f"{total_halt_loss / n_steps:.4f}",
        }
        if halt_total:
            postfix["halt_acc"] = f"{halt_correct / halt_total:.4f}"
        if total_cells:
            postfix["cell_acc"] = f"{correct_cells / total_cells:.4f}"
        progress.set_postfix(**postfix, refresh=False)
    progress.close()
    return _train_stats_from_accumulators(
        total_loss=total_loss,
        total_cell_loss=total_cell_loss,
        total_halt_loss=total_halt_loss,
        n_steps=n_steps,
        halt_correct=halt_correct,
        halt_total=halt_total,
        correct_cells=correct_cells,
        total_cells=total_cells,
        correct_puzzles_done=correct_puzzles_done,
        puzzles_done=puzzles_done,
        outer_iters_done=outer_iters_done,
        halted_done=halted_done,
        refills=refills,
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
    halt_loss_weight: float,
    use_cuda: bool,
    seed: int | None = None,
) -> EpochStats:
    if seed is not None:
        _seed_all(seed)
    model.eval()
    total_loss = 0.0
    total_cell_loss = 0.0
    total_halt_loss = 0.0
    halt_correct = 0
    halt_total = 0
    correct_cells = 0
    total_cells = 0
    correct_puzzles = 0
    outer_iters_sum = 0.0
    halted_count = 0
    n = 0
    progress = tqdm(
        loader,
        desc=_epoch_desc(epoch, epochs, phase),
        leave=False,
        unit="puzzle",
    )
    for batch in progress:
        batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
        result = rollout_eval_batch(
            model,
            batch["clues"],
            batch["answer"],
            config=rollout_config,
            halt_loss_weight=halt_loss_weight,
        )
        (
            total_loss,
            total_cell_loss,
            total_halt_loss,
            halt_correct,
            halt_total,
            correct_cells,
            total_cells,
            correct_puzzles,
            outer_iters_sum,
            halted_count,
            n,
        ) = _accumulate_eval_stats(
            result,
            batch["answer"],
            batch["clues"],
            total_loss=total_loss,
            total_cell_loss=total_cell_loss,
            total_halt_loss=total_halt_loss,
            halt_correct=halt_correct,
            halt_total=halt_total,
            correct_cells=correct_cells,
            total_cells=total_cells,
            correct_puzzles=correct_puzzles,
            outer_iters_sum=outer_iters_sum,
            halted_count=halted_count,
            n=n,
        )
        progress.set_postfix(
            loss=f"{total_loss / n:.4f}",
            cell_loss=f"{total_cell_loss / n:.4f}",
            halt_loss=f"{total_halt_loss / n:.4f}",
            cell_acc=f"{correct_cells / total_cells:.4f}",
            refresh=False,
        )
    progress.close()
    return _stats_from_accumulators(
        total_loss=total_loss,
        total_cell_loss=total_cell_loss,
        total_halt_loss=total_halt_loss,
        halt_correct=halt_correct,
        halt_total=halt_total,
        correct_cells=correct_cells,
        total_cells=total_cells,
        correct_puzzles=correct_puzzles,
        outer_iters_sum=outer_iters_sum,
        halted_count=halted_count,
        n=n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rollout sudoku model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="L2 regularization on weights only (not bias)")
    parser.add_argument("--dim", type=int, default=512, help="Embedding / mixer hidden dimension D")
    parser.add_argument("--num-blocks", type=int, default=2, help="Mixer blocks per inner step (layers in M)")
    parser.add_argument(
        "--inner-iters",
        type=int,
        default=DEFAULT_INNER_ITERS,
        help="Looped inner steps per outer round (train, val, test, viz)",
    )
    parser.add_argument(
        "--train-max-outer-iters",
        type=int,
        default=DEFAULT_MAX_OUTER_ITERS,
        help="Max outer commits per puzzle before refill (training)",
    )
    parser.add_argument(
        "--eval-max-outer-iters",
        type=int,
        default=DEFAULT_MAX_OUTER_ITERS,
        help="Max outer commits per puzzle during val/viz/test",
    )
    parser.add_argument("--train-batch-size", type=int, default=8, help="Parallel GPU slots (B)")
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=100,
        help="Optimizer steps (= outer rounds) per epoch",
    )
    parser.add_argument(
        "--halt-loss-weight",
        type=float,
        default=0.1,
        help="Weight for halt BCE loss",
    )
    parser.add_argument(
        "--rollout-mask-prob",
        type=float,
        default=0.0,
        help="Per-cell prob of masking to empty at each outer step inner-loop input (clues untouched)",
    )
    parser.add_argument(
        "--rollout-noise-prob",
        type=float,
        default=0.0,
        help="Per-cell prob of random digit noise at each outer step inner-loop input (clues untouched)",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Validation batch size (default: training batch size)",
    )
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
        help="Batch size for viz trajectory rollouts (default: train batch size)",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for augment RNG and training")
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentations")
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
    val_batch_size = args.val_batch_size or args.train_batch_size
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, **loader_kwargs)

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
    rollout_config = build_rollout_config(
        inner_iters=args.inner_iters,
        max_outer_iters=args.train_max_outer_iters,
        rollout_mask_prob=args.rollout_mask_prob,
        rollout_noise_prob=args.rollout_noise_prob,
    )
    eval_rollout_config = build_rollout_config(
        inner_iters=args.inner_iters,
        max_outer_iters=args.eval_max_outer_iters,
        rollout_mask_prob=args.rollout_mask_prob,
        rollout_noise_prob=args.rollout_noise_prob,
    )
    refill_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best_val_cell_acc = -1.0
    manifest = load_manifest(run_dir)
    viz_rows = {
        "train": train_ds.rows[: args.viz_samples],
        "validation": val_ds.rows[: args.viz_samples],
    }

    state: BatchSlotState | None = None

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        cache = PuzzleTensorCache.build(train_ds, pin_memory=use_cuda)
        if state is None:
            state = BatchSlotState.seed(
                cache,
                args.train_batch_size,
                device,
                generator=refill_generator,
            )

        train = train_epoch(
            model,
            state,
            cache,
            optimizer,
            epoch=epoch,
            epochs=args.epochs,
            rollout_config=rollout_config,
            batches_per_epoch=args.batches_per_epoch,
            halt_loss_weight=args.halt_loss_weight,
            refill_generator=refill_generator,
        )
        val = measure_split(
            model,
            val_loader,
            device,
            epoch=epoch,
            epochs=args.epochs,
            phase="val",
            rollout_config=eval_rollout_config,
            halt_loss_weight=args.halt_loss_weight,
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
                batch_size=args.viz_batch_size or args.train_batch_size,
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
        train_msg = (
            f"train_loss={train.loss:.4f} train_cell_loss={train.cell_loss:.4f} "
            f"train_halt_loss={train.halt_loss:.4f} train_halt_acc={train.halt_acc:.4f} "
            f"train_cell_acc={train.cell_acc:.4f} train_puzzle_acc={train.puzzle_acc:.4f} "
            f"train_halt_rate={train.halt_rate:.4f} train_refills={train.refills_per_step:.2f}"
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"{train_msg} val_loss={val.loss:.4f} val_cell_loss={val.cell_loss:.4f} "
            f"val_halt_loss={val.halt_loss:.4f} val_cell_acc={val.cell_acc:.4f} "
            f"val_puzzle_acc={val.puzzle_acc:.4f} val_halt_rate={val.halt_rate:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
