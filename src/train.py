from __future__ import annotations

import argparse
import json
import random
import secrets
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from amp import AmpConfig, autocast_context, resolve_amp
from augment import AugmentConfig
from dataset import PuzzleDataset, collate_puzzles, filter_rows
from encoding import cell_acc_mask
from model import MixerNextStateModel
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_MAX_OUTER_ITERS,
    DEFAULT_TRANSITION_PROB,
    BatchSlotState,
    RolloutConfig,
    RolloutResult,
    refill_done_slots,
    rollout_eval_batch,
    rollout_train_step,
)
from profiling import ProfileConfig, TrainProfiler
from viz_data import (
    load_manifest,
    save_epoch_trajectories,
    save_manifest,
    update_manifest_split,
)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
ADAPTIVE_CURRICULUM_ACC_EMA_ALPHA = 0.5


def optimizer_param_groups(
    model: MixerNextStateModel,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or name == "ema_alpha_logit":
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def update_curriculum_puzzle_acc(prev: float, epoch_puzzle_acc: float) -> float:
    """EMA of done-only train puzzle acc for adaptive curriculum p_gt upper bound."""
    a = ADAPTIVE_CURRICULUM_ACC_EMA_ALPHA
    return a * epoch_puzzle_acc + (1.0 - a) * prev

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
    transition_prob: float = DEFAULT_TRANSITION_PROB,
    curriculum_training: bool = True,
    pin_gt: bool = True,
    deep_supervision: bool = True,
    adaptive_curriculum: bool = True,
    use_ema: bool = True,
) -> RolloutConfig:
    return RolloutConfig(
        inner_iters=inner_iters,
        max_outer_iters=max_outer_iters,
        halt_threshold=halt_threshold,
        transition_prob=transition_prob,
        curriculum_training=curriculum_training,
        pin_gt=pin_gt,
        deep_supervision=deep_supervision,
        adaptive_curriculum=adaptive_curriculum,
        use_ema=use_ema,
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
class TrainMetricsAccumulator:
    total_loss: torch.Tensor
    total_cell_loss: torch.Tensor
    total_halt_loss: torch.Tensor
    halt_correct: torch.Tensor
    halt_total: torch.Tensor
    correct_cells: torch.Tensor
    total_cells: torch.Tensor
    correct_puzzles_done: torch.Tensor
    puzzles_done: torch.Tensor
    outer_iters_done: torch.Tensor
    halted_done: torch.Tensor
    refills: torch.Tensor
    n_steps: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> TrainMetricsAccumulator:
        zero = torch.zeros((), device=device)
        zero_i = torch.zeros((), device=device, dtype=torch.long)
        return cls(
            total_loss=zero.clone(),
            total_cell_loss=zero.clone(),
            total_halt_loss=zero.clone(),
            halt_correct=zero_i.clone(),
            halt_total=zero_i.clone(),
            correct_cells=zero_i.clone(),
            total_cells=zero_i.clone(),
            correct_puzzles_done=zero_i.clone(),
            puzzles_done=zero_i.clone(),
            outer_iters_done=zero.clone(),
            halted_done=zero_i.clone(),
            refills=zero_i.clone(),
            n_steps=zero_i.clone(),
        )

    def add_step(self, result: RolloutResult, state: BatchSlotState) -> None:
        b = state.digit_id.size(0)
        self.n_steps += 1
        self.total_loss += result.loss.detach()
        if result.cell_loss is not None:
            self.total_cell_loss += result.cell_loss.detach()
        if result.halt_loss is not None:
            self.total_halt_loss += result.halt_loss.detach()
        assert result.pred is not None
        assert result.halt_target is not None
        assert result.halted is not None
        assert result.done is not None

        done = result.done
        self.refills += done.sum()
        self.halt_correct += (result.halted == (result.halt_target > 0.5)).sum()
        self.halt_total += b

        mask = cell_acc_mask(state.clues)
        self.correct_cells += (result.pred[mask] == state.answer[mask]).sum()
        self.total_cells += mask.sum()

        puzzle_ok = (result.pred == state.answer).all(dim=(-2, -1))
        self.correct_puzzles_done += (puzzle_ok & done).sum()
        self.puzzles_done += done.sum()
        self.outer_iters_done += (state.outer_count * done.long()).sum()
        self.halted_done += (result.halted & done).sum()

    def finalize(self) -> TrainEpochStats:
        n = int(self.n_steps.item())
        if n == 0:
            return TrainEpochStats(loss=0.0)
        halt_total = int(self.halt_total.item())
        total_cells = int(self.total_cells.item())
        puzzles_done = int(self.puzzles_done.item())
        halted_done = int(self.halted_done.item())
        return TrainEpochStats(
            loss=self.total_loss.item() / n,
            cell_loss=self.total_cell_loss.item() / n,
            halt_loss=self.total_halt_loss.item() / n,
            cell_acc=self.correct_cells.item() / total_cells if total_cells else 0.0,
            puzzle_acc=self.correct_puzzles_done.item() / puzzles_done if puzzles_done else 0.0,
            halt_acc=self.halt_correct.item() / halt_total if halt_total else 0.0,
            avg_outer_iters=self.outer_iters_done.item() / puzzles_done if puzzles_done else 0.0,
            halt_rate=halted_done / puzzles_done if puzzles_done else 0.0,
            refills_per_step=self.refills.item() / n,
            completions_per_epoch=puzzles_done,
        )


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


@dataclass
class EvalMetricsAccumulator:
    total_loss: torch.Tensor
    total_cell_loss: torch.Tensor
    total_halt_loss: torch.Tensor
    halt_correct: torch.Tensor
    halt_total: torch.Tensor
    correct_cells: torch.Tensor
    total_cells: torch.Tensor
    correct_puzzles: torch.Tensor
    outer_iters_sum: torch.Tensor
    halted_count: torch.Tensor
    n: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> EvalMetricsAccumulator:
        zero = torch.zeros((), device=device)
        zero_i = torch.zeros((), device=device, dtype=torch.long)
        return cls(
            total_loss=zero.clone(),
            total_cell_loss=zero.clone(),
            total_halt_loss=zero.clone(),
            halt_correct=zero_i.clone(),
            halt_total=zero_i.clone(),
            correct_cells=zero_i.clone(),
            total_cells=zero_i.clone(),
            correct_puzzles=zero_i.clone(),
            outer_iters_sum=zero.clone(),
            halted_count=zero_i.clone(),
            n=zero_i.clone(),
        )

    def add_batch(
        self,
        result,
        answer: torch.Tensor,
        clues: torch.Tensor,
    ) -> None:
        batch_size = answer.size(0) if answer.dim() == 3 else 1
        self.n += batch_size
        self.total_loss += result.loss.detach() * batch_size
        self.total_cell_loss += result.cell_loss.detach() * batch_size
        self.total_halt_loss += result.halt_loss.detach() * batch_size

        preds = result.pred.unsqueeze(0) if result.pred.dim() == 2 else result.pred
        answers = answer.unsqueeze(0) if answer.dim() == 2 else answer
        clue_rows = clues.unsqueeze(0) if clues.dim() == 2 else clues
        mask = cell_acc_mask(clue_rows)
        self.correct_cells += (preds[mask] == answers[mask]).sum()
        self.total_cells += mask.sum()
        self.correct_puzzles += (preds == answers).all(dim=(-2, -1)).sum()

        self.halt_correct += result.halt_correct_rounds
        self.halt_total += result.halt_total_rounds
        outer_steps = result.outer_steps.unsqueeze(0) if result.outer_steps.dim() == 0 else result.outer_steps
        self.outer_iters_sum += outer_steps.sum()
        halted = result.halted.unsqueeze(0) if result.halted.dim() == 0 else result.halted
        self.halted_count += halted.sum()

    def finalize(self) -> EpochStats:
        n = int(self.n.item())
        if n == 0:
            return EpochStats(loss=0.0)
        halt_total = int(self.halt_total.item())
        total_cells = int(self.total_cells.item())
        return EpochStats(
            loss=self.total_loss.item() / n,
            cell_loss=self.total_cell_loss.item() / n,
            halt_loss=self.total_halt_loss.item() / n,
            cell_acc=self.correct_cells.item() / total_cells if total_cells else 0.0,
            puzzle_acc=self.correct_puzzles.item() / n,
            halt_acc=self.halt_correct.item() / halt_total if halt_total else 0.0,
            avg_outer_iters=self.outer_iters_sum.item() / n,
            halt_rate=self.halted_count.item() / n,
        )


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
    ema_alpha: float,
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
            "ema_alpha": ema_alpha,
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


def train_epoch(
    model: MixerNextStateModel,
    state: BatchSlotState,
    train_ds: PuzzleDataset,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    epochs: int,
    rollout_config: RolloutConfig,
    batches_per_epoch: int,
    halt_loss_weight: float,
    refill_generator: torch.Generator,
    profiler: TrainProfiler | None = None,
    amp: AmpConfig | None = None,
) -> TrainEpochStats:
    model.train()
    amp = amp or AmpConfig(enabled=False, dtype=None, scaler=None)
    acc = TrainMetricsAccumulator.empty(state.digit_id.device)

    progress = tqdm(
        range(batches_per_epoch),
        desc=_epoch_desc(epoch, epochs, "train"),
        leave=False,
        unit="step",
    )
    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("rollout_train_step"):
            with autocast_context(state.digit_id.device, amp):
                result = rollout_train_step(
                    model,
                    state,
                    rollout_config,
                    halt_loss_weight=halt_loss_weight,
                    backward=False,
                )
        with torch.profiler.record_function("optimizer_step"):
            if amp.scaler is not None:
                amp.scaler.scale(result.loss).backward()
                amp.scaler.step(optimizer)
                amp.scaler.update()
            else:
                result.loss.backward()
                optimizer.step()
        with torch.profiler.record_function("metrics_and_refill"):
            acc.add_step(result, state)
            assert result.done is not None
            refill_done_slots(
                state,
                result.done,
                train_ds,
                generator=refill_generator,
                dim=model.dim,
                curriculum_training=rollout_config.curriculum_training,
                pin_gt=rollout_config.pin_gt,
                adaptive_curriculum=rollout_config.adaptive_curriculum,
                curriculum_puzzle_acc=rollout_config.curriculum_puzzle_acc,
            )
        if profiler is not None:
            profiler.step()
    if profiler is not None:
        profiler.finish()
    stats = acc.finalize()
    progress.set_postfix(
        loss=f"{stats.loss:.4f}",
        cell_loss=f"{stats.cell_loss:.4f}",
        halt_loss=f"{stats.halt_loss:.4f}",
        refresh=False,
    )
    progress.close()
    return stats


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
    amp: AmpConfig | None = None,
) -> EpochStats:
    if seed is not None:
        _seed_all(seed)
    model.eval()
    amp = amp or AmpConfig(enabled=False, dtype=None, scaler=None)
    acc = EvalMetricsAccumulator.empty(device)
    progress = tqdm(
        loader,
        desc=_epoch_desc(epoch, epochs, phase),
        leave=False,
        unit="puzzle",
    )
    for batch in progress:
        batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
        with autocast_context(device, amp):
            result = rollout_eval_batch(
                model,
                batch["clues"],
                batch["answer"],
                config=rollout_config,
                halt_loss_weight=halt_loss_weight,
            )
        acc.add_batch(result, batch["answer"], batch["clues"])
    progress.close()
    return acc.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rollout sudoku model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="L2 regularization on weights only (not bias)")
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
        default=1.0,
        help="Weight for halt BCE loss",
    )
    parser.add_argument(
        "--transition-prob",
        type=float,
        default=DEFAULT_TRANSITION_PROB,
        help="Per-cell prob of committing decoded digit each outer step (clues/GT pins always commit; 1 = full grid update)",
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
    parser.add_argument(
        "--no-curriculum-training",
        action="store_true",
        help="Disable curriculum puzzle init (partial GT reveal) during training seed/refill",
    )
    parser.add_argument(
        "--no-pin-gt",
        action="store_true",
        help="Disable GT pinning during rollout (curriculum init still reveals GT; commits use model digits)",
    )
    parser.add_argument(
        "--no-deep-supervision",
        action="store_true",
        help="Use final inner loop step only for cell and halt loss (default: average all steps)",
    )
    parser.add_argument(
        "--no-adaptive-curriculum",
        action="store_true",
        help="Disable adaptive curriculum (use fixed U[0, 1] for p_gt instead of U[0, 1 - puzzle_acc])",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable outer-loop EMA memory (use current cell_embed only; equivalent to alpha=1)",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentations")
    parser.add_argument("--aug-digit-proba", type=float, default=0.5)
    parser.add_argument("--aug-rot-proba", type=float, default=0.5)
    parser.add_argument("--aug-band-proba", type=float, default=0.3)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Profile this many training steps (0 = off); writes runs/<id>/profile/trace.json",
    )
    parser.add_argument("--profile-wait", type=int, default=1, help="Profiler schedule: steps before warmup")
    parser.add_argument("--profile-warmup", type=int, default=2, help="Profiler schedule: warmup steps")
    parser.add_argument("--profile-epoch", type=int, default=1, help="Epoch to run the profiler in")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (bf16/fp16 on CUDA)",
    )
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
    use_cuda = device.type == "cuda"
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
        pin_memory=use_cuda,
    )
    # Val uses DataLoader workers; do not pin base tensors (fork + pinned memory segfaults on Jetson).
    val_ds = PuzzleDataset(rows=val_rows)
    args.train_samples = len(train_rows)
    args.val_samples_count = len(val_rows)
    loader_kwargs = {
        "collate_fn": collate_puzzles,
        "pin_memory": use_cuda,
        "num_workers": args.num_workers,
    }
    val_batch_size = args.val_batch_size or args.train_batch_size
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, **loader_kwargs)

    args.model = "looped-mixer"
    args.amp = not args.no_amp
    save_run_config(run_dir, args)
    amp = resolve_amp(device, enabled=args.amp)
    model = MixerNextStateModel(dim=args.dim, num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model, weight_decay=args.weight_decay),
        lr=args.lr,
    )
    curriculum_training = not args.no_curriculum_training
    pin_gt = not args.no_pin_gt
    adaptive_curriculum = not args.no_adaptive_curriculum
    deep_supervision = not args.no_deep_supervision
    use_ema = not args.no_ema
    rollout_config = build_rollout_config(
        inner_iters=args.inner_iters,
        max_outer_iters=args.train_max_outer_iters,
        transition_prob=args.transition_prob,
        curriculum_training=curriculum_training,
        pin_gt=pin_gt,
        deep_supervision=deep_supervision,
        adaptive_curriculum=adaptive_curriculum,
        use_ema=use_ema,
    )
    eval_rollout_config = build_rollout_config(
        inner_iters=args.inner_iters,
        max_outer_iters=args.eval_max_outer_iters,
        transition_prob=args.transition_prob,
        curriculum_training=False,
        use_ema=use_ema,
    )
    refill_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best_val_cell_acc = -1.0
    manifest = load_manifest(run_dir)
    viz_rows = {
        "train": train_ds.rows[: args.viz_samples],
        "validation": val_ds.rows[: args.viz_samples],
    }

    state: BatchSlotState | None = None
    profile_config = ProfileConfig(
        steps=args.profile_steps,
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        epoch=args.profile_epoch,
    )
    if profile_config.enabled:
        if profile_config.epoch > args.epochs:
            raise ValueError(
                f"--profile-epoch {profile_config.epoch} exceeds --epochs {args.epochs}"
            )
        if profile_config.total_steps > args.batches_per_epoch:
            raise ValueError(
                f"profile needs {profile_config.total_steps} steps "
                f"(wait + warmup + active) but --batches-per-epoch is {args.batches_per_epoch}"
            )

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        if state is None:
            state = BatchSlotState.seed(
                train_ds,
                args.train_batch_size,
                device,
                generator=refill_generator,
                dim=model.dim,
                curriculum_training=rollout_config.curriculum_training,
                pin_gt=rollout_config.pin_gt,
                adaptive_curriculum=rollout_config.adaptive_curriculum,
                curriculum_puzzle_acc=rollout_config.curriculum_puzzle_acc,
                use_ema=rollout_config.use_ema,
            )

        profiler = None
        if profile_config.enabled and epoch == profile_config.epoch:
            profiler = TrainProfiler(profile_config, output_dir=run_dir / "profile", device=device)

        train = train_epoch(
            model,
            state,
            train_ds,
            optimizer,
            epoch=epoch,
            epochs=args.epochs,
            rollout_config=rollout_config,
            batches_per_epoch=args.batches_per_epoch,
            halt_loss_weight=args.halt_loss_weight,
            refill_generator=refill_generator,
            profiler=profiler,
            amp=amp,
        )
        if rollout_config.adaptive_curriculum:
            rollout_config = replace(
                rollout_config,
                curriculum_puzzle_acc=update_curriculum_puzzle_acc(
                    rollout_config.curriculum_puzzle_acc,
                    train.puzzle_acc,
                ),
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
            amp=amp,
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
                amp=amp,
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
        ema_alpha = float(model.ema_alpha().item())
        save_epoch_metrics(
            run_dir,
            epoch=epoch,
            train=train,
            val=val,
            args=args,
            ema_alpha=ema_alpha,
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
            f"val_halt_loss={val.halt_loss:.4f} val_halt_acc={val.halt_acc:.4f} "
            f"val_cell_acc={val.cell_acc:.4f} val_puzzle_acc={val.puzzle_acc:.4f} "
            f"val_halt_rate={val.halt_rate:.4f} ema_alpha={ema_alpha:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
