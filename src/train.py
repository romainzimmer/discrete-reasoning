from __future__ import annotations

import argparse
import json
import random
import secrets
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from amp import AmpConfig, autocast_context, resolve_amp
from augment import AugmentConfig
from curriculum import (
    CURRICULUM_P_GT_LOGIT_DECAY,
    CURRICULUM_P_GT_LOGIT_STEP,
    NUM_RATING_GROUPS,
    CurriculumState,
)
from dataset import PuzzleDataset, filter_rows, sample_rows
from encoding import cell_acc_mask
from model import MixerNextStateModel
from rollout import (
    DEFAULT_INNER_ITERS,
    BatchSlotState,
    RolloutConfig,
    RolloutResult,
    refill_done_slots,
    rollout_eval_batch,
    rollout_eval_stream,
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


def optimizer_param_groups(
    model: MixerNextStateModel,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


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
    curriculum_training: bool = True,
    random_curriculum_p_gt: bool = False,
    deep_supervision: bool = True,
    curriculum_p_gt_by_group: tuple[float, ...] | None = None,
) -> RolloutConfig:
    if curriculum_p_gt_by_group is None:
        curriculum_p_gt_by_group = CurriculumState.default().p_gt_by_group()
    return RolloutConfig(
        inner_iters=inner_iters,
        max_outer_iters=max_outer_iters,
        halt_threshold=halt_threshold,
        curriculum_training=curriculum_training,
        random_curriculum_p_gt=random_curriculum_p_gt,
        deep_supervision=deep_supervision,
        curriculum_p_gt_by_group=curriculum_p_gt_by_group,
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
    avg_steps_per_puzzle: float = 0.0
    halt_rate: float = 0.0
    completions_per_epoch: int = 0
    group_puzzle_accs: tuple[float | None, ...] = (None,) * NUM_RATING_GROUPS
    group_puzzles_done: tuple[int, ...] = (0,) * NUM_RATING_GROUPS


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
    n_steps: torch.Tensor
    group_correct_puzzles_done: torch.Tensor
    group_puzzles_done: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> TrainMetricsAccumulator:
        zero = torch.zeros((), device=device)
        zero_i = torch.zeros((), device=device, dtype=torch.long)
        zero_g = torch.zeros(NUM_RATING_GROUPS, device=device, dtype=torch.long)
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
            n_steps=zero_i.clone(),
            group_correct_puzzles_done=zero_g.clone(),
            group_puzzles_done=zero_g.clone(),
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
        self.halt_correct += (result.halted == (result.halt_target > 0.5)).sum()
        self.halt_total += b

        mask = cell_acc_mask(state.clues) & done.view(-1, 1, 1)
        self.correct_cells += (result.pred[mask] == state.answer[mask]).sum()
        self.total_cells += mask.sum()

        puzzle_ok = (result.pred == state.answer).all(dim=(-2, -1))
        self.correct_puzzles_done += (puzzle_ok & done).sum()
        self.puzzles_done += done.sum()
        self.outer_iters_done += (state.outer_count * done.long()).sum()
        self.halted_done += (result.halted & done).sum()
        if done.any():
            done_groups = state.rating_group[done].long()
            correct_groups = puzzle_ok[done]
            counts = torch.bincount(done_groups, minlength=NUM_RATING_GROUPS)
            self.group_puzzles_done += counts.to(self.group_puzzles_done.dtype)
            if correct_groups.any():
                correct_counts = torch.bincount(
                    done_groups[correct_groups],
                    minlength=NUM_RATING_GROUPS,
                )
                self.group_correct_puzzles_done += correct_counts.to(
                    self.group_correct_puzzles_done.dtype
                )

    def group_puzzle_accs(self) -> list[float | None]:
        accs: list[float | None] = []
        for group in range(NUM_RATING_GROUPS):
            done = int(self.group_puzzles_done[group].item())
            if done == 0:
                accs.append(None)
            else:
                accs.append(
                    self.group_correct_puzzles_done[group].item() / done
                )
        return accs

    def finalize(self) -> TrainEpochStats:
        n = int(self.n_steps.item())
        if n == 0:
            return TrainEpochStats(loss=0.0)
        halt_total = int(self.halt_total.item())
        total_cells = int(self.total_cells.item())
        puzzles_done = int(self.puzzles_done.item())
        halted_done = int(self.halted_done.item())
        avg_outer_iters = self.outer_iters_done.item() / puzzles_done if puzzles_done else 0.0
        group_accs = tuple(self.group_puzzle_accs())
        group_done = tuple(int(self.group_puzzles_done[g].item()) for g in range(NUM_RATING_GROUPS))
        return TrainEpochStats(
            loss=self.total_loss.item() / n,
            cell_loss=self.total_cell_loss.item() / n,
            halt_loss=self.total_halt_loss.item() / n,
            cell_acc=self.correct_cells.item() / total_cells if total_cells else 0.0,
            puzzle_acc=self.correct_puzzles_done.item() / puzzles_done if puzzles_done else 0.0,
            halt_acc=self.halt_correct.item() / halt_total if halt_total else 0.0,
            avg_outer_iters=avg_outer_iters,
            avg_steps_per_puzzle=avg_outer_iters,
            halt_rate=halted_done / puzzles_done if puzzles_done else 0.0,
            completions_per_epoch=puzzles_done,
            group_puzzle_accs=group_accs,
            group_puzzles_done=group_done,
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
    avg_steps_per_puzzle: float = 0.0
    halt_rate: float = 0.0
    avg_tries: float = 0.0
    group_puzzle_accs: tuple[float | None, ...] = (None,) * NUM_RATING_GROUPS
    group_puzzles_done: tuple[int, ...] = (0,) * NUM_RATING_GROUPS


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
    tries_sum: torch.Tensor
    n: torch.Tensor
    group_correct_puzzles_done: torch.Tensor
    group_puzzles_done: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> EvalMetricsAccumulator:
        zero = torch.zeros((), device=device)
        zero_i = torch.zeros((), device=device, dtype=torch.long)
        zero_g = torch.zeros(NUM_RATING_GROUPS, device=device, dtype=torch.long)
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
            tries_sum=zero.clone(),
            n=zero_i.clone(),
            group_correct_puzzles_done=zero_g.clone(),
            group_puzzles_done=zero_g.clone(),
        )

    def add_batch(
        self,
        result,
        answer: torch.Tensor,
        clues: torch.Tensor,
        *,
        rating_groups: torch.Tensor | None = None,
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
        tries = result.tries.unsqueeze(0) if result.tries.dim() == 0 else result.tries
        self.tries_sum += tries.sum()

        if rating_groups is not None:
            groups = rating_groups.long().to(self.group_puzzles_done.device)
            puzzle_ok = (preds == answers).all(dim=(-2, -1))
            counts = torch.bincount(groups, minlength=NUM_RATING_GROUPS)
            self.group_puzzles_done += counts
            if puzzle_ok.any():
                correct_counts = torch.bincount(
                    groups[puzzle_ok],
                    minlength=NUM_RATING_GROUPS,
                )
                self.group_correct_puzzles_done += correct_counts

    def group_puzzle_accs(self) -> list[float | None]:
        accs: list[float | None] = []
        for group in range(NUM_RATING_GROUPS):
            done = int(self.group_puzzles_done[group].item())
            if done == 0:
                accs.append(None)
            else:
                accs.append(
                    self.group_correct_puzzles_done[group].item() / done
                )
        return accs

    def finalize(self) -> EpochStats:
        n = int(self.n.item())
        if n == 0:
            return EpochStats(loss=0.0)
        halt_total = int(self.halt_total.item())
        total_cells = int(self.total_cells.item())
        avg_outer_iters = self.outer_iters_sum.item() / n
        group_accs = tuple(self.group_puzzle_accs())
        group_done = tuple(int(self.group_puzzles_done[g].item()) for g in range(NUM_RATING_GROUPS))
        return EpochStats(
            loss=self.total_loss.item() / n,
            cell_loss=self.total_cell_loss.item() / n,
            halt_loss=self.total_halt_loss.item() / n,
            cell_acc=self.correct_cells.item() / total_cells if total_cells else 0.0,
            puzzle_acc=self.correct_puzzles.item() / n,
            halt_acc=self.halt_correct.item() / halt_total if halt_total else 0.0,
            avg_outer_iters=avg_outer_iters,
            avg_steps_per_puzzle=avg_outer_iters,
            halt_rate=self.halted_count.item() / n,
            avg_tries=self.tries_sum.item() / n,
            group_puzzle_accs=group_accs,
            group_puzzles_done=group_done,
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
    pool = sample_rows(rows, max_samples=max_samples, seed=seed)
    if not pool:
        return [], []
    n_val = min(val_samples, len(pool) - 1) if len(pool) > 1 else 1
    indices = list(range(len(pool)))
    rng = random.Random(seed)
    rng.shuffle(indices)
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
    curriculum_state: CurriculumState | None = None,
) -> None:
    history_path = run_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = {"run_id": run_dir.name, "args": vars(args), "epochs": []}

    history["args"] = json_safe(vars(args))
    history["epochs"] = [e for e in history["epochs"] if e["epoch"] != epoch]
    train_fields = asdict(train)
    train_group_puzzle_accs = train_fields.pop("group_puzzle_accs", ())
    train_group_puzzles_done = train_fields.pop("group_puzzles_done", ())
    val_fields = asdict(val)
    val_group_puzzle_accs = val_fields.pop("group_puzzle_accs", ())
    val_group_puzzles_done = val_fields.pop("group_puzzles_done", ())
    epoch_row = {
        "epoch": epoch,
        **{f"train_{k}": v for k, v in train_fields.items()},
        **{f"val_{k}": v for k, v in val_fields.items()},
    }
    for group in range(NUM_RATING_GROUPS):
        epoch_row[f"train_group_{group}_puzzles_done"] = train_group_puzzles_done[group]
        train_acc = train_group_puzzle_accs[group]
        if train_acc is not None:
            epoch_row[f"train_group_{group}_puzzle_acc"] = train_acc
        epoch_row[f"val_group_{group}_puzzles_done"] = val_group_puzzles_done[group]
        val_acc = val_group_puzzle_accs[group]
        if val_acc is not None:
            epoch_row[f"val_group_{group}_puzzle_acc"] = val_acc
    if curriculum_state is not None:
        epoch_row.update(curriculum_state.history_fields())
        epoch_row["curriculum_p_gt"] = curriculum_state.mean_p_gt()
    history["epochs"].append(epoch_row)
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
    curriculum_state: CurriculumState | None = None,
    best_val_cell_acc: float = -1.0,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train.loss,
        "val_loss": val.loss,
        "val_cell_acc": val.cell_acc,
        "val_puzzle_acc": val.puzzle_acc,
        "best_val_cell_acc": best_val_cell_acc,
        "args": vars(args),
    }
    if curriculum_state is not None:
        payload["curriculum_p_gt_logits"] = list(curriculum_state.logits)
        payload["curriculum_p_gt"] = curriculum_state.mean_p_gt()
        payload.update(curriculum_state.history_fields())
    torch.save(payload, path)


def load_last_checkpoint(run_dir: Path, device: torch.device) -> dict:
    path = run_dir / "last.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return torch.load(path, map_location=device, weights_only=False)


def validate_resume_epochs(completed_epoch: int, new_epochs: int) -> None:
    if new_epochs <= completed_epoch:
        raise ValueError(
            f"--epochs {new_epochs} must be greater than completed epoch {completed_epoch}"
        )


def best_val_cell_acc_from_history(run_dir: Path) -> float:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return -1.0
    history = json.loads(history_path.read_text())
    best = -1.0
    for row in history.get("epochs", []):
        best = max(best, float(row.get("val_cell_acc", -1.0)))
    return best


def _group_puzzle_accs_from_history_row(row: dict) -> list[float | None]:
    accs: list[float | None] = []
    for group in range(NUM_RATING_GROUPS):
        done_key = f"train_group_{group}_puzzles_done"
        acc_key = f"train_group_{group}_puzzle_acc"
        if done_key in row and int(row[done_key]) == 0:
            accs.append(None)
        elif acc_key in row:
            accs.append(float(row[acc_key]))
        else:
            accs.append(None)
    return accs


def curriculum_logit_step_from_args(args: dict) -> float:
    return float(args.get("curriculum_logit_step", CURRICULUM_P_GT_LOGIT_STEP))


def curriculum_logit_decay_from_args(args: dict) -> float:
    return float(args.get("curriculum_logit_decay", CURRICULUM_P_GT_LOGIT_DECAY))


def curriculum_state_from_history(run_dir: Path) -> CurriculumState:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return CurriculumState.default()
    history = json.loads(history_path.read_text())
    run_args = history.get("args", {})
    state = CurriculumState.default(
        logit_step=curriculum_logit_step_from_args(run_args),
        logit_decay=curriculum_logit_decay_from_args(run_args),
    )
    for row in sorted(history.get("epochs", []), key=lambda entry: entry["epoch"]):
        state.update_from_group_accs(_group_puzzle_accs_from_history_row(row))
    return state


def curriculum_state_for_resume(ckpt: dict, run_dir: Path) -> CurriculumState:
    run_args = ckpt.get("args", {})
    logit_step = curriculum_logit_step_from_args(run_args)
    logit_decay = curriculum_logit_decay_from_args(run_args)
    if "curriculum_p_gt_logits" in ckpt:
        logits = [float(v) for v in ckpt["curriculum_p_gt_logits"]]
        if len(logits) == NUM_RATING_GROUPS:
            return CurriculumState(
                logits=logits,
                logit_step=logit_step,
                logit_decay=logit_decay,
            )
    if "curriculum_p_gt_logit" in ckpt:
        state = CurriculumState.from_legacy_logit(float(ckpt["curriculum_p_gt_logit"]))
        state.logit_step = logit_step
        state.logit_decay = logit_decay
        return state
    if "curriculum_p_gt" in ckpt:
        state = CurriculumState.from_legacy_p_gt(float(ckpt["curriculum_p_gt"]))
        state.logit_step = logit_step
        state.logit_decay = logit_decay
        return state
    return curriculum_state_from_history(run_dir)


def best_val_cell_acc_for_resume(ckpt: dict, run_dir: Path) -> float:
    if "best_val_cell_acc" in ckpt:
        return float(ckpt["best_val_cell_acc"])
    from_history = best_val_cell_acc_from_history(run_dir)
    if from_history >= 0.0:
        return from_history
    best_path = run_dir / "best.pt"
    if best_path.is_file():
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        if "best_val_cell_acc" in best_ckpt:
            return float(best_ckpt["best_val_cell_acc"])
        return float(best_ckpt.get("val_cell_acc", -1.0))
    return -1.0


def update_history_args(run_dir: Path, args: argparse.Namespace) -> None:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"{history_path} not found")
    history = json.loads(history_path.read_text())
    history["args"] = json_safe(vars(args))
    history_path.write_text(json.dumps(history, indent=2))


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
    curriculum_p_gt: torch.Tensor | None = None,
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
                random_curriculum_p_gt=rollout_config.random_curriculum_p_gt,
                curriculum_p_gt=curriculum_p_gt,
                curriculum_p_gt_by_group=rollout_config.curriculum_p_gt_by_group,
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


def _accumulate_static_eval_batches(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answers: torch.Tensor,
    device: torch.device,
    *,
    slot_batch_size: int,
    rollout_config: RolloutConfig,
    halt_loss_weight: float,
    use_cuda: bool,
    seed: int | None,
    amp: AmpConfig,
    max_tries: int,
    rating_groups: torch.Tensor | None = None,
) -> EvalMetricsAccumulator:
    acc = EvalMetricsAccumulator.empty(device)
    clues = clues.to(device, non_blocking=use_cuda)
    answers = answers.to(device, non_blocking=use_cuda)
    for start in range(0, clues.size(0), slot_batch_size):
        end = min(start + slot_batch_size, clues.size(0))
        batch_clues = clues[start:end]
        batch_answers = answers[start:end]
        with autocast_context(device, amp):
            result = rollout_eval_batch(
                model,
                batch_clues,
                batch_answers,
                config=rollout_config,
                halt_loss_weight=halt_loss_weight,
                init_seed=seed,
                max_tries=max_tries,
            )
        batch_groups = rating_groups[start:end] if rating_groups is not None else None
        acc.add_batch(result, batch_answers, batch_clues, rating_groups=batch_groups)
    return acc


def _accumulate_stream_eval(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answers: torch.Tensor,
    device: torch.device,
    *,
    slot_batch_size: int,
    rollout_config: RolloutConfig,
    halt_loss_weight: float,
    use_cuda: bool,
    seed: int | None,
    amp: AmpConfig,
    max_tries: int,
    progress,
    rating_groups: torch.Tensor | None = None,
) -> EvalMetricsAccumulator:
    acc = EvalMetricsAccumulator.empty(device)
    clues = clues.to(device, non_blocking=use_cuda)
    answers = answers.to(device, non_blocking=use_cuda)
    with autocast_context(device, amp):
        for result, row_clues, row_answer, puzzle_idx in rollout_eval_stream(
            model,
            clues,
            answers,
            slot_batch_size=slot_batch_size,
            config=rollout_config,
            halt_loss_weight=halt_loss_weight,
            init_seed=seed,
            max_tries=max_tries,
        ):
            puzzle_group = (
                rating_groups[puzzle_idx].unsqueeze(0)
                if rating_groups is not None
                else None
            )
            acc.add_batch(
                result,
                row_answer.unsqueeze(0),
                row_clues.unsqueeze(0),
                rating_groups=puzzle_group,
            )
            progress.update(1)
    return acc


@torch.inference_mode()
def measure_split(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answers: torch.Tensor,
    device: torch.device,
    *,
    slot_batch_size: int,
    epoch: int,
    epochs: int,
    phase: str,
    rollout_config: RolloutConfig,
    halt_loss_weight: float,
    use_cuda: bool,
    seed: int | None = None,
    amp: AmpConfig | None = None,
    max_tries: int = 1,
    use_stream: bool = True,
    progress_desc: str | None = None,
    rating_groups: torch.Tensor | None = None,
) -> EpochStats:
    if seed is not None:
        _seed_all(seed)
    model.eval()
    amp = amp or AmpConfig(enabled=False, dtype=None, scaler=None)
    n_puzzles = clues.size(0)
    progress = tqdm(
        total=n_puzzles,
        desc=progress_desc or _epoch_desc(epoch, epochs, phase),
        leave=False,
        unit="puzzle",
        mininterval=0.5,
    )
    progress.set_postfix(slots=slot_batch_size, refresh=True)
    if use_stream:
        acc = _accumulate_stream_eval(
            model,
            clues,
            answers,
            device,
            slot_batch_size=slot_batch_size,
            rollout_config=rollout_config,
            halt_loss_weight=halt_loss_weight,
            use_cuda=use_cuda,
            seed=seed,
            amp=amp,
            max_tries=max_tries,
            progress=progress,
            rating_groups=rating_groups,
        )
    else:
        acc = _accumulate_static_eval_batches(
            model,
            clues,
            answers,
            device,
            slot_batch_size=slot_batch_size,
            rollout_config=rollout_config,
            halt_loss_weight=halt_loss_weight,
            use_cuda=use_cuda,
            seed=seed,
            amp=amp,
            max_tries=max_tries,
            rating_groups=rating_groups,
        )
        progress.update(n_puzzles)
    progress.close()
    return acc.finalize()


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train rollout sudoku model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="L2 regularization on weights only (not bias)")
    parser.add_argument("--dim", type=int, default=256, help="Embedding / mixer hidden dimension D")
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
        default=5,
        help="Max outer commits per puzzle before refill (training)",
    )
    parser.add_argument(
        "--eval-max-outer-iters",
        type=int,
        default=30,
        help="Max outer commits per puzzle during val/viz/test",
    )
    parser.add_argument("--train-batch-size", type=int, default=128, help="Parallel GPU slots (B)")
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random subsample size from train.csv before train/val split (seeded by --seed)",
    )
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
        "--no-adaptive-curriculum",
        action="store_true",
        help=(
            "Sample p_gt ~ U[0, 1] per puzzle at each seed/refill instead of "
            "adaptive per-rating-group p_gt"
        ),
    )
    parser.add_argument(
        "--curriculum-logit-step",
        type=float,
        default=CURRICULUM_P_GT_LOGIT_STEP,
        help="Per-epoch logit update scale for curriculum p_gt per rating group",
    )
    parser.add_argument(
        "--curriculum-logit-decay",
        type=float,
        default=CURRICULUM_P_GT_LOGIT_DECAY,
        help="Per-epoch multiplicative decay on curriculum logits before the acc update",
    )
    parser.add_argument(
        "--no-deep-supervision",
        action="store_true",
        help="Use final inner loop step only for cell and halt loss (default: average all steps)",
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
    return parser


def train_run(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    start_epoch: int = 1,
    model: MixerNextStateModel | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    best_val_cell_acc: float = -1.0,
    initial_curriculum_state: CurriculumState | None = None,
) -> None:
    if args.no_curriculum_training and args.no_adaptive_curriculum:
        raise ValueError(
            "--no-adaptive-curriculum cannot be combined with --no-curriculum-training"
        )
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
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
    val_ds = PuzzleDataset(rows=val_rows)
    args.train_samples = len(train_rows)
    args.val_samples_count = len(val_rows)
    val_batch_size = args.val_batch_size or args.train_batch_size

    if start_epoch == 1:
        args.model = "looped-mixer"
        save_run_config(run_dir, args)
    else:
        update_history_args(run_dir, args)

    args.amp = getattr(args, "amp", not getattr(args, "no_amp", False))
    amp = resolve_amp(device, enabled=args.amp)
    if model is None:
        model = MixerNextStateModel(dim=args.dim, num_blocks=args.num_blocks).to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            optimizer_param_groups(model, weight_decay=args.weight_decay),
            lr=args.lr,
        )
    curriculum_training = not args.no_curriculum_training
    random_curriculum_p_gt = curriculum_training and args.no_adaptive_curriculum
    adaptive_curriculum = curriculum_training and not args.no_adaptive_curriculum
    deep_supervision = not args.no_deep_supervision
    curriculum_logit_step = getattr(args, "curriculum_logit_step", CURRICULUM_P_GT_LOGIT_STEP)
    curriculum_logit_decay = getattr(args, "curriculum_logit_decay", CURRICULUM_P_GT_LOGIT_DECAY)
    curriculum_state: CurriculumState | None = None
    if adaptive_curriculum:
        if initial_curriculum_state is not None:
            curriculum_state = initial_curriculum_state
            curriculum_state.logit_step = curriculum_logit_step
            curriculum_state.logit_decay = curriculum_logit_decay
        else:
            curriculum_state = CurriculumState.default(
                logit_step=curriculum_logit_step,
                logit_decay=curriculum_logit_decay,
            )
    rollout_config = build_rollout_config(
        inner_iters=args.inner_iters,
        max_outer_iters=args.train_max_outer_iters,
        curriculum_training=curriculum_training,
        random_curriculum_p_gt=random_curriculum_p_gt,
        deep_supervision=deep_supervision,
        curriculum_p_gt_by_group=(
            curriculum_state.p_gt_by_group() if curriculum_state is not None else None
        ),
    )
    eval_rollout_config = build_rollout_config(
        inner_iters=args.inner_iters,
        max_outer_iters=args.eval_max_outer_iters,
        curriculum_training=False,
    )
    refill_generator = torch.Generator(device="cpu").manual_seed(args.seed)
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

    curriculum_p_gt = (
        curriculum_state.p_gt_tensor(device) if adaptive_curriculum else None
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds.set_epoch(epoch)
        if state is None:
            state = BatchSlotState.seed(
                train_ds,
                args.train_batch_size,
                device,
                generator=refill_generator,
                curriculum_training=rollout_config.curriculum_training,
                random_curriculum_p_gt=rollout_config.random_curriculum_p_gt,
                curriculum_p_gt=curriculum_p_gt,
                curriculum_p_gt_by_group=rollout_config.curriculum_p_gt_by_group,
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
            curriculum_p_gt=curriculum_p_gt,
            profiler=profiler,
            amp=amp,
        )
        if adaptive_curriculum and curriculum_state is not None:
            curriculum_state.update_from_group_accs(list(train.group_puzzle_accs))
            curriculum_p_gt = curriculum_state.p_gt_tensor(device)
            rollout_config = replace(
                rollout_config,
                curriculum_p_gt_by_group=curriculum_state.p_gt_by_group(),
            )
        val = measure_split(
            model,
            val_ds._base_clues,
            val_ds._base_answers,
            device,
            slot_batch_size=val_batch_size,
            epoch=epoch,
            epochs=args.epochs,
            phase="val",
            rollout_config=eval_rollout_config,
            halt_loss_weight=args.halt_loss_weight,
            use_cuda=use_cuda,
            seed=args.seed,
            amp=amp,
            rating_groups=val_ds._base_rating_groups,
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
        save_epoch_metrics(
            run_dir,
            epoch=epoch,
            train=train,
            val=val,
            args=args,
            curriculum_state=curriculum_state,
        )
        new_best = val.cell_acc > best_val_cell_acc
        if new_best:
            best_val_cell_acc = val.cell_acc
        ckpt_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "train": train,
            "val": val,
            "args": args,
            "curriculum_state": curriculum_state,
            "best_val_cell_acc": best_val_cell_acc,
        }
        save_checkpoint(run_dir / "last.pt", **ckpt_kwargs)
        if new_best:
            save_checkpoint(run_dir / "best.pt", **ckpt_kwargs)
        train_msg = (
            f"train_loss={train.loss:.4f} train_cell_loss={train.cell_loss:.4f} "
            f"train_halt_loss={train.halt_loss:.4f} train_halt_acc={train.halt_acc:.4f} "
            f"train_cell_acc={train.cell_acc:.4f} train_puzzle_acc={train.puzzle_acc:.4f} "
            f"train_halt_rate={train.halt_rate:.4f} "
            f"train_steps_per_puzzle={train.avg_steps_per_puzzle:.1f}"
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"{train_msg} val_loss={val.loss:.4f} val_cell_loss={val.cell_loss:.4f} "
            f"val_halt_loss={val.halt_loss:.4f} val_halt_acc={val.halt_acc:.4f} "
            f"val_cell_acc={val.cell_acc:.4f} val_puzzle_acc={val.puzzle_acc:.4f} "
            f"val_halt_rate={val.halt_rate:.4f} val_steps_per_puzzle={val.avg_steps_per_puzzle:.1f}"
            + (
                f" p_gt={curriculum_state.format_p_gt_log()}"
                if curriculum_state is not None
                else ""
            ),
            flush=True,
        )


def main() -> None:
    args = build_train_parser().parse_args()
    run_dir = make_run_dir(args.runs_dir)
    print(f"Run dir: {run_dir}")
    train_run(run_dir, args)


if __name__ == "__main__":
    main()
