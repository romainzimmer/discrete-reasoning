from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from data import tensor_to_string
from encoding import (
    attach_clue_mask,
    grid_to_onehot,
    onehot_to_grid,
    sample_decode_logits,
)
from model import NextStateModel

TrainInitMode = Literal["clues", "noisy_gt", "zero_gt", "curriculum"]
TemperatureSchedule = Literal["cosine", "exponential"]

DEFAULT_ROLLOUT_ITER = 10
DEFAULT_T_MAX = 3.0
DEFAULT_T_MIN = 0.0
DEFAULT_EXPONENTIAL_T_MIN = 1e-2


@dataclass(frozen=True)
class RolloutConfig:
    train_init: TrainInitMode = "noisy_gt"
    temperature_schedule: TemperatureSchedule = "cosine"
    t_max: float = DEFAULT_T_MAX
    t_min: float = DEFAULT_T_MIN


@dataclass(frozen=True)
class _ClueContext:
    clue_state: torch.Tensor
    clue_mask: torch.Tensor
    clue_mask_channel: torch.Tensor

    @classmethod
    def from_clues(cls, clues: torch.Tensor, clues_onehot: torch.Tensor | None = None) -> _ClueContext:
        mask = (clues > 0).unsqueeze(-1)
        return cls(
            clue_state=clues_onehot if clues_onehot is not None else grid_to_onehot(clues),
            clue_mask=mask,
            clue_mask_channel=mask.to(dtype=torch.float32),
        )


@dataclass
class RolloutResult:
    loss: torch.Tensor
    pred: torch.Tensor | None = None


def exponential_decay_rate(t_max: float, t_min: float) -> float:
    """Rate λ so T(s) = t_max * exp(-λ * s/(S-1)) reaches t_min at the last step."""
    if t_min <= 0:
        raise ValueError("t_min must be positive for exponential decay rate")
    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min")
    return math.log(t_max / t_min)


def temperature_at_step(
    step: int,
    total_steps: int,
    t_max: float,
    t_min: float,
    *,
    schedule: TemperatureSchedule = "cosine",
) -> float:
    if total_steps <= 1:
        return 0.0 if schedule == "cosine" else t_min
    t = step / (total_steps - 1)
    if schedule == "cosine":
        return t_min + (t_max - t_min) * 0.5 * (1 + math.cos(math.pi * t))
    rate = exponential_decay_rate(t_max, t_min)
    return t_max * math.exp(-rate * t)


def temperature_schedule(
    rollout_iters: int,
    t_max: float,
    t_min: float,
    *,
    schedule: TemperatureSchedule = "cosine",
) -> list[float]:
    return [
        temperature_at_step(s, rollout_iters, t_max, t_min, schedule=schedule)
        for s in range(rollout_iters)
    ]


def logits_to_state(
    logits: torch.Tensor,
    clues: torch.Tensor,
    *,
    temperature: float,
    clue_state: torch.Tensor | None = None,
    clue_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Feed back sampled-decoded logits as one-hot, detached, with clues pinned."""
    decoded = grid_to_onehot(sample_decode_logits(logits, temperature, clues=clues).detach())
    if clue_state is None:
        clue_state = grid_to_onehot(clues)
    if clue_mask is None:
        clue_mask = (clues > 0).unsqueeze(-1)
    return torch.where(clue_mask, clue_state, decoded)


def _random_zero_non_clue_grid(answer: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Zero each non-clue cell with prob p~U[0,1]; keep clue cells unchanged."""
    non_clue = clues == 0
    if answer.dim() == 2:
        p = torch.rand((), device=answer.device)
    else:
        p = torch.rand(answer.size(0), 1, 1, device=answer.device)
    zero_out = (torch.rand_like(clues, dtype=torch.float32) < p) & non_clue
    return torch.where(zero_out, torch.zeros_like(answer), answer)


def _noisy_ground_truth_initial(
    answer: torch.Tensor,
    clues: torch.Tensor,
    *,
    clues_onehot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Start from ground truth; flip whole non-clue cells to another digit with prob p~U[0,1]."""
    ctx = _ClueContext.from_clues(clues, clues_onehot)
    non_clue = clues == 0
    if answer.dim() == 2:
        p = torch.rand((), device=answer.device)
    else:
        p = torch.rand(answer.size(0), 1, 1, device=answer.device)
    flip_cell = (torch.rand_like(clues, dtype=torch.float32) < p) & non_clue
    offset = torch.randint(1, 9, answer.shape, device=answer.device)
    flipped = (answer - 1 + offset) % 9 + 1
    corrupted_grid = torch.where(flip_cell, flipped, answer)
    onehot = grid_to_onehot(corrupted_grid)
    return torch.where(ctx.clue_mask, ctx.clue_state, onehot)


def _zeroed_ground_truth_initial(
    answer: torch.Tensor,
    clues: torch.Tensor,
    *,
    clues_onehot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Start from ground truth; zero each non-clue cell with prob p~U[0,1]."""
    ctx = _ClueContext.from_clues(clues, clues_onehot)
    onehot = grid_to_onehot(_random_zero_non_clue_grid(answer, clues))
    return torch.where(ctx.clue_mask, ctx.clue_state, onehot)


def _curriculum_initial(
    answer: torch.Tensor,
    clues: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reveal GT on non-clue cells with prob (1-p), p~U[0,1]; hidden cells get random digits."""
    rollout_clues = _random_zero_non_clue_grid(answer, clues)
    grid = rollout_clues.clone()
    hidden = (rollout_clues == 0) & (clues == 0)
    if hidden.any():
        grid[hidden] = torch.randint(1, 10, (int(hidden.sum().item()),), device=clues.device)
    return grid_to_onehot(grid), rollout_clues


def _training_rollout_inputs(
    config: RolloutConfig,
    answer: torch.Tensor,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if config.train_init == "curriculum":
        initial_onehot, rollout_clues = _curriculum_initial(answer, clues)
        return initial_onehot, rollout_clues, grid_to_onehot(rollout_clues)
    initial_onehot = None
    if config.train_init == "noisy_gt":
        initial_onehot = _noisy_ground_truth_initial(answer, clues, clues_onehot=clues_onehot)
    elif config.train_init == "zero_gt":
        initial_onehot = _zeroed_ground_truth_initial(answer, clues, clues_onehot=clues_onehot)
    return initial_onehot, clues, clues_onehot


def predict_grid(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Hard argmax decode with clues pinned."""
    pred = logits.argmax(dim=-1) + 1
    return torch.where(clues > 0, clues, pred)


def target_mask(target: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Loss mask: non-clue cells that are filled in the target."""
    return (target > 0) & (clues == 0)


def _ensure_batched_clues(clues: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if clues.dim() == 2:
        return clues.unsqueeze(0), False
    return clues, True


def _ensure_batched_onehot(onehot: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if onehot.dim() == 3:
        return onehot.unsqueeze(0), False
    return onehot, True


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits[mask, :]
    targets = target[mask] - 1
    return F.cross_entropy(masked_logits, targets)


def _compute_rollout_loss_batch_mean(
    logits: torch.Tensor,
    *,
    clues: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    """Mean of per-puzzle masked CE (equal weight per puzzle)."""
    if logits.dim() == 3:
        mask = target_mask(answer, clues)
        return _masked_ce(logits, answer, mask)
    mask = target_mask(answer, clues)
    flat_logits = logits.flatten(1, 2)
    flat_targets = answer.flatten(1, 2) - 1
    flat_mask = mask.flatten(1, 2)
    b, n_cells, n_classes = flat_logits.shape
    per_cell = F.cross_entropy(
        flat_logits.reshape(-1, n_classes),
        flat_targets.reshape(-1),
        reduction="none",
    ).reshape(b, n_cells)
    per_puzzle = (per_cell * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp_min(1)
    return per_puzzle.mean()


def _advance_rollout_state(
    logits: torch.Tensor,
    clues: torch.Tensor,
    *,
    temperature: float,
    ctx: _ClueContext,
) -> torch.Tensor:
    new_onehot = logits_to_state(
        logits,
        clues,
        temperature=temperature,
        clue_state=ctx.clue_state,
        clue_mask=ctx.clue_mask,
    )
    return attach_clue_mask(new_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)


def _rollout_loop(
    model: NextStateModel,
    state: torch.Tensor,
    clues: torch.Tensor,
    ctx: _ClueContext,
    rollout_iters: int,
    config: RolloutConfig,
    *,
    answer: torch.Tensor | None = None,
    collect_state_grids: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None]:
    temperatures = temperature_schedule(
        rollout_iters,
        config.t_max,
        config.t_min,
        schedule=config.temperature_schedule,
    )
    state_grids: list[torch.Tensor] | None = [] if collect_state_grids else None
    step_losses: list[torch.Tensor] | None = [] if answer is not None else None
    logits = state.new_zeros((clues.size(0), 9, 9, 9))

    for temp in temperatures:
        logits = model(state)
        if step_losses is not None:
            step_losses.append(
                _compute_rollout_loss_batch_mean(logits, clues=clues, answer=answer)
            )
        state = _advance_rollout_state(logits, clues, temperature=temp, ctx=ctx)
        if state_grids is not None:
            state_grids.append(onehot_to_grid(state[..., :9]))

    return logits, state_grids, step_losses


def _run_rollout(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    rollout_iters: int,
    *,
    config: RolloutConfig,
    initial_onehot: torch.Tensor | None = None,
    collect_state_grids: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Fixed-step rollout with temperature-annealed sampling. Returns final logits."""
    clues, _ = _ensure_batched_clues(clues)
    clues_onehot, _ = _ensure_batched_onehot(clues_onehot)
    if initial_onehot is None:
        initial_onehot = clues_onehot
    else:
        initial_onehot, _ = _ensure_batched_onehot(initial_onehot)
    ctx = _ClueContext.from_clues(clues, clues_onehot)

    state = attach_clue_mask(initial_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    logits, state_grids, _ = _rollout_loop(
        model,
        state,
        clues,
        ctx,
        rollout_iters,
        config,
        collect_state_grids=collect_state_grids,
    )
    return logits, state_grids


def rollout_train_batch(
    model: NextStateModel,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
    answer: torch.Tensor,
    rollout_iters: int = DEFAULT_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
    compute_pred: bool = True,
) -> RolloutResult:
    """All-step rollout loss; train and eval share the same loop dynamics."""
    config = config or RolloutConfig()
    rollout_clues = clues
    rollout_clues_onehot = clues_onehot
    initial_onehot = None
    if model.training:
        initial_onehot, rollout_clues, rollout_clues_onehot = _training_rollout_inputs(
            config, answer, clues, clues_onehot
        )

    clues_batched, was_batched = _ensure_batched_clues(rollout_clues)
    clues_onehot_batched, _ = _ensure_batched_onehot(rollout_clues_onehot)
    answer_batched, _ = _ensure_batched_clues(answer)
    initial_batched = initial_onehot
    if initial_batched is not None:
        initial_batched, _ = _ensure_batched_onehot(initial_batched)

    ctx = _ClueContext.from_clues(clues_batched, clues_onehot_batched)
    if initial_batched is None:
        initial_batched = clues_onehot_batched
    state = attach_clue_mask(initial_batched, clues_batched, clue_mask_channel=ctx.clue_mask_channel)

    logits, _, step_losses = _rollout_loop(
        model,
        state,
        clues_batched,
        ctx,
        rollout_iters,
        config,
        answer=answer_batched,
    )
    assert step_losses is not None
    total_loss = torch.stack(step_losses).mean()
    eval_logits = logits
    if not was_batched:
        eval_logits = eval_logits.squeeze(0)
    pred = predict_grid(eval_logits.detach(), clues) if compute_pred else None
    return RolloutResult(loss=total_loss, pred=pred)


@torch.inference_mode()
def rollout_solve(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    rollout_iters: int = DEFAULT_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
) -> torch.Tensor:
    config = config or RolloutConfig(train_init="clues")
    logits, _ = _run_rollout(
        model,
        clues_onehot,
        clues,
        rollout_iters,
        config=config,
    )
    return predict_grid(logits, clues)


@torch.inference_mode()
def rollout_trace(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    rollout_iters: int = DEFAULT_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
) -> list[str]:
    """Rollout for viz; frames show sampled state grids."""
    config = config or RolloutConfig(train_init="clues")
    _, state_grids = _run_rollout(
        model,
        clues_onehot,
        clues,
        rollout_iters,
        config=config,
        collect_state_grids=True,
    )
    assert state_grids is not None
    grids = [tensor_to_string(clues[0] if clues.dim() == 3 else clues)]
    for grid in state_grids:
        grids.append(tensor_to_string(grid[0] if grid.dim() == 3 else grid))
    return grids
