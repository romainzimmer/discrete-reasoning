from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from data import tensor_to_string
from encoding import decode_logits, target_mask
from model import MixerNextStateModel

TrainInitMode = Literal["clues", "noisy_gt", "zero_gt", "curriculum"]

DEFAULT_INNER_ITERS = 5
DEFAULT_OUTER_ITERS = 10
DEFAULT_TRUNCATED_BPTT_STEPS = 2


@dataclass(frozen=True)
class RolloutConfig:
    train_init: TrainInitMode = "noisy_gt"
    inner_iters: int = DEFAULT_INNER_ITERS
    outer_iters: int = DEFAULT_OUTER_ITERS
    fixed_point: bool = True
    truncated_bptt_steps: int = DEFAULT_TRUNCATED_BPTT_STEPS

    def __post_init__(self) -> None:
        if self.inner_iters < 1:
            raise ValueError("inner_iters must be >= 1")
        if self.outer_iters < 1:
            raise ValueError("outer_iters must be >= 1")
        if not 0 <= self.truncated_bptt_steps <= self.inner_iters:
            raise ValueError("truncated_bptt_steps must be in [0, inner_iters]")
        if self.fixed_point and self.inner_iters < 2:
            raise ValueError("fixed_point requires inner_iters >= 2")


@dataclass(frozen=True)
class _ClueContext:
    clue_digit_ids: torch.Tensor
    clue_pin: torch.Tensor

    @classmethod
    def from_rollout_clues(cls, rollout_clues: torch.Tensor) -> _ClueContext:
        clue_pin = rollout_clues > 0
        return cls(clue_digit_ids=rollout_clues, clue_pin=clue_pin)


@dataclass
class RolloutState:
    digit_id: torch.Tensor
    clue_pin: torch.Tensor
    input_embed: torch.Tensor | None = None
    cell_embed: torch.Tensor | None = None


@dataclass
class RolloutResult:
    loss: torch.Tensor
    pred: torch.Tensor | None = None


def _pin_clue_digits(digit_id: torch.Tensor, ctx: _ClueContext) -> torch.Tensor:
    return torch.where(ctx.clue_pin, ctx.clue_digit_ids, digit_id)


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
) -> torch.Tensor:
    """Start from ground truth; flip whole non-clue cells to another digit with prob p~U[0,1]."""
    non_clue = clues == 0
    if answer.dim() == 2:
        p = torch.rand((), device=answer.device)
    else:
        p = torch.rand(answer.size(0), 1, 1, device=answer.device)
    flip_cell = (torch.rand_like(clues, dtype=torch.float32) < p) & non_clue
    offset = torch.randint(1, 9, answer.shape, device=answer.device)
    flipped = (answer - 1 + offset) % 9 + 1
    corrupted = torch.where(flip_cell, flipped, answer)
    return _pin_clue_digits(corrupted, _ClueContext.from_rollout_clues(clues))


def _zeroed_ground_truth_initial(
    answer: torch.Tensor,
    clues: torch.Tensor,
) -> torch.Tensor:
    """Start from ground truth; zero each non-clue cell with prob p~U[0,1]."""
    zeroed = _random_zero_non_clue_grid(answer, clues)
    return _pin_clue_digits(zeroed, _ClueContext.from_rollout_clues(clues))


def _curriculum_initial(
    answer: torch.Tensor,
    clues: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reveal GT on non-clue cells with prob (1-p), p~U[0,1]; hidden cells get random digits."""
    rollout_clues = _random_zero_non_clue_grid(answer, clues)
    digit_id = rollout_clues.clone()
    hidden = (rollout_clues == 0) & (clues == 0)
    if hidden.any():
        digit_id[hidden] = torch.randint(1, 10, (int(hidden.sum().item()),), device=clues.device)
    ctx = _ClueContext.from_rollout_clues(rollout_clues)
    digit_id = _pin_clue_digits(digit_id, ctx)
    return digit_id, rollout_clues


def _training_rollout_inputs(
    config: RolloutConfig,
    answer: torch.Tensor,
    clues: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if config.train_init == "curriculum":
        initial_digit_id, rollout_clues = _curriculum_initial(answer, clues)
        return initial_digit_id, rollout_clues, (rollout_clues > 0)
    initial_digit_id = None
    rollout_clues = clues
    clue_pin = clues > 0
    if config.train_init == "noisy_gt":
        initial_digit_id = _noisy_ground_truth_initial(answer, clues)
    elif config.train_init == "zero_gt":
        initial_digit_id = _zeroed_ground_truth_initial(answer, clues)
    return initial_digit_id, rollout_clues, clue_pin


def predict_grid(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Hard argmax decode with clues pinned."""
    pred = decode_logits(logits)
    return torch.where(clues > 0, clues, pred)


def _ensure_batched(grid: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if grid.dim() == 2:
        return grid.unsqueeze(0), False
    return grid, True


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits[mask, :]
    targets = target[mask]
    return F.cross_entropy(masked_logits, targets)


def _compute_rollout_loss_batch_mean(
    logits: torch.Tensor,
    *,
    clue_pin: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    """Mean of per-puzzle masked CE (equal weight per puzzle)."""
    if logits.dim() == 3:
        mask = target_mask(answer, clue_pin)
        return _masked_ce(logits, answer, mask)
    mask = target_mask(answer, clue_pin)
    flat_logits = logits.flatten(1, 2)
    flat_targets = answer.flatten(1, 2)
    flat_mask = mask.flatten(1, 2)
    b, n_cells, n_classes = flat_logits.shape
    per_cell = F.cross_entropy(
        flat_logits.reshape(-1, n_classes),
        flat_targets.reshape(-1),
        reduction="none",
    ).reshape(b, n_cells)
    per_puzzle = (per_cell * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp_min(1)
    return per_puzzle.mean()


def _compute_inner_loop_loss(
    final_logits: torch.Tensor,
    penultimate_logits: torch.Tensor | None,
    *,
    clue_pin: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    final_loss = _compute_rollout_loss_batch_mean(
        final_logits,
        clue_pin=clue_pin,
        answer=answer,
    )
    if penultimate_logits is None:
        return final_loss
    penultimate_loss = _compute_rollout_loss_batch_mean(
        penultimate_logits,
        clue_pin=clue_pin,
        answer=answer,
    )
    return (final_loss + penultimate_loss) / 2


def _outer_commit(logits: torch.Tensor, ctx: _ClueContext) -> torch.Tensor:
    decoded = decode_logits(logits).detach()
    return torch.where(ctx.clue_pin, ctx.clue_digit_ids, decoded)


def _no_grad_inner_steps(
    model: MixerNextStateModel,
    state: RolloutState,
    *,
    inner_iters: int,
    fixed_point: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert state.input_embed is not None
    penultimate_logits: torch.Tensor | None = None
    cell_embed = state.cell_embed
    for step in range(inner_iters):
        out = model(input_embed=state.input_embed, cell_embed=cell_embed)
        logits = out.logits
        cell_embed = out.cell_embed
        if fixed_point and step == inner_iters - 2:
            penultimate_logits = logits
        if step == inner_iters - 1:
            state.cell_embed = cell_embed
            return logits, penultimate_logits
    raise RuntimeError("unreachable")


def _inner_loop(
    model: MixerNextStateModel,
    state: RolloutState,
    inner_iters: int,
    *,
    truncated_bptt_steps: int | None = None,
    fixed_point: bool = True,
    use_truncated_bptt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not use_truncated_bptt:
        return _no_grad_inner_steps(
            model,
            state,
            inner_iters=inner_iters,
            fixed_point=fixed_point,
        )

    if truncated_bptt_steps is None:
        truncated_bptt_steps = DEFAULT_TRUNCATED_BPTT_STEPS
    no_grad_steps = inner_iters - 1 if truncated_bptt_steps == 0 else inner_iters - truncated_bptt_steps

    assert state.input_embed is not None
    penultimate_logits: torch.Tensor | None = None
    final_logits: torch.Tensor | None = None
    cell_embed = state.cell_embed

    for step in range(inner_iters):
        use_no_grad = step < no_grad_steps
        with torch.no_grad() if use_no_grad else nullcontext():
            out = model(input_embed=state.input_embed, cell_embed=cell_embed)
            logits = out.logits
            cell_embed = out.cell_embed
            if fixed_point and step == inner_iters - 2:
                penultimate_logits = logits
            if step == inner_iters - 1:
                final_logits = logits
        if use_no_grad and step == no_grad_steps - 1:
            cell_embed = cell_embed.detach()

    state.cell_embed = cell_embed
    assert final_logits is not None
    return final_logits, penultimate_logits if fixed_point else None


def _rollout_loop(
    model: MixerNextStateModel,
    state: RolloutState,
    rollout_clues: torch.Tensor,
    ctx: _ClueContext,
    inner_iters: int,
    outer_iters: int,
    *,
    answer: torch.Tensor | None = None,
    collect_state_grids: bool = False,
    accumulate_grad: bool = False,
    fixed_point: bool = True,
    truncated_bptt_steps: int | None = DEFAULT_TRUNCATED_BPTT_STEPS,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor]:
    state_grids: list[torch.Tensor] | None = [] if collect_state_grids else None
    outer_losses: list[torch.Tensor] = []
    logits = state.digit_id.new_zeros((rollout_clues.size(0), 9, 9, 10), dtype=torch.float32)

    for _ in range(outer_iters):
        state.input_embed = model.encode_input(state.digit_id, state.clue_pin)
        state.cell_embed = None
        logits, penultimate_logits = _inner_loop(
            model,
            state,
            inner_iters,
            truncated_bptt_steps=truncated_bptt_steps,
            fixed_point=fixed_point,
            use_truncated_bptt=accumulate_grad,
        )
        if answer is not None:
            outer_loss = _compute_inner_loop_loss(
                logits,
                penultimate_logits,
                clue_pin=ctx.clue_pin,
                answer=answer,
            )
            if accumulate_grad:
                (outer_loss / outer_iters).backward()
                outer_losses.append(outer_loss.detach())
            else:
                outer_losses.append(outer_loss)
        state.digit_id = _outer_commit(logits, ctx)
        state.cell_embed = None
        if state_grids is not None:
            state_grids.append(state.digit_id.clone())

    if answer is None:
        total_loss = logits.new_zeros(())
    elif outer_losses:
        total_loss = torch.stack(outer_losses).mean()
    else:
        total_loss = logits.new_zeros(())

    return logits, state_grids, total_loss


def _init_rollout_state(
    clues: torch.Tensor,
    *,
    initial_digit_id: torch.Tensor | None = None,
    clue_pin: torch.Tensor | None = None,
) -> RolloutState:
    if initial_digit_id is None:
        digit_id = clues.clone()
    else:
        digit_id = initial_digit_id
    if clue_pin is None:
        clue_pin = clues > 0
    return RolloutState(digit_id=digit_id, clue_pin=clue_pin)


def _run_rollout(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    *,
    config: RolloutConfig,
    initial_digit_id: torch.Tensor | None = None,
    clue_pin: torch.Tensor | None = None,
    collect_state_grids: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    clues, _ = _ensure_batched(clues)
    if initial_digit_id is not None:
        initial_digit_id, _ = _ensure_batched(initial_digit_id)
    if clue_pin is not None:
        clue_pin, _ = _ensure_batched(clue_pin)
    state = _init_rollout_state(clues, initial_digit_id=initial_digit_id, clue_pin=clue_pin)
    ctx = _ClueContext.from_rollout_clues(clues)

    logits, state_grids, _ = _rollout_loop(
        model,
        state,
        clues,
        ctx,
        config.inner_iters,
        config.outer_iters,
        collect_state_grids=collect_state_grids,
    )
    return logits, state_grids


def rollout_train_batch(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answer: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
    compute_pred: bool = True,
    accumulate_grad: bool = False,
) -> RolloutResult:
    """Rollout loss over outer loops; per-outer backward when training."""
    if model.training and not accumulate_grad:
        raise ValueError("accumulate_grad must be True when model.training")
    config = config or RolloutConfig()
    rollout_clues = clues
    clue_pin = clues > 0
    initial_digit_id = None
    if model.training:
        initial_digit_id, rollout_clues, clue_pin = _training_rollout_inputs(config, answer, clues)

    clues_batched, was_batched = _ensure_batched(rollout_clues)
    answer_batched, _ = _ensure_batched(answer)
    initial_batched = initial_digit_id
    if initial_batched is not None:
        initial_batched, _ = _ensure_batched(initial_batched)
    clue_pin_batched, _ = _ensure_batched(clue_pin)

    ctx = _ClueContext.from_rollout_clues(clues_batched)
    state = _init_rollout_state(
        clues_batched,
        initial_digit_id=initial_batched,
        clue_pin=clue_pin_batched,
    )

    logits, _, total_loss = _rollout_loop(
        model,
        state,
        clues_batched,
        ctx,
        config.inner_iters,
        config.outer_iters,
        answer=answer_batched,
        accumulate_grad=accumulate_grad,
        fixed_point=config.fixed_point,
        truncated_bptt_steps=config.truncated_bptt_steps,
    )
    eval_logits = logits
    if not was_batched:
        eval_logits = eval_logits.squeeze(0)
    pred = predict_grid(eval_logits.detach(), clues) if compute_pred else None
    return RolloutResult(loss=total_loss, pred=pred)


@torch.inference_mode()
def rollout_solve(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
) -> torch.Tensor:
    config = config or RolloutConfig(train_init="clues")
    logits, _ = _run_rollout(
        model,
        clues,
        config=config,
    )
    return predict_grid(logits, clues)


@torch.inference_mode()
def rollout_trace_batch(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
) -> list[list[str]]:
    """Rollout for viz; one frame per outer argmax commit, per puzzle."""
    config = config or RolloutConfig(train_init="clues")
    clues, _ = _ensure_batched(clues)
    _, state_grids = _run_rollout(
        model,
        clues,
        config=config,
        collect_state_grids=True,
    )
    assert state_grids is not None
    trajectories: list[list[str]] = []
    for batch_idx in range(clues.size(0)):
        grids = [tensor_to_string(clues[batch_idx])]
        for grid in state_grids:
            grids.append(tensor_to_string(grid[batch_idx]))
        trajectories.append(grids)
    return trajectories


@torch.inference_mode()
def rollout_trace(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
) -> list[str]:
    """Rollout for viz; one frame per outer argmax commit."""
    clues_b, _ = _ensure_batched(clues)
    return rollout_trace_batch(
        model,
        clues_b,
        config=config,
    )[0]
