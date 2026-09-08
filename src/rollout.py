from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from data import tensor_to_string
from encoding import attach_clue_mask, grid_to_onehot, onehot_to_grid
from model import NextStateModel

TrainInitMode = Literal["clues", "noisy_gt", "zero_gt", "curriculum"]

DEFAULT_INNER_ITERS = 5
DEFAULT_OUTER_ITERS = 10
DEFAULT_OUTER_COMMIT_PROB = 0.5


@dataclass(frozen=True)
class RolloutConfig:
    train_init: TrainInitMode = "noisy_gt"
    inner_iters: int = DEFAULT_INNER_ITERS
    outer_iters: int = DEFAULT_OUTER_ITERS
    outer_commit_prob: float = DEFAULT_OUTER_COMMIT_PROB

    def __post_init__(self) -> None:
        if self.inner_iters < 1:
            raise ValueError("inner_iters must be >= 1")
        if self.outer_iters < 1:
            raise ValueError("outer_iters must be >= 1")
        if not 0.0 < self.outer_commit_prob <= 1.0:
            raise ValueError("outer_commit_prob must be in (0, 1]")


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


def logits_to_softmax_state(
    logits: torch.Tensor,
    ctx: _ClueContext,
    rollout_clues: torch.Tensor,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    digits = torch.where(ctx.clue_mask, ctx.clue_state, probs)
    return attach_clue_mask(digits, rollout_clues, clue_mask_channel=ctx.clue_mask_channel)


def logits_to_argmax_state(
    logits: torch.Tensor,
    ctx: _ClueContext,
    rollout_clues: torch.Tensor,
    *,
    state: torch.Tensor,
    outer_commit_prob: float = DEFAULT_OUTER_COMMIT_PROB,
) -> torch.Tensor:
    decoded = grid_to_onehot(predict_grid(logits, rollout_clues).detach())
    if outer_commit_prob == 1.0:
        digits = torch.where(ctx.clue_mask, ctx.clue_state, decoded)
    else:
        prev_digits = state[..., :9]
        update = (torch.rand_like(rollout_clues, dtype=torch.float32) < outer_commit_prob) & ~ctx.clue_mask.squeeze(-1)
        blended = torch.where(update.unsqueeze(-1), decoded, prev_digits)
        digits = torch.where(ctx.clue_mask, ctx.clue_state, blended)
    return attach_clue_mask(digits, rollout_clues, clue_mask_channel=ctx.clue_mask_channel)


def _inner_loop(
    model: NextStateModel,
    state: torch.Tensor,
    ctx: _ClueContext,
    rollout_clues: torch.Tensor,
    inner_iters: int,
) -> torch.Tensor:
    for _ in range(inner_iters - 1):
        logits = model(state)
        state = logits_to_softmax_state(logits, ctx, rollout_clues)
    return model(state)


def _rollout_loop(
    model: NextStateModel,
    state: torch.Tensor,
    rollout_clues: torch.Tensor,
    ctx: _ClueContext,
    inner_iters: int,
    outer_iters: int,
    *,
    outer_commit_prob: float = DEFAULT_OUTER_COMMIT_PROB,
    answer: torch.Tensor | None = None,
    collect_state_grids: bool = False,
    accumulate_grad: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor]:
    state_grids: list[torch.Tensor] | None = [] if collect_state_grids else None
    outer_losses: list[torch.Tensor] = []
    logits = state.new_zeros((rollout_clues.size(0), 9, 9, 9))

    for _ in range(outer_iters):
        logits = _inner_loop(model, state, ctx, rollout_clues, inner_iters)
        if answer is not None:
            outer_loss = _compute_rollout_loss_batch_mean(
                logits,
                clues=rollout_clues,
                answer=answer,
            )
            if accumulate_grad:
                (outer_loss / outer_iters).backward()
                outer_losses.append(outer_loss.detach())
            else:
                outer_losses.append(outer_loss)
        state = logits_to_argmax_state(
            logits,
            ctx,
            rollout_clues,
            state=state,
            outer_commit_prob=outer_commit_prob,
        )
        if state_grids is not None:
            state_grids.append(onehot_to_grid(state[..., :9]))

    if answer is None:
        total_loss = logits.new_zeros(())
    elif outer_losses:
        total_loss = torch.stack(outer_losses).mean()
    else:
        total_loss = logits.new_zeros(())

    return logits, state_grids, total_loss


def _run_rollout(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    *,
    config: RolloutConfig,
    initial_onehot: torch.Tensor | None = None,
    collect_state_grids: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Inner/outer rollout with argmax commits. Returns final logits."""
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
        config.inner_iters,
        config.outer_iters,
        outer_commit_prob=config.outer_commit_prob,
        collect_state_grids=collect_state_grids,
    )
    return logits, state_grids


def rollout_train_batch(
    model: NextStateModel,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
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

    logits, _, total_loss = _rollout_loop(
        model,
        state,
        clues_batched,
        ctx,
        config.inner_iters,
        config.outer_iters,
        outer_commit_prob=config.outer_commit_prob,
        answer=answer_batched,
        accumulate_grad=accumulate_grad,
    )
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
    *,
    config: RolloutConfig | None = None,
) -> torch.Tensor:
    config = config or RolloutConfig(train_init="clues")
    logits, _ = _run_rollout(
        model,
        clues_onehot,
        clues,
        config=config,
    )
    return predict_grid(logits, clues)


@torch.inference_mode()
def rollout_trace(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
) -> list[str]:
    """Rollout for viz; one frame per outer argmax commit."""
    config = config or RolloutConfig(train_init="clues")
    _, state_grids = _run_rollout(
        model,
        clues_onehot,
        clues,
        config=config,
        collect_state_grids=True,
    )
    assert state_grids is not None
    grids = [tensor_to_string(clues[0] if clues.dim() == 3 else clues)]
    for grid in state_grids:
        grids.append(tensor_to_string(grid[0] if grid.dim() == 3 else grid))
    return grids
