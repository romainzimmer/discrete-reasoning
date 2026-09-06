from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from data import tensor_to_string
from encoding import attach_clue_mask, decode_logits, grid_to_onehot, onehot_to_grid
from model import NextStateModel

RolloutMode = Literal["threshold", "categorical"]
TrainInitMode = Literal["clues", "noisy_gt", "zero_gt"]


@dataclass(frozen=True)
class RolloutConfig:
    mode: RolloutMode = "threshold"
    train_init: TrainInitMode = "clues"


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
    steps: int
    hit_max_iter: bool
    cycle_length: int | None = None
    pred: torch.Tensor | None = None


def logits_to_state(
    logits: torch.Tensor,
    clues: torch.Tensor,
    *,
    threshold: bool = False,
    clue_state: torch.Tensor | None = None,
    clue_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Feed back decoded logits as one-hot, detached, with clues pinned."""
    if threshold:
        decoded = (logits > 0).float().detach()
    else:
        decoded = grid_to_onehot(decode_logits(logits).detach())
    if clue_state is None:
        clue_state = grid_to_onehot(clues)
    if clue_mask is None:
        clue_mask = (clues > 0).unsqueeze(-1)
    return torch.where(clue_mask, clue_state, decoded)


def _noisy_ground_truth_initial(
    answer: torch.Tensor,
    clues: torch.Tensor,
    *,
    clues_onehot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Start from ground truth; replace each non-clue bit with prob p~U[0,1] by random 50/50 bit."""
    ctx = _ClueContext.from_clues(clues, clues_onehot)
    onehot = grid_to_onehot(answer)
    non_clue = clues == 0
    if answer.dim() == 2:
        p = torch.rand((), device=answer.device)
    else:
        p = torch.rand(answer.size(0), 1, 1, 1, device=answer.device)
    replace = (torch.rand_like(onehot) < p) & non_clue.unsqueeze(-1)
    random_bits = (torch.rand_like(onehot) < 0.5).float()
    corrupted = torch.where(replace, random_bits, onehot)
    return torch.where(ctx.clue_mask, ctx.clue_state, corrupted)


def _noisy_ground_truth_initial_categorical(
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
    """Start from ground truth; zero each non-clue cell with prob p~U[0,1] (same as val/test empty init)."""
    ctx = _ClueContext.from_clues(clues, clues_onehot)
    onehot = grid_to_onehot(answer)
    non_clue = clues == 0
    if answer.dim() == 2:
        p = torch.rand((), device=answer.device)
    else:
        p = torch.rand(answer.size(0), 1, 1, device=answer.device)
    zero_out = (torch.rand_like(clues, dtype=torch.float32) < p) & non_clue
    zeroed = torch.where(zero_out.unsqueeze(-1), torch.zeros_like(onehot), onehot)
    return torch.where(ctx.clue_mask, ctx.clue_state, zeroed)


def _training_initial_onehot(
    config: RolloutConfig,
    answer: torch.Tensor,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
) -> torch.Tensor | None:
    if config.train_init == "noisy_gt":
        if config.mode == "categorical":
            return _noisy_ground_truth_initial_categorical(answer, clues, clues_onehot=clues_onehot)
        return _noisy_ground_truth_initial(answer, clues, clues_onehot=clues_onehot)
    if config.train_init == "zero_gt":
        return _zeroed_ground_truth_initial(answer, clues, clues_onehot=clues_onehot)
    return None


def predict_grid(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Hard argmax decode."""
    pred = decode_logits(logits)
    return torch.where(clues > 0, clues, pred)


def decode_grid(logits: torch.Tensor, clues: torch.Tensor, *, threshold: bool = False) -> torch.Tensor:
    """Decode logits to a digit grid using the active rollout mode."""
    if threshold:
        return onehot_to_grid(logits_to_state(logits, clues, threshold=True))
    return predict_grid(logits, clues)


def _threshold_state(config: RolloutConfig) -> bool:
    return config.mode == "threshold"


def final_eval_grid(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Final readout for accuracy/viz: always argmax (softmax winner)."""
    return predict_grid(logits, clues)


def target_mask(target: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Loss mask: non-clue cells that are filled in the target."""
    return (target > 0) & (clues == 0)


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    return loss_fn(logits[mask], target[mask] - 1)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: torch.nn.BCEWithLogitsLoss,
) -> torch.Tensor:
    target_onehot = grid_to_onehot(target)
    mask_exp = mask.unsqueeze(-1).expand_as(logits)
    if not mask.any():
        return logits.sum() * 0.0
    return loss_fn(logits[mask_exp], target_onehot[mask_exp])


DEFAULT_EVAL_MAX_ROLLOUT_ITER = 100
DEFAULT_TRAIN_ROLLOUT_ITER = DEFAULT_EVAL_MAX_ROLLOUT_ITER
DEFAULT_MAX_ROLLOUT_ITER = DEFAULT_EVAL_MAX_ROLLOUT_ITER


def _find_revisited_index(state_stack: torch.Tensor, new_state: torch.Tensor) -> int | None:
    """Return earliest matching index in state_stack, or None (one small GPU sync)."""
    matches = (state_stack == new_state.unsqueeze(0)).flatten(start_dim=1).all(dim=1)
    if not matches.any():
        return None
    return int(matches.nonzero(as_tuple=False)[0, 0].item())


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target_onehot = grid_to_onehot(target)
    mask_exp = mask.unsqueeze(-1).expand_as(logits)
    elem = F.binary_cross_entropy_with_logits(
        logits[mask_exp],
        target_onehot[mask_exp],
        reduction="none",
    )
    return elem.sum() / mask_exp.sum()


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits[mask, :]
    targets = target[mask] - 1
    return F.cross_entropy(masked_logits, targets)


def _run_rollout(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int,
    *,
    threshold_state: bool,
    initial_onehot: torch.Tensor | None = None,
    ctx: _ClueContext,
) -> tuple[list[torch.Tensor], int | None, bool, torch.Tensor]:
    """Collect logits; stop on revisit or max_rollout_iter."""
    if initial_onehot is None:
        initial_onehot = clues_onehot
    state = attach_clue_mask(initial_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    state_stack = state.unsqueeze(0)
    logits_list: list[torch.Tensor] = []
    cycle_length: int | None = None
    hit_max_iter = False
    last_state_in = state

    for _ in range(max_rollout_iter):
        last_state_in = state
        with torch.no_grad():
            logits = model(state)
        logits_list.append(logits)
        new_onehot = logits_to_state(
            logits,
            clues,
            threshold=threshold_state,
            clue_state=ctx.clue_state,
            clue_mask=ctx.clue_mask,
        )
        new_state = attach_clue_mask(new_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
        revisit = _find_revisited_index(state_stack, new_state)
        if revisit is not None:
            cycle_length = state_stack.size(0) - revisit
            break
        state_stack = torch.cat([state_stack, new_state.unsqueeze(0)], dim=0)
        state = new_state
    else:
        hit_max_iter = True

    return logits_list, cycle_length, hit_max_iter, last_state_in


def _run_rollout_fixed(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    rollout_iters: int,
    *,
    threshold_state: bool,
    initial_onehot: torch.Tensor | None = None,
    ctx: _ClueContext,
) -> torch.Tensor:
    """Roll out for exactly rollout_iters steps (no early stop). Returns final recurrent input."""
    if initial_onehot is None:
        initial_onehot = clues_onehot
    state = attach_clue_mask(initial_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    last_state_in = state
    for _ in range(rollout_iters):
        last_state_in = state
        with torch.no_grad():
            logits = model(state)
        new_onehot = logits_to_state(
            logits,
            clues,
            threshold=threshold_state,
            clue_state=ctx.clue_state,
            clue_mask=ctx.clue_mask,
        )
        state = attach_clue_mask(new_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    return last_state_in


def _compute_rollout_loss(
    logits: torch.Tensor,
    *,
    clues: torch.Tensor,
    answer: torch.Tensor,
    config: RolloutConfig,
) -> torch.Tensor:
    mask = target_mask(answer, clues)
    if config.mode == "threshold":
        return _masked_bce(logits, answer, mask)
    return _masked_ce(logits, answer, mask)


def rollout_train_batch(
    model: NextStateModel,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
    answer: torch.Tensor,
    loss_fn: torch.nn.Module,
    rollout_iters: int = DEFAULT_EVAL_MAX_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
    fixed_steps: bool = False,
    compute_pred: bool = True,
) -> RolloutResult:
    """Rollout; loss is computed on the final step's logits only."""
    config = config or RolloutConfig()
    threshold_state = _threshold_state(config)
    ctx = _ClueContext.from_clues(clues, clues_onehot)
    initial_onehot = (
        _training_initial_onehot(config, answer, clues, clues_onehot) if model.training else None
    )

    if fixed_steps:
        last_state_in = _run_rollout_fixed(
            model,
            clues_onehot,
            clues,
            rollout_iters,
            threshold_state=threshold_state,
            initial_onehot=initial_onehot,
            ctx=ctx,
        )
        cycle_length = None
        hit_max_iter = True
        steps = rollout_iters
    else:
        if clues.dim() == 3 and clues.size(0) > 1:
            raise ValueError("batch size > 1 requires fixed_steps=True")
        logits_list, cycle_length, hit_max_iter, last_state_in = _run_rollout(
            model,
            clues_onehot,
            clues,
            rollout_iters,
            threshold_state=threshold_state,
            initial_onehot=initial_onehot,
            ctx=ctx,
        )
        steps = len(logits_list)
        if not logits_list:
            zero = torch.zeros((), device=clues.device)
            return RolloutResult(
                loss=zero,
                steps=0,
                hit_max_iter=False,
                cycle_length=cycle_length,
                pred=None,
            )

    if model.training:
        last_logits = model(last_state_in)
    else:
        last_logits = logits_list[-1]

    total_loss = _compute_rollout_loss(
        last_logits,
        clues=clues,
        answer=answer,
        config=config,
    )

    pred = final_eval_grid(last_logits.detach(), clues) if compute_pred else None
    return RolloutResult(
        loss=total_loss,
        steps=steps,
        hit_max_iter=hit_max_iter,
        cycle_length=cycle_length,
        pred=pred,
    )


@torch.no_grad()
def rollout_solve(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int = DEFAULT_MAX_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
) -> torch.Tensor:
    """Run f until revisit or max_rollout_iter."""
    config = config or RolloutConfig()
    threshold_state = _threshold_state(config)
    ctx = _ClueContext.from_clues(clues)
    logits_list, _, _, _ = _run_rollout(
        model,
        clues_onehot,
        clues,
        max_rollout_iter,
        threshold_state=threshold_state,
        ctx=ctx,
    )
    return final_eval_grid(logits_list[-1], clues)


@torch.no_grad()
def rollout_trace(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int = DEFAULT_MAX_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
) -> list[str]:
    """Rollout with mode decode; last frame is argmax readout on the final step."""
    config = config or RolloutConfig()
    threshold_state = _threshold_state(config)
    ctx = _ClueContext.from_clues(clues)
    logits_list, _, _, _ = _run_rollout(
        model,
        clues_onehot,
        clues,
        max_rollout_iter,
        threshold_state=threshold_state,
        ctx=ctx,
    )
    grids = [tensor_to_string(clues[0] if clues.dim() == 3 else clues)]
    for i, logits in enumerate(logits_list):
        if i == len(logits_list) - 1:
            grid = final_eval_grid(logits, clues)
        else:
            grid = decode_grid(logits, clues, threshold=threshold_state)
        grids.append(tensor_to_string(grid[0] if grid.dim() == 3 else grid))
    return grids
