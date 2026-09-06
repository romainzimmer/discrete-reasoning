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
    def from_clues(cls, clues: torch.Tensor) -> _ClueContext:
        mask = (clues > 0).unsqueeze(-1)
        return cls(
            clue_state=grid_to_onehot(clues),
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


def _noisy_ground_truth_initial(answer: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Start from ground truth; replace each non-clue bit with prob p~U[0,1] by random 50/50 bit."""
    ctx = _ClueContext.from_clues(clues)
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


def _zeroed_ground_truth_initial(answer: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Start from ground truth; zero each non-clue cell with prob p~U[0,1] (same as val/test empty init)."""
    ctx = _ClueContext.from_clues(clues)
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
) -> torch.Tensor | None:
    if config.train_init == "noisy_gt":
        return _noisy_ground_truth_initial(answer, clues)
    if config.train_init == "zero_gt":
        return _zeroed_ground_truth_initial(answer, clues)
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


DEFAULT_MAX_ROLLOUT_ITER = 100


def _find_revisited_index(state_stack: torch.Tensor, new_state: torch.Tensor) -> int | None:
    """Return earliest matching index in state_stack, or None (one small GPU sync)."""
    matches = (state_stack == new_state.unsqueeze(0)).flatten(start_dim=1).all(dim=1)
    if not matches.any():
        return None
    return int(matches.nonzero(as_tuple=False)[0, 0].item())


def _stacked_masked_bce(stacked_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return stacked_logits.sum() * 0.0
    target_onehot = grid_to_onehot(target)
    steps = stacked_logits.size(0)
    target_exp = target_onehot.unsqueeze(0).expand(steps, *target_onehot.shape)
    mask_exp = mask.unsqueeze(0).unsqueeze(-1).expand_as(stacked_logits)
    elem = F.binary_cross_entropy_with_logits(
        stacked_logits[mask_exp],
        target_exp[mask_exp],
        reduction="none",
    )
    return elem.sum() / mask_exp[0].sum()


def _stacked_masked_ce(stacked_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return stacked_logits.sum() * 0.0
    steps = stacked_logits.size(0)
    masked_logits = stacked_logits[:, mask, :]
    targets = target[mask] - 1
    target_exp = targets.unsqueeze(0).expand(steps, -1)
    elem = F.cross_entropy(
        masked_logits.reshape(-1, masked_logits.size(-1)),
        target_exp.reshape(-1),
        reduction="none",
    )
    return elem.view(steps, -1).mean(dim=1).sum()


def _run_rollout(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int,
    *,
    threshold_state: bool,
    initial_onehot: torch.Tensor | None = None,
    ctx: _ClueContext,
) -> tuple[list[torch.Tensor], int | None, bool]:
    """Collect logits; stop on revisit or max_rollout_iter."""
    if initial_onehot is None:
        initial_onehot = clues_onehot
    state = attach_clue_mask(initial_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    state_stack = state.unsqueeze(0)
    logits_list: list[torch.Tensor] = []
    cycle_length: int | None = None
    hit_max_iter = False

    for _ in range(max_rollout_iter):
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

    return logits_list, cycle_length, hit_max_iter


def _run_rollout_fixed(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int,
    *,
    threshold_state: bool,
    initial_onehot: torch.Tensor | None = None,
    ctx: _ClueContext,
) -> list[torch.Tensor]:
    """Batched rollout for exactly max_rollout_iter steps (no early stop)."""
    if initial_onehot is None:
        initial_onehot = clues_onehot
    state = attach_clue_mask(initial_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    logits_list: list[torch.Tensor] = []
    for _ in range(max_rollout_iter):
        logits = model(state)
        logits_list.append(logits)
        new_onehot = logits_to_state(
            logits,
            clues,
            threshold=threshold_state,
            clue_state=ctx.clue_state,
            clue_mask=ctx.clue_mask,
        )
        state = attach_clue_mask(new_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    return logits_list


def _compute_rollout_loss(
    logits_list: list[torch.Tensor],
    *,
    clues: torch.Tensor,
    answer: torch.Tensor,
    config: RolloutConfig,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    if not logits_list:
        return torch.zeros((), device=clues.device)
    mask = target_mask(answer, clues)
    stacked_logits = torch.stack(logits_list)
    if config.mode == "threshold":
        return _stacked_masked_bce(stacked_logits, answer, mask)
    return _stacked_masked_ce(stacked_logits, answer, mask)


def rollout_train_batch(
    model: NextStateModel,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
    answer: torch.Tensor,
    loss_fn: torch.nn.Module,
    max_rollout_iter: int = DEFAULT_MAX_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
    fixed_steps: bool = False,
) -> RolloutResult:
    """Rollout with ground-truth targets at every step."""
    config = config or RolloutConfig()
    threshold_state = _threshold_state(config)
    ctx = _ClueContext.from_clues(clues)
    initial_onehot = _training_initial_onehot(config, answer, clues) if model.training else None

    if fixed_steps:
        if clues.dim() != 3 or clues.size(0) < 2:
            raise ValueError("fixed_steps requires a batch with batch size > 1")
        logits_list = _run_rollout_fixed(
            model,
            clues_onehot,
            clues,
            max_rollout_iter,
            threshold_state=threshold_state,
            initial_onehot=initial_onehot,
            ctx=ctx,
        )
        cycle_length = None
        hit_max_iter = True
        steps = max_rollout_iter
    else:
        if clues.dim() == 3 and clues.size(0) > 1:
            raise ValueError("batch size > 1 requires fixed_steps=True")
        logits_list, cycle_length, hit_max_iter = _run_rollout(
            model,
            clues_onehot,
            clues,
            max_rollout_iter,
            threshold_state=threshold_state,
            initial_onehot=initial_onehot,
            ctx=ctx,
        )
        steps = len(logits_list)

    total_loss = _compute_rollout_loss(
        logits_list,
        clues=clues,
        answer=answer,
        config=config,
        loss_fn=loss_fn,
    )

    pred = final_eval_grid(logits_list[-1].detach(), clues)
    return RolloutResult(
        loss=total_loss / max(steps, 1),
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
    logits_list, _, _ = _run_rollout(
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
    logits_list, _, _ = _run_rollout(
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
