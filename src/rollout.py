from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from data import tensor_to_string
from encoding import decode_logits, grid_to_onehot, onehot_to_grid
from model import NextStateModel

RolloutMode = Literal["threshold", "categorical"]


@dataclass(frozen=True)
class RolloutConfig:
    mode: RolloutMode = "categorical"


@dataclass
class RolloutResult:
    loss: torch.Tensor
    steps: int
    hit_max_iter: bool
    cycle_length: int | None = None


def logits_to_state(logits: torch.Tensor, clues: torch.Tensor, *, threshold: bool = False) -> torch.Tensor:
    """Feed back decoded logits as one-hot, detached, with clues pinned."""
    if threshold:
        decoded = (logits > 0).float().detach()
    else:
        decoded = grid_to_onehot(decode_logits(logits).detach())
    clue_state = grid_to_onehot(clues)
    clue_mask = (clues > 0).unsqueeze(-1)
    return torch.where(clue_mask, clue_state, decoded)


def predict_grid(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Hard argmax decode."""
    pred = decode_logits(logits)
    return torch.where(clues > 0, clues, pred)


def decode_grid(logits: torch.Tensor, clues: torch.Tensor, *, threshold: bool = False) -> torch.Tensor:
    """Decode logits to a digit grid using the active rollout mode."""
    if threshold:
        return onehot_to_grid(logits_to_state(logits, clues))
    return predict_grid(logits, clues)


def _threshold_state(config: RolloutConfig) -> bool:
    return config.mode == "threshold"


def final_eval_grid(logits: torch.Tensor, clues: torch.Tensor, config: RolloutConfig) -> torch.Tensor:
    """Final eval readout: argmax (softmax winner) for categorical, mode decode for threshold."""
    if config.mode == "categorical":
        return predict_grid(logits, clues)
    return decode_grid(logits, clues, threshold=True)


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


def _state_key(state: torch.Tensor) -> bytes:
    return state.detach().cpu().numpy().tobytes()


def _run_rollout(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int,
    *,
    threshold_state: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int | None, bool]:
    """Collect states and logits; stop on revisit or max_rollout_iter."""
    state = clues_onehot
    states = [state]
    visited: dict[bytes, int] = {_state_key(state): 0}
    logits_list: list[torch.Tensor] = []
    cycle_length: int | None = None
    hit_max_iter = False

    for _ in range(max_rollout_iter):
        logits = model(state)
        logits_list.append(logits)
        new_state = logits_to_state(logits, clues, threshold=threshold_state)
        key = _state_key(new_state)
        if key in visited:
            cycle_length = len(states) - visited[key]
            break
        states.append(new_state)
        visited[key] = len(states) - 1
        state = new_state
    else:
        hit_max_iter = True

    return states, logits_list, cycle_length, hit_max_iter


def _cycle_start_index(states: list[torch.Tensor], cycle_length: int | None) -> int | None:
    if cycle_length is None:
        return None
    return len(states) - cycle_length


def _target_for_step(
    t: int,
    *,
    rollout_grids: list[torch.Tensor],
    answer: torch.Tensor,
    cycle_start: int | None,
) -> torch.Tensor:
    """Cycle steps -> ground truth; outside cycle -> rollout at t+2 if present, else ground truth."""
    if cycle_start is not None and t >= cycle_start:
        return answer
    if t + 2 < len(rollout_grids):
        return rollout_grids[t + 2].detach()
    return answer


def _compute_rollout_loss(
    logits_list: list[torch.Tensor],
    states: list[torch.Tensor],
    *,
    clues: torch.Tensor,
    answer: torch.Tensor,
    cycle_length: int | None,
    config: RolloutConfig,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    rollout_grids = [onehot_to_grid(s) for s in states]
    cycle_start = _cycle_start_index(states, cycle_length)
    total_loss = torch.zeros((), device=clues.device)
    for t, logits in enumerate(logits_list):
        target = _target_for_step(
            t, rollout_grids=rollout_grids, answer=answer, cycle_start=cycle_start
        )
        mask = target_mask(target, clues)
        if config.mode == "threshold":
            total_loss = total_loss + masked_bce_with_logits(
                logits, target, mask, loss_fn  # type: ignore[arg-type]
            )
        else:
            total_loss = total_loss + masked_cross_entropy(
                logits, target, mask, loss_fn  # type: ignore[arg-type]
            )
    return total_loss


def rollout_train_batch(
    model: NextStateModel,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
    answer: torch.Tensor,
    loss_fn: torch.nn.Module,
    max_rollout_iter: int = DEFAULT_MAX_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
) -> RolloutResult:
    """Rollout with t+2 targets outside the cycle and answer targets inside."""
    config = config or RolloutConfig()
    threshold_state = _threshold_state(config)
    states, logits_list, cycle_length, hit_max_iter = _run_rollout(
        model, clues_onehot, clues, max_rollout_iter, threshold_state=threshold_state
    )
    total_loss = _compute_rollout_loss(
        logits_list,
        states,
        clues=clues,
        answer=answer,
        cycle_length=cycle_length,
        config=config,
        loss_fn=loss_fn,
    )

    steps = len(logits_list)
    return RolloutResult(
        loss=total_loss / max(steps, 1),
        steps=steps,
        hit_max_iter=hit_max_iter,
        cycle_length=cycle_length,
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
    _, logits_list, _, _ = _run_rollout(
        model, clues_onehot, clues, max_rollout_iter, threshold_state=threshold_state
    )
    return final_eval_grid(logits_list[-1], clues, config)


@torch.no_grad()
def rollout_trace(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int = DEFAULT_MAX_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
) -> list[str]:
    """Return grid strings at each rollout step, plus argmax final readout when it differs."""
    config = config or RolloutConfig()
    threshold_state = _threshold_state(config)
    _, logits_list, cycle_length, _ = _run_rollout(
        model, clues_onehot, clues, max_rollout_iter, threshold_state=threshold_state
    )
    grids = [tensor_to_string(clues[0] if clues.dim() == 3 else clues)]
    n_steps = len(logits_list) if cycle_length is None else len(logits_list) - 1
    for i in range(n_steps):
        grid = decode_grid(logits_list[i], clues, threshold=threshold_state)
        grids.append(tensor_to_string(grid[0] if grid.dim() == 3 else grid))
    if logits_list and config.mode == "categorical":
        final = final_eval_grid(logits_list[-1], clues, config)
        final_str = tensor_to_string(final[0] if final.dim() == 3 else final)
        if grids[-1] != final_str:
            grids.append(final_str)
    return grids
