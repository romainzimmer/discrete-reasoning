from __future__ import annotations

from dataclasses import dataclass

import torch

from data import tensor_to_string
from encoding import decode_logits, grid_to_onehot
from model import NextStateModel
from trajectory import demo_states


@dataclass
class RolloutResult:
    loss: torch.Tensor
    steps: int
    hit_max_iter: bool


def logits_to_state(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Feed back threshold-decoded logits (logit > 0), detached, with clues pinned."""
    decoded = (logits > 0).float().detach()
    clue_state = grid_to_onehot(clues)
    clue_mask = (clues > 0).unsqueeze(-1)
    return torch.where(clue_mask, clue_state, decoded)


def predict_grid(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Hard decode for inference rollout."""
    pred = decode_logits(logits)
    return torch.where(clues > 0, clues, pred)


def rollout_target(
    states: torch.Tensor,
    answer: torch.Tensor,
    t: int,
    T: torch.Tensor,
) -> torch.Tensor:
    """Target grid for f(o_t). states: (B, T_max+1, 9, 9), T: (B,) step count."""
    b = states.size(0)
    target = torch.zeros(b, 9, 9, dtype=answer.dtype, device=answer.device)
    for i in range(b):
        ti = int(T[i].item())
        if t + 2 <= ti:
            target[i] = states[i, t + 2]
        elif t >= ti - 1:
            target[i] = answer[i]
    return target


def target_mask(target: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Loss mask: non-clue cells that are filled in the target."""
    return (target > 0) & (clues == 0)


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


MAX_ROLLOUT_STEPS = 81


def rollout_train_batch(
    model: NextStateModel,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
    answer: torch.Tensor,
    loss_fn: torch.nn.BCEWithLogitsLoss,
    max_steps: int = MAX_ROLLOUT_STEPS,
) -> RolloutResult:
    """Full rollout with contracted targets; stop at fixed point or max_steps."""
    states, T = demo_states(clues, answer)
    state = clues_onehot
    total_loss = torch.zeros((), device=clues.device)
    steps = 0
    converged = False

    for t in range(max_steps):
        logits = model(state)
        target = rollout_target(states, answer, t, T)
        mask = target_mask(target, clues)
        total_loss = total_loss + masked_bce_with_logits(logits, target, mask, loss_fn)
        steps += 1

        new_state = logits_to_state(logits, clues)
        if torch.equal(new_state, state):
            converged = True
            break
        state = new_state

    return RolloutResult(
        loss=total_loss / max(steps, 1),
        steps=steps,
        hit_max_iter=not converged,
    )


@torch.no_grad()
def rollout_solve(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_steps: int = MAX_ROLLOUT_STEPS,
) -> torch.Tensor:
    """Run f until thresholded state stops changing; decode final grid with argmax."""
    state = clues_onehot
    logits = model(state)

    for _ in range(max_steps):
        new_state = logits_to_state(logits, clues)
        if torch.equal(new_state, state):
            break
        state = new_state
        logits = model(state)

    return predict_grid(logits, clues)


@torch.no_grad()
def rollout_trace(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_steps: int = MAX_ROLLOUT_STEPS,
) -> list[str]:
    """Return argmax-decoded grid strings at each rollout step."""
    grids = [tensor_to_string(clues[0] if clues.dim() == 3 else clues)]
    state = clues_onehot
    logits = model(state)

    for _ in range(max_steps):
        new_state = logits_to_state(logits, clues)
        if torch.equal(new_state, state):
            break
        state = new_state
        logits = model(state)
        grid = predict_grid(logits, clues)
        grids.append(tensor_to_string(grid[0] if grid.dim() == 3 else grid))

    return grids
