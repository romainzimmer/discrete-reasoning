from __future__ import annotations

import torch
import torch.nn.functional as F

from discrete_reasoning.encoding import decode_logits, grid_to_onehot
from discrete_reasoning.model import NextStateModel, NUM_CLASSES
from discrete_reasoning.trajectory import demo_states


def logits_to_state(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Differentiable rollout state: soft one-hot for empty cells, hard for clues."""
    probs = F.softmax(logits, dim=-1)
    clue_state = grid_to_onehot(clues)
    clue_mask = (clues > 0).unsqueeze(-1)
    return torch.where(clue_mask, clue_state, probs)


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


def target_mask(target: torch.Tensor) -> torch.Tensor:
    """Loss mask: filled cells in partial targets, all cells for full answer."""
    return target > 0


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    mask_flat = mask.reshape(-1)
    if not mask_flat.any():
        return logits.sum() * 0.0
    logits_flat = logits.reshape(-1, NUM_CLASSES)[mask_flat]
    target_flat = (target.reshape(-1)[mask_flat] - 1)
    return loss_fn(logits_flat, target_flat)


def rollout_train_batch(
    model: NextStateModel,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
    answer: torch.Tensor,
    loss_fn: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    """Full training rollout with contracted targets."""
    states, T = demo_states(clues, answer)
    state = clues_onehot
    max_t = int(T.max().item())
    total_loss = torch.zeros((), device=clues.device)

    for t in range(max_t + 1):
        active = T >= t
        if not active.any():
            break

        logits = model(state)
        target = rollout_target(states, answer, t, T)
        mask = target_mask(target) & active.unsqueeze(-1).unsqueeze(-1)

        step_loss = masked_cross_entropy(logits, target, mask, loss_fn)
        total_loss = total_loss + step_loss

        if t < max_t:
            new_state = logits_to_state(logits, clues)
            active_mask = active.view(-1, 1, 1, 1)
            state = torch.where(active_mask, new_state, state)

    return total_loss / (max_t + 1)


@torch.no_grad()
def rollout_solve(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_steps: int = 81,
) -> torch.Tensor:
    """Run f repeatedly until no change or max_steps."""
    state = clues_onehot
    prev = clues.clone()

    for _ in range(max_steps):
        logits = model(state)
        grid = predict_grid(logits, clues)
        if torch.equal(grid, prev):
            break
        prev = grid
        state = grid_to_onehot(grid)

    return prev
