from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from data import tensor_to_string
from encoding import attach_clue_mask, decode_logits, grid_to_onehot
from model import NextStateModel

TrainInitMode = Literal["clues", "noisy_gt", "zero_gt"]


@dataclass(frozen=True)
class RolloutConfig:
    train_init: TrainInitMode = "noisy_gt"


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
    steps_per_item: torch.Tensor | None = None
    hit_max_iter_per_item: torch.Tensor | None = None
    cycle_length_per_item: torch.Tensor | None = None


@dataclass(frozen=True)
class _RolloutEvalOutput:
    logits_list: list[torch.Tensor]
    best_logits: torch.Tensor
    steps_per_item: torch.Tensor
    hit_max_iter_per_item: torch.Tensor
    cycle_length_per_item: torch.Tensor


DEFAULT_EVAL_MAX_ROLLOUT_ITER = 30
DEFAULT_TRAIN_ROLLOUT_ITER = 10
DEFAULT_MAX_ROLLOUT_ITER = DEFAULT_EVAL_MAX_ROLLOUT_ITER

def logits_to_state(
    logits: torch.Tensor,
    clues: torch.Tensor,
    *,
    clue_state: torch.Tensor | None = None,
    clue_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Feed back argmax-decoded logits as one-hot, detached, with clues pinned."""
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
        return _noisy_ground_truth_initial(answer, clues, clues_onehot=clues_onehot)
    if config.train_init == "zero_gt":
        return _zeroed_ground_truth_initial(answer, clues, clues_onehot=clues_onehot)
    return None


def predict_grid(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Hard argmax decode."""
    pred = decode_logits(logits)
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


def _find_revisited_index_batched(
    state_stack: torch.Tensor,
    new_state: torch.Tensor,
    *,
    stack_len: int | None = None,
) -> torch.Tensor:
    """Earliest revisit index per batch item, or -1."""
    if stack_len is not None:
        state_stack = state_stack[:stack_len]
    if state_stack.dim() == 4:
        state_stack = state_stack.unsqueeze(1)
        new_state = new_state.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    matches = (state_stack == new_state.unsqueeze(0)).flatten(start_dim=2).all(dim=2)
    any_match = matches.any(dim=0)
    step_ids = torch.arange(matches.size(0), device=matches.device, dtype=torch.long).unsqueeze(1)
    first = torch.where(matches, step_ids, matches.size(0) + 1).min(dim=0).values
    result = torch.where(any_match, first, torch.full_like(first, -1))
    if squeeze:
        return result.squeeze(0)
    return result


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits[mask, :]
    targets = target[mask] - 1
    return F.cross_entropy(masked_logits, targets)


def _answer_state_in(
    answer: torch.Tensor,
    clues: torch.Tensor,
    *,
    ctx: _ClueContext,
) -> torch.Tensor:
    onehot = grid_to_onehot(answer)
    onehot = torch.where(ctx.clue_mask, ctx.clue_state, onehot)
    return attach_clue_mask(onehot, clues, clue_mask_channel=ctx.clue_mask_channel)


def _non_clue_mask(clues: torch.Tensor) -> torch.Tensor:
    """True for cells the model must fill (not given clues)."""
    return clues == 0


def _mean_cell_entropy(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Mean per-cell output entropy over non-clue cells; lower = more peaked."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    cell_entropy = -(probs * log_probs).sum(dim=-1)
    mask = _non_clue_mask(clues)
    if logits.dim() == 4:
        masked_entropy = torch.where(mask, cell_entropy, torch.zeros_like(cell_entropy))
        counts = mask.sum(dim=(-2, -1)).clamp_min(1)
        return masked_entropy.sum(dim=(-2, -1)) / counts
    masked_entropy = cell_entropy[mask]
    if masked_entropy.numel() == 0:
        return cell_entropy.mean()
    return masked_entropy.mean()


def _update_best_by_entropy(
    entropy: torch.Tensor,
    candidate: torch.Tensor,
    *,
    best_entropy: torch.Tensor | None,
    best: torch.Tensor | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if best is None or best_entropy is None:
        return entropy, candidate
    improved = entropy < best_entropy
    best_entropy = torch.where(improved, entropy, best_entropy)
    best = torch.where(improved.view(batch_size, 1, 1, 1), candidate, best)
    return best_entropy, best


def _compute_rollout_loss(
    logits: torch.Tensor,
    *,
    clues: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    """Per-puzzle CE over masked non-clue cells."""
    mask = target_mask(answer, clues)
    return _masked_ce(logits, answer, mask)


def _compute_rollout_loss_batch_mean(
    logits: torch.Tensor,
    *,
    clues: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    """Mean of per-puzzle masked CE (equal weight per puzzle)."""
    if logits.dim() == 3:
        return _compute_rollout_loss(logits, clues=clues, answer=answer)
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


def _dual_rollout_loss_batch_mean(
    model: NextStateModel,
    confident_logits: torch.Tensor,
    *,
    clues: torch.Tensor,
    answer: torch.Tensor,
    ctx: _ClueContext,
) -> torch.Tensor:
    """(confident-step loss + GT fixed-point loss) / 2, mean over batch."""
    target_state_in = _answer_state_in(answer, clues, ctx=ctx)
    target_logits = model(target_state_in)
    confident_loss = _compute_rollout_loss_batch_mean(
        confident_logits, clues=clues, answer=answer
    )
    target_loss = _compute_rollout_loss_batch_mean(target_logits, clues=clues, answer=answer)
    return (confident_loss + target_loss) / 2.0


def _rollout_result_stats(
    steps_per_item: torch.Tensor,
    hit_max_iter_per_item: torch.Tensor,
    cycle_length_per_item: torch.Tensor,
) -> tuple[int, bool, int | None]:
    mean_steps = int(steps_per_item.float().mean().round().item())
    hit_max_iter = bool(hit_max_iter_per_item.any().item())
    cycled = cycle_length_per_item[~torch.isnan(cycle_length_per_item)]
    cycle_length = int(cycled.float().mean().round().item()) if cycled.numel() else None
    return mean_steps, hit_max_iter, cycle_length


def _advance_rollout_state(
    logits: torch.Tensor,
    clues: torch.Tensor,
    *,
    ctx: _ClueContext,
) -> torch.Tensor:
    new_onehot = logits_to_state(
        logits,
        clues,
        clue_state=ctx.clue_state,
        clue_mask=ctx.clue_mask,
    )
    return attach_clue_mask(new_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)


@torch.no_grad()
def _fixed_rollout_entropy_loop(
    model: NextStateModel,
    state: torch.Tensor,
    clues: torch.Tensor,
    rollout_iters: int,
    *,
    clue_state: torch.Tensor,
    clue_mask: torch.Tensor,
    clue_mask_channel: torch.Tensor,
) -> torch.Tensor:
    """Return model input state at the step with lowest mean non-clue cell entropy."""
    batch_size = clues.size(0)
    best_entropy: torch.Tensor | None = None
    best_state_in = state.clone()
    ctx = _ClueContext(clue_state, clue_mask, clue_mask_channel)
    for _ in range(rollout_iters):
        logits = model(state)
        entropy = _mean_cell_entropy(logits, clues)
        best_entropy, best_state_in = _update_best_by_entropy(
            entropy,
            state,
            best_entropy=best_entropy,
            best=best_state_in,
            batch_size=batch_size,
        )
        state = _advance_rollout_state(logits, clues, ctx=ctx)
    return best_state_in


def _run_rollout(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int,
    *,
    initial_onehot: torch.Tensor | None = None,
) -> _RolloutEvalOutput:
    """Collect logits; stop on revisit or max_rollout_iter (batched)."""
    clues, _ = _ensure_batched_clues(clues)
    clues_onehot, _ = _ensure_batched_onehot(clues_onehot)
    if initial_onehot is None:
        initial_onehot = clues_onehot
    else:
        initial_onehot, _ = _ensure_batched_onehot(initial_onehot)
    ctx = _ClueContext.from_clues(clues, clues_onehot)

    batch_size = clues.size(0)
    device = clues.device
    state = attach_clue_mask(initial_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    state_stack = state.new_empty((max_rollout_iter + 1, *state.shape))
    state_stack[0] = state
    stack_len = 1
    logits_list: list[torch.Tensor] = []
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    steps_per_item = torch.zeros(batch_size, dtype=torch.long, device=device)
    hit_max_iter_per_item = torch.zeros(batch_size, dtype=torch.bool, device=device)
    cycle_length_per_item = torch.full((batch_size,), float("nan"), device=device)
    best_entropy: torch.Tensor | None = None
    best_logits: torch.Tensor | None = None

    for step in range(max_rollout_iter):
        with torch.no_grad():
            logits = model(state)
        logits_list.append(logits)
        entropy = _mean_cell_entropy(logits, clues)
        best_entropy, best_logits = _update_best_by_entropy(
            entropy,
            logits,
            best_entropy=best_entropy,
            best=best_logits,
            batch_size=batch_size,
        )

        new_state = _advance_rollout_state(logits, clues, ctx=ctx)
        revisit = _find_revisited_index_batched(state_stack, new_state, stack_len=stack_len)
        found_cycle = (revisit >= 0) & active
        cycle_length_per_item = torch.where(
            found_cycle,
            (stack_len - revisit).float(),
            cycle_length_per_item,
        )
        steps_per_item = torch.where(active, step + 1, steps_per_item)
        active = active & ~found_cycle
        state = torch.where(active.view(batch_size, 1, 1, 1), new_state, state)
        state_stack[stack_len] = new_state
        stack_len += 1
        if not active.any():
            break
    else:
        hit_max_iter_per_item = active.clone()
        steps_per_item = torch.where(active, max_rollout_iter, steps_per_item)

    if best_logits is None:
        best_logits = logits_list[-1]
    return _RolloutEvalOutput(
        logits_list=logits_list,
        best_logits=best_logits,
        steps_per_item=steps_per_item,
        hit_max_iter_per_item=hit_max_iter_per_item,
        cycle_length_per_item=cycle_length_per_item,
    )


@torch.no_grad()
def _run_rollout_fixed_select_confident_state(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    rollout_iters: int,
    *,
    initial_onehot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll out without grad; return input state at the lowest-entropy step."""
    clues, _ = _ensure_batched_clues(clues)
    clues_onehot, _ = _ensure_batched_onehot(clues_onehot)
    if initial_onehot is None:
        initial_onehot = clues_onehot
    else:
        initial_onehot, _ = _ensure_batched_onehot(initial_onehot)
    ctx = _ClueContext.from_clues(clues, clues_onehot)
    state = attach_clue_mask(initial_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)
    return _fixed_rollout_entropy_loop(
        model,
        state,
        clues,
        rollout_iters,
        clue_state=ctx.clue_state,
        clue_mask=ctx.clue_mask,
        clue_mask_channel=ctx.clue_mask_channel,
    )


def rollout_train_batch(
    model: NextStateModel,
    clues: torch.Tensor,
    clues_onehot: torch.Tensor,
    answer: torch.Tensor,
    rollout_iters: int = DEFAULT_EVAL_MAX_ROLLOUT_ITER,
    *,
    config: RolloutConfig | None = None,
    fixed_steps: bool = False,
    compute_pred: bool = True,
) -> RolloutResult:
    """Training: grad-free rollout, then two grad steps (lowest-entropy state + target fixed point).

    Eval: early-stop rollout; readout uses lowest-entropy step.
    """
    config = config or RolloutConfig()
    initial_onehot = (
        _training_initial_onehot(config, answer, clues, clues_onehot) if model.training else None
    )

    if fixed_steps:
        clues_batched, was_batched = _ensure_batched_clues(clues)
        clues_onehot_batched, _ = _ensure_batched_onehot(clues_onehot)
        answer_batched, _ = _ensure_batched_clues(answer)
        ctx_batched = _ClueContext.from_clues(clues_batched, clues_onehot_batched)
        initial_batched = initial_onehot
        if initial_batched is not None:
            initial_batched, _ = _ensure_batched_onehot(initial_batched)
        confident_state_in = _run_rollout_fixed_select_confident_state(
            model,
            clues_onehot_batched,
            clues_batched,
            rollout_iters,
            initial_onehot=initial_batched,
        )
        confident_logits = model(confident_state_in)
        total_loss = _dual_rollout_loss_batch_mean(
            model,
            confident_logits,
            clues=clues_batched,
            answer=answer_batched,
            ctx=ctx_batched,
        )
        eval_logits = confident_logits
        if not was_batched:
            eval_logits = eval_logits.squeeze(0)
        cycle_length = None
        hit_max_iter = True
        steps = rollout_iters
    else:
        clues_batched, was_batched = _ensure_batched_clues(clues)
        clues_onehot_batched, _ = _ensure_batched_onehot(clues_onehot)
        answer_batched, _ = _ensure_batched_clues(answer)
        initial_batched = initial_onehot
        if initial_batched is not None:
            initial_batched, _ = _ensure_batched_onehot(initial_batched)
        eval_out = _run_rollout(
            model,
            clues_onehot_batched,
            clues_batched,
            rollout_iters,
            initial_onehot=initial_batched,
        )
        if not eval_out.logits_list:
            zero = torch.zeros((), device=clues.device)
            return RolloutResult(
                loss=zero,
                steps=0,
                hit_max_iter=False,
                cycle_length=None,
                pred=None,
            )
        eval_logits = eval_out.best_logits
        if not was_batched:
            eval_logits = eval_logits.squeeze(0)
        ctx_batched = _ClueContext.from_clues(clues_batched, clues_onehot_batched)
        total_loss = _dual_rollout_loss_batch_mean(
            model,
            eval_out.best_logits,
            clues=clues_batched,
            answer=answer_batched,
            ctx=ctx_batched,
        )
        steps, hit_max_iter, cycle_length = _rollout_result_stats(
            eval_out.steps_per_item,
            eval_out.hit_max_iter_per_item,
            eval_out.cycle_length_per_item,
        )
        steps_per_item = eval_out.steps_per_item
        hit_max_iter_per_item = eval_out.hit_max_iter_per_item
        cycle_length_per_item = eval_out.cycle_length_per_item
        if not was_batched:
            steps_per_item = None
            hit_max_iter_per_item = None
            cycle_length_per_item = None

    pred = predict_grid(eval_logits.detach(), clues) if compute_pred else None
    return RolloutResult(
        loss=total_loss,
        steps=steps,
        hit_max_iter=hit_max_iter,
        cycle_length=cycle_length,
        pred=pred,
        steps_per_item=steps_per_item if not fixed_steps else None,
        hit_max_iter_per_item=hit_max_iter_per_item if not fixed_steps else None,
        cycle_length_per_item=cycle_length_per_item if not fixed_steps else None,
    )


@torch.no_grad()
def rollout_solve(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int = DEFAULT_MAX_ROLLOUT_ITER,
) -> torch.Tensor:
    """Run f until revisit or max_rollout_iter."""
    eval_out = _run_rollout(
        model,
        clues_onehot,
        clues,
        max_rollout_iter,
    )
    return predict_grid(eval_out.best_logits, clues)


@torch.no_grad()
def rollout_trace(
    model: NextStateModel,
    clues_onehot: torch.Tensor,
    clues: torch.Tensor,
    max_rollout_iter: int = DEFAULT_MAX_ROLLOUT_ITER,
) -> list[str]:
    """Rollout for viz; every displayed frame uses argmax decode (one digit per cell)."""
    eval_out = _run_rollout(
        model,
        clues_onehot,
        clues,
        max_rollout_iter,
    )
    grids = [tensor_to_string(clues[0] if clues.dim() == 3 else clues)]
    for logits in eval_out.logits_list:
        grid = predict_grid(logits, clues)
        grids.append(tensor_to_string(grid[0] if grid.dim() == 3 else grid))
    return grids
