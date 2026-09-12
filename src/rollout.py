from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data import tensor_to_string
from dataset import PuzzleDataset
from amp import LOSS_DTYPE, to_loss_dtype
from memory import memory_init, zero_memory
from encoding import decode_logits, target_mask
from model import MixerNextStateModel

DEFAULT_INNER_ITERS = 3
DEFAULT_MAX_OUTER_ITERS = 10


@dataclass(frozen=True)
class RolloutConfig:
    inner_iters: int = DEFAULT_INNER_ITERS
    max_outer_iters: int = DEFAULT_MAX_OUTER_ITERS
    halt_threshold: float = 0.5
    curriculum_training: bool = True
    deep_supervision: bool = True
    curriculum_p_gt: float = 0.5

    def __post_init__(self) -> None:
        if self.inner_iters < 1:
            raise ValueError("inner_iters must be >= 1")
        if self.max_outer_iters < 1:
            raise ValueError("max_outer_iters must be >= 1")


@dataclass
class BatchSlotState:
    digit_id: torch.Tensor
    clues: torch.Tensor
    answer: torch.Tensor
    clue_pin: torch.Tensor
    outer_count: torch.Tensor
    memory_embed: torch.Tensor | None = None
    pending_candidate: torch.Tensor | None = None

    @classmethod
    def seed(
        cls,
        dataset: PuzzleDataset,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator,
        curriculum_training: bool = True,
        curriculum_p_gt: float = 0.5,
    ) -> BatchSlotState:
        idx = torch.randint(len(dataset), (batch_size,), generator=generator)
        clues, answers = dataset.sample(idx)
        clues = clues.to(device, non_blocking=True)
        answers = answers.to(device, non_blocking=True)
        clue_pin = clues > 0
        if curriculum_training:
            digit_id = _curriculum_init_digit_id(
                clues,
                answers,
                clue_pin,
                p_gt=curriculum_p_gt,
            )
        else:
            digit_id = _init_digit_id_from_clues(clues, clue_pin)
        return cls(
            digit_id=digit_id,
            clues=clues,
            answer=answers,
            clue_pin=clue_pin,
            outer_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )


@dataclass
class RolloutResult:
    loss: torch.Tensor
    cell_loss: torch.Tensor | None = None
    halt_loss: torch.Tensor | None = None
    pred: torch.Tensor | None = None
    done: torch.Tensor | None = None
    halted: torch.Tensor | None = None
    halt_target: torch.Tensor | None = None
    halt_logit: torch.Tensor | None = None


@dataclass
class EvalRolloutResult:
    pred: torch.Tensor
    outer_steps: torch.Tensor
    halted: torch.Tensor
    loss: torch.Tensor
    cell_loss: torch.Tensor
    halt_loss: torch.Tensor
    halt_target: torch.Tensor
    halt_logit: torch.Tensor
    halt_correct_rounds: int
    halt_total_rounds: int
    tries: torch.Tensor


@dataclass
class _OnceEvalState:
    pred: torch.Tensor
    outer_steps: torch.Tensor
    halted: torch.Tensor
    final_logits: torch.Tensor
    final_halt_logit: torch.Tensor
    halt_correct_by_puzzle: torch.Tensor
    halt_total_by_puzzle: torch.Tensor


@dataclass
class PuzzleTrace:
    inputs: list[str]
    predictions: list[str]
    halted: bool
    outer_steps: int

    @property
    def states(self) -> list[str]:
        return self.predictions


def predict_grid(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """Hard argmax decode with clues pinned."""
    pred = decode_logits(logits)
    return torch.where(clues > 0, clues, pred)


def _ensure_batched(grid: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if grid.dim() == 2:
        return grid.unsqueeze(0), False
    return grid, True


def _halt_target(pre_commit: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
    return (pre_commit == answer).view(pre_commit.size(0), -1).all(dim=1).float()


def _predict_halt(halt_logit: torch.Tensor, *, halt_threshold: float) -> torch.Tensor:
    return torch.sigmoid(halt_logit) > halt_threshold


def _train_outer_done(
    predict_halt: torch.Tensor,
    solved: torch.Tensor,
    outer_count: torch.Tensor,
    max_outer_iters: int,
) -> torch.Tensor:
    """Training: stop a slot on model halt only when the grid is fully correct."""
    return (predict_halt & solved) | (outer_count >= max_outer_iters)


def _eval_outer_done(
    predict_halt: torch.Tensor,
    outer_count: torch.Tensor,
    max_outer_iters: int,
) -> torch.Tensor:
    """Eval: stop on model halt alone; answer is used for metrics only after rollout."""
    return predict_halt | (outer_count >= max_outer_iters)


def _compute_cell_loss(
    logits: torch.Tensor,
    *,
    clue_pin: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
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


def _compute_halt_loss(halt_logit: torch.Tensor, halt_target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(halt_logit, halt_target)


def _compute_losses(
    logits: torch.Tensor,
    halt_logit: torch.Tensor,
    *,
    clue_pin: torch.Tensor,
    answer: torch.Tensor,
    halt_target: torch.Tensor,
    halt_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = to_loss_dtype(logits)
    halt_logit = to_loss_dtype(halt_logit)
    halt_target = to_loss_dtype(halt_target)
    cell_loss = _compute_cell_loss(logits, clue_pin=clue_pin, answer=answer)
    halt_loss = _compute_halt_loss(halt_logit, halt_target)
    total_loss = cell_loss + halt_loss_weight * halt_loss
    return cell_loss, halt_loss, total_loss


def _compute_deep_supervision_losses(
    step_outputs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    clues: torch.Tensor,
    clue_pin: torch.Tensor,
    answer: torch.Tensor,
    halt_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cell_losses: list[torch.Tensor] = []
    halt_losses: list[torch.Tensor] = []
    for logits, halt_logit in step_outputs:
        logits = to_loss_dtype(logits)
        halt_logit = to_loss_dtype(halt_logit)
        cell_losses.append(_compute_cell_loss(logits, clue_pin=clue_pin, answer=answer))
        pred = predict_grid(logits, clues)
        halt_target = to_loss_dtype(_halt_target(pred, answer))
        halt_losses.append(_compute_halt_loss(halt_logit, halt_target))
    cell_loss = torch.stack(cell_losses).mean()
    halt_loss = torch.stack(halt_losses).mean()
    total_loss = cell_loss + halt_loss_weight * halt_loss
    return cell_loss, halt_loss, total_loss


def _begin_outer_step(
    digit_id: torch.Tensor,
    *,
    pending_candidate: torch.Tensor | None,
    outer_count: torch.Tensor,
) -> torch.Tensor:
    """Apply deferred full transition from the prior outer step before inner loop."""
    if pending_candidate is None:
        return digit_id
    commit_mask = (outer_count > 0).view(-1, 1, 1)
    if not commit_mask.any():
        return digit_id
    return torch.where(commit_mask, pending_candidate, digit_id)


def _puzzle_init_seed(clues_row: torch.Tensor, base_seed: int) -> int:
    mixed = base_seed & 0x7FFFFFFF
    for value in clues_row.reshape(-1).tolist():
        mixed = (mixed * 31 + int(value)) & 0x7FFFFFFF
    return mixed


def _random_fill_unpinned(
    digit_id: torch.Tensor,
    unpinned: torch.Tensor,
    *,
    init_seed: int | None = None,
) -> torch.Tensor:
    """Fill unpinned cells with uniform random digits 0-9 (0 = empty)."""
    if not unpinned.any():
        return digit_id
    if init_seed is None:
        random_digits = torch.randint(
            0, 10, digit_id.shape, device=digit_id.device, dtype=digit_id.dtype
        )
        return torch.where(unpinned, random_digits, digit_id)
    b = digit_id.size(0)
    for i in range(b):
        row_unpinned = unpinned[i]
        if not row_unpinned.any():
            continue
        gen = torch.Generator(device=digit_id.device).manual_seed(
            _puzzle_init_seed(digit_id[i], init_seed)
        )
        random_digits = torch.randint(
            0,
            10,
            digit_id[i].shape,
            device=digit_id.device,
            dtype=digit_id.dtype,
            generator=gen,
        )
        digit_id[i] = torch.where(row_unpinned, random_digits, digit_id[i])
    return digit_id


def _init_digit_id_from_clues(
    clues: torch.Tensor,
    clue_pin: torch.Tensor,
    *,
    init_seed: int | None = None,
) -> torch.Tensor:
    """Clues pinned; other cells get random digits 0-9."""
    return _random_fill_unpinned(clues.clone(), ~clue_pin, init_seed=init_seed)


def _curriculum_init_digit_id(
    clues: torch.Tensor,
    answer: torch.Tensor,
    clue_pin: torch.Tensor,
    *,
    p_gt: float = 0.5,
) -> torch.Tensor:
    """Training-only puzzle entry: partial GT reveal with fixed p_gt; unrevealed non-clue cells are random."""
    digit_id = clues.clone()
    non_clue = ~clue_pin
    reveal = non_clue & (torch.rand(clues.shape, device=clues.device) < p_gt)
    digit_id = torch.where(reveal, answer, digit_id)
    return _random_fill_unpinned(digit_id, non_clue & ~reveal)


def _inner_loop(
    model: MixerNextStateModel,
    digit_id: torch.Tensor,
    clue_pin: torch.Tensor,
    inner_iters: int,
    *,
    memory_embed: torch.Tensor | None,
    with_grad: bool = False,
    collect_steps: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]
):
    input_embed = model.encode_input(digit_id, clue_pin)
    cell_embed: torch.Tensor | None = memory_embed
    logits: torch.Tensor | None = None
    halt_logit: torch.Tensor | None = None
    step_outputs: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if collect_steps else None
    for _ in range(inner_iters):
        if with_grad:
            out = model(
                input_embed=input_embed,
                cell_embed=cell_embed,
            )
        else:
            with torch.no_grad():
                out = model(
                    input_embed=input_embed,
                    cell_embed=cell_embed,
                )
        logits = out.logits
        halt_logit = out.halt_logit
        cell_embed = out.cell_embed
        if step_outputs is not None:
            step_outputs.append((logits, halt_logit))
    assert logits is not None
    assert halt_logit is not None
    assert cell_embed is not None
    if step_outputs is not None:
        return step_outputs, cell_embed
    return logits, halt_logit, cell_embed


def rollout_train_step(
    model: MixerNextStateModel,
    state: BatchSlotState,
    config: RolloutConfig,
    *,
    halt_loss_weight: float = 1.0,
    backward: bool = True,
) -> RolloutResult:
    if not model.training:
        raise ValueError("rollout_train_step requires model.training")
    state.digit_id = _begin_outer_step(
        state.digit_id,
        pending_candidate=state.pending_candidate,
        outer_count=state.outer_count,
    )
    if config.deep_supervision:
        step_outputs, final_cell_embed = _inner_loop(
            model,
            state.digit_id,
            state.clue_pin,
            config.inner_iters,
            memory_embed=state.memory_embed,
            with_grad=True,
            collect_steps=True,
        )
        logits, halt_logit = step_outputs[-1]
        cell_loss, halt_loss, loss = _compute_deep_supervision_losses(
            step_outputs,
            clues=state.clues,
            clue_pin=state.clue_pin,
            answer=state.answer,
            halt_loss_weight=halt_loss_weight,
        )
    else:
        logits, halt_logit, final_cell_embed = _inner_loop(
            model,
            state.digit_id,
            state.clue_pin,
            config.inner_iters,
            memory_embed=state.memory_embed,
            with_grad=True,
        )
        pred = predict_grid(logits, state.clues)
        halt_target = _halt_target(pred, state.answer)
        cell_loss, halt_loss, loss = _compute_losses(
            logits,
            halt_logit,
            clue_pin=state.clue_pin,
            answer=state.answer,
            halt_target=halt_target,
            halt_loss_weight=halt_loss_weight,
        )
    if config.deep_supervision:
        pred = predict_grid(logits, state.clues)
        halt_target = _halt_target(pred, state.answer)
    predict_halt = _predict_halt(halt_logit, halt_threshold=config.halt_threshold)
    if backward:
        loss.backward()
    state.pending_candidate = pred.detach()
    state.memory_embed = memory_init(final_cell_embed)
    state.outer_count = state.outer_count + 1
    solved = halt_target > 0.5
    done = _train_outer_done(
        predict_halt,
        solved,
        state.outer_count,
        config.max_outer_iters,
    )
    return RolloutResult(
        loss=loss.detach() if backward else loss,
        cell_loss=cell_loss.detach(),
        halt_loss=halt_loss.detach(),
        pred=pred.detach(),
        done=done,
        halted=predict_halt,
        halt_target=halt_target.detach(),
        halt_logit=halt_logit.detach(),
    )


def refill_done_slots(
    state: BatchSlotState,
    done: torch.Tensor,
    dataset: PuzzleDataset,
    *,
    generator: torch.Generator,
    dim: int,
    curriculum_training: bool = True,
    curriculum_p_gt: float = 0.5,
) -> None:
    b = done.size(0)
    device = state.digit_id.device
    idx = torch.randint(len(dataset), (b,), generator=generator)
    new_clues, new_answers = dataset.sample(idx)
    new_clues = new_clues.to(device, non_blocking=True)
    new_answers = new_answers.to(device, non_blocking=True)
    done_mask = done.view(b, 1, 1)
    new_clue_pin = new_clues > 0
    if curriculum_training:
        new_digit_id = _curriculum_init_digit_id(
            new_clues,
            new_answers,
            new_clue_pin,
            p_gt=curriculum_p_gt,
        )
    else:
        new_digit_id = _init_digit_id_from_clues(new_clues, new_clue_pin)
    state.digit_id = torch.where(done_mask, new_digit_id, state.digit_id)
    state.clues = torch.where(done_mask, new_clues, state.clues)
    state.answer = torch.where(done_mask, new_answers, state.answer)
    state.clue_pin = state.clues > 0
    state.outer_count = torch.where(done, torch.zeros_like(state.outer_count), state.outer_count)
    done_mask_mem = done.view(b, 1, 1, 1)
    if state.memory_embed is not None:
        new_memory = zero_memory(b, dim, device)
        state.memory_embed = torch.where(done_mask_mem, new_memory, state.memory_embed)
    if done.any() and state.pending_candidate is not None:
        if done.all():
            state.pending_candidate = None
        else:
            state.pending_candidate = state.pending_candidate.clone()
            state.pending_candidate[done] = 0


def _pending_for_active(
    pending_candidate: torch.Tensor | None,
    slot_idx: torch.Tensor,
) -> torch.Tensor | None:
    if pending_candidate is None:
        return None
    return pending_candidate[slot_idx]


def _store_pending_candidate(
    pending_candidate: torch.Tensor | None,
    slot_idx: torch.Tensor,
    pre_commit: torch.Tensor,
    *,
    batch_size: int,
    like: torch.Tensor,
) -> torch.Tensor:
    if pending_candidate is None:
        pending_candidate = like.new_zeros((batch_size, 9, 9))
    pending_candidate[slot_idx] = pre_commit
    return pending_candidate


@dataclass
class _CompactOuterStep:
    slot_idx: torch.Tensor
    model_input: torch.Tensor
    pre_commit: torch.Tensor
    predict_halt: torch.Tensor
    logits: torch.Tensor
    halt_logit: torch.Tensor
    active_digit_id: torch.Tensor
    active_outer_count: torch.Tensor
    active_answer: torch.Tensor
    done: torch.Tensor


def _iter_compact_outer_rollout(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answer: torch.Tensor,
    *,
    config: RolloutConfig,
    init_seed: int | None = 0,
):
    clues, _ = _ensure_batched(clues)
    answer, _ = _ensure_batched(answer)
    b = clues.size(0)
    device = clues.device

    clue_pin = clues > 0
    digit_id = _init_digit_id_from_clues(clues, clue_pin, init_seed=init_seed)

    slot_idx = torch.arange(b, device=device)
    active_digit_id = digit_id
    active_clues = clues
    active_answer = answer
    active_clue_pin = clue_pin
    active_outer_count = torch.zeros(b, dtype=torch.long, device=device)
    active_memory_embed: torch.Tensor | None = None
    pending_candidate: torch.Tensor | None = None

    while slot_idx.numel() > 0:
        active_digit_id = _begin_outer_step(
            active_digit_id,
            pending_candidate=_pending_for_active(pending_candidate, slot_idx),
            outer_count=active_outer_count,
        )
        model_input = active_digit_id
        logits, halt_logit, final_cell_embed = _inner_loop(
            model,
            active_digit_id,
            active_clue_pin,
            config.inner_iters,
            memory_embed=active_memory_embed,
            with_grad=False,
        )
        pre_commit = predict_grid(logits, active_clues)
        predict_halt = _predict_halt(halt_logit, halt_threshold=config.halt_threshold)
        pending_candidate = _store_pending_candidate(
            pending_candidate,
            slot_idx,
            pre_commit,
            batch_size=b,
            like=digit_id,
        )
        active_memory_embed = memory_init(final_cell_embed)
        active_outer_count = active_outer_count + 1
        done = _eval_outer_done(predict_halt, active_outer_count, config.max_outer_iters)

        yield _CompactOuterStep(
            slot_idx=slot_idx,
            model_input=model_input,
            pre_commit=pre_commit,
            predict_halt=predict_halt,
            logits=logits,
            halt_logit=halt_logit,
            active_digit_id=active_digit_id,
            active_outer_count=active_outer_count,
            active_answer=active_answer,
            done=done,
        )

        keep = ~done
        if not keep.any():
            break
        slot_idx = slot_idx[keep]
        active_digit_id = active_digit_id[keep]
        active_clues = active_clues[keep]
        active_answer = active_answer[keep]
        active_clue_pin = active_clue_pin[keep]
        active_outer_count = active_outer_count[keep]
        active_memory_embed = active_memory_embed[keep] if active_memory_embed is not None else None


def _rollout_eval_batch_once(
    model: MixerNextStateModel,
    clues_b: torch.Tensor,
    answer_b: torch.Tensor,
    *,
    config: RolloutConfig,
    init_seed: int | None,
) -> _OnceEvalState:
    b = clues_b.size(0)
    device = clues_b.device

    out_pred = clues_b.clone()
    out_steps = torch.zeros(b, dtype=torch.long, device=device)
    out_halted = torch.zeros(b, dtype=torch.bool, device=device)
    final_logits = clues_b.new_zeros((b, 9, 9, 10), dtype=LOSS_DTYPE)
    final_halt_logit = clues_b.new_zeros((b,), dtype=LOSS_DTYPE)
    halt_correct_by_puzzle = torch.zeros(b, dtype=torch.long, device=device)
    halt_total_by_puzzle = torch.zeros(b, dtype=torch.long, device=device)

    for step in _iter_compact_outer_rollout(
        model, clues_b, answer_b, config=config, init_seed=init_seed
    ):
        halt_target_round = _halt_target(step.pre_commit, step.active_answer)
        round_correct = (step.predict_halt == (halt_target_round > 0.5)).long()
        halt_correct_by_puzzle[step.slot_idx] += round_correct
        halt_total_by_puzzle[step.slot_idx] += 1
        final_logits[step.slot_idx] = to_loss_dtype(step.logits)
        final_halt_logit[step.slot_idx] = to_loss_dtype(step.halt_logit)
        done_idx = step.slot_idx[step.done]
        out_pred[done_idx] = step.pre_commit[step.done]
        out_steps[done_idx] = step.active_outer_count[step.done]
        out_halted[done_idx] = step.predict_halt[step.done]

    return _OnceEvalState(
        pred=out_pred,
        outer_steps=out_steps,
        halted=out_halted,
        final_logits=final_logits,
        final_halt_logit=final_halt_logit,
        halt_correct_by_puzzle=halt_correct_by_puzzle,
        halt_total_by_puzzle=halt_total_by_puzzle,
    )


def _try_init_seed(init_seed: int | None, try_idx: int) -> int | None:
    if init_seed is None:
        return None
    return init_seed + try_idx


def _copy_once_state(
    out: _OnceEvalState,
    sub: _OnceEvalState,
    slot_idx: torch.Tensor,
    local_mask: torch.Tensor,
) -> None:
    local_idx = local_mask.nonzero(as_tuple=True)[0]
    accept_global = slot_idx[local_mask]
    out.pred[accept_global] = sub.pred[local_idx]
    out.outer_steps[accept_global] = sub.outer_steps[local_idx]
    out.halted[accept_global] = sub.halted[local_idx]
    out.final_logits[accept_global] = sub.final_logits[local_idx]
    out.final_halt_logit[accept_global] = sub.final_halt_logit[local_idx]
    out.halt_correct_by_puzzle[accept_global] = sub.halt_correct_by_puzzle[local_idx]
    out.halt_total_by_puzzle[accept_global] = sub.halt_total_by_puzzle[local_idx]


def _once_state_to_result(
    state: _OnceEvalState,
    clues_b: torch.Tensor,
    answer_b: torch.Tensor,
    *,
    halt_loss_weight: float,
    tries: torch.Tensor,
    was_batched: bool,
) -> EvalRolloutResult:
    clue_pin = clues_b > 0
    halt_target = _halt_target(predict_grid(state.final_logits, clues_b), answer_b)
    cell_loss, halt_loss, loss = _compute_losses(
        state.final_logits,
        state.final_halt_logit,
        clue_pin=clue_pin,
        answer=answer_b,
        halt_target=halt_target,
        halt_loss_weight=halt_loss_weight,
    )
    halt_correct_rounds = int(state.halt_correct_by_puzzle.sum().item())
    halt_total_rounds = int(state.halt_total_by_puzzle.sum().item())
    if not was_batched:
        return EvalRolloutResult(
            pred=state.pred.squeeze(0),
            outer_steps=state.outer_steps.squeeze(0),
            halted=state.halted.squeeze(0),
            loss=loss,
            cell_loss=cell_loss,
            halt_loss=halt_loss,
            halt_target=halt_target.squeeze(0),
            halt_logit=state.final_halt_logit.squeeze(0),
            halt_correct_rounds=halt_correct_rounds,
            halt_total_rounds=halt_total_rounds,
            tries=tries.squeeze(0),
        )
    return EvalRolloutResult(
        pred=state.pred,
        outer_steps=state.outer_steps,
        halted=state.halted,
        loss=loss,
        cell_loss=cell_loss,
        halt_loss=halt_loss,
        halt_target=halt_target,
        halt_logit=state.final_halt_logit,
        halt_correct_rounds=halt_correct_rounds,
        halt_total_rounds=halt_total_rounds,
        tries=tries,
    )


@dataclass
class _StreamSlot:
    puzzle_idx: int
    try_idx: int
    clues: torch.Tensor
    answer: torch.Tensor
    digit_id: torch.Tensor
    outer_count: int
    memory_embed: torch.Tensor | None
    pending_candidate: torch.Tensor | None
    pred: torch.Tensor
    outer_steps: int
    halted: bool
    final_logits: torch.Tensor
    final_halt_logit: torch.Tensor
    halt_correct: int
    halt_total: int


def _begin_outer_step_row(
    digit_id: torch.Tensor,
    pending_candidate: torch.Tensor | None,
    outer_count: int,
) -> torch.Tensor:
    if pending_candidate is None or outer_count == 0:
        return digit_id
    return pending_candidate


def _new_stream_slot(
    puzzle_idx: int,
    clues: torch.Tensor,
    answer: torch.Tensor,
    *,
    init_seed: int | None,
    try_idx: int,
) -> _StreamSlot:
    clue_pin = clues > 0
    digit_id = _init_digit_id_from_clues(
        clues.unsqueeze(0),
        clue_pin.unsqueeze(0),
        init_seed=_try_init_seed(init_seed, try_idx),
    ).squeeze(0)
    return _StreamSlot(
        puzzle_idx=puzzle_idx,
        try_idx=try_idx,
        clues=clues,
        answer=answer,
        digit_id=digit_id,
        outer_count=0,
        memory_embed=None,
        pending_candidate=None,
        pred=clues.clone(),
        outer_steps=0,
        halted=False,
        final_logits=clues.new_zeros((9, 9, 10), dtype=LOSS_DTYPE),
        final_halt_logit=clues.new_zeros((), dtype=LOSS_DTYPE),
        halt_correct=0,
        halt_total=0,
    )


def _slot_to_eval_result(
    slot: _StreamSlot,
    *,
    halt_loss_weight: float,
) -> EvalRolloutResult:
    once = _OnceEvalState(
        pred=slot.pred.unsqueeze(0),
        outer_steps=torch.tensor([slot.outer_steps], device=slot.clues.device),
        halted=torch.tensor([slot.halted], device=slot.clues.device),
        final_logits=slot.final_logits.unsqueeze(0),
        final_halt_logit=slot.final_halt_logit.unsqueeze(0),
        halt_correct_by_puzzle=torch.tensor([slot.halt_correct], device=slot.clues.device),
        halt_total_by_puzzle=torch.tensor([slot.halt_total], device=slot.clues.device),
    )
    tries = torch.tensor([slot.try_idx + 1], device=slot.clues.device, dtype=torch.long)
    return _once_state_to_result(
        once,
        slot.clues.unsqueeze(0),
        slot.answer.unsqueeze(0),
        halt_loss_weight=halt_loss_weight,
        tries=tries,
        was_batched=False,
    )


def _stack_active_memory_embed(
    slots: list[_StreamSlot],
    *,
    dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not slots or all(s.memory_embed is None for s in slots):
        return None
    rows: list[torch.Tensor] = []
    for slot in slots:
        if slot.memory_embed is None:
            rows.append(zero_memory(1, dim, device).squeeze(0))
        else:
            rows.append(slot.memory_embed)
    return torch.stack(rows)


def _record_stream_outer_metrics(
    slot: _StreamSlot,
    *,
    pre_commit: torch.Tensor,
    predict_halt: bool,
    logits: torch.Tensor,
    halt_logit: torch.Tensor,
) -> None:
    halt_target_round = _halt_target(pre_commit.unsqueeze(0), slot.answer.unsqueeze(0)).item()
    slot.halt_correct += int(predict_halt == (halt_target_round > 0.5))
    slot.halt_total += 1
    slot.final_logits = to_loss_dtype(logits)
    slot.final_halt_logit = to_loss_dtype(halt_logit)


@torch.inference_mode()
def rollout_eval_stream(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answers: torch.Tensor,
    *,
    slot_batch_size: int,
    config: RolloutConfig | None = None,
    halt_loss_weight: float = 1.0,
    init_seed: int | None = 0,
    max_tries: int = 1,
):
    """Evaluate puzzles with B parallel slots; refill freed slots from the queue."""
    config = config or RolloutConfig()
    if slot_batch_size < 1:
        raise ValueError("slot_batch_size must be >= 1")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")

    clues_b, _ = _ensure_batched(clues)
    answers_b, _ = _ensure_batched(answers)
    if clues_b.size(0) != answers_b.size(0):
        raise ValueError("clues and answers must have the same batch size")
    n_puzzles = clues_b.size(0)
    if n_puzzles == 0:
        return

    queue_pos = 0
    slots: list[_StreamSlot] = []

    def enqueue_next() -> bool:
        nonlocal queue_pos
        if queue_pos >= n_puzzles:
            return False
        slots.append(
            _new_stream_slot(
                queue_pos,
                clues_b[queue_pos],
                answers_b[queue_pos],
                init_seed=init_seed,
                try_idx=0,
            )
        )
        queue_pos += 1
        return True

    while len(slots) < slot_batch_size:
        if not enqueue_next():
            break

    while slots:
        active_digit_id = torch.stack(
            [
                _begin_outer_step_row(s.digit_id, s.pending_candidate, s.outer_count)
                for s in slots
            ]
        )
        active_clues = torch.stack([s.clues for s in slots])
        active_clue_pin = active_clues > 0
        active_outer_count = torch.tensor(
            [s.outer_count for s in slots],
            device=clues_b.device,
            dtype=torch.long,
        )
        active_memory = _stack_active_memory_embed(
            slots,
            dim=model.dim,
            device=clues_b.device,
        )

        logits, halt_logit, final_cell_embed = _inner_loop(
            model,
            active_digit_id,
            active_clue_pin,
            config.inner_iters,
            memory_embed=active_memory,
            with_grad=False,
        )
        pre_commit = predict_grid(logits, active_clues)
        predict_halt = _predict_halt(halt_logit, halt_threshold=config.halt_threshold)
        next_outer_count = active_outer_count + 1
        done = _eval_outer_done(predict_halt, next_outer_count, config.max_outer_iters)

        next_slots: list[_StreamSlot] = []
        refill_count = 0
        for i, slot in enumerate(slots):
            _record_stream_outer_metrics(
                slot,
                pre_commit=pre_commit[i],
                predict_halt=bool(predict_halt[i].item()),
                logits=logits[i],
                halt_logit=halt_logit[i],
            )
            slot.outer_count = int(next_outer_count[i].item())
            slot.pending_candidate = pre_commit[i]
            slot.memory_embed = final_cell_embed[i]
            slot.digit_id = active_digit_id[i]

            if not done[i]:
                next_slots.append(slot)
                continue

            slot.pred = pre_commit[i]
            slot.outer_steps = slot.outer_count
            slot.halted = bool(predict_halt[i].item())
            if slot.halted or slot.try_idx + 1 >= max_tries:
                yield (
                    _slot_to_eval_result(slot, halt_loss_weight=halt_loss_weight),
                    slot.clues,
                    slot.answer,
                )
                refill_count += 1
            else:
                next_slots.append(
                    _new_stream_slot(
                        slot.puzzle_idx,
                        slot.clues,
                        slot.answer,
                        init_seed=init_seed,
                        try_idx=slot.try_idx + 1,
                    )
                )

        slots = next_slots
        for _ in range(refill_count):
            if not enqueue_next():
                break


def _rollout_eval_batch_multi_try(
    model: MixerNextStateModel,
    clues_b: torch.Tensor,
    answer_b: torch.Tensor,
    *,
    config: RolloutConfig,
    halt_loss_weight: float,
    init_seed: int | None,
    max_tries: int,
    was_batched: bool,
) -> EvalRolloutResult:
    b = clues_b.size(0)
    device = clues_b.device
    pending = torch.ones(b, dtype=torch.bool, device=device)
    tries = torch.zeros(b, dtype=torch.long, device=device)
    merged = _OnceEvalState(
        pred=clues_b.clone(),
        outer_steps=torch.zeros(b, dtype=torch.long, device=device),
        halted=torch.zeros(b, dtype=torch.bool, device=device),
        final_logits=clues_b.new_zeros((b, 9, 9, 10), dtype=LOSS_DTYPE),
        final_halt_logit=clues_b.new_zeros((b,), dtype=LOSS_DTYPE),
        halt_correct_by_puzzle=torch.zeros(b, dtype=torch.long, device=device),
        halt_total_by_puzzle=torch.zeros(b, dtype=torch.long, device=device),
    )

    for try_idx in range(max_tries):
        if not pending.any():
            break
        slot_idx = pending.nonzero(as_tuple=True)[0]
        sub = _rollout_eval_batch_once(
            model,
            clues_b[slot_idx],
            answer_b[slot_idx],
            config=config,
            init_seed=_try_init_seed(init_seed, try_idx),
        )
        tries[slot_idx] += 1
        sub_halted = sub.halted
        if try_idx == max_tries - 1:
            accept = torch.ones(sub_halted.size(0), dtype=torch.bool, device=device)
        else:
            accept = sub_halted
        if accept.any():
            _copy_once_state(merged, sub, slot_idx, accept)
            pending[slot_idx[accept]] = False

    return _once_state_to_result(
        merged,
        clues_b,
        answer_b,
        halt_loss_weight=halt_loss_weight,
        tries=tries,
        was_batched=was_batched,
    )


@torch.inference_mode()
def rollout_eval_batch(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answer: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
    halt_loss_weight: float = 1.0,
    init_seed: int | None = 0,
    max_tries: int = 1,
) -> EvalRolloutResult:
    config = config or RolloutConfig()
    clues_b, was_batched = _ensure_batched(clues)
    answer_b, _ = _ensure_batched(answer)
    b = clues_b.size(0)
    if max_tries > 1:
        return _rollout_eval_batch_multi_try(
            model,
            clues_b,
            answer_b,
            config=config,
            halt_loss_weight=halt_loss_weight,
            init_seed=init_seed,
            max_tries=max_tries,
            was_batched=was_batched,
        )

    state = _rollout_eval_batch_once(
        model, clues_b, answer_b, config=config, init_seed=init_seed
    )
    tries = torch.ones(b, dtype=torch.long, device=clues_b.device)
    return _once_state_to_result(
        state,
        clues_b,
        answer_b,
        halt_loss_weight=halt_loss_weight,
        tries=tries,
        was_batched=was_batched,
    )


@torch.inference_mode()
def rollout_solve(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
    init_seed: int | None = 0,
) -> torch.Tensor:
    config = config or RolloutConfig()
    clues_b, was_batched = _ensure_batched(clues)
    answer = clues_b.clone()
    result = rollout_eval_batch(model, clues_b, answer, config=config, init_seed=init_seed)
    pred = result.pred
    if not was_batched:
        pred = pred.squeeze(0)
    return pred


@torch.inference_mode()
def rollout_trace_batch(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
    init_seed: int | None = 0,
) -> list[PuzzleTrace]:
    """Rollout for viz; one frame per outer step: model input + clean prediction."""
    config = config or RolloutConfig()
    clues, _ = _ensure_batched(clues)
    b = clues.size(0)
    device = clues.device

    input_frames: list[list[str]] = [[] for _ in range(b)]
    pred_frames: list[list[str]] = [[] for _ in range(b)]
    out_halted = torch.zeros(b, dtype=torch.bool, device=device)
    out_steps = torch.zeros(b, dtype=torch.long, device=device)

    for step in _iter_compact_outer_rollout(
        model, clues, clues, config=config, init_seed=init_seed
    ):
        for local_i, global_i in enumerate(step.slot_idx.tolist()):
            input_frames[global_i].append(tensor_to_string(step.model_input[local_i]))
            pred_frames[global_i].append(tensor_to_string(step.pre_commit[local_i]))
        if step.done.any():
            done_idx = step.slot_idx[step.done]
            out_halted[done_idx] = step.predict_halt[step.done]
            out_steps[done_idx] = step.active_outer_count[step.done]

    return [
        PuzzleTrace(
            inputs=input_frames[i],
            predictions=pred_frames[i],
            halted=bool(out_halted[i].item()),
            outer_steps=int(out_steps[i].item()),
        )
        for i in range(b)
    ]


@torch.inference_mode()
def rollout_trace(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
    init_seed: int | None = 0,
) -> list[str]:
    clues_b, _ = _ensure_batched(clues)
    return rollout_trace_batch(model, clues_b, config=config, init_seed=init_seed)[0].states
