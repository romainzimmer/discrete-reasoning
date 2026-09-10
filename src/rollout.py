from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data import tensor_to_string
from dataset import PuzzleDataset
from amp import LOSS_DTYPE, to_loss_dtype
from ema import ema_update, memory_init, zero_ema
from encoding import decode_logits, target_mask
from model import MixerNextStateModel

DEFAULT_INNER_ITERS = 5
DEFAULT_MAX_OUTER_ITERS = 10
DEFAULT_TRANSITION_PROB = 0.5
DEFAULT_TRANSITION_NOISE_PROB = 0.0


@dataclass(frozen=True)
class RolloutConfig:
    inner_iters: int = DEFAULT_INNER_ITERS
    max_outer_iters: int = DEFAULT_MAX_OUTER_ITERS
    halt_threshold: float = 0.5
    transition_prob: float = DEFAULT_TRANSITION_PROB
    transition_noise_prob: float = DEFAULT_TRANSITION_NOISE_PROB
    curriculum_training: bool = True
    pin_gt: bool = True
    deep_supervision: bool = True
    adaptive_curriculum: bool = True
    curriculum_puzzle_acc: float = 0.0

    def __post_init__(self) -> None:
        if self.inner_iters < 1:
            raise ValueError("inner_iters must be >= 1")
        if self.max_outer_iters < 1:
            raise ValueError("max_outer_iters must be >= 1")
        if not 0.0 < self.transition_prob <= 1.0:
            raise ValueError("transition_prob must be in (0, 1]")
        if not 0.0 <= self.transition_noise_prob <= 1.0:
            raise ValueError("transition_noise_prob must be in [0, 1]")


@dataclass(frozen=True)
class _PinContext:
    pin: torch.Tensor
    pin_digit_ids: torch.Tensor

    @classmethod
    def from_state(
        cls,
        clues: torch.Tensor,
        answer: torch.Tensor,
        gt_pin: torch.Tensor,
        *,
        pin_gt: bool = True,
    ) -> _PinContext:
        clue_pin = clues > 0
        pin = clue_pin | (gt_pin if pin_gt else False)
        pin_digit_ids = torch.where(clue_pin, clues, answer)
        return cls(pin=pin, pin_digit_ids=pin_digit_ids)


@dataclass
class BatchSlotState:
    digit_id: torch.Tensor
    clues: torch.Tensor
    answer: torch.Tensor
    clue_pin: torch.Tensor
    gt_pin: torch.Tensor
    pin_ctx: _PinContext
    outer_count: torch.Tensor
    memory_embed: torch.Tensor | None = None
    ema_embed: torch.Tensor | None = None
    pending_candidate: torch.Tensor | None = None

    @classmethod
    def seed(
        cls,
        dataset: PuzzleDataset,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator,
        dim: int,
        curriculum_training: bool = True,
        pin_gt: bool = True,
        adaptive_curriculum: bool = True,
        curriculum_puzzle_acc: float = 0.0,
    ) -> BatchSlotState:
        idx = torch.randint(len(dataset), (batch_size,), generator=generator)
        clues, answers = dataset.sample(idx)
        clues = clues.to(device, non_blocking=True)
        answers = answers.to(device, non_blocking=True)
        clue_pin = clues > 0
        if curriculum_training:
            digit_id, gt_pin = _curriculum_init_digit_id(
                clues,
                answers,
                clue_pin,
                puzzle_acc=curriculum_puzzle_acc,
                adaptive=adaptive_curriculum,
            )
        else:
            digit_id = clues.clone()
            gt_pin = torch.zeros_like(clue_pin)
        pin_ctx = _PinContext.from_state(clues, answers, gt_pin, pin_gt=pin_gt)
        ema_embed = zero_ema(batch_size, dim, device)
        return cls(
            digit_id=digit_id,
            clues=clues,
            answer=answers,
            clue_pin=clue_pin,
            gt_pin=gt_pin,
            pin_ctx=pin_ctx,
            outer_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            ema_embed=ema_embed,
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


def _noise_committed_digits(
    committed: torch.Tensor,
    ctx: _PinContext,
    *,
    transition_noise_prob: float,
) -> torch.Tensor:
    """Randomly replace unpinned committed cells with digits 0-9 (incl. empty)."""
    if transition_noise_prob <= 0.0:
        return committed
    noise_mask = ~ctx.pin & (
        torch.rand(committed.shape, device=committed.device) < transition_noise_prob
    )
    noisy = torch.randint(
        0, 10, committed.shape, device=committed.device, dtype=committed.dtype
    )
    return torch.where(noise_mask, noisy, committed)


def _transition_board(
    candidate: torch.Tensor,
    ctx: _PinContext,
    prev_digit_id: torch.Tensor,
    *,
    transition_prob: float,
    transition_noise_prob: float,
) -> torch.Tensor:
    """Apply masked transition + noise; candidate is clean pred with pins reapplied."""
    candidate = torch.where(ctx.pin, ctx.pin_digit_ids, candidate)
    if transition_prob >= 1.0:
        committed = candidate
    else:
        transition_mask = ~ctx.pin & (
            torch.rand(prev_digit_id.shape, device=prev_digit_id.device) < transition_prob
        )
        committed = torch.where(ctx.pin | transition_mask, candidate, prev_digit_id)
    return _noise_committed_digits(
        committed, ctx, transition_noise_prob=transition_noise_prob
    )


def _begin_outer_step(
    digit_id: torch.Tensor,
    pin_ctx: _PinContext,
    *,
    pending_candidate: torch.Tensor | None,
    outer_count: torch.Tensor,
    transition_prob: float,
    transition_noise_prob: float,
) -> torch.Tensor:
    """Apply deferred transition+noise from the prior outer step before inner loop."""
    if pending_candidate is None:
        return digit_id
    commit_mask = (outer_count > 0).view(-1, 1, 1)
    if not commit_mask.any():
        return digit_id
    committed = _transition_board(
        pending_candidate,
        pin_ctx,
        digit_id,
        transition_prob=transition_prob,
        transition_noise_prob=transition_noise_prob,
    )
    return torch.where(commit_mask, committed, digit_id)


def _curriculum_init_digit_id(
    clues: torch.Tensor,
    answer: torch.Tensor,
    clue_pin: torch.Tensor,
    *,
    puzzle_acc: float = 0.0,
    adaptive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-only puzzle entry: partial GT reveal; unrevealed non-clue cells stay empty."""
    digit_id = clues.clone()
    non_clue = ~clue_pin
    b, device = clues.size(0), clues.device

    p_gt = torch.rand(b, device=device)
    if adaptive:
        upper = max(0.0, 1.0 - puzzle_acc)
        p_gt = p_gt * upper
    reveal = non_clue & (torch.rand(clues.shape, device=device) < p_gt.view(b, 1, 1))
    digit_id = torch.where(reveal, answer, digit_id)
    return digit_id, reveal


def _inner_loop(
    model: MixerNextStateModel,
    digit_id: torch.Tensor,
    clue_pin: torch.Tensor,
    inner_iters: int,
    *,
    memory_embed: torch.Tensor | None,
    ema_embed: torch.Tensor | None,
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
    loop_ema = ema_embed
    step_outputs: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if collect_steps else None
    for _ in range(inner_iters):
        if with_grad:
            out = model(
                input_embed=input_embed,
                cell_embed=cell_embed,
                ema_embed=loop_ema,
            )
        else:
            with torch.no_grad():
                out = model(
                    input_embed=input_embed,
                    cell_embed=cell_embed,
                    ema_embed=loop_ema,
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
        state.pin_ctx,
        pending_candidate=state.pending_candidate,
        outer_count=state.outer_count,
        transition_prob=config.transition_prob,
        transition_noise_prob=config.transition_noise_prob,
    )
    if config.deep_supervision:
        step_outputs, final_cell_embed = _inner_loop(
            model,
            state.digit_id,
            state.clue_pin,
            config.inner_iters,
            memory_embed=state.memory_embed,
            ema_embed=state.ema_embed,
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
            ema_embed=state.ema_embed,
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
    if state.ema_embed is not None:
        state.ema_embed = ema_update(
            state.ema_embed,
            final_cell_embed,
            float(model.ema_alpha().detach().item()),
        )
    state.outer_count = state.outer_count + 1
    solved = halt_target > 0.5
    done = (predict_halt & solved) | (state.outer_count >= config.max_outer_iters)
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
    pin_gt: bool = True,
    adaptive_curriculum: bool = True,
    curriculum_puzzle_acc: float = 0.0,
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
        new_digit_id, new_gt_pin = _curriculum_init_digit_id(
            new_clues,
            new_answers,
            new_clue_pin,
            puzzle_acc=curriculum_puzzle_acc,
            adaptive=adaptive_curriculum,
        )
    else:
        new_digit_id = new_clues
        new_gt_pin = torch.zeros_like(new_clue_pin)
    state.digit_id = torch.where(done_mask, new_digit_id, state.digit_id)
    state.clues = torch.where(done_mask, new_clues, state.clues)
    state.answer = torch.where(done_mask, new_answers, state.answer)
    state.clue_pin = state.clues > 0
    state.gt_pin = torch.where(done_mask, new_gt_pin, state.gt_pin)
    state.pin_ctx = _PinContext.from_state(
        state.clues, state.answer, state.gt_pin, pin_gt=pin_gt
    )
    state.outer_count = torch.where(done, torch.zeros_like(state.outer_count), state.outer_count)
    done_mask_mem = done.view(b, 1, 1, 1)
    if state.memory_embed is not None:
        new_memory = zero_ema(b, dim, device)
        state.memory_embed = torch.where(done_mask_mem, new_memory, state.memory_embed)
    if state.ema_embed is not None:
        new_ema = zero_ema(b, dim, device)
        state.ema_embed = torch.where(done_mask_mem, new_ema, state.ema_embed)
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
):
    clues, _ = _ensure_batched(clues)
    answer, _ = _ensure_batched(answer)
    b = clues.size(0)
    device = clues.device

    digit_id = clues.clone()
    clue_pin = clues > 0
    gt_pin = torch.zeros_like(clue_pin)

    slot_idx = torch.arange(b, device=device)
    active_digit_id = digit_id
    active_clues = clues
    active_answer = answer
    active_clue_pin = clue_pin
    active_outer_count = torch.zeros(b, dtype=torch.long, device=device)
    active_ctx = _PinContext.from_state(clues, answer, gt_pin)
    active_memory_embed: torch.Tensor | None = None
    active_ema_embed = zero_ema(b, model.dim, device)
    pending_candidate: torch.Tensor | None = None

    while slot_idx.numel() > 0:
        active_gt_pin = gt_pin[slot_idx]
        active_digit_id = _begin_outer_step(
            active_digit_id,
            active_ctx,
            pending_candidate=_pending_for_active(pending_candidate, slot_idx),
            outer_count=active_outer_count,
            transition_prob=config.transition_prob,
            transition_noise_prob=config.transition_noise_prob,
        )
        model_input = active_digit_id
        logits, halt_logit, final_cell_embed = _inner_loop(
            model,
            active_digit_id,
            active_clue_pin,
            config.inner_iters,
            memory_embed=active_memory_embed,
            ema_embed=active_ema_embed,
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
        if active_ema_embed is not None:
            active_ema_embed = ema_update(
                active_ema_embed,
                final_cell_embed,
                float(model.ema_alpha().detach().item()),
            )
        active_outer_count = active_outer_count + 1
        done = predict_halt | (active_outer_count >= config.max_outer_iters)

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
        active_gt_pin = active_gt_pin[keep]
        active_outer_count = active_outer_count[keep]
        active_memory_embed = active_memory_embed[keep] if active_memory_embed is not None else None
        if active_ema_embed is not None:
            active_ema_embed = active_ema_embed[keep]
        active_ctx = _PinContext.from_state(active_clues, active_answer, active_gt_pin)


@torch.inference_mode()
def rollout_eval_batch(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answer: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
    halt_loss_weight: float = 1.0,
) -> EvalRolloutResult:
    config = config or RolloutConfig()
    clues_b, was_batched = _ensure_batched(clues)
    answer_b, _ = _ensure_batched(answer)
    b = clues_b.size(0)
    device = clues_b.device
    clue_pin = clues_b > 0

    out_pred = clues_b.clone()
    out_steps = torch.zeros(b, dtype=torch.long, device=device)
    out_halted = torch.zeros(b, dtype=torch.bool, device=device)
    final_logits = clues_b.new_zeros((b, 9, 9, 10), dtype=LOSS_DTYPE)
    final_halt_logit = clues_b.new_zeros((b,), dtype=LOSS_DTYPE)
    halt_correct_rounds = torch.zeros((), device=device, dtype=torch.long)
    halt_total_rounds = 0

    for step in _iter_compact_outer_rollout(model, clues_b, answer_b, config=config):
        halt_target_round = _halt_target(step.pre_commit, step.active_answer)
        halt_correct_rounds += (step.predict_halt == (halt_target_round > 0.5)).sum()
        halt_total_rounds += step.pre_commit.size(0)
        final_logits[step.slot_idx] = to_loss_dtype(step.logits)
        final_halt_logit[step.slot_idx] = to_loss_dtype(step.halt_logit)
        done_idx = step.slot_idx[step.done]
        out_pred[done_idx] = step.pre_commit[step.done]
        out_steps[done_idx] = step.active_outer_count[step.done]
        out_halted[done_idx] = step.predict_halt[step.done]

    pre_commit_final = predict_grid(final_logits, clues_b)
    halt_target = _halt_target(pre_commit_final, answer_b)
    cell_loss, halt_loss, loss = _compute_losses(
        final_logits,
        final_halt_logit,
        clue_pin=clue_pin,
        answer=answer_b,
        halt_target=halt_target,
        halt_loss_weight=halt_loss_weight,
    )

    if not was_batched:
        return EvalRolloutResult(
            pred=out_pred.squeeze(0),
            outer_steps=out_steps.squeeze(0),
            halted=out_halted.squeeze(0),
            loss=loss,
            cell_loss=cell_loss,
            halt_loss=halt_loss,
            halt_target=halt_target.squeeze(0),
            halt_logit=final_halt_logit.squeeze(0),
            halt_correct_rounds=int(halt_correct_rounds.item()),
            halt_total_rounds=halt_total_rounds,
        )
    return EvalRolloutResult(
        pred=out_pred,
        outer_steps=out_steps,
        halted=out_halted,
        loss=loss,
        cell_loss=cell_loss,
        halt_loss=halt_loss,
        halt_target=halt_target,
        halt_logit=final_halt_logit,
        halt_correct_rounds=int(halt_correct_rounds.item()),
        halt_total_rounds=halt_total_rounds,
    )


@torch.inference_mode()
def rollout_solve(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
) -> torch.Tensor:
    config = config or RolloutConfig()
    clues_b, was_batched = _ensure_batched(clues)
    answer = clues_b.clone()
    result = rollout_eval_batch(model, clues_b, answer, config=config)
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

    for step in _iter_compact_outer_rollout(model, clues, clues, config=config):
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
) -> list[str]:
    clues_b, _ = _ensure_batched(clues)
    return rollout_trace_batch(model, clues_b, config=config)[0].states
