from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data import tensor_to_string
from dataset import PuzzleTensorCache
from encoding import decode_logits, target_mask
from model import MixerNextStateModel

DEFAULT_INNER_ITERS = 5
DEFAULT_MAX_OUTER_ITERS = 10


@dataclass(frozen=True)
class RolloutConfig:
    inner_iters: int = DEFAULT_INNER_ITERS
    max_outer_iters: int = DEFAULT_MAX_OUTER_ITERS
    halt_threshold: float = 0.5
    rollout_mask_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.inner_iters < 1:
            raise ValueError("inner_iters must be >= 1")
        if self.max_outer_iters < 1:
            raise ValueError("max_outer_iters must be >= 1")
        if not 0.0 <= self.rollout_mask_prob <= 1.0:
            raise ValueError("rollout_mask_prob must be in [0, 1]")


@dataclass(frozen=True)
class _ClueContext:
    clue_digit_ids: torch.Tensor
    clue_pin: torch.Tensor

    @classmethod
    def from_clues(cls, clues: torch.Tensor) -> _ClueContext:
        clue_pin = clues > 0
        return cls(clue_digit_ids=clues, clue_pin=clue_pin)


@dataclass
class BatchSlotState:
    digit_id: torch.Tensor
    clues: torch.Tensor
    answer: torch.Tensor
    clue_pin: torch.Tensor
    outer_count: torch.Tensor

    @classmethod
    def seed(
        cls,
        cache: PuzzleTensorCache,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator,
    ) -> BatchSlotState:
        idx = torch.randint(cache.clues.size(0), (batch_size,), generator=generator)
        clues = cache.clues[idx].to(device, non_blocking=True)
        answers = cache.answers[idx].to(device, non_blocking=True)
        clue_pin = clues > 0
        return cls(
            digit_id=clues.clone(),
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


@dataclass
class PuzzleTrace:
    states: list[str]
    halted: bool
    outer_steps: int


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
    cell_loss = _compute_cell_loss(logits, clue_pin=clue_pin, answer=answer)
    halt_loss = _compute_halt_loss(halt_logit, halt_target)
    total_loss = cell_loss + halt_loss_weight * halt_loss
    return cell_loss, halt_loss, total_loss


def _outer_commit(logits: torch.Tensor, ctx: _ClueContext) -> torch.Tensor:
    decoded = decode_logits(logits).detach()
    return torch.where(ctx.clue_pin, ctx.clue_digit_ids, decoded)


def _masked_digits_for_inner_loop(
    digit_id: torch.Tensor,
    clue_pin: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """Randomly mask committed digits to empty for inner-loop input only (clues untouched)."""
    if prob <= 0.0:
        return digit_id
    mutable = ~clue_pin
    mask = mutable & (torch.rand(digit_id.shape, device=digit_id.device) < prob)
    return torch.where(mask, torch.zeros_like(digit_id), digit_id)


def _inner_loop(
    model: MixerNextStateModel,
    digit_id: torch.Tensor,
    clue_pin: torch.Tensor,
    inner_iters: int,
    *,
    rollout_mask_prob: float = 0.0,
    with_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loop_digit_id = _masked_digits_for_inner_loop(digit_id, clue_pin, rollout_mask_prob)
    input_embed = model.encode_input(loop_digit_id, clue_pin)
    cell_embed: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    halt_logit: torch.Tensor | None = None
    for _ in range(inner_iters):
        if with_grad:
            out = model(input_embed=input_embed, cell_embed=cell_embed)
        else:
            with torch.no_grad():
                out = model(input_embed=input_embed, cell_embed=cell_embed)
        logits = out.logits
        halt_logit = out.halt_logit
        cell_embed = out.cell_embed
    assert logits is not None
    assert halt_logit is not None
    return logits, halt_logit, cell_embed


def rollout_train_step(
    model: MixerNextStateModel,
    state: BatchSlotState,
    config: RolloutConfig,
    *,
    halt_loss_weight: float = 0.1,
) -> RolloutResult:
    if not model.training:
        raise ValueError("rollout_train_step requires model.training")
    ctx = _ClueContext.from_clues(state.clues)
    logits, halt_logit, _ = _inner_loop(
        model,
        state.digit_id,
        state.clue_pin,
        config.inner_iters,
        rollout_mask_prob=config.rollout_mask_prob,
        with_grad=True,
    )
    pre_commit = predict_grid(logits, state.clues)
    halt_target = _halt_target(pre_commit, state.answer)
    predict_halt = _predict_halt(halt_logit, halt_threshold=config.halt_threshold)
    cell_loss, halt_loss, loss = _compute_losses(
        logits,
        halt_logit,
        clue_pin=ctx.clue_pin,
        answer=state.answer,
        halt_target=halt_target,
        halt_loss_weight=halt_loss_weight,
    )
    loss.backward()
    state.digit_id = _outer_commit(logits, ctx)
    state.outer_count = state.outer_count + 1
    done = predict_halt | (state.outer_count >= config.max_outer_iters)
    return RolloutResult(
        loss=loss.detach(),
        cell_loss=cell_loss.detach(),
        halt_loss=halt_loss.detach(),
        pred=pre_commit.detach(),
        done=done,
        halted=predict_halt,
        halt_target=halt_target.detach(),
        halt_logit=halt_logit.detach(),
    )


def refill_done_slots(
    state: BatchSlotState,
    done: torch.Tensor,
    cache: PuzzleTensorCache,
    *,
    generator: torch.Generator,
) -> None:
    k = int(done.sum().item())
    if k == 0:
        return
    device = state.digit_id.device
    idx = torch.randint(cache.clues.size(0), (k,), generator=generator)
    clues = cache.clues[idx].to(device, non_blocking=True)
    answers = cache.answers[idx].to(device, non_blocking=True)
    clue_pin = clues > 0
    state.digit_id[done] = clues
    state.clues[done] = clues
    state.answer[done] = answers
    state.clue_pin[done] = clue_pin
    state.outer_count[done] = 0


@torch.inference_mode()
def rollout_eval_batch(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answer: torch.Tensor,
    *,
    config: RolloutConfig | None = None,
    halt_loss_weight: float = 0.1,
) -> EvalRolloutResult:
    config = config or RolloutConfig()
    clues_b, was_batched = _ensure_batched(clues)
    answer_b, _ = _ensure_batched(answer)
    b = clues_b.size(0)
    device = clues_b.device

    digit_id = clues_b.clone()
    clue_pin = clues_b > 0
    ctx = _ClueContext.from_clues(clues_b)
    outer_count = torch.zeros(b, dtype=torch.long, device=device)

    out_pred = digit_id.clone()
    out_steps = torch.zeros(b, dtype=torch.long, device=device)
    out_halted = torch.zeros(b, dtype=torch.bool, device=device)
    final_logits = digit_id.new_zeros((b, 9, 9, 10), dtype=torch.float32)
    final_halt_logit = digit_id.new_zeros((b,), dtype=torch.float32)

    slot_idx = torch.arange(b, device=device)
    active_digit_id = digit_id
    active_clues = clues_b
    active_answer = answer_b
    active_clue_pin = clue_pin
    active_outer_count = outer_count
    active_ctx = ctx
    halt_correct_rounds = 0
    halt_total_rounds = 0

    while slot_idx.numel() > 0:
        logits, halt_logit, _ = _inner_loop(
            model,
            active_digit_id,
            active_clue_pin,
            config.inner_iters,
            rollout_mask_prob=config.rollout_mask_prob,
            with_grad=False,
        )
        pre_commit = predict_grid(logits, active_clues)
        halt_target_round = _halt_target(pre_commit, active_answer)
        predict_halt = _predict_halt(halt_logit, halt_threshold=config.halt_threshold)
        halt_correct_rounds += int((predict_halt == (halt_target_round > 0.5)).sum().item())
        halt_total_rounds += pre_commit.size(0)
        committed = _outer_commit(logits, active_ctx)
        active_outer_count = active_outer_count + 1
        done = predict_halt | (active_outer_count >= config.max_outer_iters)

        final_logits[slot_idx] = logits
        final_halt_logit[slot_idx] = halt_logit

        if done.any():
            done_idx = slot_idx[done]
            out_pred[done_idx] = pre_commit[done]
            out_steps[done_idx] = active_outer_count[done]
            out_halted[done_idx] = predict_halt[done]

        keep = ~done
        if not keep.any():
            break
        slot_idx = slot_idx[keep]
        active_digit_id = committed[keep]
        active_clues = active_clues[keep]
        active_answer = active_answer[keep]
        active_clue_pin = active_clue_pin[keep]
        active_outer_count = active_outer_count[keep]
        active_ctx = _ClueContext.from_clues(active_clues)

    pre_commit_final = predict_grid(final_logits, clues_b)
    halt_target = _halt_target(pre_commit_final, answer_b)
    cell_loss, halt_loss, loss = _compute_losses(
        final_logits,
        final_halt_logit,
        clue_pin=ctx.clue_pin,
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
            halt_correct_rounds=halt_correct_rounds,
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
        halt_correct_rounds=halt_correct_rounds,
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
    """Rollout for viz; one frame per outer argmax commit, per puzzle."""
    config = config or RolloutConfig()
    clues, _ = _ensure_batched(clues)
    b = clues.size(0)
    device = clues.device

    digit_id = clues.clone()
    clue_pin = clues > 0
    ctx = _ClueContext.from_clues(clues)
    outer_count = torch.zeros(b, dtype=torch.long, device=device)
    trajectories: list[list[str]] = [[tensor_to_string(clues[i])] for i in range(b)]
    out_halted = torch.zeros(b, dtype=torch.bool, device=device)
    out_steps = torch.zeros(b, dtype=torch.long, device=device)

    slot_idx = torch.arange(b, device=device)
    active_digit_id = digit_id
    active_clues = clues
    active_clue_pin = clue_pin
    active_outer_count = outer_count
    active_ctx = ctx

    while slot_idx.numel() > 0:
        logits, halt_logit, _ = _inner_loop(
            model,
            active_digit_id,
            active_clue_pin,
            config.inner_iters,
            rollout_mask_prob=config.rollout_mask_prob,
            with_grad=False,
        )
        committed = _outer_commit(logits, active_ctx)
        predict_halt = _predict_halt(halt_logit, halt_threshold=config.halt_threshold)
        active_outer_count = active_outer_count + 1
        done = predict_halt | (active_outer_count >= config.max_outer_iters)

        for local_i, global_i in enumerate(slot_idx.tolist()):
            trajectories[global_i].append(tensor_to_string(committed[local_i]))

        if done.any():
            done_idx = slot_idx[done]
            out_halted[done_idx] = predict_halt[done]
            out_steps[done_idx] = active_outer_count[done]

        keep = ~done
        if not keep.any():
            break
        slot_idx = slot_idx[keep]
        active_digit_id = committed[keep]
        active_clues = active_clues[keep]
        active_clue_pin = active_clue_pin[keep]
        active_outer_count = active_outer_count[keep]
        active_ctx = _ClueContext.from_clues(active_clues)

    return [
        PuzzleTrace(
            states=trajectories[i],
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
