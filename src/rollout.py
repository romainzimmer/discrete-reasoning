from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from curriculum import CurriculumState
from data import tensor_to_string
from dataset import PuzzleDataset
from amp import LOSS_DTYPE, to_loss_dtype
from memory import memory_init, zero_memory
from encoding import decode_logits, target_mask
from model import MixerNextStateModel

DEFAULT_INNER_ITERS = 6
DEFAULT_MAX_OUTER_ITERS = 10


@dataclass(frozen=True)
class RolloutConfig:
    inner_iters: int = DEFAULT_INNER_ITERS
    max_outer_iters: int = DEFAULT_MAX_OUTER_ITERS
    halt_after_stable_outer_steps: int = 3
    gt_reveal: bool = True
    random_gt_reveal_p_gt: bool = False
    deep_supervision: bool = True
    random_init: bool = False

    def __post_init__(self) -> None:
        if self.random_gt_reveal_p_gt and not self.gt_reveal:
            raise ValueError("random_gt_reveal_p_gt requires gt_reveal")
        if self.inner_iters < 1:
            raise ValueError("inner_iters must be >= 1")
        if self.max_outer_iters < 1:
            raise ValueError("max_outer_iters must be >= 1")
        if self.halt_after_stable_outer_steps < 2:
            raise ValueError("halt_after_stable_outer_steps must be >= 2")


@dataclass
class BatchSlotState:
    digit_id: torch.Tensor
    clues: torch.Tensor
    answer: torch.Tensor
    clue_pin: torch.Tensor
    outer_count: torch.Tensor
    rating_group: torch.Tensor
    memory_embed: torch.Tensor | None = None
    pending_candidate: torch.Tensor | None = None
    commit_prior: torch.Tensor | None = None
    commit_streak: torch.Tensor | None = None

    @classmethod
    def seed(
        cls,
        dataset: PuzzleDataset,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator,
        gt_reveal: bool = True,
        random_gt_reveal_p_gt: bool = False,
        gt_reveal_p_gt_caps: torch.Tensor | None = None,
        random_init: bool = False,
    ) -> BatchSlotState:
        idx = torch.randint(len(dataset), (batch_size,), generator=generator)
        clues, answers, rating_groups = dataset.sample(idx)
        clues = clues.to(device, non_blocking=True)
        answers = answers.to(device, non_blocking=True)
        rating_group = rating_groups.to(device, non_blocking=True)
        clue_pin = clues > 0
        digit_id = _gt_reveal_digit_id_for_seed_refill(
            clues,
            answers,
            clue_pin,
            rating_group=rating_group,
            gt_reveal=gt_reveal,
            random_gt_reveal_p_gt=random_gt_reveal_p_gt,
            gt_reveal_p_gt_caps=gt_reveal_p_gt_caps,
            random_init=random_init,
            generator=generator,
        )
        return cls(
            digit_id=digit_id,
            clues=clues,
            answer=answers,
            clue_pin=clue_pin,
            rating_group=rating_group,
            outer_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )


@dataclass
class RolloutResult:
    loss: torch.Tensor
    cell_loss: torch.Tensor | None = None
    pred: torch.Tensor | None = None
    done: torch.Tensor | None = None
    halted: torch.Tensor | None = None


@dataclass
class EvalRolloutResult:
    pred: torch.Tensor
    outer_steps: torch.Tensor
    halted: torch.Tensor
    loss: torch.Tensor
    cell_loss: torch.Tensor
    tries: torch.Tensor


@dataclass
class _OnceEvalState:
    pred: torch.Tensor
    outer_steps: torch.Tensor
    halted: torch.Tensor
    final_logits: torch.Tensor


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


def _grid_solved(pre_commit: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
    return (pre_commit == answer).view(pre_commit.size(0), -1).all(dim=1)


def _update_commit_stability(
    pre_commit: torch.Tensor,
    commit_prior: torch.Tensor,
    commit_streak: torch.Tensor,
    *,
    halt_after_stable_outer_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched stability update. commit_streak==0 means no prior snapshot for that row."""
    uninitialized = commit_streak == 0
    same = torch.zeros(pre_commit.size(0), dtype=torch.bool, device=pre_commit.device)
    if (~uninitialized).any():
        init_idx = (~uninitialized).nonzero(as_tuple=True)[0]
        n_init = init_idx.numel()
        same[init_idx] = (pre_commit[init_idx] == commit_prior[init_idx]).view(n_init, -1).all(
            dim=1
        )
    new_streak = torch.where(
        uninitialized,
        torch.ones_like(commit_streak),
        torch.where(same, commit_streak + 1, torch.ones_like(commit_streak)),
    )
    needs_prior_write = uninitialized | ~same
    if needs_prior_write.any():
        commit_prior = commit_prior.clone()
        commit_prior[needs_prior_write] = pre_commit[needs_prior_write]
    implicit_halt = new_streak >= halt_after_stable_outer_steps
    return commit_prior, new_streak, implicit_halt


def _update_commit_stability_row(
    pre_commit: torch.Tensor,
    commit_prior: torch.Tensor | None,
    commit_streak: int,
    *,
    halt_after_stable_outer_steps: int,
) -> tuple[torch.Tensor, int, bool]:
    if commit_streak == 0 or commit_prior is None:
        return pre_commit.clone(), 1, 1 >= halt_after_stable_outer_steps
    if torch.equal(pre_commit, commit_prior):
        commit_streak += 1
    else:
        commit_prior = pre_commit.clone()
        commit_streak = 1
    return commit_prior, commit_streak, commit_streak >= halt_after_stable_outer_steps


def _train_outer_done(
    implicit_halt: torch.Tensor,
    solved: torch.Tensor,
    outer_count: torch.Tensor,
    max_outer_iters: int,
) -> torch.Tensor:
    """Training: stop a slot on stable halt only when the grid is fully correct."""
    return (implicit_halt & solved) | (outer_count >= max_outer_iters)


def _eval_outer_done(
    implicit_halt: torch.Tensor,
    outer_count: torch.Tensor,
    max_outer_iters: int,
) -> torch.Tensor:
    """Eval: stop on stable halt alone; answer is used for metrics only after rollout."""
    return implicit_halt | (outer_count >= max_outer_iters)


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


def _compute_deep_supervision_cell_losses(
    step_outputs: list[torch.Tensor],
    *,
    clue_pin: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    cell_losses = [
        _compute_cell_loss(to_loss_dtype(logits), clue_pin=clue_pin, answer=answer)
        for logits in step_outputs
    ]
    return torch.stack(cell_losses).mean()


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


def _rand(
    size,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if generator is None:
        return torch.rand(size, device=device, dtype=dtype)
    if generator.device.type == "cpu" and device.type != "cpu":
        return torch.rand(size, dtype=dtype, generator=generator).to(device)
    return torch.rand(size, device=device, dtype=dtype, generator=generator)


def _randint(
    low: int,
    high: int,
    size,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.long,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if generator is None:
        return torch.randint(low, high, size, device=device, dtype=dtype)
    if generator.device.type == "cpu" and device.type != "cpu":
        return torch.randint(low, high, size, dtype=dtype, generator=generator).to(device)
    return torch.randint(low, high, size, device=device, dtype=dtype, generator=generator)


def _random_fill_unpinned(
    digit_id: torch.Tensor,
    unpinned: torch.Tensor,
    *,
    init_seed: int | None = None,
    generator: torch.Generator | None = None,
    random_init: bool = False,
) -> torch.Tensor:
    """Fill unpinned cells with uniform random digits 0-9 (0 = empty)."""
    if not random_init or not unpinned.any():
        return digit_id
    if init_seed is None:
        random_digits = _randint(
            0,
            10,
            digit_id.shape,
            device=digit_id.device,
            dtype=digit_id.dtype,
            generator=generator,
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
    generator: torch.Generator | None = None,
    random_init: bool = False,
) -> torch.Tensor:
    """Clues pinned; other cells get random digits 0-9 or stay empty."""
    return _random_fill_unpinned(
        clues.clone(),
        ~clue_pin,
        init_seed=init_seed,
        generator=generator,
        random_init=random_init,
    )


def _sample_uniform_p_gt(
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return _rand((batch_size,), device=device, generator=generator)


def _sample_uniform_p_gt_up_to(
    caps: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample p_gt ~ U[0, cap_i] for each slot."""
    return _rand(caps.shape, device=caps.device, generator=generator) * caps


def _gt_reveal_p_gt_for_slots(
    *,
    rating_group: torch.Tensor,
    device: torch.device,
    gt_reveal: bool,
    random_gt_reveal_p_gt: bool,
    gt_reveal_p_gt_caps: torch.Tensor | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    if not gt_reveal:
        return None
    if random_gt_reveal_p_gt:
        return _sample_uniform_p_gt(rating_group.size(0), device, generator=generator)
    if gt_reveal_p_gt_caps is None:
        gt_reveal_p_gt_caps = CurriculumState.default().p_gt_cap_tensor(device)
    caps_by_slot = gt_reveal_p_gt_caps[rating_group]
    return _sample_uniform_p_gt_up_to(caps_by_slot, generator=generator)


def _gt_reveal_digit_id_for_seed_refill(
    clues: torch.Tensor,
    answers: torch.Tensor,
    clue_pin: torch.Tensor,
    *,
    rating_group: torch.Tensor,
    gt_reveal: bool,
    random_gt_reveal_p_gt: bool = False,
    gt_reveal_p_gt_caps: torch.Tensor | None = None,
    random_init: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not gt_reveal:
        return _init_digit_id_from_clues(
            clues, clue_pin, generator=generator, random_init=random_init
        )
    p_gt = _gt_reveal_p_gt_for_slots(
        rating_group=rating_group,
        device=clues.device,
        gt_reveal=gt_reveal,
        random_gt_reveal_p_gt=random_gt_reveal_p_gt,
        gt_reveal_p_gt_caps=gt_reveal_p_gt_caps,
        generator=generator,
    )
    assert p_gt is not None
    return _gt_reveal_init_digit_id(
        clues,
        answers,
        clue_pin,
        p_gt=p_gt,
        random_init=random_init,
        generator=generator,
    )


def _gt_reveal_init_digit_id(
    clues: torch.Tensor,
    answer: torch.Tensor,
    clue_pin: torch.Tensor,
    *,
    p_gt: float | torch.Tensor = 0.25,
    random_init: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Training-only puzzle entry: partial GT reveal; unrevealed non-clue cells are random or empty."""
    digit_id = clues.clone()
    non_clue = ~clue_pin
    if isinstance(p_gt, torch.Tensor):
        if p_gt.dim() != 1:
            raise ValueError("per-puzzle p_gt must have shape (B,)")
        reveal_p = p_gt.to(device=clues.device, dtype=torch.float32).view(-1, 1, 1)
    else:
        reveal_p = p_gt
    reveal = non_clue & (
        _rand(clues.shape, device=clues.device, generator=generator) < reveal_p
    )
    digit_id = torch.where(reveal, answer, digit_id)
    return _random_fill_unpinned(
        digit_id,
        non_clue & ~reveal,
        generator=generator,
        random_init=random_init,
    )


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
    tuple[torch.Tensor, torch.Tensor]
    | tuple[list[torch.Tensor], torch.Tensor]
):
    input_embed = model.encode_input(digit_id, clue_pin)
    cell_embed: torch.Tensor | None = memory_embed
    logits: torch.Tensor | None = None
    step_outputs: list[torch.Tensor] | None = [] if collect_steps else None
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
        cell_embed = out.cell_embed
        if step_outputs is not None:
            step_outputs.append(logits)
    assert logits is not None
    assert cell_embed is not None
    if step_outputs is not None:
        return step_outputs, cell_embed
    return logits, cell_embed


def rollout_train_step(
    model: MixerNextStateModel,
    state: BatchSlotState,
    config: RolloutConfig,
    *,
    backward: bool = True,
) -> RolloutResult:
    if not model.training:
        raise ValueError("rollout_train_step requires model.training")
    b = state.digit_id.size(0)
    device = state.digit_id.device
    if state.commit_prior is None:
        state.commit_prior = state.digit_id.new_zeros((b, 9, 9))
    if state.commit_streak is None:
        state.commit_streak = torch.zeros(b, dtype=torch.long, device=device)
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
        logits = step_outputs[-1]
        cell_loss = _compute_deep_supervision_cell_losses(
            step_outputs,
            clue_pin=state.clue_pin,
            answer=state.answer,
        )
        loss = cell_loss
    else:
        logits, final_cell_embed = _inner_loop(
            model,
            state.digit_id,
            state.clue_pin,
            config.inner_iters,
            memory_embed=state.memory_embed,
            with_grad=True,
        )
        cell_loss = _compute_cell_loss(
            to_loss_dtype(logits),
            clue_pin=state.clue_pin,
            answer=state.answer,
        )
        loss = cell_loss
    pred = predict_grid(logits, state.clues).detach()
    state.commit_prior, state.commit_streak, implicit_halt = _update_commit_stability(
        pred,
        state.commit_prior,
        state.commit_streak,
        halt_after_stable_outer_steps=config.halt_after_stable_outer_steps,
    )
    if backward:
        loss.backward()
    state.pending_candidate = pred
    state.memory_embed = memory_init(final_cell_embed)
    state.outer_count = state.outer_count + 1
    solved = _grid_solved(pred, state.answer)
    done = _train_outer_done(
        implicit_halt,
        solved,
        state.outer_count,
        config.max_outer_iters,
    )
    halted = implicit_halt & solved
    return RolloutResult(
        loss=loss.detach() if backward else loss,
        cell_loss=cell_loss.detach(),
        pred=pred,
        done=done,
        halted=halted,
    )


def refill_done_slots(
    state: BatchSlotState,
    done: torch.Tensor,
    dataset: PuzzleDataset,
    *,
    generator: torch.Generator,
    dim: int,
    gt_reveal: bool = True,
    random_gt_reveal_p_gt: bool = False,
    gt_reveal_p_gt_caps: torch.Tensor | None = None,
    random_init: bool = False,
) -> None:
    if not done.any():
        return
    b = done.size(0)
    device = state.digit_id.device
    done_flat = done.nonzero(as_tuple=True)[0]
    n_done = int(done_flat.numel())
    idx = torch.randint(len(dataset), (n_done,), generator=generator)
    new_clues, new_answers, new_rating_groups = dataset.sample(idx)
    new_clues = new_clues.to(device, non_blocking=True)
    new_answers = new_answers.to(device, non_blocking=True)
    new_rating_groups = new_rating_groups.to(device, non_blocking=True)
    new_clue_pin = new_clues > 0
    new_digit_id = _gt_reveal_digit_id_for_seed_refill(
        new_clues,
        new_answers,
        new_clue_pin,
        rating_group=new_rating_groups,
        gt_reveal=gt_reveal,
        random_gt_reveal_p_gt=random_gt_reveal_p_gt,
        gt_reveal_p_gt_caps=gt_reveal_p_gt_caps,
        random_init=random_init,
        generator=generator,
    )
    state.digit_id[done_flat] = new_digit_id
    state.clues[done_flat] = new_clues
    state.answer[done_flat] = new_answers
    state.rating_group[done_flat] = new_rating_groups
    state.clue_pin = state.clues > 0
    state.outer_count[done_flat] = 0
    if state.memory_embed is not None:
        state.memory_embed[done_flat] = zero_memory(n_done, dim, device)
    if state.pending_candidate is not None:
        if done.all():
            state.pending_candidate = None
        else:
            state.pending_candidate = state.pending_candidate.clone()
            state.pending_candidate[done_flat] = 0
    if state.commit_streak is not None:
        state.commit_streak = state.commit_streak.clone()
        state.commit_streak[done_flat] = 0


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
    implicit_halt: torch.Tensor
    logits: torch.Tensor
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
    digit_id = _init_digit_id_from_clues(
        clues, clue_pin, init_seed=init_seed, random_init=config.random_init
    )

    slot_idx = torch.arange(b, device=device)
    active_digit_id = digit_id
    active_clues = clues
    active_answer = answer
    active_clue_pin = clue_pin
    active_outer_count = torch.zeros(b, dtype=torch.long, device=device)
    active_memory_embed: torch.Tensor | None = None
    pending_candidate: torch.Tensor | None = None
    commit_prior = digit_id.new_zeros((b, 9, 9))
    commit_streak = torch.zeros(b, dtype=torch.long, device=device)

    while slot_idx.numel() > 0:
        active_digit_id = _begin_outer_step(
            active_digit_id,
            pending_candidate=_pending_for_active(pending_candidate, slot_idx),
            outer_count=active_outer_count,
        )
        model_input = active_digit_id
        logits, final_cell_embed = _inner_loop(
            model,
            active_digit_id,
            active_clue_pin,
            config.inner_iters,
            memory_embed=active_memory_embed,
            with_grad=False,
        )
        pre_commit = predict_grid(logits, active_clues)
        prior_active = commit_prior[slot_idx]
        streak_active = commit_streak[slot_idx]
        prior_active, streak_active, implicit_halt = _update_commit_stability(
            pre_commit,
            prior_active,
            streak_active,
            halt_after_stable_outer_steps=config.halt_after_stable_outer_steps,
        )
        commit_prior[slot_idx] = prior_active
        commit_streak[slot_idx] = streak_active
        pending_candidate = _store_pending_candidate(
            pending_candidate,
            slot_idx,
            pre_commit,
            batch_size=b,
            like=digit_id,
        )
        active_memory_embed = memory_init(final_cell_embed)
        active_outer_count = active_outer_count + 1
        done = _eval_outer_done(implicit_halt, active_outer_count, config.max_outer_iters)

        yield _CompactOuterStep(
            slot_idx=slot_idx,
            model_input=model_input,
            pre_commit=pre_commit,
            implicit_halt=implicit_halt,
            logits=logits,
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

    for step in _iter_compact_outer_rollout(
        model, clues_b, answer_b, config=config, init_seed=init_seed
    ):
        final_logits[step.slot_idx] = to_loss_dtype(step.logits)
        if step.done.any():
            done_idx = step.slot_idx[step.done]
            out_pred[done_idx] = step.pre_commit[step.done]
            out_steps[done_idx] = step.active_outer_count[step.done]
            out_halted[done_idx] = step.implicit_halt[step.done]

    return _OnceEvalState(
        pred=out_pred,
        outer_steps=out_steps,
        halted=out_halted,
        final_logits=final_logits,
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


def _once_state_to_result(
    state: _OnceEvalState,
    clues_b: torch.Tensor,
    answer_b: torch.Tensor,
    *,
    tries: torch.Tensor,
    was_batched: bool,
) -> EvalRolloutResult:
    clue_pin = clues_b > 0
    cell_loss = _compute_cell_loss(
        state.final_logits,
        clue_pin=clue_pin,
        answer=answer_b,
    )
    loss = cell_loss
    if not was_batched:
        return EvalRolloutResult(
            pred=state.pred.squeeze(0),
            outer_steps=state.outer_steps.squeeze(0),
            halted=state.halted.squeeze(0),
            loss=loss,
            cell_loss=cell_loss,
            tries=tries.squeeze(0),
        )
    return EvalRolloutResult(
        pred=state.pred,
        outer_steps=state.outer_steps,
        halted=state.halted,
        loss=loss,
        cell_loss=cell_loss,
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
    commit_prior: torch.Tensor | None = None
    commit_streak: int = 0


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
    random_init: bool = False,
) -> _StreamSlot:
    clue_pin = clues > 0
    digit_id = _init_digit_id_from_clues(
        clues.unsqueeze(0),
        clue_pin.unsqueeze(0),
        init_seed=_try_init_seed(init_seed, try_idx),
        random_init=random_init,
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
    )


def _slot_to_eval_result(
    slot: _StreamSlot,
) -> EvalRolloutResult:
    once = _OnceEvalState(
        pred=slot.pred.unsqueeze(0),
        outer_steps=torch.tensor([slot.outer_steps], device=slot.clues.device),
        halted=torch.tensor([slot.halted], device=slot.clues.device),
        final_logits=slot.final_logits.unsqueeze(0),
    )
    tries = torch.tensor([slot.try_idx + 1], device=slot.clues.device, dtype=torch.long)
    return _once_state_to_result(
        once,
        slot.clues.unsqueeze(0),
        slot.answer.unsqueeze(0),
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


@torch.inference_mode()
def rollout_eval_stream(
    model: MixerNextStateModel,
    clues: torch.Tensor,
    answers: torch.Tensor,
    *,
    slot_batch_size: int,
    config: RolloutConfig | None = None,
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
                random_init=config.random_init,
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

        logits, final_cell_embed = _inner_loop(
            model,
            active_digit_id,
            active_clue_pin,
            config.inner_iters,
            memory_embed=active_memory,
            with_grad=False,
        )
        pre_commit = predict_grid(logits, active_clues)
        implicit_halt_list: list[bool] = []
        for i, slot in enumerate(slots):
            prior, streak, implicit = _update_commit_stability_row(
                pre_commit[i],
                slot.commit_prior,
                slot.commit_streak,
                halt_after_stable_outer_steps=config.halt_after_stable_outer_steps,
            )
            slot.commit_prior = prior
            slot.commit_streak = streak
            implicit_halt_list.append(implicit)
            slot.final_logits = to_loss_dtype(logits[i])
        implicit_halt = torch.tensor(implicit_halt_list, device=clues_b.device, dtype=torch.bool)
        next_outer_count = active_outer_count + 1
        done = _eval_outer_done(implicit_halt, next_outer_count, config.max_outer_iters)

        next_slots: list[_StreamSlot] = []
        refill_count = 0
        for i, slot in enumerate(slots):
            slot.outer_count = int(next_outer_count[i].item())
            slot.pending_candidate = pre_commit[i]
            slot.memory_embed = final_cell_embed[i]
            slot.digit_id = active_digit_id[i]

            if not done[i]:
                next_slots.append(slot)
                continue

            slot.pred = pre_commit[i]
            slot.outer_steps = slot.outer_count
            slot.halted = bool(implicit_halt[i].item())
            if slot.halted or slot.try_idx + 1 >= max_tries:
                yield (
                    _slot_to_eval_result(slot),
                    slot.clues,
                    slot.answer,
                    slot.puzzle_idx,
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
                        random_init=config.random_init,
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
            out_halted[done_idx] = step.implicit_halt[step.done]
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
