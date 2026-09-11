from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from dataset import PuzzleDataset
from encoding import decode_logits
from model import MixerNextStateModel
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_MAX_OUTER_ITERS,
    BatchSlotState,
    RolloutConfig,
    _OnceEvalState,
    _PinContext,
    _copy_once_state,
    _compute_cell_loss,
    _compute_deep_supervision_losses,
    _compute_halt_loss,
    _compute_losses,
    _halt_target,
    _curriculum_init_digit_id,
    _inner_loop,
    _predict_halt,
    _transition_board,
    predict_grid,
    refill_done_slots,
    rollout_eval_batch,
    rollout_solve,
    rollout_trace,
    rollout_trace_batch,
    rollout_train_step,
)


def _first_param(model: MixerNextStateModel) -> torch.Tensor:
    return model.encoder.digit_embed.weight


def _baseline_config(**kwargs) -> RolloutConfig:
    kwargs.setdefault("transition_prob", 1.0)
    return RolloutConfig(**kwargs)


def _deep_supervision_inner_return(
    logits: torch.Tensor,
    halt_logit: torch.Tensor,
    cell_embed: torch.Tensor,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    return ([(logits, halt_logit)], cell_embed)


def _tiny_batch():
    clues = torch.tensor(
        [
            [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ]
        ]
    )
    answer = torch.full((1, 9, 9), 1)
    return clues, answer


def _make_state(
    clues: torch.Tensor,
    answer: torch.Tensor,
    *,
    gt_pin: torch.Tensor | None = None,
) -> BatchSlotState:
    clue_pin = clues > 0
    if gt_pin is None:
        gt_pin = torch.zeros_like(clue_pin)
    pin_ctx = _PinContext.from_state(clues, answer, gt_pin)
    return BatchSlotState(
        digit_id=clues.clone(),
        clues=clues,
        answer=answer,
        clue_pin=clue_pin,
        gt_pin=gt_pin,
        pin_ctx=pin_ctx,
        outer_count=torch.zeros(clues.size(0), dtype=torch.long),
    )


def _tiny_dataset() -> PuzzleDataset:
    clues, answer = _tiny_batch()
    clues2 = clues.clone()
    clues2[0, 0, 2] = 4
    return PuzzleDataset.from_tensors(
        torch.cat([clues, clues2]),
        torch.cat([answer, answer]),
    )


def test_rollout_config_validation():
    with pytest.raises(ValueError):
        RolloutConfig(inner_iters=0)
    with pytest.raises(ValueError):
        RolloutConfig(max_outer_iters=0)
    with pytest.raises(ValueError):
        RolloutConfig(transition_prob=0.0)
    with pytest.raises(ValueError):
        RolloutConfig(transition_prob=1.1)


def test_defaults():
    config = RolloutConfig()
    assert config.inner_iters == DEFAULT_INNER_ITERS
    assert config.max_outer_iters == DEFAULT_MAX_OUTER_ITERS
    assert config.deep_supervision is True
    assert config.adaptive_curriculum is True
    assert config.curriculum_puzzle_acc == 0.0


def test_one_outer_per_step():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    count_before = state.outer_count.item()
    rollout_train_step(model, state, config)
    assert state.outer_count.item() == count_before + 1


def test_state_persists_across_steps():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    digit_before = state.digit_id.clone()
    rollout_train_step(model, state, config)
    rollout_train_step(model, state, config)
    assert not torch.equal(state.digit_id, digit_before)
    assert state.outer_count.item() == 2


def test_state_persists_across_epochs():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32    )
    config = _baseline_config(inner_iters=2, max_outer_iters=100, halt_threshold=1.1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    rollout_train_step(model, state, config)
    assert state.outer_count.item() == 1
    memory_after_step = state.memory_embed.clone()
    # epoch boundary: same dataset, no re-seed
    dataset.set_epoch(2)
    rollout_train_step(model, state, config)
    assert state.outer_count.item() == 2
    assert state.memory_embed is not None
    assert not torch.equal(state.memory_embed, memory_after_step)


def test_clues_init_without_curriculum():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(42)
    state = BatchSlotState.seed(
        dataset,
        batch_size=2,
        device=torch.device("cpu"),
        generator=gen,
        dim=32,
        curriculum_training=False,
    )
    assert torch.equal(state.digit_id[state.clue_pin], state.clues[state.clue_pin])
    assert not torch.equal(state.digit_id[~state.clue_pin], state.clues[~state.clue_pin])


def test_halt_target_matches_grid():
    clues, answer = _tiny_batch()
    logits = torch.zeros(1, 9, 9, 10)
    pre_commit = predict_grid(logits, clues)
    target = _halt_target(pre_commit, answer)
    assert target.item() == float((pre_commit == answer).all())


def test_oracle_halt_model_continues():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10)
    with patch("rollout._predict_halt", return_value=torch.zeros(1, dtype=torch.bool)):
        with patch("rollout._halt_target", return_value=torch.ones(1)):
            result = rollout_train_step(model, state, config)
            assert result.done is not None
            assert not result.done.any()


def test_wrong_predict_halt_does_not_finish():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10)
    with patch("rollout._predict_halt", return_value=torch.ones(1, dtype=torch.bool)):
        with patch("rollout._halt_target", return_value=torch.zeros(1)):
            result = rollout_train_step(model, state, config)
            assert result.done is not None
            assert not result.done.any()


def test_refill_after_done():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32    )
    original_clues = state.clues.clone()
    done = torch.tensor([True])
    refill_done_slots(
        state, done, dataset, generator=gen, dim=32, curriculum_training=False
    )
    assert not torch.equal(state.clues, original_clues)
    assert state.outer_count.item() == 0
    assert torch.equal(state.digit_id[state.clue_pin], state.clues[state.clue_pin])
    assert not torch.equal(state.digit_id[~state.clue_pin], state.clues[~state.clue_pin])


def test_max_outer_forces_refill():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32    )
    state.outer_count[0] = 9
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    result = rollout_train_step(model, state, config)
    assert result.done is not None
    assert result.done.all()
    original_clues = state.clues.clone()
    refill_done_slots(
        state, result.done, dataset, generator=gen, dim=32, curriculum_training=False
    )
    assert not torch.equal(state.clues, original_clues)


def _curriculum_rand_side_effect():
    """Return p_gt=0, cell draws=1 so curriculum init is deterministic."""
    values = [0.0, 1.0]
    idx = 0

    def _side_effect(shape, *, device=None):
        nonlocal idx
        val = values[min(idx, len(values) - 1)]
        idx += 1
        size = shape if isinstance(shape, tuple) else (shape,)
        return torch.full(size, val, device=device)

    return _side_effect


def test_curriculum_init_preserves_clues():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    with patch("rollout.torch.rand", side_effect=_curriculum_rand_side_effect()):
        digit_id, gt_pin = _curriculum_init_digit_id(clues, answer, clue_pin)
    assert torch.equal(digit_id[clue_pin], clues[clue_pin])
    assert not gt_pin.any()


def test_curriculum_init_gt_reveal():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    calls = 0

    def _rand(shape, *, device=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([1.0], device=device)
        size = shape if isinstance(shape, tuple) else (shape,)
        return torch.zeros(size, device=device)

    with patch("rollout.torch.rand", side_effect=_rand):
        digit_id, gt_pin = _curriculum_init_digit_id(clues, answer, clue_pin)
    assert torch.equal(digit_id[~clue_pin], answer[~clue_pin])
    assert torch.equal(gt_pin, ~clue_pin)


def test_curriculum_init_random_when_no_reveal():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    calls = 0

    def _rand(shape, *, device=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([0.0], device=device)
        size = shape if isinstance(shape, tuple) else (shape,)
        return torch.zeros(size, device=device)

    with patch("rollout.torch.rand", side_effect=_rand):
        with patch(
            "rollout.torch.randint",
            return_value=torch.full(clues.shape, 7, dtype=clues.dtype),
        ):
            digit_id, gt_pin = _curriculum_init_digit_id(clues, answer, clue_pin)
    assert torch.equal(digit_id[clue_pin], clues[clue_pin])
    assert torch.equal(digit_id[~clue_pin], torch.full_like(clues, 7)[~clue_pin])
    assert not gt_pin.any()


def test_curriculum_seed_fills_cells():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(42)
    state = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32    )
    assert not torch.equal(state.digit_id, state.clues)
    assert torch.equal(state.digit_id[state.clue_pin], state.clues[state.clue_pin])


def test_curriculum_refill_fills_cells():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset,
        batch_size=1,
        device=torch.device("cpu"),
        generator=gen,
        dim=32,
        curriculum_training=False,
    )
    torch.manual_seed(42)
    refill_done_slots(state, torch.tensor([True]), dataset, generator=gen, dim=32)
    assert not torch.equal(state.digit_id, state.clues)
    assert torch.equal(state.digit_id[state.clue_pin], state.clues[state.clue_pin])


def test_eval_starts_from_random_non_clue():
    clues, answer = _tiny_batch()
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clue_pin = clues > 0
    with patch("rollout._inner_loop") as mock_inner:
        mock_inner.return_value = (
            torch.zeros(1, 9, 9, 10),
            torch.zeros(1),
            torch.zeros(1, 9, 9, 32),
        )
        with patch(
            "rollout.torch.randint",
            return_value=torch.full(clues.shape, 4, dtype=clues.dtype),
        ):
            rollout_eval_batch(model, clues, answer, config=_baseline_config(inner_iters=1, max_outer_iters=1))
    call_digit_id = mock_inner.call_args.args[1]
    assert torch.equal(call_digit_id[clue_pin], clues[clue_pin])
    assert torch.equal(call_digit_id[~clue_pin], torch.full_like(clues, 4)[~clue_pin])


def test_halt_stops_eval_early():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=100, halt_threshold=0.5)
    with patch("rollout._predict_halt", side_effect=[torch.tensor([False]), torch.tensor([True])]):
        result = rollout_eval_batch(model, clues, answer, config=config)
    assert result.outer_steps.item() == 2


def test_eval_multi_try_stops_at_first_halt():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=1, max_outer_iters=1, halt_threshold=0.5)
    with patch(
        "rollout._predict_halt",
        side_effect=[torch.tensor([False]), torch.tensor([True])],
    ):
        result = rollout_eval_batch(model, clues, answer, config=config, max_tries=3, init_seed=0)
    assert result.tries.item() == 2
    assert result.halted.item() is True


def test_eval_multi_try_keeps_last_try_without_halt():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=1, max_outer_iters=2, halt_threshold=0.5)
    with patch("rollout._predict_halt", return_value=torch.tensor([False])):
        result = rollout_eval_batch(model, clues, answer, config=config, max_tries=3, init_seed=0)
    assert result.tries.item() == 3
    assert result.halted.item() is False
    assert result.outer_steps.item() == 2


def test_eval_multi_try_uses_different_init_seeds():
    import rollout as rollout_module

    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=1, max_outer_iters=1)
    seeds_seen: list[int | None] = []
    original_init = rollout_module._init_digit_id_from_clues

    def track_init(clues_tensor, clue_pin, *, init_seed=None):
        seeds_seen.append(init_seed)
        return original_init(clues_tensor, clue_pin, init_seed=init_seed)

    with patch.object(rollout_module, "_init_digit_id_from_clues", side_effect=track_init):
        with patch("rollout._predict_halt", return_value=torch.tensor([False])):
            rollout_eval_batch(model, clues, answer, config=config, max_tries=2, init_seed=7)
    assert seeds_seen == [7, 8]


def test_copy_once_state_scatters_partial_accept():
    out = _OnceEvalState(
        pred=torch.zeros(2, 9, 9, dtype=torch.long),
        outer_steps=torch.zeros(2, dtype=torch.long),
        halted=torch.zeros(2, dtype=torch.bool),
        final_logits=torch.zeros(2, 9, 9, 10),
        final_halt_logit=torch.zeros(2),
        halt_correct_by_puzzle=torch.zeros(2, dtype=torch.long),
        halt_total_by_puzzle=torch.zeros(2, dtype=torch.long),
    )
    sub = _OnceEvalState(
        pred=torch.stack([torch.full((9, 9), 1, dtype=torch.long), torch.full((9, 9), 2, dtype=torch.long)]),
        outer_steps=torch.tensor([10, 20]),
        halted=torch.tensor([True, False]),
        final_logits=torch.zeros(2, 9, 9, 10),
        final_halt_logit=torch.zeros(2),
        halt_correct_by_puzzle=torch.tensor([3, 7]),
        halt_total_by_puzzle=torch.tensor([4, 8]),
    )
    _copy_once_state(
        out,
        sub,
        slot_idx=torch.tensor([0, 1]),
        local_mask=torch.tensor([True, False]),
    )
    assert out.halted.tolist() == [True, False]
    assert out.outer_steps.tolist() == [10, 0]
    assert out.halt_correct_by_puzzle.tolist() == [3, 0]
    assert out.pred[0, 0, 0].item() == 1
    assert out.pred[1, 0, 0].item() == 0


def test_eval_compact_scatter():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    clues2 = clues.clone()
    clues2[0, 0, 2] = 4
    clues_b = torch.cat([clues, clues2], dim=0)
    answer_b = torch.cat([answer, answer], dim=0)
    config = _baseline_config(inner_iters=2, max_outer_iters=3)
    result = rollout_eval_batch(model, clues_b, answer_b, config=config)
    assert result.pred.shape == (2, 9, 9)
    assert result.outer_steps.shape == (2,)


def test_eval_batch_casts_autocast_logits_to_float32():
    import rollout as rollout_module

    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=3)
    original_inner = rollout_module._inner_loop

    def bf16_inner(*args, **kwargs):
        logits, halt_logit, cell_embed = original_inner(*args, **kwargs)
        return logits.to(torch.bfloat16), halt_logit.to(torch.bfloat16), cell_embed

    with patch.object(rollout_module, "_inner_loop", bf16_inner):
        result = rollout_eval_batch(model, clues, answer, config=config)
    assert result.loss.dtype == torch.float32


def test_train_step_loss_is_float32_under_autocast_logits():
    import rollout as rollout_module

    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    original_inner = rollout_module._inner_loop

    def bf16_inner(*args, **kwargs):
        out = original_inner(*args, **kwargs)
        if isinstance(out[0], list):
            step_outputs, cell_embed = out
            return (
                [
                    (logits.to(torch.bfloat16), halt.to(torch.bfloat16))
                    for logits, halt in step_outputs
                ],
                cell_embed,
            )
        logits, halt_logit, cell_embed = out
        return logits.to(torch.bfloat16), halt_logit.to(torch.bfloat16), cell_embed

    with patch.object(rollout_module, "_inner_loop", bf16_inner):
        result = rollout_train_step(model, state, config, backward=False)
    assert result.loss.dtype == torch.float32
    assert result.cell_loss is not None
    assert result.cell_loss.dtype == torch.float32
    assert result.halt_loss is not None
    assert result.halt_loss.dtype == torch.float32


def test_no_grad_across_steps():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    rollout_train_step(model, state, config)
    assert not state.digit_id.requires_grad


def test_memory_embed_persisted_detached():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    assert state.memory_embed is None
    rollout_train_step(model, state, config)
    assert state.memory_embed is not None
    assert not state.memory_embed.requires_grad
    memory_after_step1 = state.memory_embed.clone()
    rollout_train_step(model, state, config)
    assert state.memory_embed is not None
    assert not torch.equal(state.memory_embed, memory_after_step1)


def test_seeded_refill():
    dataset = _tiny_dataset()
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    state1 = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen1, dim=32    )
    state2 = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen2, dim=32    )
    done = torch.tensor([True])
    refill_done_slots(state1, done, dataset, generator=gen1, dim=32)
    refill_done_slots(state2, done, dataset, generator=gen2, dim=32)
    assert torch.equal(state1.clues, state2.clues)


def test_refill_no_op_when_not_done():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset, batch_size=2, device=torch.device("cpu"), generator=gen, dim=32    )
    before = (
        state.digit_id.clone(),
        state.clues.clone(),
        state.answer.clone(),
        state.outer_count.clone(),
    )
    refill_done_slots(state, torch.tensor([False, False]), dataset, generator=gen, dim=32)
    assert torch.equal(state.digit_id, before[0])
    assert torch.equal(state.clues, before[1])
    assert torch.equal(state.answer, before[2])
    assert torch.equal(state.outer_count, before[3])


def test_eval_pred_matches_rollout_solve():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=3)
    result = rollout_eval_batch(model, clues, answer, config=config)
    for idx in range(clues.size(0)):
        assert torch.equal(result.pred[idx], rollout_solve(model, clues[idx], config=config))


def test_inner_loop_grad():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=3, max_outer_iters=10, halt_threshold=1.1)
    result = rollout_train_step(model, state, config)
    assert result.loss.item() > 0
    assert _first_param(model).grad is not None
    assert _first_param(model).grad.abs().sum().item() > 0


def test_rollout_fixed_steps_and_clues_pinned():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=4)
    result = rollout_eval_batch(model, clues, answer, config=config)
    assert torch.all(result.pred[clues > 0] == clues[clues > 0])


def test_reproducible_eval():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=3)
    r1 = rollout_eval_batch(model, clues, answer, config=config)
    r2 = rollout_eval_batch(model, clues, answer, config=config)
    assert torch.equal(r1.pred, r2.pred)
    assert r1.loss.item() == r2.loss.item()


def test_rollout_trace_frame_count():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, _ = _tiny_batch()
    max_outer_iters = 3
    config = _baseline_config(inner_iters=2, max_outer_iters=max_outer_iters)
    grids = rollout_trace(model, clues, config=config)
    assert 1 <= len(grids) <= max_outer_iters


def test_rollout_trace_batch_matches_single():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, _ = _tiny_batch()
    clues2 = clues.clone()
    clues2[0, 0, 2] = 4
    clues_b = torch.cat([clues, clues2], dim=0)
    config = _baseline_config(inner_iters=2, max_outer_iters=3)
    torch.manual_seed(0)
    single_first = rollout_trace(model, clues, config=config)
    torch.manual_seed(0)
    single_second = rollout_trace(model, clues2, config=config)
    torch.manual_seed(0)
    batched = rollout_trace_batch(model, clues_b, config=config)
    assert len(batched) == 2
    assert batched[0].states == single_first
    assert batched[1].states == single_second


def _transition_from_logits(logits, ctx, prev, **kwargs):
    decoded = decode_logits(logits)
    candidate = torch.where(ctx.pin, ctx.pin_digit_ids, decoded)
    return _transition_board(candidate, ctx, prev, **kwargs)


def test_transition_board_matches_full_decode():
    clues = torch.zeros(9, 9, dtype=torch.long)
    gt_pin = torch.zeros(9, 9, dtype=torch.bool)
    prev = clues.unsqueeze(0)
    ctx = _PinContext.from_state(prev, prev, gt_pin.unsqueeze(0))
    logits = torch.zeros(1, 9, 9, 10)
    logits[0, 0, 0, 5] = 10.0
    committed = _transition_from_logits(logits, ctx, prev, transition_prob=1.0)
    assert committed[0, 0, 0] == 5


def test_transition_board_pins_gt():
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    answer = torch.full((1, 9, 9), 3)
    gt_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    gt_pin[0, 0, 0] = True
    ctx = _PinContext.from_state(clues, answer, gt_pin)
    logits = torch.zeros(1, 9, 9, 10)
    logits[0, 0, 0, 7] = 10.0
    committed = _transition_from_logits(logits, ctx, clues, transition_prob=1.0)
    assert committed[0, 0, 0] == 3


def test_transition_board_unpins_gt_when_disabled():
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    answer = torch.full((1, 9, 9), 3)
    gt_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    gt_pin[0, 0, 0] = True
    ctx = _PinContext.from_state(clues, answer, gt_pin, pin_gt=False)
    logits = torch.zeros(1, 9, 9, 10)
    logits[0, 0, 0, 7] = 10.0
    committed = _transition_from_logits(logits, ctx, clues, transition_prob=1.0)
    assert committed[0, 0, 0] == 7


def test_transition_board_partial_transition():
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    gt_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    prev = clues.clone()
    prev[0, 0, 0] = 2
    prev[0, 0, 1] = 3
    ctx = _PinContext.from_state(clues, clues, gt_pin)
    logits = torch.zeros(1, 9, 9, 10)
    logits[0, 0, 0, 5] = 10.0
    logits[0, 0, 1, 6] = 10.0
    with patch(
        "rollout.torch.rand",
        return_value=torch.tensor([[[0.2, 0.8] + [0.0] * 7] * 9]),
    ):
        committed = _transition_from_logits(logits, ctx, prev, transition_prob=0.5)
    assert committed[0, 0, 0] == 5
    assert committed[0, 0, 1] == 3


def test_predict_grid_gt_unpinned():
    clues = torch.zeros(9, 9, dtype=torch.long)
    gt_pin = torch.zeros(9, 9, dtype=torch.bool)
    gt_pin[0, 0] = True
    logits = torch.zeros(9, 9, 10)
    logits[0, 0, 7] = 10.0
    pred = predict_grid(logits, clues)
    assert pred[0, 0] == 7


def test_halt_requires_gt_logits():
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    answer = torch.ones(1, 9, 9, dtype=torch.long)
    logits = torch.zeros(1, 9, 9, 10)
    logits[..., 2] = 10.0
    pred = predict_grid(logits, clues)
    assert _halt_target(pred, answer).item() == 0.0
    logits = torch.zeros(1, 9, 9, 10)
    logits[..., 1] = 10.0
    pred = predict_grid(logits, clues)
    assert _halt_target(pred, answer).item() == 1.0


def test_curriculum_gt_pinned_after_commit():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    gt_pin = ~clue_pin
    pin_ctx = _PinContext.from_state(clues, answer, gt_pin)
    state = _make_state(clues, answer, gt_pin=gt_pin)
    state.digit_id = torch.where(gt_pin, answer, clues)
    state.pin_ctx = pin_ctx
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    with patch("rollout._inner_loop") as mock_inner:
        logits = torch.zeros(1, 9, 9, 10)
        logits[..., 2] = 10.0
        mock_inner.return_value = _deep_supervision_inner_return(
            logits, torch.zeros(1), torch.zeros(1, 9, 9, 32)
        )
        rollout_train_step(model, state, _baseline_config(), backward=False)
    assert torch.equal(state.digit_id[gt_pin], answer[gt_pin])


def test_curriculum_gt_unpinned_after_commit():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    gt_pin = ~clue_pin
    pin_ctx = _PinContext.from_state(clues, answer, gt_pin, pin_gt=False)
    state = _make_state(clues, answer, gt_pin=gt_pin)
    state.digit_id = torch.where(gt_pin, answer, clues)
    state.pin_ctx = pin_ctx
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    with patch("rollout._inner_loop") as mock_inner:
        logits = torch.zeros(1, 9, 9, 10)
        logits[..., 2] = 10.0
        mock_inner.return_value = _deep_supervision_inner_return(
            logits, torch.zeros(1), torch.zeros(1, 9, 9, 32)
        )
        rollout_train_step(model, state, _baseline_config(pin_gt=False), backward=False)
        rollout_train_step(model, state, _baseline_config(pin_gt=False), backward=False)
    assert torch.equal(state.digit_id[gt_pin], torch.full_like(state.digit_id[gt_pin], 2))


def test_gt_cells_in_loss():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    gt_pin = ~clue_pin
    logits = torch.zeros(1, 9, 9, 10)
    logits[..., 2] = 10.0
    loss_wrong = _compute_cell_loss(logits, clue_pin=clue_pin, answer=answer)
    logits[..., 1] = 10.0
    loss_right = _compute_cell_loss(logits, clue_pin=clue_pin, answer=answer)
    assert loss_wrong.item() > loss_right.item()


def test_train_step_order_of_ops_with_gt_pin():
    """Wrong GT logits block halt but commit still pins board state."""
    clues, answer = _tiny_batch()
    gt_pin = ~(clues > 0)
    state = _make_state(clues, answer, gt_pin=gt_pin)
    state.digit_id = torch.where(gt_pin, answer, clues)
    state.pin_ctx = _PinContext.from_state(clues, answer, gt_pin)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    with patch("rollout._inner_loop") as mock_inner:
        logits = torch.zeros(1, 9, 9, 10)
        logits[..., 2] = 10.0
        mock_inner.return_value = _deep_supervision_inner_return(
            logits, torch.zeros(1), torch.zeros(1, 9, 9, 32)
        )
        result = rollout_train_step(model, state, _baseline_config(), backward=False)
    assert result.halt_target is not None
    assert result.pred is not None
    assert result.cell_loss is not None
    assert result.halt_target.item() == 0.0
    assert result.cell_loss.item() > 0.0
    assert torch.equal(state.digit_id[gt_pin], answer[gt_pin])
    assert not torch.equal(result.pred[gt_pin], answer[gt_pin])


def test_gt_pin_not_in_encode_clue_pin():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    gt_pin = ~clue_pin
    digit_id = torch.where(gt_pin, answer, clues)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    with patch.object(model, "encode_input", wraps=model.encode_input) as mock_encode:
        _inner_loop(
            model,
            digit_id,
            clue_pin,
            1,
            memory_embed=None,
        )
    passed_clue_pin = mock_encode.call_args[0][1]
    assert torch.equal(passed_clue_pin, clue_pin)
    assert not (passed_clue_pin & gt_pin).any()


def test_train_seed_without_pin_gt_keeps_curriculum_init():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    calls = 0

    def _rand(shape, *, device=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([1.0], device=device)
        size = shape if isinstance(shape, tuple) else (shape,)
        return torch.zeros(size, device=device)

    with patch("rollout.torch.rand", side_effect=_rand):
        state = BatchSlotState.seed(
            dataset, 1, torch.device("cpu"), generator=gen, dim=32, pin_gt=False
        )
    assert state.gt_pin.any()
    assert torch.equal(state.pin_ctx.pin, state.clue_pin)
    assert torch.equal(state.digit_id[state.gt_pin], state.answer[state.gt_pin])


def test_train_seed_sets_gt_pin():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    calls = 0

    def _rand(shape, *, device=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([1.0], device=device)
        size = shape if isinstance(shape, tuple) else (shape,)
        return torch.zeros(size, device=device)

    with patch("rollout.torch.rand", side_effect=_rand):
        state = BatchSlotState.seed(
            dataset, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32        )
    assert state.gt_pin.any()
    assert state.gt_pin.sum() == (~state.clue_pin).sum()
    assert torch.equal(state.pin_ctx.pin, state.clue_pin | state.gt_pin)


def test_train_seed_without_curriculum_has_no_gt_pin():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset,
        batch_size=1,
        device=torch.device("cpu"),
        generator=gen,
        dim=32,
        curriculum_training=False,
    )
    assert not state.gt_pin.any()
    assert torch.equal(state.pin_ctx.pin, state.clue_pin)


def test_eval_outer_commit_clue_pin_only():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    from rollout import _transition_board

    with patch("rollout._transition_board", wraps=_transition_board) as mock_commit:
        with patch("rollout._inner_loop") as mock_inner:
            logits = torch.zeros(1, 9, 9, 10)
            logits[..., 1] = 10.0
            mock_inner.return_value = (logits, torch.zeros(1), torch.zeros(1, 9, 9, 32))
            rollout_eval_batch(
                model, clues, answer, config=_baseline_config(inner_iters=1, max_outer_iters=2)
            )
    ctx = mock_commit.call_args[0][1]
    assert torch.equal(ctx.pin, clues > 0)


def test_mutable_non_gt_cell_overwritable_on_commit():
    clues, answer = _tiny_batch()
    gt_pin = torch.zeros_like(clues > 0)
    state = _make_state(clues, answer, gt_pin=gt_pin)
    state.digit_id = clues.clone()
    state.digit_id[0, 0, 2] = 7
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    with patch("rollout._inner_loop") as mock_inner:
        logits = torch.zeros(1, 9, 9, 10)
        logits[0, 0, 2, 4] = 10.0
        mock_inner.return_value = _deep_supervision_inner_return(
            logits, torch.zeros(1), torch.zeros(1, 9, 9, 32)
        )
        rollout_train_step(model, state, _baseline_config(), backward=False)
        rollout_train_step(model, state, _baseline_config(), backward=False)
    assert state.digit_id[0, 0, 2] == 4
    assert not state.gt_pin[0, 0, 2]


def test_curriculum_partial_reveal_gt_pin():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    calls = 0

    def _rand(shape, *, device=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([0.5], device=device)
        size = shape if isinstance(shape, tuple) else (shape,)
        out = torch.full(size, 0.7, device=device)
        half = out.view(out.size(0), -1).size(1) // 2
        out.view(out.size(0), -1)[:, :half] = 0.3
        return out

    with patch("rollout.torch.rand", side_effect=_rand):
        _, gt_pin = _curriculum_init_digit_id(clues, answer, clue_pin)
    assert gt_pin.any()
    assert (~gt_pin & ~clue_pin).any()
    assert gt_pin.sum() < (~clue_pin).sum()
    assert not (gt_pin & clue_pin).any()
    empty = (clues == 0) & ~gt_pin
    assert empty.any()


def test_refill_updates_gt_pin_and_pin_ctx():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset,
        batch_size=1,
        device=torch.device("cpu"),
        generator=gen,
        dim=32,
        curriculum_training=False,
    )
    assert not state.gt_pin.any()
    calls = 0

    def _rand(shape, *, device=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor([1.0], device=device)
        size = shape if isinstance(shape, tuple) else (shape,)
        return torch.zeros(size, device=device)

    with patch("rollout.torch.rand", side_effect=_rand):
        refill_done_slots(state, torch.tensor([True]), dataset, generator=gen, dim=32)
    assert state.gt_pin.any()
    assert torch.equal(state.pin_ctx.pin, state.clue_pin | state.gt_pin)
    assert torch.equal(state.digit_id[state.gt_pin], state.answer[state.gt_pin])


def test_predict_grid_pins_clues():
    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 3
    clues[1, 1] = 8
    logits = torch.zeros(9, 9, 10)
    logits[0, 0, 4] = 10.0
    logits[1, 1, 2] = 10.0
    logits[2, 2, 7] = 10.0
    pred = predict_grid(logits, clues)
    assert pred[0, 0] == 3
    assert pred[1, 1] == 8
    assert pred[2, 2] == 7


def test_predict_halt_threshold():
    halt_logit = torch.tensor([10.0, -10.0])
    assert torch.equal(_predict_halt(halt_logit, halt_threshold=0.5), torch.tensor([True, False]))


def test_eval_halt_acc_counts_all_rounds():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    max_outer = 4
    config = _baseline_config(inner_iters=2, max_outer_iters=max_outer, halt_threshold=1.1)
    result = rollout_eval_batch(model, clues, answer, config=config)
    assert result.halt_total_rounds == max_outer
    assert result.outer_steps.item() == max_outer


def test_trace_includes_halt_metadata():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, _ = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=3)
    trace = rollout_trace_batch(model, clues, config=config)[0]
    assert len(trace.predictions) == trace.outer_steps
    assert len(trace.inputs) == len(trace.predictions)
    assert isinstance(trace.halted, bool)
    halt_logit = torch.tensor([10.0, -10.0])
    assert torch.equal(_predict_halt(halt_logit, halt_threshold=0.5), torch.tensor([True, False]))


def test_build_rollout_config():
    from train import build_rollout_config

    config = build_rollout_config(
        inner_iters=2,
        max_outer_iters=3,
        transition_prob=0.75,
    )
    assert config.inner_iters == 2
    assert config.max_outer_iters == 3
    assert config.transition_prob == 0.75
    assert config.curriculum_training is True
    assert config.pin_gt is True


def test_eval_rollout_config_disables_curriculum():
    from train import build_rollout_config

    train_config = build_rollout_config(inner_iters=2, max_outer_iters=3, curriculum_training=True)
    eval_config = build_rollout_config(inner_iters=2, max_outer_iters=3, curriculum_training=False)
    assert train_config.curriculum_training is True
    assert eval_config.curriculum_training is False


def test_train_metrics_done_only_puzzle_acc():
    from train import TrainMetricsAccumulator

    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    result = rollout_train_step(
        model,
        state,
        _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1),
    )
    result.done = torch.tensor([False])
    acc = TrainMetricsAccumulator.empty(torch.device("cpu"))
    acc.add_step(result, state)
    assert int(acc.puzzles_done.item()) == 0
    assert int(acc.correct_puzzles_done.item()) == 0


def test_eval_rollout_reaches_max_outer_iters():
    torch.manual_seed(0)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=3, halt_threshold=1.1)
    result = rollout_eval_batch(model, clues, answer, config=config)
    assert result.outer_steps.item() == 3


def test_train_step_stores_memory_embed():
    clues, answer = _tiny_batch()
    state = BatchSlotState.seed(
        _tiny_dataset(),
        batch_size=1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
        dim=32,
    )
    config = RolloutConfig(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    with patch("rollout._inner_loop") as mock_inner:
        final_cell = torch.randn(1, 9, 9, 32)
        mock_inner.return_value = _deep_supervision_inner_return(
            torch.zeros(1, 9, 9, 10), torch.zeros(1), final_cell
        )
        rollout_train_step(model, state, config, backward=False)
    assert state.memory_embed is not None
    assert not state.memory_embed.requires_grad
    assert torch.allclose(state.memory_embed, final_cell.detach())


def test_refill_zeros_memory_embed():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32,
    )
    state.memory_embed = torch.randn(1, 9, 9, 32)
    refill_done_slots(state, torch.tensor([True]), dataset, generator=gen, dim=32)
    assert state.memory_embed is not None
    assert state.memory_embed.sum().item() == 0.0


def test_deep_supervision_affects_loss():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=3, max_outer_iters=10, halt_threshold=1.1, deep_supervision=True)
    with patch("rollout._compute_cell_loss", wraps=_compute_cell_loss) as mock_cell:
        with patch("rollout._compute_halt_loss", wraps=_compute_halt_loss) as mock_halt:
            rollout_train_step(model, state, config, backward=False)
    assert mock_cell.call_count == 3
    assert mock_halt.call_count == 3


def test_deep_supervision_off_calls_loss_once():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(
        inner_iters=3, max_outer_iters=10, halt_threshold=1.1, deep_supervision=False
    )
    with patch("rollout._compute_cell_loss", wraps=_compute_cell_loss) as mock_cell:
        with patch("rollout._compute_halt_loss", wraps=_compute_halt_loss) as mock_halt:
            rollout_train_step(model, state, config, backward=False)
    assert mock_cell.call_count == 1
    assert mock_halt.call_count == 1


def test_deep_supervision_grad_all_steps():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=3, max_outer_iters=10, halt_threshold=1.1, deep_supervision=True)
    result = rollout_train_step(model, state, config)
    assert result.loss.item() > 0
    assert _first_param(model).grad is not None
    assert _first_param(model).grad.abs().sum().item() > 0


def test_deep_supervision_losses_synthetic():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    b = clues.size(0)
    wrong_logits = torch.zeros(b, 9, 9, 10)
    wrong_logits[..., 2] = 10.0
    right_logits = torch.zeros(b, 9, 9, 10)
    right_logits[..., 1] = 10.0
    answer = predict_grid(right_logits, clues)
    halt_logit = torch.full((b,), 2.0)
    step_outputs = [(wrong_logits, halt_logit), (right_logits, halt_logit)]
    _, deep_halt, _ = _compute_deep_supervision_losses(
        step_outputs,
        clues=clues,
        clue_pin=clue_pin,
        answer=answer,
        halt_loss_weight=1.0,
    )
    final_pred = predict_grid(right_logits, clues)
    final_halt_target = _halt_target(final_pred, answer)
    _, final_halt, _ = _compute_losses(
        right_logits,
        halt_logit,
        clue_pin=clue_pin,
        answer=answer,
        halt_target=final_halt_target,
        halt_loss_weight=1.0,
    )
    assert deep_halt.item() > final_halt.item()


def test_deep_supervision_regression_single_step():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state_off = _make_state(clues, answer)
    state_on = _make_state(clues, answer)
    config_off = _baseline_config(
        inner_iters=1, max_outer_iters=10, halt_threshold=1.1, deep_supervision=False
    )
    config_on = _baseline_config(
        inner_iters=1, max_outer_iters=10, halt_threshold=1.1, deep_supervision=True
    )
    torch.manual_seed(0)
    result_off = rollout_train_step(model, state_off, config_off, backward=False)
    torch.manual_seed(0)
    result_on = rollout_train_step(model, state_on, config_on, backward=False)
    assert result_off.loss.item() == result_on.loss.item()


def test_adaptive_curriculum_p_gt_upper_bound():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    call_count = 0

    def rand_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if len(args) == 1 and isinstance(args[0], int):
            return torch.ones(args[0])
        shape = args[0] if len(args) == 1 else args
        return torch.full(shape, 0.5)

    with patch("rollout.torch.rand", side_effect=rand_side_effect):
        _, gt_pin_adaptive = _curriculum_init_digit_id(
            clues, answer, clue_pin, puzzle_acc=0.6, adaptive=True
        )
        call_count = 0
        _, gt_pin_fixed = _curriculum_init_digit_id(
            clues, answer, clue_pin, puzzle_acc=0.6, adaptive=False
        )
    assert not gt_pin_adaptive.any()
    assert gt_pin_fixed.any()


def test_adaptive_curriculum_zero_acc_matches_uniform():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    torch.manual_seed(42)
    digit_fixed, gt_fixed = _curriculum_init_digit_id(clues, answer, clue_pin, adaptive=False)
    torch.manual_seed(42)
    digit_adaptive, gt_adaptive = _curriculum_init_digit_id(
        clues, answer, clue_pin, puzzle_acc=0.0, adaptive=True
    )
    assert torch.equal(digit_fixed, digit_adaptive)
    assert torch.equal(gt_fixed, gt_adaptive)


def test_adaptive_curriculum_full_acc_no_reveal():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    digit_id, gt_pin = _curriculum_init_digit_id(
        clues, answer, clue_pin, puzzle_acc=1.0, adaptive=True
    )
    assert torch.equal(digit_id[clue_pin], clues[clue_pin])
    assert not torch.equal(digit_id[~clue_pin], clues[~clue_pin])
    assert not gt_pin.any()


def test_adaptive_curriculum_config_update():
    from dataclasses import replace

    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    config = replace(RolloutConfig(), curriculum_puzzle_acc=0.8, adaptive_curriculum=True)

    def rand_reveal(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], int):
            return torch.ones(args[0])
        shape = args[0] if len(args) == 1 else args
        return torch.full(shape, 0.15)

    def rand_no_reveal(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], int):
            return torch.ones(args[0])
        shape = args[0] if len(args) == 1 else args
        return torch.full(shape, 0.25)

    with patch("rollout.torch.rand", side_effect=rand_reveal):
        _, gt_pin = _curriculum_init_digit_id(
            clues,
            answer,
            clue_pin,
            puzzle_acc=config.curriculum_puzzle_acc,
            adaptive=config.adaptive_curriculum,
        )
    with patch("rollout.torch.rand", side_effect=rand_no_reveal):
        _, gt_pin_tight = _curriculum_init_digit_id(
            clues,
            answer,
            clue_pin,
            puzzle_acc=config.curriculum_puzzle_acc,
            adaptive=config.adaptive_curriculum,
        )
    assert gt_pin.any()
    assert not gt_pin_tight.any()


def test_build_rollout_config_new_flags():
    from train import build_rollout_config

    config = build_rollout_config(
        inner_iters=2,
        max_outer_iters=3,
        deep_supervision=True,
        adaptive_curriculum=True,
    )
    assert config.deep_supervision is True
    assert config.adaptive_curriculum is True
