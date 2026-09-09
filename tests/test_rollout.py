from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from dataset import PuzzleDataset, PuzzleTensorCache
from model import MixerNextStateModel
from ema import DEFAULT_EMA_ALPHA, ema_combine
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_MAX_OUTER_ITERS,
    BatchSlotState,
    RolloutConfig,
    _ClueContext,
    _halt_target,
    _apply_rollout_mask,
    _apply_rollout_noise,
    _digits_for_inner_loop,
    _outer_commit,
    _predict_halt,
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
    kwargs.setdefault("ema_alpha", 1.0)
    return RolloutConfig(**kwargs)


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


def _make_state(clues: torch.Tensor, answer: torch.Tensor) -> BatchSlotState:
    clue_pin = clues > 0
    return BatchSlotState(
        digit_id=clues.clone(),
        clues=clues,
        answer=answer,
        clue_pin=clue_pin,
        outer_count=torch.zeros(clues.size(0), dtype=torch.long),
    )


def _tiny_cache() -> PuzzleTensorCache:
    clues, answer = _tiny_batch()
    clues2 = clues.clone()
    clues2[0, 0, 2] = 4
    return PuzzleTensorCache(clues=torch.cat([clues, clues2]), answers=torch.cat([answer, answer]))


def test_rollout_config_validation():
    with pytest.raises(ValueError):
        RolloutConfig(inner_iters=0)
    with pytest.raises(ValueError):
        RolloutConfig(max_outer_iters=0)
    with pytest.raises(ValueError):
        RolloutConfig(rollout_mask_prob=1.1)
    with pytest.raises(ValueError):
        RolloutConfig(rollout_noise_prob=1.1)
    with pytest.raises(ValueError):
        RolloutConfig(ema_alpha=0.0)


def test_rollout_mask_and_noise():
    clues, _ = _tiny_batch()
    clue_pin = clues > 0
    filled = clues.clone()
    filled[0, 0, 2] = 7
    masked = _apply_rollout_mask(filled, clue_pin, prob=1.0)
    assert torch.equal(masked[clue_pin], filled[clue_pin])
    assert (masked[~clue_pin] == 0).all()
    noisy = _apply_rollout_noise(filled, clue_pin, prob=1.0)
    assert torch.equal(noisy[clue_pin], filled[clue_pin])
    assert not torch.equal(noisy, filled)
    assert (noisy[~clue_pin] >= 1).all()
    assert torch.equal(
        _digits_for_inner_loop(filled, clue_pin, rollout_mask_prob=0.0, rollout_noise_prob=0.0),
        filled,
    )


def test_defaults():
    config = RolloutConfig()
    assert config.inner_iters == DEFAULT_INNER_ITERS
    assert config.max_outer_iters == DEFAULT_MAX_OUTER_ITERS
    assert config.ema_alpha == DEFAULT_EMA_ALPHA


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
    assert not torch.equal(state.digit_id, digit_before)
    assert state.outer_count.item() == 1


def test_state_persists_across_epochs():
    cache = _tiny_cache()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        cache, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32, ema_alpha=1.0
    )
    config = _baseline_config(inner_iters=2, max_outer_iters=100, halt_threshold=1.1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    rollout_train_step(model, state, config)
    assert state.outer_count.item() == 1
    digit_after_step = state.digit_id.clone()
    # epoch boundary: new cache object, no re-seed
    PuzzleTensorCache(clues=cache.clues, answers=cache.answers)
    rollout_train_step(model, state, config)
    assert state.outer_count.item() == 2
    assert not torch.equal(state.digit_id, digit_after_step)


def test_always_clues_init():
    cache = _tiny_cache()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        cache, batch_size=2, device=torch.device("cpu"), generator=gen, dim=32, ema_alpha=1.0
    )
    assert torch.equal(state.digit_id, state.clues)


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


def test_refill_after_done():
    cache = _tiny_cache()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        cache, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32, ema_alpha=1.0
    )
    original_clues = state.clues.clone()
    done = torch.tensor([True])
    refill_done_slots(state, done, cache, generator=gen, dim=32, ema_alpha=1.0)
    assert not torch.equal(state.clues, original_clues)
    assert state.outer_count.item() == 0
    assert torch.equal(state.digit_id, state.clues)


def test_max_outer_forces_refill():
    cache = _tiny_cache()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        cache, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32, ema_alpha=1.0
    )
    state.outer_count[0] = 9
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    result = rollout_train_step(model, state, config)
    assert result.done is not None
    assert result.done.all()
    original_clues = state.clues.clone()
    refill_done_slots(state, result.done, cache, generator=gen, dim=32, ema_alpha=1.0)
    assert not torch.equal(state.clues, original_clues)


def test_halt_stops_eval_early():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=100, halt_threshold=0.5)
    with patch("rollout._predict_halt", side_effect=[torch.tensor([False]), torch.tensor([True])]):
        result = rollout_eval_batch(model, clues, answer, config=config)
    assert result.outer_steps.item() == 2


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
        logits, halt_logit, cell_embed = original_inner(*args, **kwargs)
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


def test_cell_embed_not_persisted():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    rollout_train_step(model, state, config)
    # state has no cell_embed field — fresh encode each step
    assert not hasattr(state, "cell_embed")


def test_seeded_refill():
    cache = _tiny_cache()
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    state1 = BatchSlotState.seed(
        cache, batch_size=1, device=torch.device("cpu"), generator=gen1, dim=32, ema_alpha=1.0
    )
    state2 = BatchSlotState.seed(
        cache, batch_size=1, device=torch.device("cpu"), generator=gen2, dim=32, ema_alpha=1.0
    )
    done = torch.tensor([True])
    refill_done_slots(state1, done, cache, generator=gen1, dim=32, ema_alpha=1.0)
    refill_done_slots(state2, done, cache, generator=gen2, dim=32, ema_alpha=1.0)
    assert torch.equal(state1.clues, state2.clues)


def test_refill_no_op_when_not_done():
    cache = _tiny_cache()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        cache, batch_size=2, device=torch.device("cpu"), generator=gen, dim=32, ema_alpha=1.0
    )
    before = (
        state.digit_id.clone(),
        state.clues.clone(),
        state.answer.clone(),
        state.outer_count.clone(),
    )
    refill_done_slots(state, torch.tensor([False, False]), cache, generator=gen, dim=32, ema_alpha=1.0)
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
    assert 2 <= len(grids) <= max_outer_iters + 1


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


def test_outer_commit_matches_full_decode():
    clues = torch.zeros(9, 9, dtype=torch.long)
    ctx = _ClueContext.from_clues(clues.unsqueeze(0))
    logits = torch.zeros(1, 9, 9, 10)
    logits[0, 0, 0, 5] = 10.0
    committed = _outer_commit(logits, ctx)
    assert committed[0, 0, 0] == 5


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
    assert len(trace.states) == trace.outer_steps + 1
    assert isinstance(trace.halted, bool)
    halt_logit = torch.tensor([10.0, -10.0])
    assert torch.equal(_predict_halt(halt_logit, halt_threshold=0.5), torch.tensor([True, False]))


def test_build_rollout_config():
    from train import build_rollout_config

    config = build_rollout_config(
        inner_iters=2,
        max_outer_iters=3,
        rollout_mask_prob=0.2,
        rollout_noise_prob=0.05,
        ema_alpha=0.05,
    )
    assert config.inner_iters == 2
    assert config.max_outer_iters == 3
    assert config.rollout_mask_prob == 0.2
    assert config.rollout_noise_prob == 0.05
    assert config.ema_alpha == 0.05


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


def test_ema_alpha_one_regression():
    torch.manual_seed(0)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = _baseline_config(inner_iters=2, max_outer_iters=3, halt_threshold=1.1)
    result = rollout_eval_batch(model, clues, answer, config=config)
    assert result.outer_steps.item() == 3


def test_ema_alpha_one_state_has_no_ema_embed():
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    rollout_train_step(model, state, config)
    assert state.ema_embed is None


def test_ema_embed_updated_after_train_step():
    clues, answer = _tiny_batch()
    state = BatchSlotState.seed(
        _tiny_cache(),
        batch_size=1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
        dim=32,
        ema_alpha=0.05,
    )
    config = RolloutConfig(inner_iters=2, max_outer_iters=10, halt_threshold=1.1, ema_alpha=0.05)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    rollout_train_step(model, state, config)
    assert state.ema_embed is not None
    assert not state.ema_embed.requires_grad
    assert state.ema_embed.abs().sum().item() > 0


def test_ema_end_of_inner_matches_stored_ema():
    clues, answer = _tiny_batch()
    ema_alpha = 0.05
    state = BatchSlotState.seed(
        _tiny_cache(),
        batch_size=1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
        dim=32,
        ema_alpha=ema_alpha,
    )
    ema_before = state.ema_embed.clone()
    config = RolloutConfig(inner_iters=2, max_outer_iters=10, halt_threshold=1.1, ema_alpha=ema_alpha)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    with patch("rollout._inner_loop") as mock_inner:
        final_cell = torch.randn(1, 9, 9, 32)
        mock_inner.return_value = (torch.zeros(1, 9, 9, 10), torch.zeros(1), final_cell)
        rollout_train_step(model, state, config, backward=False)
    expected = ema_combine(final_cell, ema_before, ema_alpha)
    assert torch.allclose(state.ema_embed, expected)


def test_refill_zeros_ema_embed():
    cache = _tiny_cache()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        cache, batch_size=1, device=torch.device("cpu"), generator=gen, dim=32, ema_alpha=0.05
    )
    state.ema_embed.fill_(1.0)
    refill_done_slots(state, torch.tensor([True]), cache, generator=gen, dim=32, ema_alpha=0.05)
    assert state.ema_embed.sum().item() == 0.0
