from __future__ import annotations

import pytest
import torch

from encoding import attach_clue_mask, decode_logits, grid_to_onehot, onehot_to_grid, sample_decode_logits
from model import NextStateModel
from rollout import (
    DEFAULT_T_MAX,
    RolloutConfig,
    _ClueContext,
    _curriculum_initial,
    _rollout_loop,
    predict_grid,
    rollout_train_batch,
    rollout_trace,
    temperature_at_step,
    temperature_schedule,
)
from encoding import attach_clue_mask


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
    clues_onehot = grid_to_onehot(clues)
    return clues, clues_onehot, answer


def test_temperature_endpoints_and_monotonic():
    for total in (2, 10, 30):
        temps = temperature_schedule(total, t_max=3.0, t_min=0.0)
        assert len(temps) == total
        assert temps[0] == pytest.approx(3.0)
        assert temps[-1] == pytest.approx(0.0)
        assert all(temps[i] >= temps[i + 1] for i in range(total - 1))


def test_temperature_adapts_to_length():
    assert temperature_at_step(1, 10, 3.0, 0.0) != temperature_at_step(1, 30, 3.0, 0.0)


def test_temperature_single_step():
    assert temperature_at_step(0, 1, 3.0, 0.0) == 0.0


def test_sample_decode_t0_matches_argmax():
    logits = torch.randn(2, 9, 9, 9)
    assert torch.equal(sample_decode_logits(logits, 0.0), decode_logits(logits))


def test_sample_decode_high_temperature_more_uniform():
    logits = torch.zeros(1, 9, 9, 9)
    torch.manual_seed(0)
    high = sample_decode_logits(logits, 10.0)
    assert high.unique().numel() > 1


def test_rollout_fixed_steps_and_clues_pinned():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", t_max=2.0)
    with torch.inference_mode():
        result = rollout_train_batch(
            model,
            clues[0],
            clues_onehot[0],
            answer[0],
            rollout_iters=4,
            config=config,
        )
    assert result.pred is not None
    assert torch.all(result.pred[clues[0] > 0] == clues[0][clues[0] > 0])


def test_grad_isolation_and_backward():
    model = NextStateModel(width=32, num_blocks=1)
    model.train()
    clues, clues_onehot, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", t_max=1.5)
    result = rollout_train_batch(
        model,
        clues,
        clues_onehot,
        answer,
        rollout_iters=3,
        config=config,
        compute_pred=False,
    )
    result.loss.backward()
    assert model.net[0].weight.grad is not None
    assert model.net[0].weight.grad.abs().sum().item() > 0


def test_curriculum_init():
    clues = torch.tensor(
        [
            [5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    answer = torch.ones(9, 9)
    torch.manual_seed(0)
    onehot = _curriculum_initial(answer, clues)
    grid = onehot_to_grid(onehot)
    assert grid[0, 0] == 5
    assert torch.all((grid[clues == 0] >= 1) & (grid[clues == 0] <= 9))


def test_reproducible_eval_with_seed():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", t_max=2.0)

    torch.manual_seed(42)
    r1 = rollout_train_batch(
        model, clues, clues_onehot, answer, rollout_iters=5, config=config
    )
    torch.manual_seed(42)
    r2 = rollout_train_batch(
        model, clues, clues_onehot, answer, rollout_iters=5, config=config
    )
    assert torch.equal(r1.pred, r2.pred)
    assert r1.loss.item() == r2.loss.item()


def test_sample_decode_pins_clue_cells():
    logits = torch.randn(1, 9, 9, 9)
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clues[0, 0, 0] = 5
    clues[0, 1, 1] = 7
    torch.manual_seed(0)
    decoded = sample_decode_logits(logits, 5.0, clues=clues)
    assert decoded[0, 0, 0] == 5
    assert decoded[0, 1, 1] == 7


def test_rollout_trace_frame_count():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, _ = _tiny_batch()
    config = RolloutConfig(train_init="clues", t_max=DEFAULT_T_MAX)
    torch.manual_seed(1)
    grids = rollout_trace(model, clues_onehot, clues, rollout_iters=3, config=config)
    assert len(grids) == 4  # clues + 3 sampled steps


def test_sampled_state_differs_from_argmax_at_high_temp():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, _ = _tiny_batch()
    config = RolloutConfig(train_init="clues", t_max=DEFAULT_T_MAX)
    clues_b = clues.unsqueeze(0)
    clues_onehot_b = clues_onehot.unsqueeze(0)
    clue_ctx = _ClueContext.from_clues(clues_b, clues_onehot_b)

    torch.manual_seed(1)
    with torch.inference_mode():
        state = attach_clue_mask(clues_onehot_b, clues_b, clue_mask_channel=clue_ctx.clue_mask_channel)
        _, state_grids, _ = _rollout_loop(
            model,
            state,
            clues_b,
            clue_ctx,
            2,
            config,
            collect_state_grids=True,
        )
        first_logits = model(state)
    assert state_grids is not None
    argmax_grid = predict_grid(first_logits, clues_b)[0]
    sampled_grid = state_grids[0][0]
    non_clue = clues == 0
    assert not torch.equal(sampled_grid[non_clue], argmax_grid[non_clue])
