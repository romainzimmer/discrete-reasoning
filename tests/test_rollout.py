from __future__ import annotations

import pytest
import torch

from encoding import decode_logits, target_mask
from model import MixerNextStateModel
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_OUTER_ITERS,
    DEFAULT_TRUNCATED_BPTT_STEPS,
    RolloutConfig,
    RolloutState,
    _ClueContext,
    _curriculum_initial,
    _inner_loop,
    _init_rollout_state,
    _outer_commit,
    _rollout_loop,
    predict_grid,
    rollout_train_batch,
    rollout_trace,
    rollout_trace_batch,
)


def _first_param(model: MixerNextStateModel) -> torch.Tensor:
    return model.encoder.digit_embed.weight


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


def test_rollout_config_validation():
    with pytest.raises(ValueError):
        RolloutConfig(inner_iters=0)
    with pytest.raises(ValueError):
        RolloutConfig(outer_iters=0)
    with pytest.raises(ValueError):
        RolloutConfig(inner_iters=3, truncated_bptt_steps=4)
    with pytest.raises(ValueError):
        RolloutConfig(inner_iters=1, fixed_point=True)


def test_truncated_bptt_zero_detaches_inner_prefix():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    state = _init_rollout_state(clues)
    state.input_embed = model.encode_input(state.digit_id, state.clue_pin)
    state.cell_embed = state.input_embed.detach().requires_grad_(True)

    logits, _ = _inner_loop(
        model,
        state,
        inner_iters=3,
        truncated_bptt_steps=0,
        use_truncated_bptt=True,
    )
    logits.sum().backward()
    assert state.cell_embed.grad is None


def test_truncated_bptt_full_matches_default_inner_loop():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, _ = _tiny_batch()
    state = _init_rollout_state(clues)
    state.input_embed = model.encode_input(state.digit_id, state.clue_pin)

    torch.manual_seed(0)
    default_logits, _ = _inner_loop(model, state, inner_iters=3)
    torch.manual_seed(0)
    state2 = _init_rollout_state(clues)
    state2.input_embed = model.encode_input(state2.digit_id, state2.clue_pin)
    full_bptt_logits, _ = _inner_loop(
        model,
        state2,
        inner_iters=3,
        truncated_bptt_steps=3,
        use_truncated_bptt=True,
    )
    assert torch.allclose(default_logits, full_bptt_logits)


def test_fixed_point_changes_eval_loss():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    base_config = RolloutConfig(train_init="clues", inner_iters=3, outer_iters=1, fixed_point=False)
    fixed_config = RolloutConfig(
        train_init="clues",
        inner_iters=3,
        outer_iters=1,
        fixed_point=True,
    )
    base = rollout_train_batch(model, clues, answer, config=base_config)
    fixed = rollout_train_batch(model, clues, answer, config=fixed_config)
    assert fixed.loss.item() != base.loss.item()


def test_fixed_point_with_truncated_bptt_trains():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    config = RolloutConfig(
        train_init="clues",
        inner_iters=4,
        outer_iters=1,
        fixed_point=True,
        truncated_bptt_steps=2,
    )
    result = rollout_train_batch(
        model,
        clues,
        answer,
        config=config,
        compute_pred=False,
        accumulate_grad=True,
    )
    assert result.loss.item() > 0
    assert _first_param(model).grad is not None
    assert _first_param(model).grad.abs().sum().item() > 0


def test_rollout_fixed_steps_and_clues_pinned():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=4)
    with torch.inference_mode():
        result = rollout_train_batch(
            model,
            clues[0],
            answer[0],
            config=config,
        )
    assert result.pred is not None
    assert torch.all(result.pred[clues[0] > 0] == clues[0][clues[0] > 0])


def test_inner_bptt_grad():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=3, outer_iters=1)
    result = rollout_train_batch(
        model,
        clues,
        answer,
        config=config,
        compute_pred=False,
        accumulate_grad=True,
    )
    assert result.loss.item() > 0
    assert _first_param(model).grad is not None
    assert _first_param(model).grad.abs().sum().item() > 0


def test_outer_detach_isolates_blocks():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    ctx = _ClueContext.from_rollout_clues(clues)
    state = _init_rollout_state(clues)
    state.input_embed = model.encode_input(state.digit_id, state.clue_pin)

    logits_o1, _ = _inner_loop(model, state, inner_iters=2)
    digit_o1 = _outer_commit(logits_o1, ctx)
    assert not digit_o1.requires_grad

    state_o2 = RolloutState(
        digit_id=digit_o1,
        clue_pin=state.clue_pin,
        input_embed=model.encode_input(digit_o1, state.clue_pin),
        cell_embed=None,
    )
    cell_leaf = state_o2.input_embed.detach().requires_grad_(True)
    state_o2.cell_embed = cell_leaf
    logits_o2, _ = _inner_loop(model, state_o2, inner_iters=2)
    logits_o2.sum().backward()
    assert cell_leaf.grad is not None
    assert state.input_embed.grad is None


def test_per_outer_backward_matches_stacked_mean():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=3)
    ctx = _ClueContext.from_rollout_clues(clues)
    state = _init_rollout_state(clues)

    torch.manual_seed(0)
    result_acc = rollout_train_batch(
        model,
        clues,
        answer,
        config=config,
        compute_pred=False,
        accumulate_grad=True,
    )
    grad_acc = _first_param(model).grad.clone()

    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    _, _, stacked_loss = _rollout_loop(
        model,
        state,
        clues,
        ctx,
        config.inner_iters,
        config.outer_iters,
        answer=answer,
        accumulate_grad=False,
    )
    stacked_loss.backward()
    grad_stack = _first_param(model).grad

    assert result_acc.loss.item() == pytest.approx(stacked_loss.item())
    assert torch.allclose(grad_acc, grad_stack, rtol=1e-5, atol=1e-5)


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
    answer = torch.full((9, 9), 3)
    answer[0, 0] = 5
    torch.manual_seed(0)
    digit_id, rollout_clues = _curriculum_initial(answer, clues)
    assert digit_id[0, 0] == 5
    assert torch.all(rollout_clues[clues > 0] == clues[clues > 0])
    revealed = (rollout_clues > 0) & (clues == 0)
    if revealed.any():
        assert torch.all(rollout_clues[revealed] == answer[revealed])
    hidden = (rollout_clues == 0) & (clues == 0)
    if hidden.any():
        assert torch.all((digit_id[hidden] >= 1) & (digit_id[hidden] <= 9))


def test_curriculum_pins_revealed_cells_in_state():
    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 5
    answer = torch.full((9, 9), 4)
    answer[0, 0] = 5
    torch.manual_seed(1)
    digit_id, rollout_clues = _curriculum_initial(answer, clues)
    ctx = _ClueContext.from_rollout_clues(rollout_clues.unsqueeze(0))
    logits = torch.randn(1, 9, 9, 10)
    committed = _outer_commit(logits, ctx)
    for grid in (digit_id, committed[0]):
        assert grid[0, 0] == 5
        revealed = (rollout_clues > 0) & (clues == 0)
        if revealed.any():
            assert torch.all(grid[revealed] == answer[revealed])


def test_curriculum_excludes_revealed_from_loss_mask():
    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 5
    answer = torch.full((9, 9), 4)
    answer[0, 0] = 5
    answer[0, 1] = 6
    torch.manual_seed(2)
    _, rollout_clues = _curriculum_initial(answer, clues)
    clue_pin = rollout_clues > 0
    mask = target_mask(answer, clue_pin)
    revealed = (rollout_clues > 0) & (clues == 0)
    if revealed.any():
        assert not mask[revealed].any()


def test_reproducible_eval():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=3)

    r1 = rollout_train_batch(model, clues, answer, config=config)
    r2 = rollout_train_batch(model, clues, answer, config=config)
    assert torch.equal(r1.pred, r2.pred)
    assert r1.loss.item() == r2.loss.item()


def test_rollout_trace_frame_count():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, _ = _tiny_batch()
    outer_iters = 3
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=outer_iters)
    grids = rollout_trace(model, clues, config=config)
    assert len(grids) == outer_iters + 1


def test_rollout_trace_batch_matches_single():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, _ = _tiny_batch()
    clues2 = clues.clone()
    clues2[0, 0, 2] = 4
    clues_b = torch.cat([clues, clues2], dim=0)
    config = RolloutConfig(
        train_init="clues",
        inner_iters=2,
        outer_iters=3,
    )
    torch.manual_seed(0)
    single_first = rollout_trace(model, clues, config=config)
    torch.manual_seed(0)
    single_second = rollout_trace(model, clues2, config=config)
    torch.manual_seed(0)
    batched = rollout_trace_batch(model, clues_b, config=config)
    assert len(batched) == 2
    assert batched[0] == single_first
    assert batched[1] == single_second


def test_inner_one_outer_n_commits():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    ctx = _ClueContext.from_rollout_clues(clues)
    state = _init_rollout_state(clues)
    outer_iters = 4
    _, state_grids, losses = _rollout_loop(
        model,
        state,
        clues,
        ctx,
        inner_iters=1,
        outer_iters=outer_iters,
        answer=answer,
        collect_state_grids=True,
    )
    assert state_grids is not None
    assert len(state_grids) == outer_iters
    assert losses.ndim == 0


def test_defaults():
    config = RolloutConfig()
    assert config.inner_iters == DEFAULT_INNER_ITERS
    assert config.outer_iters == DEFAULT_OUTER_ITERS
    assert config.fixed_point is True
    assert config.truncated_bptt_steps == DEFAULT_TRUNCATED_BPTT_STEPS


def test_outer_commit_matches_full_decode():
    clues = torch.zeros(9, 9, dtype=torch.long)
    ctx = _ClueContext.from_rollout_clues(clues.unsqueeze(0))
    digit_id = clues.clone()
    logits = torch.zeros(1, 9, 9, 10)
    logits[0, 0, 0, 5] = 10.0  # predict digit 5
    committed = _outer_commit(logits, ctx)
    assert committed[0, 0, 0] == 5


def test_curriculum_training_rollout():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
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
    answer = torch.full((9, 9), 4)
    answer[0, 0] = 5
    config = RolloutConfig(train_init="curriculum", inner_iters=2, outer_iters=2)
    torch.manual_seed(0)
    result = rollout_train_batch(
        model,
        clues,
        answer,
        config=config,
        compute_pred=False,
        accumulate_grad=True,
    )
    assert result.loss.item() > 0
    assert _first_param(model).grad is not None


def test_accumulate_grad_required_in_training():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    with pytest.raises(ValueError, match="accumulate_grad"):
        rollout_train_batch(
            model,
            clues,
            answer,
            config=RolloutConfig(
                train_init="clues",
                inner_iters=2,
                outer_iters=1,
                fixed_point=False,
                truncated_bptt_steps=1,
            ),
        )
