from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from encoding import decode_logits, target_mask
from model import MixerNextStateModel
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_OUTER_ITERS,
    RolloutConfig,
    RolloutState,
    _ClueContext,
    _compute_rollout_loss_batch_mean,
    _curriculum_initial,
    _inner_loop,
    _init_rollout_state,
    _noisy_ground_truth_initial,
    _outer_commit,
    _rollout_loop,
    _training_rollout_inputs,
    _zeroed_ground_truth_initial,
    predict_grid,
    rollout_train_batch,
    rollout_solve,
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


def test_inner_loop_grad():
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

    logits_o1 = _inner_loop(model, state, inner_iters=2)
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
    logits_o2 = _inner_loop(model, state_o2, inner_iters=2, with_grad=True)
    logits_o2.sum().backward()
    assert cell_leaf.grad is not None
    assert state.input_embed.grad is None


def test_last_outer_backward_matches_rollout_train_batch():
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
    _, _, direct_loss = _rollout_loop(
        model,
        state,
        clues,
        ctx,
        config.inner_iters,
        config.outer_iters,
        answer=answer,
        accumulate_grad=True,
    )
    grad_direct = _first_param(model).grad

    assert result_acc.loss.item() == pytest.approx(direct_loss.item())
    assert torch.allclose(grad_acc, grad_direct, rtol=1e-5, atol=1e-5)


def test_earlier_outer_loops_do_not_backprop():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=3)
    ctx = _ClueContext.from_rollout_clues(clues)
    state = _init_rollout_state(clues)
    state.input_embed = model.encode_input(state.digit_id, state.clue_pin)

    with torch.no_grad():
        logits_o1 = _inner_loop(model, state, inner_iters=2)
        digit_o1 = _outer_commit(logits_o1, ctx)
    state.digit_id = digit_o1
    state.input_embed = model.encode_input(state.digit_id, state.clue_pin)
    state.cell_embed = None

    with torch.no_grad():
        logits_o2 = _inner_loop(model, state, inner_iters=2)
        digit_o2 = _outer_commit(logits_o2, ctx)
    state.digit_id = digit_o2
    state.input_embed = model.encode_input(state.digit_id, state.clue_pin)
    state.cell_embed = None

    logits_o3 = _inner_loop(model, state, inner_iters=2, with_grad=True)
    loss = _compute_rollout_loss_batch_mean(
        logits_o3,
        clue_pin=ctx.clue_pin,
        answer=answer,
    )
    loss.backward()
    assert _first_param(model).grad is not None
    assert state.input_embed.grad is None


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
        assert torch.all((digit_id[hidden] >= 0) & (digit_id[hidden] <= 9))


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


def test_curriculum_hidden_can_sample_empty():
    clues = torch.zeros(9, 9, dtype=torch.long)
    answer = torch.full((9, 9), 4)
    with (
        patch("rollout.torch.rand", return_value=torch.tensor(1.0)),
        patch(
            "rollout.torch.rand_like",
            return_value=torch.full_like(clues, 0.5, dtype=torch.float32),
        ),
        patch("rollout.torch.randint", return_value=torch.zeros((), dtype=torch.long)),
    ):
        digit_id, _ = _curriculum_initial(answer, clues)
    hidden = clues == 0
    assert torch.all(digit_id[hidden] == 0)


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


def test_outer_commit_matches_full_decode():
    clues = torch.zeros(9, 9, dtype=torch.long)
    ctx = _ClueContext.from_rollout_clues(clues.unsqueeze(0))
    digit_id = clues.clone()
    logits = torch.zeros(1, 9, 9, 10)
    logits[0, 0, 0, 5] = 10.0  # predict digit 5
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


def _simple_clue_answer():
    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 5
    answer = torch.full((9, 9), 4)
    answer[0, 0] = 5
    return clues, answer


def test_noisy_gt_pins_clues_and_fills_non_clue():
    clues, answer = _simple_clue_answer()
    torch.manual_seed(0)
    initial = _noisy_ground_truth_initial(answer, clues)
    assert initial[0, 0] == 5
    non_clue = clues == 0
    assert torch.all((initial[non_clue] >= 0) & (initial[non_clue] <= 9))


def test_noisy_gt_flip_can_sample_empty():
    clues, answer = _simple_clue_answer()
    with (
        patch("rollout.torch.rand", return_value=torch.tensor(1.0)),
        patch(
            "rollout.torch.rand_like",
            return_value=torch.full_like(clues, 0.5, dtype=torch.float32),
        ),
        patch("rollout.torch.randint", return_value=torch.full_like(answer, 6)),
    ):
        initial = _noisy_ground_truth_initial(answer, clues)
    non_clue = clues == 0
    assert torch.any(initial[non_clue] == 0)


def test_noisy_gt_flips_change_digit_when_p_one():
    clues, answer = _simple_clue_answer()
    with (
        patch("rollout.torch.rand", return_value=torch.tensor(1.0)),
        patch(
            "rollout.torch.rand_like",
            return_value=torch.full_like(clues, 0.5, dtype=torch.float32),
        ),
    ):
        initial = _noisy_ground_truth_initial(answer, clues)
    non_clue = clues == 0
    assert torch.all(initial[non_clue] != answer[non_clue])


def test_noisy_gt_keeps_answer_when_p_zero():
    clues, answer = _simple_clue_answer()
    with patch("rollout.torch.rand", return_value=torch.tensor(0.0)):
        initial = _noisy_ground_truth_initial(answer, clues)
    assert torch.equal(initial, answer)


def test_zero_gt_zeros_non_clue_when_p_one():
    clues, answer = _simple_clue_answer()
    with (
        patch("rollout.torch.rand", return_value=torch.tensor(1.0)),
        patch(
            "rollout.torch.rand_like",
            return_value=torch.full_like(clues, 0.5, dtype=torch.float32),
        ),
    ):
        initial = _zeroed_ground_truth_initial(answer, clues)
    assert initial[0, 0] == 5
    non_clue = clues == 0
    assert torch.all(initial[non_clue] == 0)


def test_zero_gt_keeps_answer_when_p_zero():
    clues, answer = _simple_clue_answer()
    with patch("rollout.torch.rand", return_value=torch.tensor(0.0)):
        initial = _zeroed_ground_truth_initial(answer, clues)
    assert torch.equal(initial, answer)


@pytest.mark.parametrize("train_init", ["clues", "noisy_gt", "zero_gt", "curriculum"])
def test_training_rollout_inputs_dispatch(train_init: str):
    clues, answer = _simple_clue_answer()
    config = RolloutConfig(train_init=train_init)
    initial_digit_id, rollout_clues, clue_pin = _training_rollout_inputs(config, answer, clues)

    if train_init == "clues":
        assert initial_digit_id is None
        assert torch.equal(rollout_clues, clues)
        assert torch.equal(clue_pin, clues > 0)
    elif train_init == "noisy_gt":
        assert initial_digit_id is not None
        assert torch.equal(rollout_clues, clues)
        assert torch.equal(clue_pin, clues > 0)
        assert torch.all(initial_digit_id[clues > 0] == clues[clues > 0])
    elif train_init == "zero_gt":
        assert initial_digit_id is not None
        assert torch.equal(rollout_clues, clues)
        assert torch.equal(clue_pin, clues > 0)
    elif train_init == "curriculum":
        assert initial_digit_id is not None
        assert torch.all(rollout_clues[clues > 0] == clues[clues > 0])
        assert torch.equal(clue_pin, rollout_clues > 0)


def test_eval_skips_train_init():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    clues_config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=2)
    noisy_config = RolloutConfig(train_init="noisy_gt", inner_iters=2, outer_iters=2)
    torch.manual_seed(0)
    clues_result = rollout_train_batch(model, clues, answer, config=clues_config)
    torch.manual_seed(0)
    noisy_result = rollout_train_batch(model, clues, answer, config=noisy_config)
    assert clues_result.loss.item() == noisy_result.loss.item()
    assert torch.equal(clues_result.pred, noisy_result.pred)


@pytest.mark.parametrize("train_init", ["noisy_gt", "zero_gt"])
def test_train_init_training_rollout(train_init: str):
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    config = RolloutConfig(train_init=train_init, inner_iters=2, outer_iters=1)
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


def test_noisy_gt_differs_from_clues_initial():
    clues, answer = _simple_clue_answer()
    with (
        patch("rollout.torch.rand", return_value=torch.tensor(1.0)),
        patch(
            "rollout.torch.rand_like",
            return_value=torch.full_like(clues, 0.5, dtype=torch.float32),
        ),
    ):
        noisy_initial, _, _ = _training_rollout_inputs(
            RolloutConfig(train_init="noisy_gt"),
            answer,
            clues,
        )
    clues_initial, _, _ = _training_rollout_inputs(
        RolloutConfig(train_init="clues"),
        answer,
        clues,
    )
    assert clues_initial is None
    assert noisy_initial is not None
    assert not torch.equal(noisy_initial, clues)


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
            ),
        )


@pytest.mark.parametrize("train_init", ["clues", "noisy_gt", "zero_gt", "curriculum"])
def test_all_train_inits_with_multiple_outer_iters(train_init: str):
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues, answer = _tiny_batch()
    config = RolloutConfig(train_init=train_init, inner_iters=2, outer_iters=3)
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


@pytest.mark.parametrize("outer_iters", [1, 3])
def test_eval_pred_matches_rollout_solve(outer_iters: int):
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=outer_iters)
    result = rollout_train_batch(model, clues, answer, config=config)
    assert result.pred is not None
    for idx in range(clues.size(0)):
        assert torch.equal(result.pred[idx], rollout_solve(model, clues[idx], config=config))


def test_build_rollout_config():
    from train import build_rollout_config

    config = build_rollout_config(train_init="noisy_gt", inner_iters=2, outer_iters=3)
    assert config.train_init == "noisy_gt"
    assert config.inner_iters == 2
    assert config.outer_iters == 3
