from __future__ import annotations

import pytest
import torch

from encoding import attach_clue_mask, grid_to_onehot, onehot_to_grid
from model import NextStateModel
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_OUTER_COMMIT_PROB,
    DEFAULT_OUTER_ITERS,
    DEFAULT_TRUNCATED_BPTT_STEPS,
    RolloutConfig,
    _ClueContext,
    _curriculum_initial,
    _inner_loop,
    _rollout_loop,
    logits_to_argmax_state,
    logits_to_softmax_state,
    predict_grid,
    rollout_train_batch,
    rollout_trace,
    target_mask,
)


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


def test_rollout_config_validation():
    with pytest.raises(ValueError):
        RolloutConfig(inner_iters=0)
    with pytest.raises(ValueError):
        RolloutConfig(outer_iters=0)
    with pytest.raises(ValueError):
        RolloutConfig(outer_commit_prob=1.5)
    with pytest.raises(ValueError):
        RolloutConfig(outer_commit_prob=0.0)
    with pytest.raises(ValueError):
        RolloutConfig(inner_iters=3, truncated_bptt_steps=4)
    with pytest.raises(ValueError):
        RolloutConfig(inner_iters=1, fixed_point=True)


def test_truncated_bptt_zero_detaches_inner_prefix():
    model = NextStateModel(width=32, num_blocks=1)
    model.train()
    clues, clues_onehot, answer = _tiny_batch()
    clues_b = clues
    clues_onehot_b = clues_onehot
    ctx = _ClueContext.from_clues(clues_b, clues_onehot_b)
    state = attach_clue_mask(clues_onehot_b, clues_b, clue_mask_channel=ctx.clue_mask_channel)
    state_leaf = state.detach().requires_grad_(True)

    logits, _ = _inner_loop(
        model,
        state_leaf,
        ctx,
        clues_b,
        inner_iters=3,
        truncated_bptt_steps=0,
        use_truncated_bptt=True,
    )
    logits.sum().backward()
    assert state_leaf.grad is None


def test_truncated_bptt_full_matches_default_inner_loop():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, _ = _tiny_batch()
    ctx = _ClueContext.from_clues(clues, clues_onehot)
    state = attach_clue_mask(clues_onehot, clues, clue_mask_channel=ctx.clue_mask_channel)

    torch.manual_seed(0)
    default_logits, _ = _inner_loop(model, state, ctx, clues, inner_iters=3)
    torch.manual_seed(0)
    full_bptt_logits, _ = _inner_loop(
        model,
        state,
        ctx,
        clues,
        inner_iters=3,
        truncated_bptt_steps=3,
        use_truncated_bptt=True,
    )
    assert torch.allclose(default_logits, full_bptt_logits)


def test_fixed_point_changes_eval_loss():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, answer = _tiny_batch()
    base_config = RolloutConfig(train_init="clues", inner_iters=3, outer_iters=1, fixed_point=False)
    fixed_config = RolloutConfig(
        train_init="clues",
        inner_iters=3,
        outer_iters=1,
        fixed_point=True,
    )
    base = rollout_train_batch(model, clues, clues_onehot, answer, config=base_config)
    fixed = rollout_train_batch(model, clues, clues_onehot, answer, config=fixed_config)
    assert fixed.loss.item() != base.loss.item()


def test_fixed_point_with_truncated_bptt_trains():
    model = NextStateModel(width=32, num_blocks=1)
    model.train()
    clues, clues_onehot, answer = _tiny_batch()
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
        clues_onehot,
        answer,
        config=config,
        compute_pred=False,
        accumulate_grad=True,
    )
    assert result.loss.item() > 0
    assert model.net[0].weight.grad is not None
    assert model.net[0].weight.grad.abs().sum().item() > 0


def test_rollout_fixed_steps_and_clues_pinned():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=4)
    with torch.inference_mode():
        result = rollout_train_batch(
            model,
            clues[0],
            clues_onehot[0],
            answer[0],
            config=config,
        )
    assert result.pred is not None
    assert torch.all(result.pred[clues[0] > 0] == clues[0][clues[0] > 0])


def test_inner_bptt_grad():
    model = NextStateModel(width=32, num_blocks=1)
    model.train()
    clues, clues_onehot, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=3, outer_iters=1)
    result = rollout_train_batch(
        model,
        clues,
        clues_onehot,
        answer,
        config=config,
        compute_pred=False,
        accumulate_grad=True,
    )
    assert result.loss.item() > 0
    assert model.net[0].weight.grad is not None
    assert model.net[0].weight.grad.abs().sum().item() > 0


def test_outer_detach_isolates_blocks():
    model = NextStateModel(width=32, num_blocks=1)
    model.train()
    clues, clues_onehot, answer = _tiny_batch()
    clues_b = clues
    clues_onehot_b = clues_onehot
    ctx = _ClueContext.from_clues(clues_b, clues_onehot_b)
    state = attach_clue_mask(clues_onehot_b, clues_b, clue_mask_channel=ctx.clue_mask_channel)

    logits_o1, _ = _inner_loop(model, state, ctx, clues_b, inner_iters=2)
    state_o1 = logits_to_argmax_state(logits_o1, ctx, clues_b, state=state, outer_commit_prob=1.0)
    assert not state_o1.requires_grad

    state_o1_leaf = state_o1.detach().requires_grad_(True)
    logits_o2, _ = _inner_loop(model, state_o1_leaf, ctx, clues_b, inner_iters=2)
    logits_o2.sum().backward()
    assert state_o1_leaf.grad is not None
    assert state.grad is None


def test_per_outer_backward_matches_stacked_mean():
    model = NextStateModel(width=32, num_blocks=1)
    model.train()
    clues, clues_onehot, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=3, outer_commit_prob=1.0)
    clues_b = clues
    clues_onehot_b = clues_onehot
    ctx = _ClueContext.from_clues(clues_b, clues_onehot_b)
    state = attach_clue_mask(clues_onehot_b, clues_b, clue_mask_channel=ctx.clue_mask_channel)

    torch.manual_seed(0)
    result_acc = rollout_train_batch(
        model,
        clues,
        clues_onehot,
        answer,
        config=config,
        compute_pred=False,
        accumulate_grad=True,
    )
    grad_acc = model.net[0].weight.grad.clone()

    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    _, _, stacked_loss = _rollout_loop(
        model,
        state,
        clues_b,
        ctx,
        config.inner_iters,
        config.outer_iters,
        outer_commit_prob=config.outer_commit_prob,
        answer=answer,
        accumulate_grad=False,
    )
    stacked_loss.backward()
    grad_stack = model.net[0].weight.grad

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
    onehot, rollout_clues = _curriculum_initial(answer, clues)
    grid = onehot_to_grid(onehot)
    assert grid[0, 0] == 5
    assert torch.all(rollout_clues[clues > 0] == clues[clues > 0])
    revealed = (rollout_clues > 0) & (clues == 0)
    if revealed.any():
        assert torch.all(rollout_clues[revealed] == answer[revealed])
    hidden = (rollout_clues == 0) & (clues == 0)
    if hidden.any():
        assert torch.all((grid[hidden] >= 1) & (grid[hidden] <= 9))


def test_curriculum_pins_revealed_cells_in_state():
    clues = torch.zeros(9, 9, dtype=torch.long)
    clues[0, 0] = 5
    answer = torch.full((9, 9), 4)
    answer[0, 0] = 5
    torch.manual_seed(1)
    initial_onehot, rollout_clues = _curriculum_initial(answer, clues)
    ctx = _ClueContext.from_clues(rollout_clues.unsqueeze(0), grid_to_onehot(rollout_clues).unsqueeze(0))
    logits = torch.randn(1, 9, 9, 9)
    softmax_state = logits_to_softmax_state(logits, ctx, rollout_clues.unsqueeze(0))
    argmax_state = logits_to_argmax_state(
        logits,
        ctx,
        rollout_clues.unsqueeze(0),
        state=softmax_state,
        outer_commit_prob=1.0,
    )
    for state in (softmax_state, argmax_state):
        grid = onehot_to_grid(state[0, ..., :9])
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
    mask = target_mask(answer, rollout_clues)
    revealed = (rollout_clues > 0) & (clues == 0)
    if revealed.any():
        assert not mask[revealed].any()


def test_reproducible_eval():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, answer = _tiny_batch()
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=3, outer_commit_prob=1.0)

    r1 = rollout_train_batch(model, clues, clues_onehot, answer, config=config)
    r2 = rollout_train_batch(model, clues, clues_onehot, answer, config=config)
    assert torch.equal(r1.pred, r2.pred)
    assert r1.loss.item() == r2.loss.item()


def test_rollout_trace_frame_count():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, _ = _tiny_batch()
    outer_iters = 3
    config = RolloutConfig(train_init="clues", inner_iters=2, outer_iters=outer_iters)
    grids = rollout_trace(model, clues_onehot, clues, config=config)
    assert len(grids) == outer_iters + 1


def test_inner_one_outer_n_commits():
    model = NextStateModel(width=32, num_blocks=1)
    model.eval()
    clues, clues_onehot, answer = _tiny_batch()
    clues_b = clues
    clues_onehot_b = clues_onehot
    ctx = _ClueContext.from_clues(clues_b, clues_onehot_b)
    state = attach_clue_mask(clues_onehot_b, clues_b, clue_mask_channel=ctx.clue_mask_channel)
    outer_iters = 4
    _, state_grids, losses = _rollout_loop(
        model,
        state,
        clues_b,
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
    assert config.outer_commit_prob == DEFAULT_OUTER_COMMIT_PROB
    assert config.fixed_point is True
    assert config.truncated_bptt_steps == DEFAULT_TRUNCATED_BPTT_STEPS


def test_outer_commit_prob_one_matches_full_decode():
    clues = torch.zeros(9, 9, dtype=torch.long)
    ctx = _ClueContext.from_clues(clues.unsqueeze(0), grid_to_onehot(clues).unsqueeze(0))
    state = attach_clue_mask(grid_to_onehot(clues).unsqueeze(0), clues.unsqueeze(0), clue_mask_channel=ctx.clue_mask_channel)
    state[..., 0, 0, 2] = 1.0
    logits = torch.zeros(1, 9, 9, 9)
    logits[0, 0, 0, 4] = 10.0  # predict digit 5
    committed = logits_to_argmax_state(
        logits,
        ctx,
        clues.unsqueeze(0),
        state=state,
        outer_commit_prob=1.0,
    )
    assert onehot_to_grid(committed[0, ..., :9])[0, 0] == 5


def test_curriculum_training_rollout():
    model = NextStateModel(width=32, num_blocks=1)
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
    clues_onehot = grid_to_onehot(clues)
    config = RolloutConfig(train_init="curriculum", inner_iters=2, outer_iters=2)
    torch.manual_seed(0)
    result = rollout_train_batch(
        model,
        clues,
        clues_onehot,
        answer,
        config=config,
        compute_pred=False,
        accumulate_grad=True,
    )
    assert result.loss.item() > 0
    assert model.net[0].weight.grad is not None


def test_accumulate_grad_required_in_training():
    model = NextStateModel(width=32, num_blocks=1)
    model.train()
    clues, clues_onehot, answer = _tiny_batch()
    with pytest.raises(ValueError, match="accumulate_grad"):
        rollout_train_batch(
            model,
            clues,
            clues_onehot,
            answer,
            config=RolloutConfig(train_init="clues", inner_iters=1, outer_iters=1),
        )
