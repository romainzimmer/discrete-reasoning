from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from dataset import PuzzleDataset
from memory import inner_halpern_alphas, inner_halpern_input
from model import MixerNextStateModel, ModelOutput
from rollout import (
    DEFAULT_INNER_ITERS,
    DEFAULT_MAX_OUTER_ITERS,
    BatchSlotState,
    RolloutConfig,
    _OnceEvalState,
    _copy_once_state,
    _compute_cell_loss,
    _compute_deep_supervision_losses,
    _compute_halt_loss,
    _compute_losses,
    _halt_target,
    _gt_reveal_init_digit_id,
    _gt_reveal_p_gt_for_slots,
    _sample_uniform_p_gt,
    _sample_uniform_p_gt_up_to,
    _inner_loop,
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
) -> BatchSlotState:
    clue_pin = clues > 0
    return BatchSlotState(
        digit_id=clues.clone(),
        clues=clues,
        answer=answer,
        clue_pin=clue_pin,
        rating_group=torch.zeros(clues.size(0), dtype=torch.long),
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
    with pytest.raises(ValueError, match="random_gt_reveal_p_gt requires gt_reveal"):
        RolloutConfig(gt_reveal=False, random_gt_reveal_p_gt=True)


def test_defaults():
    config = RolloutConfig()
    assert config.inner_iters == DEFAULT_INNER_ITERS
    assert config.max_outer_iters == DEFAULT_MAX_OUTER_ITERS
    assert config.deep_supervision is True
    assert config.gt_reveal is True
    assert config.random_init is False
    assert config.random_gt_reveal_p_gt is False


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
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen)
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
        gt_reveal=False,
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
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen)
    original_digit_id = state.digit_id.clone()
    done = torch.tensor([True])
    refill_done_slots(
        state,
        done,
        dataset,
        generator=gen,
        dim=32,
        gt_reveal=False,
        random_init=True,
    )
    assert not torch.equal(state.digit_id, original_digit_id)
    assert state.outer_count.item() == 0
    assert torch.equal(state.digit_id[state.clue_pin], state.clues[state.clue_pin])
    assert not torch.equal(state.digit_id[~state.clue_pin], state.clues[~state.clue_pin])


def test_max_outer_forces_refill():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen)
    state.outer_count[0] = 9
    config = _baseline_config(inner_iters=2, max_outer_iters=10, halt_threshold=1.1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    result = rollout_train_step(model, state, config)
    assert result.done is not None
    assert result.done.all()
    original_digit_id = state.digit_id.clone()
    refill_done_slots(
        state, result.done, dataset, generator=gen, dim=32, gt_reveal=False
    )
    assert not torch.equal(state.digit_id, original_digit_id)


def test_curriculum_init_preserves_clues():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    with patch("rollout.torch.rand", return_value=torch.ones(clues.shape)):
        digit_id = _gt_reveal_init_digit_id(clues, answer, clue_pin, p_gt=0.0)
    assert torch.equal(digit_id[clue_pin], clues[clue_pin])


def test_curriculum_init_gt_reveal():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    with patch("rollout.torch.rand", return_value=torch.zeros(clues.shape)):
        digit_id = _gt_reveal_init_digit_id(clues, answer, clue_pin, p_gt=1.0)
    assert torch.equal(digit_id[~clue_pin], answer[~clue_pin])


def test_curriculum_init_random_when_no_reveal():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    with patch("rollout.torch.rand", return_value=torch.ones(clues.shape)):
        with patch(
            "rollout.torch.randint",
            return_value=torch.full(clues.shape, 7, dtype=clues.dtype),
        ):
            digit_id = _gt_reveal_init_digit_id(
                clues, answer, clue_pin, p_gt=0.0, random_init=True
            )
    assert torch.equal(digit_id[clue_pin], clues[clue_pin])
    assert torch.equal(digit_id[~clue_pin], torch.full_like(clues, 7)[~clue_pin])


def test_gt_reveal_empty_init_leaves_unrevealed_cells_empty():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    with patch("rollout.torch.rand", return_value=torch.ones(clues.shape)):
        digit_id = _gt_reveal_init_digit_id(
            clues, answer, clue_pin, p_gt=0.0, random_init=False
        )
    assert torch.equal(digit_id[clue_pin], clues[clue_pin])
    assert torch.equal(digit_id[~clue_pin], torch.zeros_like(clues)[~clue_pin])


def test_gt_reveal_empty_init_still_reveals_gt():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    with patch("rollout.torch.rand", return_value=torch.zeros(clues.shape)):
        digit_id = _gt_reveal_init_digit_id(
            clues, answer, clue_pin, p_gt=1.0, random_init=False
        )
    assert torch.equal(digit_id[~clue_pin], answer[~clue_pin])


def test_eval_empty_init_leaves_non_clue_cells_empty():
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
        rollout_eval_batch(
            model,
            clues,
            answer,
            config=_baseline_config(inner_iters=1, max_outer_iters=1),
        )
    call_digit_id = mock_inner.call_args.args[1]
    assert torch.equal(call_digit_id[clue_pin], clues[clue_pin])
    assert torch.equal(call_digit_id[~clue_pin], torch.zeros_like(clues)[~clue_pin])


def test_curriculum_seed_fills_cells():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(42)
    state = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen)
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
        gt_reveal=False,
        random_init=True,
    )
    torch.manual_seed(42)
    refill_done_slots(
        state,
        torch.tensor([True]),
        dataset,
        generator=gen,
        dim=32,
        random_init=True,
    )
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
            rollout_eval_batch(
                model,
                clues,
                answer,
                config=_baseline_config(inner_iters=1, max_outer_iters=1, random_init=True),
            )
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

    def track_init(clues_tensor, clue_pin, *, init_seed=None, random_init=True, **kwargs):
        seeds_seen.append(init_seed)
        return original_init(
            clues_tensor, clue_pin, init_seed=init_seed, random_init=random_init
        )

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
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen1)
    state2 = BatchSlotState.seed(
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen2)
    done = torch.tensor([True])
    refill_done_slots(state1, done, dataset, generator=gen1, dim=32)
    refill_done_slots(state2, done, dataset, generator=gen2, dim=32)
    assert torch.equal(state1.clues, state2.clues)


def test_refill_no_op_when_not_done():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset, batch_size=2, device=torch.device("cpu"), generator=gen)
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


def test_predict_grid_gt_unpinned():
    clues = torch.zeros(9, 9, dtype=torch.long)
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


def test_curriculum_gt_overwritten_after_commit():
    clues, answer = _tiny_batch()
    non_clue = ~(clues > 0)
    state = _make_state(clues, answer)
    state.digit_id = torch.where(non_clue, answer, clues)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    with patch("rollout._inner_loop") as mock_inner:
        logits = torch.zeros(1, 9, 9, 10)
        logits[..., 2] = 10.0
        mock_inner.return_value = _deep_supervision_inner_return(
            logits, torch.zeros(1), torch.zeros(1, 9, 9, 32)
        )
        rollout_train_step(model, state, _baseline_config(), backward=False)
        rollout_train_step(model, state, _baseline_config(), backward=False)
    assert torch.equal(state.digit_id[non_clue], torch.full_like(state.digit_id[non_clue], 2))
    assert torch.equal(state.digit_id[state.clue_pin], clues[state.clue_pin])


def test_gt_cells_in_loss():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    logits = torch.zeros(1, 9, 9, 10)
    logits[..., 2] = 10.0
    loss_wrong = _compute_cell_loss(logits, clue_pin=clue_pin, answer=answer)
    logits[..., 1] = 10.0
    loss_right = _compute_cell_loss(logits, clue_pin=clue_pin, answer=answer)
    assert loss_wrong.item() > loss_right.item()


def test_train_step_wrong_gt_blocks_halt():
    clues, answer = _tiny_batch()
    non_clue = ~(clues > 0)
    state = _make_state(clues, answer)
    state.digit_id = torch.where(non_clue, answer, clues)
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
    assert torch.equal(state.digit_id[non_clue], answer[non_clue])
    assert not torch.equal(result.pred[non_clue], answer[non_clue])


def test_gt_reveal_not_in_encode_clue_pin():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    revealed = ~clue_pin
    digit_id = torch.where(revealed, answer, clues)
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
    assert not (passed_clue_pin & revealed).any()


def test_train_seed_gt_reveal_reveals_gt():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    with patch("rollout._sample_uniform_p_gt_up_to", return_value=torch.tensor([1.0])):
        with patch("rollout.torch.rand", return_value=torch.zeros(1, 9, 9)):
            state = BatchSlotState.seed(
                dataset,
                batch_size=1,
                device=torch.device("cpu"),
                generator=gen,
                gt_reveal_p_gt_caps=torch.ones(5),
            )
    assert torch.equal(state.digit_id[~state.clue_pin], state.answer[~state.clue_pin])
    assert torch.equal(state.digit_id[state.clue_pin], state.clues[state.clue_pin])


def test_sample_uniform_p_gt_returns_batch_on_device():
    p_gt = _sample_uniform_p_gt(3, torch.device("cpu"))
    assert p_gt.shape == (3,)
    assert p_gt.dtype == torch.float32
    assert torch.all(p_gt >= 0.0)
    assert torch.all(p_gt <= 1.0)


def test_sample_uniform_p_gt_is_reproducible_with_generator():
    gen1 = torch.Generator().manual_seed(11)
    gen2 = torch.Generator().manual_seed(11)
    p_gt_a = _sample_uniform_p_gt(4, torch.device("cpu"), generator=gen1)
    p_gt_b = _sample_uniform_p_gt(4, torch.device("cpu"), generator=gen2)
    assert torch.equal(p_gt_a, p_gt_b)


def test_curriculum_init_is_reproducible_with_generator():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    gen1 = torch.Generator().manual_seed(7)
    gen2 = torch.Generator().manual_seed(7)
    digit_a = _gt_reveal_init_digit_id(
        clues, answer, clue_pin, p_gt=0.5, generator=gen1
    )
    digit_b = _gt_reveal_init_digit_id(
        clues, answer, clue_pin, p_gt=0.5, generator=gen2
    )
    assert torch.equal(digit_a, digit_b)


def test_gt_reveal_p_gt_for_slots_adaptive_samples_within_cap():
    with patch("rollout._rand", return_value=torch.tensor([0.5, 0.25])):
        p_gt = _gt_reveal_p_gt_for_slots(
            rating_group=torch.tensor([0, 1]),
            device=torch.device("cpu"),
            gt_reveal=True,
            random_gt_reveal_p_gt=False,
            gt_reveal_p_gt_caps=torch.tensor([0.4, 0.8, 0.5, 0.5, 0.5]),
        )
    assert p_gt is not None
    assert torch.equal(p_gt, torch.tensor([0.2, 0.2]))


def test_sample_uniform_p_gt_up_to_scales_by_cap():
    caps = torch.tensor([0.0, 0.5, 1.0])
    with patch("rollout._rand", return_value=torch.tensor([0.4, 0.6, 0.8])):
        p_gt = _sample_uniform_p_gt_up_to(caps)
    assert torch.equal(p_gt, torch.tensor([0.0, 0.3, 0.8]))


def test_gt_reveal_p_gt_for_slots_random_ignores_caps():
    with patch("rollout._sample_uniform_p_gt", return_value=torch.tensor([0.3, 0.7])) as sample:
        p_gt = _gt_reveal_p_gt_for_slots(
            rating_group=torch.tensor([0, 4]),
            device=torch.device("cpu"),
            gt_reveal=True,
            random_gt_reveal_p_gt=True,
            gt_reveal_p_gt_caps=torch.tensor([0.5] * 5),
        )
    sample.assert_called_once()
    assert p_gt is not None
    assert torch.equal(p_gt, torch.tensor([0.3, 0.7]))


def test_train_seed_random_gt_reveal_samples_uniform_p_gt():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    with patch("rollout._sample_uniform_p_gt", return_value=torch.tensor([1.0])) as sample:
        with patch("rollout.torch.rand", return_value=torch.zeros(1, 9, 9)):
            state = BatchSlotState.seed(
                dataset,
                batch_size=1,
                device=torch.device("cpu"),
                generator=gen,
                random_gt_reveal_p_gt=True,
            )
        sample.assert_called_once_with(
            1, torch.device("cpu"), generator=gen
        )
    assert torch.equal(state.digit_id[~state.clue_pin], state.answer[~state.clue_pin])


def test_train_seed_random_gt_reveal_zero_p_gt_skips_gt_reveal():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    with patch("rollout._sample_uniform_p_gt", return_value=torch.tensor([0.0])):
        state = BatchSlotState.seed(
            dataset,
            batch_size=1,
            device=torch.device("cpu"),
            generator=gen,
            random_gt_reveal_p_gt=True,
        )
    assert not torch.equal(state.digit_id[~state.clue_pin], state.answer[~state.clue_pin])
    assert torch.equal(state.digit_id[state.clue_pin], state.clues[state.clue_pin])


def test_train_seed_without_gt_reveal_keeps_clues_only():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset,
        batch_size=1,
        device=torch.device("cpu"),
        generator=gen,
        gt_reveal=False,
    )
    assert torch.equal(state.digit_id, state.clues)


def test_mutable_non_clue_cell_overwritable_on_commit():
    clues, answer = _tiny_batch()
    state = _make_state(clues, answer)
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


def test_curriculum_partial_reveal():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    out = torch.full(clues.shape, 0.7)
    half = out.view(out.size(0), -1).size(1) // 2
    out.view(out.size(0), -1)[:, :half] = 0.3

    with patch("rollout.torch.rand", return_value=out):
        with patch("rollout.torch.randint", return_value=torch.zeros_like(clues)):
            digit_id = _gt_reveal_init_digit_id(clues, answer, clue_pin, p_gt=0.5)
    revealed = (digit_id == answer) & ~clue_pin
    empty = (digit_id == 0) & ~clue_pin
    assert revealed.any()
    assert empty.any()
    assert revealed.sum() < (~clue_pin).sum()
    assert not (revealed & clue_pin).any()


def test_refill_gt_reveal_reveals_gt():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset,
        batch_size=1,
        device=torch.device("cpu"),
        generator=gen,
        gt_reveal=False,
    )
    assert not torch.equal(state.digit_id[~state.clue_pin], state.answer[~state.clue_pin])
    with patch("rollout._sample_uniform_p_gt_up_to", return_value=torch.tensor([1.0])):
        with patch("rollout.torch.rand", return_value=torch.zeros(1, 9, 9)):
            refill_done_slots(
                state,
                torch.tensor([True]),
                dataset,
                generator=gen,
                dim=32,
                gt_reveal=True,
                gt_reveal_p_gt_caps=torch.ones(5),
            )
    assert torch.equal(state.digit_id[~state.clue_pin], state.answer[~state.clue_pin])


def test_refill_random_gt_reveal_resamples_uniform_p_gt():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset,
        batch_size=1,
        device=torch.device("cpu"),
        generator=gen,
        gt_reveal=False,
    )
    with patch("rollout._sample_uniform_p_gt", return_value=torch.tensor([1.0])) as sample:
        with patch("rollout.torch.rand", return_value=torch.zeros(1, 9, 9)):
            refill_done_slots(
                state,
                torch.tensor([True]),
                dataset,
                generator=gen,
                dim=32,
                gt_reveal=True,
                random_gt_reveal_p_gt=True,
            )
        sample.assert_called_once_with(
            1, torch.device("cpu"), generator=gen
        )
    assert torch.equal(state.digit_id[~state.clue_pin], state.answer[~state.clue_pin])


def test_refill_only_samples_done_slots():
    dataset = _tiny_dataset()
    gen = torch.Generator().manual_seed(0)
    state = BatchSlotState.seed(
        dataset,
        batch_size=4,
        device=torch.device("cpu"),
        generator=gen,
        gt_reveal=False,
    )
    before_clues = state.clues.clone()
    state.outer_count = torch.tensor([3, 5, 7, 9])
    with patch.object(dataset, "sample", wraps=dataset.sample) as sample_mock:
        refill_done_slots(
            state,
            torch.tensor([False, True, False, True]),
            dataset,
            generator=gen,
            dim=32,
            gt_reveal=False,
        )
    assert sample_mock.call_args[0][0].numel() == 2
    assert torch.equal(state.clues[0], before_clues[0])
    assert torch.equal(state.clues[2], before_clues[2])
    assert state.outer_count.tolist() == [3, 0, 7, 0]


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
    )
    assert config.inner_iters == 2
    assert config.max_outer_iters == 3
    assert config.gt_reveal is True


def test_eval_rollout_config_disables_gt_reveal():
    from train import build_rollout_config

    train_config = build_rollout_config(inner_iters=2, max_outer_iters=3, gt_reveal=True)
    eval_config = build_rollout_config(inner_iters=2, max_outer_iters=3, gt_reveal=False)
    assert train_config.gt_reveal is True
    assert eval_config.gt_reveal is False


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
    assert int(acc.total_cells.item()) == 0
    assert int(acc.correct_cells.item()) == 0


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
        dataset, batch_size=1, device=torch.device("cpu"), generator=gen,
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


def test_curriculum_low_p_gt_skips_reveal_at_threshold():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0

    with patch("rollout.torch.rand", return_value=torch.full(clues.shape, 0.5)):
        with patch("rollout.torch.randint", return_value=torch.zeros_like(clues)):
            digit_id = _gt_reveal_init_digit_id(clues, answer, clue_pin, p_gt=0.4)
    assert torch.equal(digit_id[~clue_pin], torch.zeros_like(answer[~clue_pin]))


def test_curriculum_p_gt_one_reveals_all():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0

    with patch("rollout.torch.rand", return_value=torch.full(clues.shape, 0.5)):
        with patch("rollout.torch.randint", return_value=torch.zeros_like(clues)):
            digit_id = _gt_reveal_init_digit_id(clues, answer, clue_pin, p_gt=1.0)
    assert torch.equal(digit_id[~clue_pin], answer[~clue_pin])


def test_curriculum_p_gt_zero_fills_non_clue_cells():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    digit_id = _gt_reveal_init_digit_id(
        clues, answer, clue_pin, p_gt=0.0, random_init=True
    )
    assert torch.equal(digit_id[clue_pin], clues[clue_pin])
    assert not torch.equal(digit_id[~clue_pin], clues[~clue_pin])


def test_curriculum_higher_p_gt_reveals_more():
    clues, answer = _tiny_batch()
    clue_pin = clues > 0
    draws = torch.full(clues.shape, 0.4)

    with patch("rollout.torch.rand", return_value=draws):
        with patch("rollout.torch.randint", return_value=torch.zeros_like(clues)):
            digit_id_low = _gt_reveal_init_digit_id(clues, answer, clue_pin, p_gt=0.3)
            digit_id_high = _gt_reveal_init_digit_id(clues, answer, clue_pin, p_gt=0.5)
    assert (digit_id_high == answer)[~clue_pin].sum() > (digit_id_low == answer)[~clue_pin].sum()


def test_gt_reveal_init_accepts_per_puzzle_p_gt_tensor():
    clues, answer = _tiny_batch()
    clues = clues.repeat(2, 1, 1)
    answer = answer.repeat(2, 1, 1)
    clue_pin = clues > 0
    p_gt = torch.tensor([0.0, 1.0])

    with patch("rollout.torch.rand", return_value=torch.full(clues.shape, 0.5)):
        with patch("rollout.torch.randint", return_value=torch.zeros_like(clues)):
            digit_id = _gt_reveal_init_digit_id(clues, answer, clue_pin, p_gt=p_gt)

    assert not torch.equal(digit_id[0, ~clue_pin[0]], answer[0, ~clue_pin[0]])
    assert torch.equal(digit_id[1, ~clue_pin[1]], answer[1, ~clue_pin[1]])


def test_build_rollout_config_new_flags():
    from train import build_rollout_config

    config = build_rollout_config(
        inner_iters=2,
        max_outer_iters=3,
        deep_supervision=True,
    )
    assert config.deep_supervision is True


def test_inner_loop_halpern_carry_sequence():
    model = MixerNextStateModel(dim=4, num_blocks=1)
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    anchor = torch.ones(1, 9, 9, 4)
    n = 3
    alphas = inner_halpern_alphas(n)
    base = 10.0
    seen_inputs: list[torch.Tensor | None] = []

    def fake_forward(self, *, input_embed, cell_embed=None):
        seen_inputs.append(cell_embed)
        step = len(seen_inputs) - 1
        h_new = torch.full((1, 9, 9, self.dim), base + step, device=input_embed.device)
        return ModelOutput(
            cell_embed=h_new,
            logits=torch.zeros(1, 9, 9, 10, device=input_embed.device),
            halt_logit=torch.zeros(1, device=input_embed.device),
        )

    with patch.object(MixerNextStateModel, "forward", fake_forward):
        *_, final_carry = _inner_loop(
            model, clues, clue_pin, n, memory_embed=anchor, with_grad=False
        )

    assert torch.equal(seen_inputs[0], anchor)
    carry: torch.Tensor | None = anchor
    for t in range(n):
        h_new = torch.full((1, 9, 9, 4), base + t)
        carry = h_new
        if t + 1 < n:
            expected_in = inner_halpern_input(
                anchor=anchor,
                carry=carry,
                alpha=alphas[t + 1],
            )
            assert torch.allclose(seen_inputs[t + 1], expected_in)
    assert torch.allclose(final_carry, carry)


def test_inner_loop_halpern_zero_anchor():
    model = MixerNextStateModel(dim=4, num_blocks=1)
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    n = 6
    h_new = torch.full((1, 9, 9, 4), 3.0)
    seen: list[torch.Tensor | None] = []

    def fake_forward(self, *, input_embed, cell_embed=None):
        seen.append(cell_embed)
        return ModelOutput(
            cell_embed=h_new,
            logits=torch.zeros(1, 9, 9, 10, device=input_embed.device),
            halt_logit=torch.zeros(1, device=input_embed.device),
        )

    with patch.object(MixerNextStateModel, "forward", fake_forward):
        *_, carry = _inner_loop(model, clues, clue_pin, n, memory_embed=None, with_grad=False)

    assert seen[0] is None
    assert torch.allclose(seen[1], h_new * (1.0 - inner_halpern_alphas(n)[1]))
    assert torch.equal(carry, h_new)


def test_inner_loop_halpern_mixed_anchor_batch():
    model = MixerNextStateModel(dim=4, num_blocks=1)
    clues = torch.zeros(2, 9, 9, dtype=torch.long)
    clue_pin = torch.zeros(2, 9, 9, dtype=torch.bool)
    anchor = torch.stack(
        [
            torch.ones(9, 9, 4),
            torch.zeros(9, 9, 4),
        ]
    )
    n = 3
    alpha = inner_halpern_alphas(n)[1]
    h_new = torch.stack(
        [
            torch.full((9, 9, 4), 2.0),
            torch.full((9, 9, 4), 4.0),
        ]
    )
    seen: list[torch.Tensor | None] = []

    def fake_forward(self, *, input_embed, cell_embed=None):
        seen.append(cell_embed)
        return ModelOutput(
            cell_embed=h_new,
            logits=torch.zeros(2, 9, 9, 10, device=input_embed.device),
            halt_logit=torch.zeros(2, device=input_embed.device),
        )

    with patch.object(MixerNextStateModel, "forward", fake_forward):
        *_, carry = _inner_loop(model, clues, clue_pin, n, memory_embed=anchor, with_grad=False)

    expected_row0 = alpha * 1.0 + (1.0 - alpha) * 2.0
    expected_row1 = (1.0 - alpha) * 4.0
    assert torch.allclose(seen[1][0], torch.full((9, 9, 4), expected_row0))
    assert torch.allclose(seen[1][1], torch.full((9, 9, 4), expected_row1))
    assert not torch.allclose(seen[1][0], seen[1][1])
    assert torch.equal(carry, h_new)


def test_inner_loop_halpern_t_in_1():
    model = MixerNextStateModel(dim=4, num_blocks=1)
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    anchor = torch.full((1, 9, 9, 4), 7.0)
    h_new = torch.full((1, 9, 9, 4), 11.0)
    seen: list[torch.Tensor | None] = []

    def fake_forward(self, *, input_embed, cell_embed=None):
        seen.append(cell_embed)
        return ModelOutput(
            cell_embed=h_new,
            logits=torch.zeros(1, 9, 9, 10, device=input_embed.device),
            halt_logit=torch.zeros(1, device=input_embed.device),
        )

    with patch.object(MixerNextStateModel, "forward", fake_forward):
        *_, carry = _inner_loop(model, clues, clue_pin, 1, memory_embed=anchor, with_grad=False)

    assert torch.equal(seen[0], anchor)
    assert torch.equal(carry, h_new)


def test_inner_loop_halpern_t_in_2():
    model = MixerNextStateModel(dim=4, num_blocks=1)
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    anchor = torch.ones(1, 9, 9, 4)
    outputs = [
        torch.full((1, 9, 9, 4), 2.0),
        torch.full((1, 9, 9, 4), 5.0),
    ]
    step = {"i": 0}
    seen: list[torch.Tensor | None] = []

    def fake_forward(self, *, input_embed, cell_embed=None):
        seen.append(cell_embed)
        h_new = outputs[step["i"]]
        step["i"] += 1
        return ModelOutput(
            cell_embed=h_new,
            logits=torch.zeros(1, 9, 9, 10, device=input_embed.device),
            halt_logit=torch.zeros(1, device=input_embed.device),
        )

    with patch.object(MixerNextStateModel, "forward", fake_forward):
        *_, carry = _inner_loop(model, clues, clue_pin, 2, memory_embed=anchor, with_grad=False)

    assert torch.equal(seen[0], anchor)
    assert torch.equal(seen[1], outputs[0])
    assert torch.equal(carry, outputs[1])
