from __future__ import annotations

from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from dataset import PuzzleDataset
from model import MixerNextStateModel
from rollout import (
    BatchSlotState,
    RolloutConfig,
    _eval_outer_done,
    _init_digit_id_from_clues,
    _inner_loop,
    _train_outer_done,
    rollout_eval_batch,
    rollout_eval_stream,
    rollout_train_step,
)
from train import (
    _accumulate_static_eval_batches,
    _accumulate_stream_eval,
    build_rollout_config,
    measure_split,
)


def _assert_row_allclose(solo: torch.Tensor, batched_row: torch.Tensor) -> None:
    assert torch.allclose(solo, batched_row, rtol=1e-5, atol=1e-6)


def _three_puzzle_dataset() -> PuzzleDataset:
    clues = torch.zeros(3, 9, 9, dtype=torch.long)
    clues[:, 0, 0] = torch.tensor([5, 6, 7])
    answers = torch.full((3, 9, 9), 1, dtype=torch.long)
    return PuzzleDataset.from_tensors(clues, answers)


def _run_measure(
    model: MixerNextStateModel,
    dataset: PuzzleDataset,
    *,
    slot_batch_size: int,
    use_stream: bool,
    max_tries: int = 1,
) -> dict:
    config = build_rollout_config(inner_iters=1, max_outer_iters=3, curriculum_training=False)
    stats = measure_split(
        model,
        dataset._base_clues,
        dataset._base_answers,
        torch.device("cpu"),
        slot_batch_size=slot_batch_size,
        epoch=1,
        epochs=1,
        phase="test",
        rollout_config=config,
        halt_loss_weight=1.0,
        use_cuda=False,
        seed=0,
        max_tries=max_tries,
        use_stream=use_stream,
    )
    return asdict(stats)


def test_inner_loop_row_outputs_independent_of_batch_size():
    """Each puzzle row gets the same forward outputs alone or batched with others."""
    torch.manual_seed(0)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues = torch.zeros(3, 9, 9, dtype=torch.long)
    clues[:, 0, 0] = torch.tensor([5, 6, 7])
    clue_pin = clues > 0
    digit_id = _init_digit_id_from_clues(clues, clue_pin, init_seed=0)

    solo_first: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i in range(3):
        solo_first.append(
            _inner_loop(
                model,
                digit_id[i : i + 1],
                clue_pin[i : i + 1],
                2,
                memory_embed=None,
                with_grad=False,
            )
        )

    batched_first = _inner_loop(
        model,
        digit_id,
        clue_pin,
        2,
        memory_embed=None,
        with_grad=False,
    )

    for i in range(3):
        solo_logits, solo_halt, solo_mem = solo_first[i]
        _assert_row_allclose(solo_logits, batched_first[0][i : i + 1])
        _assert_row_allclose(solo_halt, batched_first[1][i : i + 1])
        _assert_row_allclose(solo_mem, batched_first[2][i : i + 1])

    solo_second: list[torch.Tensor] = []
    for i in range(3):
        logits, _, _ = _inner_loop(
            model,
            digit_id[i : i + 1],
            clue_pin[i : i + 1],
            2,
            memory_embed=solo_first[i][2],
            with_grad=False,
        )
        solo_second.append(logits)

    batched_second_logits, _, _ = _inner_loop(
        model,
        digit_id,
        clue_pin,
        2,
        memory_embed=batched_first[2],
        with_grad=False,
    )
    for i in range(3):
        _assert_row_allclose(solo_second[i], batched_second_logits[i : i + 1])


def test_rollout_eval_batch_row_independent_of_batch_size():
    """Full single-try eval gives identical per-puzzle preds alone vs batched."""
    torch.manual_seed(1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()
    clues = dataset._base_clues
    answers = dataset._base_answers
    config = RolloutConfig(inner_iters=1, max_outer_iters=2, curriculum_training=False)

    solo_preds = []
    for i in range(clues.size(0)):
        solo_preds.append(
            rollout_eval_batch(
                model,
                clues[i : i + 1],
                answers[i : i + 1],
                config=config,
                init_seed=0,
            ).pred
        )

    batched = rollout_eval_batch(
        model,
        clues,
        answers,
        config=config,
        init_seed=0,
    )
    for i in range(clues.size(0)):
        assert torch.equal(solo_preds[i].squeeze(0), batched.pred[i])


def test_eval_outer_done_ignores_correctness():
    predict_halt = torch.tensor([True])
    outer_count = torch.tensor([1])
    solved = torch.tensor([False])
    assert _eval_outer_done(predict_halt, outer_count, 10).item() is True
    assert _train_outer_done(predict_halt, solved, outer_count, 10).item() is False


def test_eval_halt_without_correct_grid_stops():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _three_puzzle_dataset()._base_clues[:1], _three_puzzle_dataset()._base_answers[:1]
    config = RolloutConfig(inner_iters=1, max_outer_iters=10, curriculum_training=False)
    with patch("rollout._predict_halt", return_value=torch.tensor([True])):
        result = rollout_eval_batch(model, clues, answer, config=config)
    assert result.outer_steps.item() == 1
    assert result.halted.item() is True
    assert not torch.equal(result.pred, answer)


def test_eval_stream_never_passes_answer_to_inner_loop():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()
    answers = dataset._base_answers
    answer_ptrs = {answers[i].data_ptr() for i in range(answers.size(0))}

    def track_inner(model_obj, digit_id, clue_pin, inner_iters, **kwargs):
        for value in (digit_id, clue_pin, kwargs.get("memory_embed")):
            if isinstance(value, torch.Tensor) and value.data_ptr() in answer_ptrs:
                raise AssertionError("answer tensor leaked into eval forward path")
        return (
            torch.zeros(digit_id.size(0), 9, 9, 10),
            torch.zeros(digit_id.size(0)),
            torch.zeros(digit_id.size(0), 9, 9, 32),
        )

    with patch("rollout._inner_loop", side_effect=track_inner):
        with patch("rollout._init_digit_id_from_clues", wraps=_init_digit_id_from_clues) as init_mock:
            list(
                rollout_eval_stream(
                    model,
                    dataset._base_clues,
                    dataset._base_answers,
                    slot_batch_size=2,
                    config=RolloutConfig(inner_iters=1, max_outer_iters=2, curriculum_training=False),
                    init_seed=0,
                )
            )
    for call in init_mock.call_args_list:
        assert call.args[1] is not answers
        assert answers.data_ptr() not in {a.data_ptr() for a in call.args if isinstance(a, torch.Tensor)}


def test_stream_refill_mixed_memory_slots():
    """Refilled slots start with memory_embed=None while others carry memory."""
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()
    step = {"n": 0}

    def halt_first_slot_once(halt_logit, *, halt_threshold=0.5):
        step["n"] += 1
        b = halt_logit.size(0)
        if step["n"] == 1:
            out = torch.zeros(b, dtype=torch.bool)
            out[0] = True
            return out
        return torch.zeros(b, dtype=torch.bool)

    with patch("rollout._predict_halt", side_effect=halt_first_slot_once):
        results = list(
            rollout_eval_stream(
                model,
                dataset._base_clues,
                dataset._base_answers,
                slot_batch_size=2,
                config=RolloutConfig(inner_iters=1, max_outer_iters=3, curriculum_training=False),
                init_seed=0,
            )
        )
    assert len(results) == 3


def test_stream_yield_clues_match_dataset_puzzles():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()

    def halt_all(halt_logit, *, halt_threshold=0.5):
        return torch.ones(halt_logit.size(0), dtype=torch.bool, device=halt_logit.device)

    with patch("rollout._predict_halt", side_effect=halt_all):
        results = list(
            rollout_eval_stream(
                model,
                dataset._base_clues,
                dataset._base_answers,
                slot_batch_size=2,
                config=RolloutConfig(inner_iters=1, max_outer_iters=1, curriculum_training=False),
                init_seed=0,
            )
        )
    matched = set()
    for _, row_clues, row_answer in results:
        for idx in range(dataset._base_clues.size(0)):
            if torch.equal(row_clues, dataset._base_clues[idx]) and torch.equal(
                row_answer, dataset._base_answers[idx]
            ):
                matched.add(idx)
    assert matched == {0, 1, 2}


def test_stream_max_tries_retries_before_new_puzzle():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clues[0, 0, 0] = 5
    answers = torch.full((1, 9, 9), 1, dtype=torch.long)
    config = RolloutConfig(inner_iters=1, max_outer_iters=1, curriculum_training=False)

    with patch("rollout._predict_halt", return_value=torch.tensor([False])):
        stream_results = list(
            rollout_eval_stream(
                model,
                clues,
                answers,
                slot_batch_size=1,
                config=config,
                init_seed=3,
                max_tries=2,
            )
        )
        batch_result = rollout_eval_batch(
            model,
            clues,
            answers,
            config=config,
            init_seed=3,
            max_tries=2,
        )
    assert len(stream_results) == 1
    stream_result = stream_results[0][0]
    assert stream_result.tries.item() == batch_result.tries.item() == 2
    assert stream_result.halted.item() == batch_result.halted.item() is False


def test_eval_stops_at_max_outer_without_halt():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    clues, answer = _three_puzzle_dataset()._base_clues[:1], _three_puzzle_dataset()._base_answers[:1]
    config = RolloutConfig(inner_iters=1, max_outer_iters=2, curriculum_training=False)
    with patch("rollout._predict_halt", return_value=torch.tensor([False])):
        result = rollout_eval_batch(model, clues, answer, config=config)
    assert result.outer_steps.item() == 2
    assert result.halted.item() is False


def test_train_stops_when_halt_and_correct():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    answer = torch.full((1, 9, 9), 2, dtype=torch.long)
    state = BatchSlotState(
        digit_id=answer.clone(),
        clues=clues,
        answer=answer,
        clue_pin=clues > 0,
        outer_count=torch.zeros(1, dtype=torch.long),
    )
    config = RolloutConfig(inner_iters=1, max_outer_iters=10, deep_supervision=False)
    with patch("rollout._predict_halt", return_value=torch.tensor([True])):
        with patch("rollout._inner_loop") as mock_inner:
            logits = torch.zeros(1, 9, 9, 10)
            logits[..., 2] = 10.0
            mock_inner.return_value = (logits, torch.zeros(1), torch.zeros(1, 9, 9, 32))
            result = rollout_train_step(model, state, config, backward=False)
    assert result.done is not None
    assert result.done.all()


def test_eval_stream_evaluates_all_puzzles():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()
    def halt_all(halt_logit, *, halt_threshold=0.5):
        return torch.ones(halt_logit.size(0), dtype=torch.bool, device=halt_logit.device)

    with patch("rollout._predict_halt", side_effect=halt_all):
        results = list(
            rollout_eval_stream(
                model,
                dataset._base_clues,
                dataset._base_answers,
                slot_batch_size=2,
                config=RolloutConfig(inner_iters=1, max_outer_iters=5, curriculum_training=False),
                init_seed=0,
            )
        )
    assert len(results) == 3


def test_measure_split_stream_matches_static_metrics():
    torch.manual_seed(0)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()
    stream_stats = _run_measure(model, dataset, slot_batch_size=2, use_stream=True)
    torch.manual_seed(0)
    static_stats = _run_measure(model, dataset, slot_batch_size=2, use_stream=False)
    _assert_stats_equal(stream_stats, static_stats)


def test_measure_split_stream_matches_static_after_refill():
    torch.manual_seed(2)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()
    step = {"n": 0}

    def halt_first_slot_once(halt_logit, *, halt_threshold=0.5):
        step["n"] += 1
        b = halt_logit.size(0)
        if step["n"] == 1:
            out = torch.zeros(b, dtype=torch.bool)
            out[0] = True
            return out
        return torch.zeros(b, dtype=torch.bool)

    with patch("rollout._predict_halt", side_effect=halt_first_slot_once):
        stream_stats = _run_measure(model, dataset, slot_batch_size=2, use_stream=True)
    step["n"] = 0
    torch.manual_seed(2)
    with patch("rollout._predict_halt", side_effect=halt_first_slot_once):
        static_stats = _run_measure(model, dataset, slot_batch_size=2, use_stream=False)
    _assert_stats_equal(stream_stats, static_stats)


def _assert_stats_equal(stream_stats: dict, static_stats: dict) -> None:
    for key in stream_stats:
        assert stream_stats[key] == pytest.approx(static_stats[key])


def test_measure_split_stream_matches_static_with_max_tries():
    torch.manual_seed(1)
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()
    stream_stats = _run_measure(model, dataset, slot_batch_size=2, use_stream=True, max_tries=2)
    torch.manual_seed(1)
    static_stats = _run_measure(model, dataset, slot_batch_size=2, use_stream=False, max_tries=2)
    _assert_stats_equal(stream_stats, static_stats)


def test_train_done_requires_model_halt_and_correct_grid():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.train()
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clues[0, 0, 0] = 5
    answer = torch.full((1, 9, 9), 1, dtype=torch.long)
    state = BatchSlotState(
        digit_id=clues.clone(),
        clues=clues,
        answer=answer,
        clue_pin=clues > 0,
        outer_count=torch.zeros(1, dtype=torch.long),
    )
    config = RolloutConfig(inner_iters=1, max_outer_iters=10, deep_supervision=False)
    with patch("rollout._predict_halt", return_value=torch.tensor([True])):
        with patch("rollout._inner_loop") as mock_inner:
            logits = torch.zeros(1, 9, 9, 10)
            logits[..., 2] = 10.0
            mock_inner.return_value = (logits, torch.zeros(1), torch.zeros(1, 9, 9, 32))
            result = rollout_train_step(model, state, config, backward=False)
    assert result.done is not None
    assert not result.done.any()


def test_stream_accumulator_matches_manual_add_batch():
    model = MixerNextStateModel(dim=32, num_blocks=1)
    model.eval()
    dataset = _three_puzzle_dataset()
    config = build_rollout_config(inner_iters=1, max_outer_iters=2, curriculum_training=False)
    amp = type("Amp", (), {"enabled": False, "dtype": None, "scaler": None})()

    class _Progress:
        def update(self, _n: int) -> None:
            return None

    stream_acc = _accumulate_stream_eval(
        model,
        dataset._base_clues,
        dataset._base_answers,
        torch.device("cpu"),
        slot_batch_size=2,
        rollout_config=config,
        halt_loss_weight=1.0,
        use_cuda=False,
        seed=0,
        amp=amp,
        max_tries=1,
        progress=_Progress(),
    )
    static_acc = _accumulate_static_eval_batches(
        model,
        dataset._base_clues,
        dataset._base_answers,
        torch.device("cpu"),
        slot_batch_size=2,
        rollout_config=config,
        halt_loss_weight=1.0,
        use_cuda=False,
        seed=0,
        amp=amp,
        max_tries=1,
    )
    _assert_stats_equal(asdict(stream_acc.finalize()), asdict(static_acc.finalize()))
