from __future__ import annotations

import torch

from rollout import EvalRolloutResult
from train import EvalMetricsAccumulator


def test_eval_metrics_accumulator():
    acc = EvalMetricsAccumulator.empty(torch.device("cpu"))
    clues = torch.zeros(2, 9, 9, dtype=torch.long)
    clues[:, 0, 0] = 5
    answer = torch.full((2, 9, 9), 4)
    pred = answer.clone()
    pred[0, 0, 1] = 9
    result = EvalRolloutResult(
        pred=pred,
        outer_steps=torch.tensor([2, 3]),
        halted=torch.tensor([True, False]),
        loss=torch.tensor(1.5),
        cell_loss=torch.tensor(1.4),
        halt_loss=torch.tensor(0.1),
        halt_target=torch.zeros(2),
        halt_logit=torch.zeros(2),
        halt_correct_rounds=3,
        halt_total_rounds=10,
        tries=torch.tensor([2, 3]),
    )
    acc.add_batch(result, answer, clues)
    stats = acc.finalize()
    assert stats.loss == 1.5
    assert stats.puzzle_acc == 0.5
    assert stats.halt_acc == 0.3
    assert stats.avg_outer_iters == 2.5
    assert stats.halt_rate == 0.5
    assert stats.avg_tries == 2.5
