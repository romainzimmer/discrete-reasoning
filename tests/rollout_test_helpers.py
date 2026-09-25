from __future__ import annotations

import torch

import rollout

_REAL_UPDATE_COMMIT_STABILITY = rollout._update_commit_stability
_REAL_UPDATE_COMMIT_STABILITY_ROW = rollout._update_commit_stability_row


def patch_implicit_halt_responses(responses: list[torch.Tensor]):
    """Patch _update_commit_stability to force implicit_halt per outer step."""
    iterator = iter(responses)

    def _wrapped(
        pre_commit: torch.Tensor,
        commit_prior: torch.Tensor,
        commit_streak: torch.Tensor,
        *,
        halt_after_stable_outer_steps: int,
    ):
        prior, streak, _ = _REAL_UPDATE_COMMIT_STABILITY(
            pre_commit,
            commit_prior,
            commit_streak,
            halt_after_stable_outer_steps=halt_after_stable_outer_steps,
        )
        try:
            implicit = next(iterator)
        except StopIteration:
            implicit = torch.zeros(pre_commit.size(0), dtype=torch.bool, device=pre_commit.device)
        if implicit.dim() == 0:
            implicit = implicit.expand(pre_commit.size(0))
        return prior, streak, implicit

    return _wrapped


def patch_implicit_halt_row(value: bool):
    def _wrapped(
        pre_commit: torch.Tensor,
        commit_prior: torch.Tensor | None,
        commit_streak: int,
        *,
        halt_after_stable_outer_steps: int,
    ):
        prior, streak, _ = _REAL_UPDATE_COMMIT_STABILITY_ROW(
            pre_commit,
            commit_prior,
            commit_streak,
            halt_after_stable_outer_steps=halt_after_stable_outer_steps,
        )
        return prior, streak, value

    return _wrapped


def patch_implicit_halt_row_first_call_only():
    calls = {"n": 0}

    def _wrapped(
        pre_commit: torch.Tensor,
        commit_prior: torch.Tensor | None,
        commit_streak: int,
        *,
        halt_after_stable_outer_steps: int,
    ):
        calls["n"] += 1
        prior, streak, _ = _REAL_UPDATE_COMMIT_STABILITY_ROW(
            pre_commit,
            commit_prior,
            commit_streak,
            halt_after_stable_outer_steps=halt_after_stable_outer_steps,
        )
        return prior, streak, calls["n"] == 1

    return _wrapped
