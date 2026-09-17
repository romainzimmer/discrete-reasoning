from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from memory import inner_halpern_alphas, inner_halpern_input
from model import MixerNextStateModel
from rollout import _inner_loop


def _expected_alphas(n: int) -> list[float]:
    if n == 1:
        return [0.0]
    return [(n - 1 - t) / (n - 1) for t in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 6, 10])
def test_inner_halpern_alphas_mapping(n: int):
    alphas = inner_halpern_alphas(n)
    expected = _expected_alphas(n)
    assert len(alphas) == n
    assert alphas == expected


def test_inner_halpern_alphas_invalid():
    with pytest.raises(ValueError, match="inner_iters must be positive"):
        inner_halpern_alphas(0)


@pytest.mark.parametrize("n", [1, 2, 3, 6, 10])
def test_inner_halpern_boundary_last(n: int):
    assert inner_halpern_alphas(n)[n - 1] == 0.0
    anchor = torch.ones(1, 9, 9, 4)
    carry = torch.full((1, 9, 9, 4), 2.0)
    assert torch.equal(inner_halpern_input(anchor=anchor, carry=carry, alpha=0.0), carry)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, [0.0]),
        (2, [1.0, 0.0]),
        (3, [1.0, 0.5, 0.0]),
        (6, [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
    ],
)
def test_inner_halpern_alpha_sequence(n: int, expected: list[float]):
    alphas = inner_halpern_alphas(n)
    assert alphas == expected
    positive = [a for a in alphas if a > 0]
    for prev, curr in zip(positive, positive[1:]):
        assert curr < prev


def test_inner_halpern_input():
    carry = torch.full((1, 9, 9, 4), 3.0)
    assert torch.equal(inner_halpern_input(anchor=None, carry=carry, alpha=0.0), carry)

    scaled = inner_halpern_input(anchor=None, carry=carry, alpha=0.5)
    assert torch.allclose(scaled, torch.full_like(carry, 1.5))

    anchor = torch.ones(1, 9, 9, 4)
    zeros = torch.zeros(1, 9, 9, 4)
    blended = inner_halpern_input(anchor=anchor, carry=zeros, alpha=0.5)
    assert blended.dtype == torch.float32
    assert torch.allclose(blended, torch.full((1, 9, 9, 4), 0.5))

    assert inner_halpern_input(anchor=anchor, carry=carry, alpha=1.0) is anchor


def test_inner_halpern_input_dtype():
    anchor = torch.ones(1, 9, 9, 4, dtype=torch.float32)
    carry = torch.full((1, 9, 9, 4), 2.0, dtype=torch.bfloat16)
    blended = inner_halpern_input(anchor=anchor, carry=carry, alpha=0.25)
    assert blended.dtype == torch.float32
    assert torch.allclose(blended, torch.full((1, 9, 9, 4), 1.75))


def test_inner_halpern_init_k0():
    blend_calls: list[int] = []

    def track_input(**kwargs):
        blend_calls.append(1)
        return inner_halpern_input(**kwargs)

    model = MixerNextStateModel(dim=8, num_blocks=1)
    model.eval()
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    memory = torch.randn(1, 9, 9, 8)

    with patch("rollout.inner_halpern_input", side_effect=track_input):
        _inner_loop(model, clues, clue_pin, 3, memory_embed=memory, with_grad=False)

    assert len(blend_calls) == 3


def test_inner_halpern_boundary_first():
    model = MixerNextStateModel(dim=8, num_blocks=1)
    model.eval()
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.bool)
    memory = torch.randn(1, 9, 9, 8)

    seen: list[torch.Tensor | None] = []
    original_forward = model.forward

    def track_forward(*, input_embed, cell_embed=None):
        seen.append(cell_embed)
        return original_forward(input_embed=input_embed, cell_embed=cell_embed)

    with patch.object(model, "forward", side_effect=track_forward):
        _inner_loop(model, clues, clue_pin, 2, memory_embed=memory, with_grad=False)

    assert len(seen) == 2
    assert torch.equal(seen[0], memory)

    seen.clear()
    with patch.object(model, "forward", side_effect=track_forward):
        _inner_loop(model, clues, clue_pin, 2, memory_embed=None, with_grad=False)

    assert len(seen) == 2
    assert seen[0] is None
