from __future__ import annotations

from unittest.mock import patch

import torch

from memory import inner_step_input
from model import MixerNextStateModel
from rollout import _inner_loop


def test_inner_step_input_first_step():
    memory = torch.ones(1, 9, 9, 4)
    assert torch.equal(inner_step_input(memory=memory, last_output=None), memory)
    assert inner_step_input(memory=None, last_output=None) is None


def test_inner_step_input_adds_last_output():
    memory = torch.ones(1, 9, 9, 4)
    last_output = torch.full((1, 9, 9, 4), 2.0)
    expected = torch.full((1, 9, 9, 4), 3.0)
    assert torch.equal(inner_step_input(memory=memory, last_output=last_output), expected)


def test_inner_step_input_no_memory():
    last_output = torch.full((1, 9, 9, 4), 4.0)
    assert torch.equal(inner_step_input(memory=None, last_output=last_output), last_output)


def test_inner_loop_first_step_uses_memory():
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
        _inner_loop(model, clues, clue_pin, 3, memory_embed=memory, with_grad=False)

    assert len(seen) == 3
    assert torch.equal(seen[0], memory)


def test_inner_loop_first_step_no_memory():
    model = MixerNextStateModel(dim=8, num_blocks=1)
    model.eval()
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.bool)

    seen: list[torch.Tensor | None] = []
    original_forward = model.forward

    def track_forward(*, input_embed, cell_embed=None):
        seen.append(cell_embed)
        return original_forward(input_embed=input_embed, cell_embed=cell_embed)

    with patch.object(model, "forward", side_effect=track_forward):
        _inner_loop(model, clues, clue_pin, 2, memory_embed=None, with_grad=False)

    assert len(seen) == 2
    assert seen[0] is None
