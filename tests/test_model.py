from __future__ import annotations

import torch

from encoding import NUM_VOCAB
from model import MixerNextStateModel, _swiglu_hidden_dim


def test_swiglu_hidden_dim_matches_trm():
    assert _swiglu_hidden_dim(512) == 1536
    assert _swiglu_hidden_dim(32) == 256


def test_mixer_block_stack():
    model = MixerNextStateModel(dim=32, num_blocks=3)
    assert len(model.blocks) == 3


def test_h_plus_p_forward():
    model = MixerNextStateModel(dim=16, num_blocks=1)
    digit_id = torch.randint(0, 10, (1, 9, 9))
    clue_pin = (digit_id > 0).long()
    p = model.encode_input(digit_id, clue_pin)
    out0 = model(input_embed=p, cell_embed=None)
    out1 = model(input_embed=p, cell_embed=out0.cell_embed)
    assert out0.logits.shape == (1, 9, 9, NUM_VOCAB)
    assert not torch.allclose(out0.logits, out1.logits)
