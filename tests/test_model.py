from __future__ import annotations

import torch

from encoding import NUM_VOCAB, decode_logits, target_mask
from model import MixerNextStateModel, _swiglu_hidden_dim
from rollout import predict_grid


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


def test_encode_input_adds_clue_type_embed():
    model = MixerNextStateModel(dim=16, num_blocks=1)
    digit_id = torch.zeros(1, 9, 9, dtype=torch.long)
    digit_id[0, 0, 0] = 5
    digit_id[0, 0, 1] = 5
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.long)
    clue_pin[0, 0, 0] = 1
    encoded = model.encode_input(digit_id, clue_pin)
    digit_only = model.encoder.digit_embed(digit_id.reshape(1, 81))
    clue_embed = model.encoder.clue_type_embed(torch.tensor([1]))
    non_clue_embed = model.encoder.clue_type_embed(torch.tensor([0]))
    assert torch.allclose(encoded[0, 0], digit_only[0, 0] + clue_embed[0])
    assert torch.allclose(encoded[0, 1], digit_only[0, 1] + non_clue_embed[0])
    assert not torch.allclose(encoded[0, 0], encoded[0, 1])


def test_encode_input_accepts_bool_clue_pin():
    model = MixerNextStateModel(dim=16, num_blocks=1)
    digit_id = torch.full((1, 9, 9), 3, dtype=torch.long)
    clue_pin_bool = torch.zeros(1, 9, 9, dtype=torch.bool)
    clue_pin_bool[0, 0, 0] = True
    clue_pin_long = clue_pin_bool.long()
    assert torch.allclose(
        model.encode_input(digit_id, clue_pin_bool),
        model.encode_input(digit_id, clue_pin_long),
    )


def test_encode_decode_round_trip_pins_clues():
    model = MixerNextStateModel(dim=16, num_blocks=1)
    model.eval()
    clues = torch.zeros(1, 9, 9, dtype=torch.long)
    clues[0, 0, 0] = 7
    digit_id = clues.clone()
    clue_pin = clues > 0
    logits = model(input_embed=model.encode_input(digit_id, clue_pin)).logits
    pred = predict_grid(logits, clues)
    assert pred[0, 0, 0] == 7
    assert pred.shape == (1, 9, 9)
