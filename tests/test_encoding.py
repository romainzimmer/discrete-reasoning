from __future__ import annotations

import torch

from encoding import NUM_VOCAB, decode_logits, target_mask


def test_decode_logits_ten_classes():
    logits = torch.zeros(1, 9, 9, NUM_VOCAB)
    logits[0, 0, 0, 7] = 10.0
    assert decode_logits(logits)[0, 0, 0] == 7


def test_target_mask_excludes_clue_cells():
    answer = torch.full((9, 9), 4)
    clue_pin = torch.zeros(9, 9, dtype=torch.bool)
    clue_pin[0, 0] = True
    mask = target_mask(answer, clue_pin)
    assert not mask[0, 0]
    assert mask[0, 1]
