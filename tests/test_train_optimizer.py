from __future__ import annotations

from model import MixerNextStateModel
from train import optimizer_param_groups


def test_optimizer_param_groups_split_decay_and_no_decay() -> None:
    model = MixerNextStateModel(dim=32, num_blocks=1)
    groups = optimizer_param_groups(model, weight_decay=0.1)
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0

    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            assert id(param) in no_decay_ids
            assert id(param) not in decay_ids
        else:
            assert id(param) in decay_ids
            assert id(param) not in no_decay_ids

    assert id(model.encoder.digit_embed.weight) in decay_ids
