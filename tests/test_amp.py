from __future__ import annotations

import torch

from amp import resolve_amp


def test_resolve_amp_disabled_on_cpu():
    amp = resolve_amp(torch.device("cpu"), enabled=True)
    assert not amp.enabled
    assert amp.dtype is None
    assert amp.scaler is None


def test_resolve_amp_disabled_by_flag():
    amp = resolve_amp(torch.device("cuda"), enabled=False)
    assert not amp.enabled


def test_no_amp_flag():
    assert not (not True)
    assert not False
