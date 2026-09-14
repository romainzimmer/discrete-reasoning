from __future__ import annotations

from unittest.mock import patch

import torch

from amp import LOSS_DTYPE, resolve_amp, to_loss_dtype


def test_resolve_amp_disabled_on_cpu():
    amp = resolve_amp(torch.device("cpu"), enabled=True)
    assert not amp.enabled
    assert amp.dtype is None
    assert amp.scaler is None


def test_resolve_amp_disabled_by_flag():
    amp = resolve_amp(torch.device("cuda"), enabled=False)
    assert not amp.enabled


def test_resolve_amp_uses_scaler_only_for_fp16():
    with patch("amp.torch.cuda.is_bf16_supported", return_value=True):
        amp = resolve_amp(torch.device("cuda"), enabled=True)
    assert amp.enabled
    assert amp.dtype == torch.bfloat16
    assert amp.scaler is None

    with patch("amp.torch.cuda.is_bf16_supported", return_value=False):
        amp = resolve_amp(torch.device("cuda"), enabled=True)
    assert amp.enabled
    assert amp.dtype == torch.float16
    assert amp.scaler is not None


def test_to_loss_dtype():
    x = torch.tensor([1.0], dtype=torch.bfloat16)
    y = to_loss_dtype(x)
    assert y.dtype == LOSS_DTYPE
    z = to_loss_dtype(y)
    assert z is y
