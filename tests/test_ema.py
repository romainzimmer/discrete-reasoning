from __future__ import annotations

import pytest
import torch

from ema import ema_combine, ema_update, validate_ema_alpha, zero_ema
from rollout import RolloutConfig


def test_ema_combine_alpha_point_three():
    h = torch.tensor([1.0, 0.0])
    ema = torch.tensor([0.0, 1.0])
    out = ema_combine(h, ema, 0.3)
    assert torch.allclose(out, torch.tensor([0.3, 0.7]))


def test_ema_combine_alpha_one():
    h = torch.tensor([2.0, 3.0])
    ema = torch.tensor([9.0, 9.0])
    assert torch.allclose(ema_combine(h, ema, 1.0), h)


def test_ema_update_matches_combine():
    ema_k = torch.randn(1, 9, 9, 4)
    h_final = torch.randn(1, 9, 9, 4)
    alpha = 0.2
    updated = ema_update(ema_k, h_final, alpha)
    expected = ema_combine(h_final.detach(), ema_k, alpha)
    assert torch.allclose(updated, expected)
    assert not updated.requires_grad


def test_zero_ema_shape():
    ema = zero_ema(2, 8, torch.device("cpu"))
    assert ema.shape == (2, 9, 9, 8)
    assert ema.dtype == torch.float32
    assert ema.sum().item() == 0.0


def test_ema_update_stores_float32_from_bf16():
    ema = zero_ema(1, 4, torch.device("cpu"))
    value = torch.randn(1, 9, 9, 4, dtype=torch.bfloat16)
    updated = ema_update(ema, value, 0.1)
    assert updated.dtype == torch.float32


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_validate_ema_alpha_rejects_invalid(alpha: float):
    with pytest.raises(ValueError):
        validate_ema_alpha(alpha)


def test_rollout_config_rejects_invalid_ema_alpha():
    with pytest.raises(ValueError):
        RolloutConfig(ema_alpha=0.0)
