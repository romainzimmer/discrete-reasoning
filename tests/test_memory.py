from __future__ import annotations

import torch

from memory import memory_init, zero_memory


def test_zero_memory_shape():
    mem = zero_memory(2, 8, torch.device("cpu"))
    assert mem.shape == (2, 9, 9, 8)
    assert mem.dtype == torch.float32
    assert mem.sum().item() == 0.0


def test_memory_init_detached():
    value = torch.randn(1, 9, 9, 4, requires_grad=True)
    stored = memory_init(value)
    assert stored.dtype == torch.float32
    assert not stored.requires_grad
    assert torch.allclose(stored, value.detach())
