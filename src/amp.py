from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

# Losses and cross-step memory stay in fp32 under autocast for stability.
LOSS_DTYPE = torch.float32
STORAGE_DTYPE = torch.float32


def _make_grad_scaler() -> Any:
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


@dataclass(frozen=True)
class AmpConfig:
    enabled: bool
    dtype: torch.dtype | None
    scaler: Any | None


def to_loss_dtype(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == LOSS_DTYPE else x.to(LOSS_DTYPE)


def to_storage_dtype(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == STORAGE_DTYPE else x.to(STORAGE_DTYPE)


def resolve_amp(device: torch.device, *, enabled: bool) -> AmpConfig:
    if not enabled or device.type != "cuda":
        return AmpConfig(enabled=False, dtype=None, scaler=None)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # bf16 has enough range; GradScaler is only needed for fp16.
    scaler = _make_grad_scaler() if dtype == torch.float16 else None
    return AmpConfig(enabled=True, dtype=dtype, scaler=scaler)


@contextmanager
def autocast_context(device: torch.device, amp: AmpConfig):
    if amp.enabled and amp.dtype is not None:
        with torch.autocast(device_type=device.type, dtype=amp.dtype):
            yield
    else:
        yield
