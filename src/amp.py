from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AmpConfig:
    enabled: bool
    dtype: torch.dtype | None
    scaler: torch.amp.GradScaler | None


def resolve_amp(device: torch.device, *, enabled: bool) -> AmpConfig:
    if not enabled or device.type != "cuda":
        return AmpConfig(enabled=False, dtype=None, scaler=None)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    return AmpConfig(enabled=True, dtype=dtype, scaler=scaler)


@contextmanager
def autocast_context(device: torch.device, amp: AmpConfig):
    if amp.enabled and amp.dtype is not None:
        with torch.autocast(device_type=device.type, dtype=amp.dtype):
            yield
    else:
        yield
