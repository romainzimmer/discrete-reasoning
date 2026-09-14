from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ProfileConfig:
    steps: int = 0
    wait: int = 1
    warmup: int = 2
    epoch: int = 1

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("profile steps must be >= 0")
        if self.wait < 0 or self.warmup < 0:
            raise ValueError("profile wait/warmup must be >= 0")
        if self.epoch < 1:
            raise ValueError("profile epoch must be >= 1")

    @property
    def enabled(self) -> bool:
        return self.steps > 0

    @property
    def total_steps(self) -> int:
        return self.wait + self.warmup + self.steps


class TrainProfiler:
    """torch.profiler wrapper for a slice of training steps in one epoch."""

    def __init__(self, config: ProfileConfig, *, output_dir: Path, device: torch.device) -> None:
        if not config.enabled:
            raise ValueError("TrainProfiler requires profile steps > 0")
        self._config = config
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._device = device
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        schedule = torch.profiler.schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.steps,
            repeat=1,
        )
        self._prof = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=True,
            profile_memory=device.type == "cuda",
        )
        self._prof.__enter__()
        self._finished = False

    def step(self) -> None:
        self._prof.step()

    def finish(self) -> Path:
        if self._finished:
            raise RuntimeError("TrainProfiler.finish already called")
        self._finished = True
        self._prof.__exit__(None, None, None)
        trace_path = self._output_dir / "trace.json"
        self._prof.export_chrome_trace(str(trace_path))
        sort_key = "self_cuda_time_total" if self._device.type == "cuda" else "self_cpu_time_total"
        print(self._prof.key_averages().table(sort_by=sort_key, row_limit=20))
        print(f"Profile trace: {trace_path}")
        print("Open chrome://tracing (or edge://tracing) and load trace.json")
        return trace_path
