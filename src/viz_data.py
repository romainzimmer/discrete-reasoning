from __future__ import annotations

import json
from pathlib import Path

import torch

from data import puzzle_to_tensor
from model import MixerNextStateModel
from rollout import RolloutConfig, rollout_trace_batch


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"splits": {}}


def save_manifest(run_dir: Path, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def save_trajectory(run_dir: Path, split: str, epoch: int, puzzle_index: int, payload: dict) -> None:
    path = (
        run_dir
        / "trajectories"
        / split
        / f"epoch_{epoch:04d}"
        / f"puzzle_{puzzle_index:04d}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _trajectory_payload(
    row: dict,
    states: list[str],
    *,
    split: str,
    epoch: int,
    puzzle_index: int,
) -> dict:
    return {
        "question": row["question"],
        "answer": row["answer"],
        "states": states,
        "meta": {
            "split": split,
            "epoch": epoch,
            "puzzle_index": puzzle_index,
            "source": row["source"],
            "rating": row["rating"],
        },
    }


def save_epoch_trajectories(
    model: MixerNextStateModel,
    rows: list[dict],
    *,
    split: str,
    epoch: int,
    run_dir: Path,
    device: torch.device,
    rollout_config: RolloutConfig,
    batch_size: int | None = None,
) -> list[int]:
    if not rows:
        return []
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    model.eval()
    chunk_size = batch_size or len(rows)
    puzzle_indices: list[int] = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        clues = torch.stack([puzzle_to_tensor(row["question"]) for row in chunk]).to(device)
        trajectories = rollout_trace_batch(
            model,
            clues,
            config=rollout_config,
        )
        for offset, (row, states) in enumerate(zip(chunk, trajectories)):
            puzzle_index = start + offset
            save_trajectory(
                run_dir,
                split,
                epoch,
                puzzle_index,
                _trajectory_payload(
                    row,
                    states,
                    split=split,
                    epoch=epoch,
                    puzzle_index=puzzle_index,
                ),
            )
            puzzle_indices.append(puzzle_index)
    return puzzle_indices


def update_manifest_split(manifest: dict, split: str, epoch: int, puzzle_indices: list[int]) -> None:
    manifest.setdefault("splits", {}).setdefault(split, {})[str(epoch)] = puzzle_indices
