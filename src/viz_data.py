from __future__ import annotations

import json
from pathlib import Path

import torch

from data import puzzle_to_tensor
from encoding import grid_to_onehot
from model import NextStateModel
from rollout import RolloutConfig, rollout_trace


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


def build_trajectory(
    model: NextStateModel,
    row: dict,
    *,
    split: str,
    epoch: int,
    puzzle_index: int,
    device: torch.device,
    rollout_config: RolloutConfig,
) -> dict:
    clues = puzzle_to_tensor(row["question"]).unsqueeze(0).to(device)
    states = rollout_trace(
        model,
        grid_to_onehot(clues),
        clues,
        config=rollout_config,
    )
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
    model: NextStateModel,
    rows: list[dict],
    *,
    split: str,
    epoch: int,
    run_dir: Path,
    device: torch.device,
    rollout_config: RolloutConfig,
) -> list[int]:
    puzzle_indices: list[int] = []
    for puzzle_index, row in enumerate(rows):
        payload = build_trajectory(
            model,
            row,
            split=split,
            epoch=epoch,
            puzzle_index=puzzle_index,
            device=device,
            rollout_config=rollout_config,
        )
        save_trajectory(run_dir, split, epoch, puzzle_index, payload)
        puzzle_indices.append(puzzle_index)
    return puzzle_indices


def update_manifest_split(manifest: dict, split: str, epoch: int, puzzle_indices: list[int]) -> None:
    manifest.setdefault("splits", {}).setdefault(split, {})[str(epoch)] = puzzle_indices
