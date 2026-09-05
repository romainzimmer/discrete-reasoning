from __future__ import annotations

import argparse
from pathlib import Path

from data import load_split
from trajectory import Trajectory, demo_trajectory

VIZ_DIR = Path(__file__).resolve().parents[1] / "viz"
TRAJECTORY_JSON = VIZ_DIR / "trajectory.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trajectory for the web viewer")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=TRAJECTORY_JSON)
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="Load trajectory JSON produced by the solver instead of demo",
    )
    args = parser.parse_args()

    if args.from_file is not None:
        traj = Trajectory.load(args.from_file)
    else:
        row = load_split(args.split)[args.index]
        traj = demo_trajectory(
            question=row["question"],
            answer=row["answer"],
            source=row["source"],
            rating=row["rating"],
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traj.save(args.output)
    print(f"Exported trajectory ({len(traj.steps)} steps) to {args.output}")


if __name__ == "__main__":
    main()
