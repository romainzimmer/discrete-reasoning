from __future__ import annotations

import argparse
import json
from pathlib import Path

from discrete_reasoning.data import load_split

VIZ_DIR = Path(__file__).resolve().parents[2] / "viz"
PUZZLES_JSON = VIZ_DIR / "puzzles.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export puzzles for the web viewer")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--max-rating", type=int, default=None)
    args = parser.parse_args()

    rows = load_split(args.split)
    if args.min_rating is not None:
        rows = [r for r in rows if r["rating"] >= args.min_rating]
    if args.max_rating is not None:
        rows = [r for r in rows if r["rating"] <= args.max_rating]
    rows = rows[: args.n]

    puzzles = [
        {
            "source": row["source"],
            "question": row["question"],
            "answer": row["answer"],
            "rating": row["rating"],
        }
        for row in rows
    ]

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    PUZZLES_JSON.write_text(json.dumps(puzzles, indent=2))
    print(f"Exported {len(puzzles)} puzzles to {PUZZLES_JSON}")


if __name__ == "__main__":
    main()
