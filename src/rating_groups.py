from __future__ import annotations

NUM_RATING_GROUPS = 5

# Quintile bins from sapientinc/sudoku-extreme train.csv (~3.83M puzzles).
RATING_GROUP_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 11),
    (12, 23),
    (24, 38),
    (39, 10_000),
)

GROUP_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")


def rating_group(rating: int) -> int:
    for group, (lo, hi) in enumerate(RATING_GROUP_BOUNDS):
        if lo <= rating <= hi:
            return group
    return NUM_RATING_GROUPS - 1


def rating_group_label(group: int) -> str:
    lo, hi = RATING_GROUP_BOUNDS[group]
    return f"G{group} [{lo},{hi}]"
