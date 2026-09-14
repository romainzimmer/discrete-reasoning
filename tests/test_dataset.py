from __future__ import annotations

from dataset import sample_rows
from train import split_train_val


def _rows(n: int) -> list[dict]:
    return [{"rating": i, "question": "q", "answer": "a", "source": "t"} for i in range(n)]


def test_sample_rows_is_reproducible_with_seed() -> None:
    rows = _rows(20)
    a = sample_rows(rows, max_samples=5, seed=7)
    b = sample_rows(rows, max_samples=5, seed=7)
    assert a == b
    assert len(a) == 5
    assert {row["rating"] for row in a} != set(range(5))


def test_sample_rows_keeps_all_when_below_cap() -> None:
    rows = _rows(3)
    assert sample_rows(rows, max_samples=10, seed=0) == rows


def test_split_train_val_samples_pool_before_holdout() -> None:
    rows = _rows(20)
    train_a, val_a = split_train_val(rows, val_samples=4, max_samples=10, seed=3)
    train_b, val_b = split_train_val(rows, val_samples=4, max_samples=10, seed=3)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) + len(val_a) == 10
    assert len(val_a) == 4
    assert {row["rating"] for row in train_a + val_a} != set(range(10))
