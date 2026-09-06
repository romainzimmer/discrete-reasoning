from __future__ import annotations

import torch

from augment import (
    AugmentConfig,
    _band_stack_perm,
    _permute_digits,
    _rotate_grid,
    apply_augment,
)
from data import answer_to_tensor, puzzle_to_tensor
from dataset import PuzzleDataset
from encoding import attach_clue_mask, clue_mask, grid_to_onehot

SOLVED = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)

CLUES = (
    "53..7...2"
    "6..195..."
    ".98....6."
    "8...6...3"
    "4..8.3..1"
    "7...2...6"
    ".6....28."
    "...419..5"
    "....8..79"
)


def _is_valid_complete(grid: torch.Tensor) -> bool:
    for i in range(9):
        if len(set(grid[i].tolist())) != 9:
            return False
        if len(set(grid[:, i].tolist())) != 9:
            return False
    for br in range(3):
        for bc in range(3):
            box = grid[br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3].reshape(-1)
            if len(set(box.tolist())) != 9:
                return False
    return True


def _box_multisets(grid: torch.Tensor) -> list[list[int]]:
    boxes: list[list[int]] = []
    for br in range(3):
        for bc in range(3):
            box = grid[br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3].reshape(-1)
            boxes.append(sorted(box.tolist()))
    return boxes


def _sample_row(question: str = CLUES, answer: str = SOLVED) -> dict:
    return {"question": question, "answer": answer, "rating": 0}


class TestPermuteDigits:
    def test_inverse_recovers_grid(self) -> None:
        grid = puzzle_to_tensor(CLUES)
        perm = torch.tensor([3, 7, 1, 9, 4, 2, 8, 5, 6])
        permuted = _permute_digits(grid, perm)
        inv = torch.empty(9, dtype=torch.long)
        inv[perm - 1] = torch.arange(1, 10)
        recovered = _permute_digits(permuted, inv)
        assert torch.equal(grid, recovered)

    def test_zeros_unchanged(self) -> None:
        grid = puzzle_to_tensor(CLUES)
        perm = torch.randperm(9) + 1
        out = _permute_digits(grid, perm)
        assert torch.equal(grid == 0, out == 0)


class TestRotateGrid:
    def test_each_k_preserves_valid_complete_grid(self) -> None:
        answer = answer_to_tensor(SOLVED)
        assert _is_valid_complete(answer)
        for k in (1, 2, 3):
            rotated = _rotate_grid(answer, k)
            assert _is_valid_complete(rotated)


class TestBandStackPerm:
    def test_fixed_perm_preserves_box_multisets(self) -> None:
        answer = answer_to_tensor(SOLVED)
        band_perm = torch.tensor([2, 0, 1])
        within = torch.tensor([[1, 0, 2], [2, 1, 0], [0, 2, 1]])
        stack_perm = torch.tensor([1, 2, 0])
        col_within = torch.tensor([[2, 0, 1], [1, 2, 0], [0, 1, 2]])
        permuted = _band_stack_perm(answer, band_perm, within, stack_perm, col_within)
        assert _box_multisets(answer) == _box_multisets(permuted)
        assert not torch.equal(answer, permuted)

    def test_clues_and_answer_stay_aligned(self) -> None:
        clues = puzzle_to_tensor(CLUES)
        answer = answer_to_tensor(SOLVED)
        band_perm = torch.tensor([1, 2, 0])
        within = torch.tensor([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
        stack_perm = torch.tensor([2, 1, 0])
        col_within = torch.tensor([[1, 0, 2], [0, 2, 1], [2, 1, 0]])
        aug_clues = _band_stack_perm(clues, band_perm, within, stack_perm, col_within)
        aug_answer = _band_stack_perm(answer, band_perm, within, stack_perm, col_within)
        clue_positions = aug_clues > 0
        assert torch.equal(aug_answer[clue_positions], aug_clues[clue_positions])


class TestApplyAugment:
    def test_zero_probabilities_are_identity(self) -> None:
        clues = puzzle_to_tensor(CLUES)
        answer = answer_to_tensor(SOLVED)
        config = AugmentConfig(p_digit=0.0, p_rot=0.0, p_band=0.0)
        aug_clues, aug_answer = apply_augment(clues, answer, config)
        assert torch.equal(clues, aug_clues)
        assert torch.equal(answer, aug_answer)

    def test_seeded_calls_are_reproducible(self) -> None:
        clues = puzzle_to_tensor(CLUES)
        answer = answer_to_tensor(SOLVED)
        config = AugmentConfig(p_digit=0.5, p_rot=0.5, p_band=0.3)
        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)
        out_a = apply_augment(clues, answer, config, generator=gen_a)
        out_b = apply_augment(clues, answer, config, generator=gen_b)
        assert torch.equal(out_a[0], out_b[0])
        assert torch.equal(out_a[1], out_b[1])


class TestEncodingSync:
    def test_clues_onehot_matches_augmented_grid(self) -> None:
        ds = PuzzleDataset(
            rows=[_sample_row()],
            augment=True,
            aug_config=AugmentConfig(p_digit=1.0, p_rot=1.0, p_band=1.0),
            aug_seed=7,
        )
        item = ds[0]
        assert torch.equal(item["clues_onehot"], grid_to_onehot(item["clues"]))

    def test_clue_cells_match_answer(self) -> None:
        ds = PuzzleDataset(
            rows=[_sample_row()],
            augment=True,
            aug_config=AugmentConfig(p_digit=1.0, p_rot=0.0, p_band=0.0),
            aug_seed=11,
        )
        item = ds[0]
        clues = item["clues"]
        answer = item["answer"]
        clue_positions = clues > 0
        assert torch.equal(answer[clue_positions], clues[clue_positions])

    def test_clue_mask_matches_nonzero_clues(self) -> None:
        ds = PuzzleDataset(
            rows=[_sample_row()],
            augment=True,
            aug_config=AugmentConfig(p_digit=0.0, p_rot=1.0, p_band=1.0),
            aug_seed=3,
        )
        item = ds[0]
        clues = item["clues"]
        mask = clue_mask(clues)
        assert torch.equal(mask.squeeze(-1) > 0, clues > 0)

    def test_attach_clue_mask_uses_augmented_clues(self) -> None:
        ds = PuzzleDataset(
            rows=[_sample_row()],
            augment=True,
            aug_config=AugmentConfig(p_digit=0.0, p_rot=1.0, p_band=0.0),
            aug_seed=5,
        )
        item = ds[0]
        state = attach_clue_mask(item["clues_onehot"], item["clues"])
        assert torch.equal(state[..., -1] > 0, item["clues"] > 0)

    def test_spatial_augment_relocates_clue_mask(self) -> None:
        clues = puzzle_to_tensor(CLUES)
        answer = answer_to_tensor(SOLVED)
        original_mask = clues > 0
        band_perm = torch.tensor([2, 0, 1])
        within = torch.tensor([[1, 0, 2], [2, 1, 0], [0, 2, 1]])
        stack_perm = torch.tensor([1, 2, 0])
        col_within = torch.tensor([[2, 0, 1], [1, 2, 0], [0, 1, 2]])
        aug_clues = _band_stack_perm(clues, band_perm, within, stack_perm, col_within)
        aug_answer = _band_stack_perm(answer, band_perm, within, stack_perm, col_within)
        assert int(original_mask.sum()) == int((aug_clues > 0).sum())
        assert torch.equal(aug_answer[aug_clues > 0], aug_clues[aug_clues > 0])


class TestDatasetWiring:
    def test_augment_false_ignores_config(self) -> None:
        row = _sample_row()
        ds = PuzzleDataset(
            rows=[row],
            augment=False,
            aug_config=AugmentConfig(p_digit=1.0, p_rot=1.0, p_band=1.0),
            aug_seed=0,
        )
        expected_clues = puzzle_to_tensor(row["question"])
        expected_answer = answer_to_tensor(row["answer"])
        item = ds[0]
        assert torch.equal(item["clues"], expected_clues)
        assert torch.equal(item["answer"], expected_answer)

    def test_augment_true_applies_transform(self) -> None:
        ds = PuzzleDataset(
            rows=[_sample_row()],
            augment=True,
            aug_config=AugmentConfig(p_digit=1.0, p_rot=1.0, p_band=1.0),
            aug_seed=99,
        )
        item = ds[0]
        assert not torch.equal(item["clues"], puzzle_to_tensor(CLUES))
        assert not torch.equal(item["answer"], answer_to_tensor(SOLVED))

    def test_default_augment_is_off(self) -> None:
        row = _sample_row()
        ds = PuzzleDataset(rows=[row])
        item = ds[0]
        assert torch.equal(item["clues"], puzzle_to_tensor(row["question"]))
