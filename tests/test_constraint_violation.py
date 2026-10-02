from __future__ import annotations

import torch

from encoding import constraint_violation_mask


def _violations(grid: list[list[int]]) -> list[list[bool]]:
    t = torch.tensor(grid, dtype=torch.long)
    return constraint_violation_mask(t).tolist()


def _violation_cells(grid: list[list[int]]) -> set[tuple[int, int]]:
    v = _violations(grid)
    return {(r, c) for r in range(9) for c in range(9) if v[r][c]}


def test_empty_and_zeros_never_violate():
    grid = [[0] * 9 for _ in range(9)]
    assert _violations(grid) == [[False] * 9 for _ in range(9)]


def test_unique_filled_grid_has_no_violations():
    # One complete row of distinct digits; rest empty.
    grid = [[0] * 9 for _ in range(9)]
    grid[0] = list(range(1, 10))
    assert _violations(grid) == [[False] * 9 for _ in range(9)]


def test_row_duplicate_marks_both_cells():
    grid = [[0] * 9 for _ in range(9)]
    grid[4][2] = 7
    grid[4][8] = 7
    assert _violation_cells(grid) == {(4, 2), (4, 8)}


def test_row_duplicate_first_row_adjacent():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][3] = 2
    grid[0][4] = 2
    assert _violation_cells(grid) == {(0, 3), (0, 4)}


def test_row_duplicate_last_row():
    grid = [[0] * 9 for _ in range(9)]
    grid[8][0] = 1
    grid[8][6] = 1
    assert _violation_cells(grid) == {(8, 0), (8, 6)}


def test_column_duplicate_marks_both_cells():
    grid = [[0] * 9 for _ in range(9)]
    grid[1][5] = 3
    grid[6][5] = 3
    assert _violation_cells(grid) == {(1, 5), (6, 5)}


def test_column_duplicate_left_and_right_edges():
    grid = [[0] * 9 for _ in range(9)]
    grid[2][0] = 8
    grid[7][0] = 8
    grid[1][8] = 5
    grid[5][8] = 5
    assert _violation_cells(grid) == {(2, 0), (7, 0), (1, 8), (5, 8)}


def test_column_triple_duplicate():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][6] = 1
    grid[4][6] = 1
    grid[8][6] = 1
    assert _violation_cells(grid) == {(0, 6), (4, 6), (8, 6)}


def test_block_duplicate_marks_both_cells():
    # Top-left block: (0,0) and (2,2) share block, not row or column.
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 9
    grid[2][2] = 9
    assert _violation_cells(grid) == {(0, 0), (2, 2)}


def test_block_duplicate_center_block_diagonal():
    grid = [[0] * 9 for _ in range(9)]
    grid[3][3] = 6
    grid[5][5] = 6
    assert _violation_cells(grid) == {(3, 3), (5, 5)}


def test_block_duplicate_top_right_block():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][7] = 4
    grid[2][8] = 4
    assert _violation_cells(grid) == {(0, 7), (2, 8)}


def test_block_duplicate_bottom_left_block():
    grid = [[0] * 9 for _ in range(9)]
    grid[6][0] = 2
    grid[8][1] = 2
    assert _violation_cells(grid) == {(6, 0), (8, 1)}


def test_block_triple_duplicate():
    grid = [[0] * 9 for _ in range(9)]
    grid[1][1] = 7
    grid[1][2] = 7
    grid[2][0] = 7
    assert _violation_cells(grid) == {(1, 1), (1, 2), (2, 0)}


def test_cell_violates_row_and_column():
    grid = [[0] * 9 for _ in range(9)]
    grid[4][4] = 5
    grid[4][7] = 5
    grid[7][4] = 5
    assert _violation_cells(grid) == {(4, 4), (4, 7), (7, 4)}


def test_multiple_disjoint_violations():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 1
    grid[0][8] = 1
    grid[8][0] = 2
    grid[8][8] = 2
    grid[3][3] = 3
    grid[5][4] = 3
    assert _violation_cells(grid) == {
        (0, 0),
        (0, 8),
        (8, 0),
        (8, 8),
        (3, 3),
        (5, 4),
    }


def test_triple_duplicate_in_row():
    grid = [[0] * 9 for _ in range(9)]
    grid[3][1] = 4
    grid[3][4] = 4
    grid[3][7] = 4
    v = _violations(grid)
    assert v[3][1] and v[3][4] and v[3][7]


def test_batch_matches_per_item():
    g0 = torch.zeros(9, 9, dtype=torch.long)
    g0[0, 0] = 1
    g0[0, 1] = 1
    g1 = torch.zeros(9, 9, dtype=torch.long)
    g1[8, 8] = 5
    batch = torch.stack([g0, g1])
    v = constraint_violation_mask(batch)
    assert v.shape == (2, 9, 9)
    assert v[0, 0, 0] and v[0, 0, 1]
    assert not v[1].any()


def test_encode_input_adds_constraint_violation_embed():
    from model import MixerNextStateModel

    model = MixerNextStateModel(dim=16, num_blocks=1)
    digit_id = torch.zeros(1, 9, 9, dtype=torch.long)
    digit_id[0, 2, 2] = 6
    digit_id[0, 2, 7] = 6
    clue_pin = torch.zeros(1, 9, 9, dtype=torch.long)
    encoded = model.encode_input(digit_id, clue_pin)
    flat = digit_id.reshape(1, 81)
    viol = constraint_violation_mask(digit_id).reshape(1, 81).long()
    expected = (
        model.encoder.digit_embed(flat)
        + model.encoder.clue_type_embed(torch.zeros(1, 81, dtype=torch.long))
        + model.encoder.constraint_violation_embed(viol)
    )
    assert torch.allclose(encoded, expected)

    digit_id[0, 2, 7] = 0
    encoded_ok = model.encode_input(digit_id, clue_pin)
    viol_ok = constraint_violation_mask(digit_id).reshape(1, 81).long()
    expected_ok = (
        model.encoder.digit_embed(digit_id.reshape(1, 81))
        + model.encoder.clue_type_embed(torch.zeros(1, 81, dtype=torch.long))
        + model.encoder.constraint_violation_embed(viol_ok)
    )
    assert torch.allclose(encoded_ok, expected_ok)
    assert not viol_ok.any()
