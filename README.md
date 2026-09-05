# discrete-reasoning

Experiments on [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme).

## Setup

```bash
uv sync
```

## Download dataset

```bash
uv run download-dataset
```

Writes `data/train.csv` and `data/test.csv` (~798 MB).

## Visualize puzzles

```bash
uv run export-viz --n 100
uv run python -m http.server --directory viz
```

Open http://localhost:8000

Export options:

```bash
uv run export-viz --split test --n 50 --min-rating 50
```

## Visualize solver trajectory

The solver builds a trajectory with `Trajectory.add_fill(row, col, value)` and saves JSON:

```python
from discrete_reasoning.trajectory import Trajectory

traj = Trajectory(question=row["question"], answer=row["answer"])
traj.add_fill(0, 2, 5)  # each fill is one step
traj.save("viz/trajectory.json")
```

Export a demo trajectory (reveals answer cell-by-cell):

```bash
uv run export-trajectory --index 0
uv run python -m http.server --directory viz
```

Or export a solver-produced file:

```bash
uv run export-trajectory --from-file path/to/trajectory.json
```

The viewer colors clues white, correct fills green, wrong fills red.

## Python usage

```python
from discrete_reasoning.data import load_split, puzzle_to_tensor, answer_to_tensor

row = load_split("test")[0]
x = puzzle_to_tensor(row["question"])  # (9, 9), 0 = empty
y = answer_to_tensor(row["answer"])    # (9, 9), 1-9
```
