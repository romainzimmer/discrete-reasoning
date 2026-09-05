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

## Train and visualize runs

```bash
uv run train --min-rating 0 --max-rating 0 --max-samples 100 --epochs 30
uv run python -m http.server
```

Open http://localhost:8000/viz/

Training writes each run under `runs/<run_id>/` (checkpoints, metrics, trajectories).

## Python usage

```python
from data import load_split, puzzle_to_tensor, answer_to_tensor

row = load_split("test")[0]
x = puzzle_to_tensor(row["question"])  # (9, 9), 0 = empty
y = answer_to_tensor(row["answer"])    # (9, 9), 1-9
```
