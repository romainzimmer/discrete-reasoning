# Getting started

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/). CUDA is recommended for training; CPU works for smoke tests and pytest.

## Install

```bash
uv sync
```

## Download, train, viz

```bash
uv run download-dataset
```

Writes `data/train.csv` and `data/test.csv` (~798 MB).

```bash
uv run train \
  --min-rating 0 --max-rating 0 \
  --max-samples 100 --epochs 30 \
  --dim 512 --num-blocks 2 \
  --inner-iters 5 --train-max-outer-iters 10 \
  --eval-max-outer-iters 10 \
  --train-batch-size 8 --batches-per-epoch 100

uv run python -m http.server 8000
```

Open [http://localhost:8000/viz/](http://localhost:8000/viz/). Serve from the **repo root** (see [runs.md](runs.md)).

## System notes

| Platform | Notes |
|----------|-------|
| Linux + CUDA | Recommended for training |
| macOS Apple Silicon | Works; use default torch from PyPI |
| Intel Mac (x86_64) | `numpy<2` and `torch<2.3` pinned in `pyproject.toml` |
| CPU only | Fine for `--max-samples 100` smoke tests |

## Troubleshooting

**Out of memory**: lower `--train-batch-size` (Jetson: try 64 or 32).

**`data/train.csv` not found**: run `uv run download-dataset`.

**Viz shows “No training runs”**: train first; serve from repo root.

**Empty trajectory**: pick a split/epoch/puzzle that exists in `manifest.json`.
