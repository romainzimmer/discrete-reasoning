# CLI

All commands via `uv run`:

```bash
uv run download-dataset
uv run train [options]
uv run eval <run-or-checkpoint> [options]
uv run resume <run-dir> --epochs N
```

## download-dataset

Fetches [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) into `data/train.csv` and `data/test.csv`.

## train

Main training loop. Key flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--dim` | 256 | Hidden dimension D |
| `--num-blocks` | 2 | Mixer blocks per inner step |
| `--inner-iters` | 6 | Inner mixer steps per outer round |
| `--train-max-outer-iters` | 5 | Max outer commits before slot refill (train) |
| `--eval-max-outer-iters` | 30 | Max outer commits (val / test / viz) |
| `--train-batch-size` | 128 | Parallel puzzle slots |
| `--batches-per-epoch` | 100 | Optimizer steps per epoch |
| `--max-samples` | all | Random subsample from train.csv |
| `--min-rating` / `--max-rating` | none | Filter by puzzle difficulty |
| `--seed` | 0 | RNG for augment, subsampling, refill |
| `--no-augment` | off | Disable training augmentations |
| `--no-gt-reveal` | off | Clues + random fill only (no partial GT reveal) |
| `--no-deep-supervision` | off | Final inner step only for loss |
| `--no-amp` | off | Disable mixed precision on CUDA |

Run `uv run train --help` for the full list.

Outputs go to `runs/<run-id>/`. See [runs.md](runs.md).

## eval

Evaluate on `test.csv`:

```bash
uv run eval runs/<run-id>                  # uses best.pt
uv run eval runs/<run-id>/last.pt
uv run eval runs/<run-id> --max-test-samples 1000
uv run eval runs/<run-id> --min-rating 5 --max-rating 9
uv run eval runs/<run-id> --max-tries 10   # random restarts until halt
uv run eval runs/<run-id> --sweep           # inner/outer/tries ablations
```

Model and rollout settings default from the checkpoint. Test rating filters are independent of training.

## resume

Continue from `runs/<run-id>/last.pt`. `--epochs` is the new **total** (must exceed the completed epoch):

```bash
uv run resume runs/<run-id> --epochs 100
```

## Profiling

PyTorch profiler (`wait + warmup + active` must fit in `--batches-per-epoch`):

```bash
uv run train \
  --epochs 1 --batches-per-epoch 20 \
  --profile-steps 5 --profile-wait 1 --profile-warmup 2 \
  --max-samples 100 --min-rating 0 --max-rating 0
```

Writes `runs/<run-id>/profile/trace.json`. Open in `chrome://tracing`.

On Jetson, see [jetson/README.md](../jetson/README.md) for Docker commands and Nsight Systems.

## Reproducibility

`--seed` controls training augment RNG, train/val subsampling, slot refill, and eval subsampling. Same seed + same hardware should give matching metrics; small FP differences across devices are normal.
