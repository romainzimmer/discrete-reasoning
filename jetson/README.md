# Jetson

Docker setup for JetPack 7.2.1 (L4T r39.2.1). Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

All commands run from `jetson/`:

```bash
cd jetson
```

## Build

Only needed once (or when dependencies change). Source code is bind-mounted from `../src`, so Python changes do not require a rebuild.

```bash
docker compose build
```

Override base image or PyTorch index if needed:

```bash
BASE_IMAGE=whitesscott/l4t-jetpack:r39.2.1 \
TORCH_INDEX=https://download.pytorch.org/whl/cu132 \
docker compose build
```

For JetPack 6 (L4T r36.4.x):

```bash
BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
TORCH_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126 \
docker compose build
```

## Download dataset

```bash
docker compose run --rm download
```

Writes `data/train.csv` and `data/test.csv` (~798 MB).

## Train

```bash
docker compose run --rm train --epochs 150 --width 512 --num-blocks 2 --batch-size 512 --num-workers 3 --val-samples 1024 --train-init noisy-gt --max-samples 52224 --train-inner-iters 5 --train-outer-iters 10 --eval-inner-iters 5 --eval-outer-iters 10
```

Quick test (easy sudoku, 1k train cap):

```bash
docker compose run --rm train --epochs 5 --width 512 --num-blocks 2 --train-inner-iters 2 --train-outer-iters 3 --eval-inner-iters 2 --eval-outer-iters 3 --batch-size 64 --num-workers 1 --max-samples 6464 --min-rating 0 --max-rating 0 --val-samples 64 --train-init noisy-gt
```

Checkpoints and trajectories are written to `runs/`.

## Test

Evaluate `best.pt` from a run (reuses model and rollout settings from the checkpoint; test rating filters are independent of training):

```bash
docker compose run --rm eval runs/20260906-145132-bda4748d
```

Cap test puzzles:

```bash
docker compose run --rm eval runs/20260906-145132-bda4748d --max-test-samples 1000
```

Filter test by rating (omit both flags to evaluate all ratings):

```bash
docker compose run --rm eval runs/20260906-145132-bda4748d --min-rating 5 --max-rating 9
```

## Visualize

On the Jetson:

```bash
docker compose up viz
```

Configure a jetson SSH config then forward port 8000:

```bash
ssh -L 8000:localhost:8000 jetson
```

Then open http://localhost:8000/viz/
