# Jetson

Docker setup for JetPack 7.2.1 (L4T r39.2.1). Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

All commands run from `jetson/`:

```bash
cd jetson
```

## Model

Training uses a **mixer-looped** model: embed the grid → looped MLP-Mixer updates `h_{t+1} = M(h_t + P)` → unembed to 10-way logits.

- **`--dim`**: embedding / mixer hidden dimension (D); SwiGLU channel-mix width `H = round(4·D·2/3)` rounded to 256 (TRM)
- **`--num-blocks`**: mixer layers inside each inner step (depth of M)
- **`--train-inner-iters` / `--eval-inner-iters`**: looped `h + P` steps per outer argmax commit
- **`--train-outer-iters` / `--eval-outer-iters`**: outer argmax commits per puzzle

Runs save `args.model: mixer-looped` in `history.json`. **Old checkpoints from before this migration cannot be loaded by `eval`.**

The mixer keeps a recurrent `cell_embed` per batch; it uses more GPU memory than the old flat model. If you hit OOM, lower `--batch-size` (512 may need tuning on Jetson).

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

Main run (mixer-looped; reduce `--batch-size` if OOM):

```bash
docker compose run --rm train   --epochs 150   --dim 256   --num-blocks 1   --batch-size 128   --num-workers 3   --val-samples 1024   --train-init noisy-gt  --max-samples 13824   --train-inner-iters 8   --train-outer-iters 1   --eval-inner-iters 8   --eval-outer-iters 8
```

Quick test (easy sudoku, ~1k train cap):

```bash
docker compose run --rm train \
  --epochs 5 \
  --dim 512 \
  --num-blocks 1 \
  --train-inner-iters 2 \
  --train-outer-iters 1 \
  --eval-inner-iters 2 \
  --eval-outer-iters 3 \
  --batch-size 64 \
  --num-workers 1 \
  --max-samples 6464 \
  --min-rating 0 \
  --max-rating 0 \
  --val-samples 64 \
  --train-init noisy-gt
```

Checkpoints and trajectories are written to `runs/`. Trajectory JSON is unchanged: one frame per **outer** commit in `states[]`.

## Test

Evaluate `best.pt` from a run trained after the mixer migration (`args.model: mixer-looped` in `history.json`). Reuses model and rollout settings from the checkpoint; test rating filters are independent of training:

```bash
docker compose run --rm eval runs/<run-id>
```

Cap test puzzles:

```bash
docker compose run --rm eval runs/<run-id> --max-test-samples 1000
```

Filter test by rating (omit both flags to evaluate all ratings):

```bash
docker compose run --rm eval runs/<run-id> --min-rating 5 --max-rating 9
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

## References

- [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/html/2510.04871v1) — MLP-Mixer blocks
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/pdf/2311.12424) — looped update `Y_{t+1} = M(Y_t + P)`
