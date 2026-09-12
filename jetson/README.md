# Jetson

Docker setup for JetPack 7.2.1 (L4T r39.2.1). Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

All commands run from `jetson/`:

```bash
cd jetson
```

## Model

Training uses a **looped-mixer** model: embed the grid → looped MLP-Mixer updates `h_{t+1} = M(h_t + P)` → unembed to 10-way logits.

- **`--dim`**: embedding / mixer hidden dimension (D); SwiGLU channel-mix width `H = round(4·D·2/3)` rounded to 256 (TRM)
- **`--num-blocks`**: mixer layers inside each inner step (depth of M)
- **`--inner-iters`**: looped `h + P` steps per outer argmax commit (train, val, test, viz)
- **`--train-max-outer-iters` / `--eval-max-outer-iters`**: max outer commits per puzzle (halt or cap)
- **`--batches-per-epoch`**: optimizer steps per epoch (one outer round per step)

Runs save `args.model: looped-mixer` in `history.json`. **Old checkpoints from before this migration cannot be loaded by `eval`.**

Training keeps a persistent grid state per batch slot (`digit_id`, `outer_count`); `cell_embed` is re-encoded each step. If you hit OOM, lower `--train-batch-size` (512 may need tuning on Jetson).

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

Main run (looped-mixer; reduce `--train-batch-size` if OOM):

```bash
docker compose run --rm train --epochs 1000 --dim 256 --num-blocks 2 --train-batch-size 256 --num-workers 3 --max-samples 100000 --val-samples 512 --inner-iters 3 --train-max-outer-iters 30 --eval-max-outer-iters 30 --batches-per-epoch 300
```

Quick test (easy sudoku, ~1k train cap):

```bash
docker compose run --rm train \
  --epochs 5 \
  --dim 256 \
  --num-blocks 2 \
  --inner-iters 3 \
  --train-max-outer-iters 3 \
  --eval-max-outer-iters 3 \
  --train-batch-size 64 \
  --batches-per-epoch 50 \
  --num-workers 1 \
  --max-samples 5000 \
  --min-rating 0 \
  --max-rating 0 \
  --val-samples 64
```

Checkpoints and trajectories are written to `runs/`. Trajectory JSON is unchanged: one frame per **outer** commit in `states[]`.

## Resume

Continue a run from `runs/<run-id>/last.pt`. `--epochs` is the new total (not added to the previous value); it must be greater than the completed epoch:

```bash
docker compose run --rm resume runs/<run-id> --epochs 1000
```

## Profile

### PyTorch profiler

Short run with `--profile-steps` (`wait + warmup + active` must fit in `--batches-per-epoch`):

```bash
docker compose run --rm train \
  --epochs 1 \
  --batches-per-epoch 20 \
  --dim 512 \
  --num-blocks 2 \
  --inner-iters 3 \
  --train-batch-size 64 \
  --profile-steps 5 \
  --profile-wait 1 \
  --profile-warmup 2 \
  --max-samples 100 \
  --min-rating 0 \
  --max-rating 0
```

Writes `runs/<run-id>/profile/trace.json`. Copy to your laptop and open in `chrome://tracing`:

```bash
scp jetson:<repo>/runs/<run-id>/profile/trace.json ~/Downloads/trace.json
```

### Nsight Systems

Requires Nsight Systems on the **Jetson host** at `/opt/nvidia/nsight-systems` (JetPack dev tools). The script profiles **`train` inside the container** (not the docker CLI), mounts host `nsys`, and writes `runs/nsys/<timestamp>.nsys-rep`:

```bash
./nsys-profile.sh
```

Custom train args (no `--profile-steps`; PyTorch profiler stays off by default):

```bash
./nsys-profile.sh --epochs 1 --batches-per-epoch 20 --dim 512 --num-blocks 2 --inner-iters 3 --train-batch-size 64 --max-samples 100 --min-rating 0 --max-rating 0 --val-samples 10 --viz-samples 0
```

If auto-detect fails: `NSYS=/usr/local/cuda/bin/nsys ./nsys-profile.sh`. Uses the `train-profile` compose service (`privileged: true` in `docker-compose.yml`); try `sudo ./nsys-profile.sh` on permission errors.

In the report you should see a **`python` / `train` process**, **CUDA** rows, and NVTX ranges (`rollout_train_step`, `metrics_and_refill`). If you only see `docker` and no GPU data, the capture failed — check warnings in Nsight.

View on your laptop: install [Nsight Systems macOS Host](https://developer.nvidia.com/nsight-systems/get-started) (Mac version ≥ Jetson `nsys --version`), then:

```bash
scp jetson:<repo>/runs/nsys/<timestamp>.nsys-rep ~/Downloads/
```

Open the file in **NVIDIA Nsight Systems** (File → Open).

## Test

Evaluate a checkpoint from a run trained after the mixer migration (`args.model: looped-mixer` in `history.json`). Pass a run directory (uses `best.pt`) or a specific checkpoint path. Reuses model and rollout settings from the checkpoint; test rating filters are independent of training:

```bash
docker compose run --rm eval runs/<run-id>
docker compose run --rm eval runs/<run-id>/last.pt
docker compose run --rm eval runs/<run-id>/epochs/0010.pt
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

- [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/abs/2510.04871) — MLP-Mixer blocks
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424) — looped update `Y_{t+1} = M(Y_t + P)`
- [Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning](https://arxiv.org/abs/2609.01449) — persistent hidden state, anytime iterative solving
