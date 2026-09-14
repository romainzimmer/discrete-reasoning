# Jetson

Docker setup for JetPack 7.2.1 (L4T r39.2.1). Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

All commands run from `jetson/`:

```bash
cd jetson
```

Model and training details: [docs/method.md](../docs/method.md). CLI flags: [docs/cli.md](../docs/cli.md).

If you hit OOM, lower `--train-batch-size` (512 may need tuning on Jetson).

## Build

Only needed once (or when dependencies change). Source is bind-mounted from `../src`.

```bash
docker compose build
```

Override base image or PyTorch index:

```bash
BASE_IMAGE=whitesscott/l4t-jetpack:r39.2.1 \
TORCH_INDEX=https://download.pytorch.org/whl/cu132 \
docker compose build
```

JetPack 6 (L4T r36.4.x):

```bash
BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
TORCH_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126 \
docker compose build
```



## Download dataset

```bash
docker compose run --rm download
```



## Train

```bash
docker compose run --rm train --epochs 1000 --dim 256 --num-blocks 2 --train-batch-size 128 --num-workers 3 --max-samples 100000 --val-samples 512 --inner-iters 6 --train-max-outer-iters 5 --eval-max-outer-iters 30 --batches-per-epoch 300
```

Quick test:

```bash
docker compose run --rm train \
  --epochs 5 --dim 256 --num-blocks 2 \
  --inner-iters 3 --train-max-outer-iters 3 --eval-max-outer-iters 3 \
  --train-batch-size 64 --batches-per-epoch 50 --num-workers 1 \
  --max-samples 5000 --min-rating 0 --max-rating 0 --val-samples 64
```



## Resume

```bash
docker compose run --rm resume runs/<run-id> --epochs 1000
```

`--epochs` is the new total; must exceed the completed epoch.

## Eval

```bash
docker compose run --rm eval runs/<run-id>
docker compose run --rm eval runs/<run-id> --max-test-samples 1000
docker compose run --rm eval runs/<run-id> --min-rating 5 --max-rating 9
```



## Profile

PyTorch profiler:

```bash
docker compose run --rm train \
  --epochs 1 --batches-per-epoch 20 --dim 512 --num-blocks 2 --inner-iters 3 \
  --train-batch-size 64 --profile-steps 5 --profile-wait 1 --profile-warmup 2 \
  --max-samples 100 --min-rating 0 --max-rating 0
```

Copy `runs/<run-id>/profile/trace.json` to your laptop and open in `chrome://tracing`.

### Nsight Systems

Requires Nsight on the host at `/opt/nvidia/nsight-systems`:

```bash
./nsys-profile.sh
```

Custom train args:

```bash
./nsys-profile.sh --epochs 1 --batches-per-epoch 20 --dim 512 --num-blocks 2 --inner-iters 3 --train-batch-size 64 --max-samples 100 --min-rating 0 --max-rating 0 --val-samples 10 --viz-samples 0
```

View on laptop: [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started), then open the `.nsys-rep` from `runs/nsys/`.

## Visualize

```bash
docker compose up viz
```

SSH port forward:

```bash
ssh -L 8000:localhost:8000 jetson
```

Open [http://localhost:8000/viz/](http://localhost:8000/viz/)