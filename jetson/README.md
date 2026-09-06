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
docker compose run --rm train --min-rating 0 --max-rating 0 --max-samples 100 --epochs 30
```

Checkpoints and trajectories are written to `runs/`.

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
