#!/usr/bin/env bash
# Profile training with Nsight Systems inside the train container (run from jetson/ on the Jetson host).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/runs/nsys"
mkdir -p "$OUTPUT_DIR"

timestamp="$(date +%Y%m%d-%H%M%S)"
output="/workspace/runs/nsys/${timestamp}"

find_nsys_tegra() {
  if [[ -n "${NSYS:-}" ]]; then
    echo "$NSYS"
    return
  fi
  find /opt/nvidia/nsight-systems -path '*/target-linux-tegra-armv8/nsys' -type f 2>/dev/null | head -1
}

nsys_bin="$(find_nsys_tegra || true)"
if [[ -z "$nsys_bin" ]]; then
  echo "tegra nsys not found under /opt/nvidia/nsight-systems (install Nsight Systems from JetPack)." >&2
  echo "Or set NSYS=/opt/nvidia/nsight-systems/<ver>/target-linux-tegra-armv8/nsys" >&2
  exit 1
fi

if [[ ! -d /opt/nvidia/nsight-systems ]]; then
  echo "/opt/nvidia/nsight-systems missing on host; cannot mount into container." >&2
  exit 1
fi

cd "$SCRIPT_DIR"

if [[ $# -eq 0 ]]; then
  set -- \
    --epochs 1 \
    --batches-per-epoch 20 \
    --dim 512 \
    --num-blocks 2 \
    --inner-iters 3 \
    --train-batch-size 64 \
    --max-samples 100 \
    --min-rating 0 \
    --max-rating 0 \
    --val-samples 10 \
    --viz-samples 0
fi

host_report="${OUTPUT_DIR}/${timestamp}.nsys-rep"
echo "nsys (tegra): $nsys_bin"
echo "report: $host_report"
echo "train args: $*"

exec docker compose run --rm --privileged \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  --entrypoint "$nsys_bin" \
  train \
  profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o "$output" \
  -- \
  train "$@"
