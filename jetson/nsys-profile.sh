#!/usr/bin/env bash
# Profile training with Nsight Systems (run on the Jetson host, from jetson/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/runs/nsys"
mkdir -p "$OUTPUT_DIR"

timestamp="$(date +%Y%m%d-%H%M%S)"
output="${OUTPUT_DIR}/${timestamp}"

find_nsys() {
  if [[ -n "${NSYS:-}" ]]; then
    echo "$NSYS"
    return
  fi
  if command -v nsys >/dev/null 2>&1; then
    command -v nsys
    return
  fi
  find /opt/nvidia/nsight-systems -path '*/target-linux-tegra-armv8/nsys' -type f 2>/dev/null | head -1
}

nsys_bin="$(find_nsys || true)"
if [[ -z "$nsys_bin" ]]; then
  echo "nsys not found. Install Nsight Systems (JetPack dev tools) or set NSYS=/path/to/nsys" >&2
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

echo "nsys: $nsys_bin"
echo "report: ${output}.nsys-rep"
echo "train args: $*"

exec "$nsys_bin" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o "$output" \
  docker compose run --rm train "$@"
