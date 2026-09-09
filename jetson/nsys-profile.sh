#!/usr/bin/env bash
# Profile training with Nsight Systems inside the train container (run from jetson/ on the Jetson host).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/runs/nsys"
mkdir -p "$OUTPUT_DIR"

timestamp="$(date +%Y%m%d-%H%M%S)"
output="/workspace/runs/nsys/${timestamp}"

find_nsys_bin() {
  if [[ -n "${NSYS:-}" ]]; then
    if [[ -x "$NSYS" ]]; then
      echo "$NSYS"
      return 0
    fi
    echo "NSYS is set but not executable: $NSYS" >&2
    return 1
  fi

  local -a candidates=()
  local path

  shopt -s nullglob
  for path in \
    /usr/local/cuda/bin/nsys \
    /opt/nvidia/nsight-systems/*/target-linux-tegra-armv8/nsys \
    /opt/nvidia/nsight-systems-cli/*/target-linux-tegra-armv8/nsys \
    /opt/nvidia/nsight-systems-cli/*/bin/nsys; do
    [[ -x "$path" ]] && candidates+=("$path")
  done
  shopt -u nullglob

  if ((${#candidates[@]})); then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi

  if command -v nsys >/dev/null 2>&1; then
    path="$(readlink -f "$(command -v nsys)" 2>/dev/null || command -v nsys)"
    if [[ -x "$path" ]]; then
      echo "$path"
      return 0
    fi
  fi

  return 1
}

nsys_mount_for() {
  local bin="$1"
  if [[ "$bin" == /usr/local/cuda/* ]]; then
    echo /usr/local/cuda
  elif [[ "$bin" == /opt/nvidia/nsight-systems/* ]]; then
    echo /opt/nvidia/nsight-systems
  elif [[ "$bin" == /opt/nvidia/nsight-systems-cli/* ]]; then
    local rest="${bin#/opt/nvidia/nsight-systems-cli/}"
    echo "/opt/nvidia/nsight-systems-cli/${rest%%/*}"
  elif [[ "$bin" == */bin/nsys ]]; then
    dirname "$(dirname "$bin")"
  else
    dirname "$bin"
  fi
}

print_install_hint() {
  cat >&2 <<'EOF'
Install Nsight Systems on the Jetson host, then re-run:

  sudo apt update
  apt search nsight-systems          # pick the version matching your JetPack
  sudo apt install nsight-systems-2026.3   # example for JP 7.x

Verify:

  dpkg -L nsight-systems-* | grep '/nsys$'
  # or: ls /opt/nvidia/nsight-systems-cli/*/bin/nsys

Override manually:

  NSYS=/path/to/nsys ./nsys-profile.sh
EOF
}

nsys_bin="$(find_nsys_bin || true)"
if [[ -z "$nsys_bin" ]]; then
  echo "nsys not found on this Jetson." >&2
  print_install_hint
  exit 1
fi

nsys_mount="$(nsys_mount_for "$nsys_bin")"
if [[ ! -d "$nsys_mount" ]]; then
  echo "nsys mount dir missing: $nsys_mount" >&2
  print_install_hint
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
echo "nsys: $nsys_bin"
echo "mount: $nsys_mount"
echo "report: $host_report"
echo "train args: $*"

exec docker compose run --rm --privileged \
  -v "${nsys_mount}:${nsys_mount}:ro" \
  --entrypoint "$nsys_bin" \
  train \
  profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o "$output" \
  -- \
  train "$@"
