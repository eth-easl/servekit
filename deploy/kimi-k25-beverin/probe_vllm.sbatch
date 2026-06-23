#!/bin/bash
#SBATCH --job-name=kimi-k25-probe
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k25-vllm}"
export EDF_PATH="${DEPLOY_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs"

echo "probe started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"
echo "EDF_PATH=${EDF_PATH}"

srun --environment=kimi-k25-rocm \
  bash -lc 'set -euo pipefail
    echo "inside container: $(hostname)"
    python3 - <<'"'"'PY'"'"'
import importlib.metadata
import os
import shutil

for exe in ("python3", "vllm", "ray", "rocm-smi"):
    print(f"{exe}={shutil.which(exe)}")

for package in ("vllm", "torch", "ray"):
    try:
        print(f"{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")

print("HF_HOME=" + os.environ.get("HF_HOME", ""))
print("VLLM_ROCM_USE_AITER=" + os.environ.get("VLLM_ROCM_USE_AITER", ""))
PY
    rocm-smi --showproductname || true
  '

echo "probe finished: $(date --iso-8601=seconds)"
