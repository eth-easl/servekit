#!/bin/bash
#SBATCH --job-name=glm-52-fp8-probe
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-52-fp8-vllm/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-52-fp8-vllm}"
SCRIPT_DIR="${DEPLOY_DIR}/deploy/glm-52-fp8-beverin"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2-FP8}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs"

echo "probe started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"
echo "model=${MODEL}"
echo "EDF_PATH=${EDF_PATH}"

srun --environment=glm-52-fp8-rocm \
  bash -lc 'set -euo pipefail
    echo "inside container: $(hostname)"
    python3 - <<'"'"'PY'"'"'
import importlib.metadata
import json
import os
import pathlib
import shutil

model = pathlib.Path(os.environ.get("MODEL", "/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2-FP8"))
for exe in ("python3", "vllm", "ray", "rocm-smi"):
    print(f"{exe}={shutil.which(exe)}")

for package in ("vllm", "torch", "transformers", "ray"):
    try:
        print(f"{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")

print("HF_HOME=" + os.environ.get("HF_HOME", ""))
print("VLLM_ROCM_USE_AITER=" + os.environ.get("VLLM_ROCM_USE_AITER", ""))
print("VLLM_ROCM_USE_AITER_MLA=" + os.environ.get("VLLM_ROCM_USE_AITER_MLA", ""))

config = json.loads((model / "config.json").read_text())
print("model_type=" + str(config.get("model_type")))
print("architectures=" + str(config.get("architectures")))
print("quant_method=" + str(config.get("quantization_config", {}).get("quant_method")))
print("fmt=" + str(config.get("quantization_config", {}).get("fmt")))
print("num_hidden_layers=" + str(config.get("num_hidden_layers")))
print("max_position_embeddings=" + str(config.get("max_position_embeddings")))
PY
    rocm-smi --showproductname || true
  '

echo "probe finished: $(date --iso-8601=seconds)"
