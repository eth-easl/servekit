#!/bin/bash
#SBATCH --job-name=vllm-coldstart-cuda
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/%x-%j.out

set -euo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
export DEPLOY_DIR
export EDF_PATH="${DEPLOY_DIR}/deploy/glm-47-flash-bristen-vllm:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"

# NOTE: Each run is a SEPARATE srun step so the container (and CUDA driver context)
# restarts cleanly between serves. A single srun with multiple sequential vllm serves
# leaves GPU KV-cache memory allocated between runs (kill -9 is not sufficient to
# release enroot/CUDA shared allocations within the same container session).
# pip install is repeated in each step but is fast after the first (pip uses its local
# cache at ${DEPLOY_DIR}/cache/pip which persists on capstor across steps).

export PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
mkdir -p "${PIP_CACHE_DIR}"

echo "=== warm-up run (populate Triton/vLLM caches; not measured) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  cd "${DEPLOY_DIR}"
  PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  python3 -c "import transformers; print(\"transformers:\", transformers.__version__)"
  MODE=graph DEADLINE=1800 bash snapshot/recipe/_vllm_coldstart_cuda.sh || true
' || true

echo "=== measured: graph run #1 (warm cache) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  cd "${DEPLOY_DIR}"
  PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  RUN_TAG=graphs1 bash snapshot/recipe/_vllm_measure_cuda.sh
'

echo "=== measured: graph run #2 (reproducibility) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  cd "${DEPLOY_DIR}"
  PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  RUN_TAG=graphs2 bash snapshot/recipe/_vllm_measure_cuda.sh
'

echo "=== measured: enforce-eager (no capture) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  cd "${DEPLOY_DIR}"
  PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  RUN_TAG=eager EXTRA_ARGS=--enforce-eager bash snapshot/recipe/_vllm_measure_cuda.sh
'
