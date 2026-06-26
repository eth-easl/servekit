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

srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  cd "${DEPLOY_DIR}"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  python -c "import transformers; print(\"transformers:\", transformers.__version__)"
  echo "=== warm-up run (populate Triton/vLLM caches; not measured) ==="
  MODE=graph DEADLINE=1800 bash snapshot/recipe/_vllm_coldstart_cuda.sh || true
  echo "=== measured: graph run #1 (warm cache) ==="
  RUN_TAG=graphs1 bash snapshot/recipe/_vllm_measure_cuda.sh
  echo "=== measured: graph run #2 (reproducibility) ==="
  RUN_TAG=graphs2 bash snapshot/recipe/_vllm_measure_cuda.sh
  echo "=== measured: enforce-eager (no capture) ==="
  RUN_TAG=eager EXTRA_ARGS=--enforce-eager bash snapshot/recipe/_vllm_measure_cuda.sh
'
