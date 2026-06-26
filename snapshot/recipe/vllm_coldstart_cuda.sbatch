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

# G4 cross-check: three independent measures of capture-phase cost.
# Measure 1 (per-graph sum) and Measure 3 (cg_skip total) run here.
# Measure 2 (graph − eager delta) comes from the Task-2 runs above.
# All three should agree within a reasonable margin (~32s expected).

echo "=== instrumented: per-graph capture timing (VLLM_CG_INSTRUMENT) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  cd "${DEPLOY_DIR}"
  PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  mkdir -p "${DEPLOY_DIR}/measure-init"
  VLLM_CG_INSTRUMENT="${DEPLOY_DIR}/measure-init/cg_per_graph.csv" \
    MODE=graph bash snapshot/recipe/_vllm_coldstart_cuda.sh
  echo "--- per-graph capture_wall_ms sum ---"
  awk -F, "NR>1 && \$6!=\"\" {s+=\$6} END{printf \"PER_GRAPH_CAPTURE_MS_SUM=%.0f\n\", s}" \
    "${DEPLOY_DIR}/measure-init/cg_per_graph.csv" || true
  echo "--- cg_per_graph.csv row count ---"
  awk -F, "NR>1" "${DEPLOY_DIR}/measure-init/cg_per_graph.csv" | wc -l || true
'

echo "=== instrumented: total capture-phase cost (VLLM_CG_SKIP_CAPTURE=measure) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  cd "${DEPLOY_DIR}"
  PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  VLLM_CG_SKIP_CAPTURE=measure MODE=graph bash snapshot/recipe/_vllm_coldstart_cuda.sh
  echo "--- cg_skip output from vllm log ---"
  grep -E "CAPTURE_PHASE_S|cg_skip|cginst_skip" \
    "${DEPLOY_DIR}/logs/vllm_coldstart_graph_${SLURM_JOB_ID:-local}.log" || true
'
