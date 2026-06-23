#!/bin/bash
#SBATCH --job-name=snapshot-bench
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k25-vllm}"
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
SNAPSHOT_FILE="${SNAPSHOT_FILE:-${DEPLOY_DIR}/snapshot/synthetic.snap}"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs"

echo "snapshot bench started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"
echo "snapshot_file=${SNAPSHOT_FILE}"

# Process 1: capture the (non-scaled) workload to a file and exit.
srun --environment=snapshot-rocm \
  bash -lc 'set -euo pipefail
    test -x snapshot/build/snapshot || {
      cmake -S snapshot -B snapshot/build -DSNAPSHOT_BACKEND=HIP
      cmake --build snapshot/build -j"${SLURM_CPUS_PER_TASK:-8}"
    }
    echo "--- capture (process 1) ---"
    snapshot/build/snapshot capture "'"${SNAPSHOT_FILE}"'"
  '

# Process 2 (fresh process): restore from the file, rebuild, launch, and assert
# the output is bit-identical to a freshly computed reference. This is the
# two-process acceptance gate.
srun --environment=snapshot-rocm \
  bash -lc 'set -euo pipefail
    echo "--- restore + bit-identical gate (process 2) ---"
    snapshot/build/snapshot restore "'"${SNAPSHOT_FILE}"'"
  '

# Timing: cold capture (incl. warmup) vs warm restore, on the larger scaled graph.
srun --environment=snapshot-rocm \
  bash -lc 'set -euo pipefail
    echo "--- bench (cold capture vs warm restore) ---"
    snapshot/build/snapshot bench "'"${SNAPSHOT_FILE}"'" --iters 20 --scaled
  '

echo "snapshot bench finished: $(date --iso-8601=seconds)"
