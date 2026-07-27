#!/bin/bash
# Decompose the 8.47 s -> 4.37 s gap on the REAL staged checkpoint, no engine.
#
#   sbatch shm-weight-loading-exp2/scripts/shm_loader_bench.sbatch
#
#SBATCH --job-name=e2-loaderbench
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=shm-weight-loading-exp2/results/%x-%j.out

set -uo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp2"
SHARDED_SRC="${SHARDED_SRC:-/iopsstor/scratch/cscs/yboughizane/sharded-ckpt/llama70b-tp4}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b-sharded}"
NODE="$(hostname)"
JSON="${EXP_DIR}/results/loaderbench-${SLURM_JOB_ID}-${NODE}.jsonl"

echo "job=${SLURM_JOB_ID} node=${NODE} started=$(date --iso-8601=seconds)"

INNER=$(cat <<EOF
set -uo pipefail
# NUMA-aware staging, matching the numaboth arm this is explaining.
bash "${EXP_DIR}/scripts/stage_to_shm_numa.sh" "${SHARDED_SRC}" "${SHM_DEST}" 64

echo
echo "=== strategies (4 concurrent ranks, real rank-local shards) ==="
run() { python3 "${EXP_DIR}/scripts/shm_loader_bench.py" --dir "${SHM_DEST}" --json "${JSON}" "\$@" || echo "  ^ FAILED: \$*"; }

# Repeat each strategy: 76366 vs 76367 disagreed by up to 1.7 s on the same
# strategy, so single cells are not trustworthy at the margin we now care about.
for rep in 1 2; do
  run --strategy pinned --threads 8 --gather torch
  run --strategy pinned --gather memmove
  run --strategy pinned --gather mt --gather-threads 4
  run --strategy pinned --gather mt --gather-threads 8
  run --strategy pinned --gather mt --gather-threads 16
  run --strategy pinned --gather mt --gather-threads 8 --chunk-mib 128
done
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task=128 --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER}"

echo
echo "=== reference: e2e numaboth (job 76364) gated weight_loading = 8.47 s; target < 5.00 s ==="
echo "finished=$(date --iso-8601=seconds)"
