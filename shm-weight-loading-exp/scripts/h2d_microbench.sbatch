#!/bin/bash
# 2x2 H2D microbenchmark: does pinning and/or contiguity fix the copy that
# dominates weight_loading? See ../NOTES.md. No engine, no model load --
# minutes, not a full cold start.
#
# Runs twice on purpose:
#   1 proc  - the clean per-rank ceiling
#   4 procs - the real situation, 4 TP ranks contending for PCIe/host bandwidth
# The 4-proc number is the one to compare against the measured ~2.0 GB/s/rank.
#
#   sbatch shm-weight-loading-exp/scripts/h2d_microbench.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=shm-weight-loading-exp/results/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp"
TENSORS="${TENSORS:-16}"

NODE="$(hostname)"
echo "job=${SLURM_JOB_ID} node=${NODE} tensors=${TENSORS}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
echo "=== /dev/shm capacity ==="
df -h /dev/shm
echo "=== tmpfs mount options (is it huge=?) ==="
grep -E ' /dev/shm ' /proc/mounts || true
echo

# ONE shared file, each rank narrowing its own slice -- exactly what the real
# loader does (all 4 TP ranks mmap the same shard and take different slices of
# the same tensors). Per-rank files would 4x the tmpfs footprint and lose the
# page sharing, which is not the situation we are trying to measure.
# The solo run below creates the file; the 4-proc phase then reuses it
# (build_file returns early when the size already matches).
SHM_FILE=/dev/shm/h2dbench.bin

echo "############ 1 process (clean ceiling) ############"
python3 "${EXP_DIR}/scripts/h2d_microbench.py" \
  --path \${SHM_FILE} --tensors ${TENSORS} --gpu 0 --rank 0 --tag solo

echo
echo "############ 4 processes, DEFAULT threads (oversubscribed) ############"
for g in 0 1 2 3; do
  python3 "${EXP_DIR}/scripts/h2d_microbench.py" \
    --path \${SHM_FILE} --tensors ${TENSORS} --gpu \${g} --rank \${g} --tag d\${g} &
done
wait

echo
echo "############ 4 processes, 16 threads each (64 cores / 4 procs) ############"
# Job 75718 showed AGGREGATE gather throughput going DOWN from 1 proc to 4
# (22.56 -> 9.11 GB/s) while the driver's own pageable copy scaled UP
# (8.36 -> 16.06). Aggregate moving backwards is not a bandwidth wall, it is
# contention: torch's CPU copy_ parallelizes over OMP, and 4 procs x 64
# default threads on 64 cores thrashes. This phase pins each proc to its
# fair share so the two explanations can be told apart.
export OMP_NUM_THREADS=16
for g in 0 1 2 3; do
  python3 "${EXP_DIR}/scripts/h2d_microbench.py" \
    --path \${SHM_FILE} --tensors ${TENSORS} --gpu \${g} --rank \${g} \
    --torch-threads 16 --tag t\${g} &
done
wait

rm -f \${SHM_FILE}
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "finished=$(date --iso-8601=seconds)"
