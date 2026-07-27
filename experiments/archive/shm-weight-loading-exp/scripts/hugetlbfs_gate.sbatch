#!/bin/bash
# Phase 0 gate for STEP4_PLAN.md -- build nothing until this passes.
# Two independent checks in hugetlbfs_gate.py (0a: is hugetlbfs usable, 0b: is
# cudaHostRegister dramatically faster on 2 MB pages), plus a free nsys
# cuda_api_sum pass over the existing job-76112 trace (splits time inside
# cudaMemcpy* from time outside it -- faults + iterator).
#
#   sbatch shm-weight-loading-exp/scripts/hugetlbfs_gate.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=shm-weight-loading-exp/results/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp"
GB="${GB:-8}"

NODE="$(hostname)"
echo "job=${SLURM_JOB_ID} node=${NODE} gb=${GB}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
echo "=== hugetlbfs pool ==="
grep -E 'HugePages|Hugepagesize' /proc/meminfo
cat /proc/sys/vm/nr_overcommit_hugepages 2>/dev/null || true
echo

python3 "${EXP_DIR}/scripts/hugetlbfs_gate.py" --gb ${GB} --gpu 0
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== free: nsys cuda_api_sum on existing job-76112 trace ==="
REP="${EXP_DIR}/results/nsys-76112-nid002292.nsys-rep"
if [[ -f "${REP}" ]]; then
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 \
    --environment="${EDF}" \
    nsys stats --report cuda_api_sum "${REP}" || true
else
  echo "missing ${REP}, skipping"
fi

echo "finished=$(date --iso-8601=seconds)"
