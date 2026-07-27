#!/bin/bash
# Standalone smoke test of hugepage_stage_daemon.py + hugepage_client.py
# against a real sharded_state checkpoint, no sglang involved.
#
#   sbatch hugepage-sharded-loading-exp/scripts/smoke_test_daemon.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=hugepage-sharded-loading-exp/results/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/hugepage-sharded-loading-exp"
CKPT="${CKPT:-/iopsstor/scratch/cscs/yboughizane/sharded-ckpt/llama70b-tp4}"
SOCK="/tmp/hugepage_smoke_${SLURM_JOB_ID}.sock"
READY="/tmp/hugepage_smoke_${SLURM_JOB_ID}.ready"

NODE="$(hostname)"
echo "job=${SLURM_JOB_ID} node=${NODE} ckpt=${CKPT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
python3 "${EXP_DIR}/scripts/hugepage_stage_daemon.py" "${CKPT}" \
  --sock "${SOCK}" --ready-file "${READY}" &
DAEMON_PID=\$!

for i in \$(seq 1 60); do
  [ -f "${READY}" ] && break
  sleep 1
done
[ -f "${READY}" ] || { echo "FATAL: daemon never became ready" >&2; exit 1; }

python3 "${EXP_DIR}/scripts/smoke_test_daemon.py" "${CKPT}" --sock "${SOCK}" --rank 0
RC=\$?

kill \$DAEMON_PID 2>/dev/null || true
wait \$DAEMON_PID 2>/dev/null || true
exit \$RC
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

echo "finished=$(date --iso-8601=seconds)"
