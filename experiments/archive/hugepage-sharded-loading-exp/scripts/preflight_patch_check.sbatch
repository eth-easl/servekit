#!/bin/bash
# Cheap, GPU-less preflight: apply hugepage_sharded_h2d.patch inside the
# container via the shared patch harness, confirm it swaps in against the
# REAL installed sglang (not just our local clone), and that the loader
# module + our new scripts import cleanly with PYTHONPATH set.
#
#   sbatch hugepage-sharded-loading-exp/scripts/preflight_patch_check.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=hugepage-sharded-loading-exp/results/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
EXP_DIR="${SLURM_SUBMIT_DIR}/hugepage-sharded-loading-exp"

echo "job=${SLURM_JOB_ID} node=$(hostname)"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${LUSTRE_EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" \
  "${EXP_DIR}/scripts/hugepage_sharded_h2d.patch"

echo "=== import check (no GPU, no CUDA context needed for this) ==="
PYTHONPATH="${EXP_DIR}/scripts:\${PYTHONPATH:-}" python3 -c "
import hugepage_client
import hugepage_safetensors
import hugepage_stager
import hugepage_fd_broker
from sglang.srt.model_loader import loader
print('all imports OK')
print('SGLANG_HUGEPAGE_SHARDED_SOCK knob present:',
      'SGLANG_HUGEPAGE_SHARDED_SOCK' in open(loader.__file__).read())
"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

echo "finished=$(date --iso-8601=seconds)"
