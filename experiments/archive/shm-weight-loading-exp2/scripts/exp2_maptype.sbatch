#!/bin/bash
#SBATCH --job-name=e2-maptype
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=shm-weight-loading-exp2/results/%x-%j.out
set -uo pipefail
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp2"
SHARDED_SRC=/iopsstor/scratch/cscs/yboughizane/sharded-ckpt/llama70b-tp4
SHM_DEST=/dev/shm/llama70b-sharded
INNER=$(cat <<INNEREOF
set -uo pipefail
bash "${EXP_DIR}/scripts/stage_to_shm_numa.sh" "${SHARDED_SRC}" "${SHM_DEST}" 64 >/dev/null 2>&1
G="python3 ${EXP_DIR}/scripts/exp2_gate.py --test maptype --dir ${SHM_DEST}"
echo "=== what makes registration slow: page size, or the mapping itself? ==="
\$G --map anon_priv
\$G --map anon_shared
\$G --map memfd
\$G --map shm_shared
\$G --map shm_priv
INNEREOF
)
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-task=4 \
  --environment="${EDF}" bash -c "${INNER}"
echo "reference: anon 4K 13.28 GB/s vs /dev/shm 4K 3.57 GB/s (job 76379)"
