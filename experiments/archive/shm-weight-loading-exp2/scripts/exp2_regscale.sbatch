#!/bin/bash
# Does cudaHostRegister's RATE depend on how much is already registered?
# 1 shard (5.3 GB) -> 9.65 GB/s; 7 shards (35.3 GB) -> 3.57 GB/s. If the rate
# is a function of total registered bytes, then a hybrid design that registers
# only a FRACTION runs at the fast rate, not the slow one -- which is the
# difference between the hybrid missing 5 s and clearing it.
#SBATCH --job-name=e2-regscale
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
G="python3 ${EXP_DIR}/scripts/exp2_gate.py --test shards --dir ${SHM_DEST}"
echo "=== registration rate vs total bytes registered (4 ranks, real shards) ==="
for n in 1 2 3 4 5 7; do \$G --files \$n; done
INNEREOF
)
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-task=4 \
  --environment="${EDF}" bash -c "${INNER}"
