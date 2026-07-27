#!/bin/bash
# Phase 1.5 validation for STEP4_PLAN.md: proves the merged design (stage
# into memfd(MFD_HUGETLB), share the fd with separately-SPAWNED workers over
# a Unix-socket/SCM_RIGHTS broker, reconstruct tensors without safe_open)
# before touching SGLang itself.
#
#   sbatch shm-weight-loading-exp/scripts/hugepage_shared_pipeline.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=shm-weight-loading-exp/results/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp"
SRC="${SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
# Two of the smaller real shards -- enough to prove correctness + the
# fd-sharing mechanism without paying for the full 141 GB stage.
FILES="${FILES:-model-00029-of-00030.safetensors model-00030-of-00030.safetensors}"

NODE="$(hostname)"
echo "job=${SLURM_JOB_ID} node=${NODE} files=${FILES}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
python3 "${EXP_DIR}/scripts/test_hugepage_shared_pipeline.py" \
  "${SRC}" ${FILES} --nworkers 4 --gpu
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

echo "finished=$(date --iso-8601=seconds)"
