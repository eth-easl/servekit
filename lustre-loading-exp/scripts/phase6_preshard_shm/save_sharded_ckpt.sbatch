#!/bin/bash
# ONE-TIME: write a TP=4 pre-sharded checkpoint for phase6_preshard_shm.
#
# Loads the model normally at TP=4, then has each rank save its OWN shard via
# ShardedStateLoader.save_model -> model-rank-{rank}-part-{part}.safetensors.
# Metadata (config.json, tokenizer, ...) is copied alongside, so the output
# dir is a drop-in --model-path.
#
# Why this matters: the TP de-stride is done ONCE here, offline, and frozen
# into each rank's file. At serve time each rank then reads only its own
# ~35 GB, already contiguous and in final layout -- no strided narrow, no 4x
# redundancy, no cross-rank collective.
#
# The output is the SAME total size as the original (~141 GB), just
# partitioned, so it replaces the original for staging rather than adding
# to it.
#
# NOTE: shards are TP-size-locked. These load only at --tensor-parallel-size 4.
#
# Output goes to /capstor/store/cscs/swissai/infra01/cold-start-experiments,
# not scratch -- that shared dir already holds other experiments' large
# artifacts (see llama70b_c*_s*M siblings).
#
#   sbatch lustre-loading-exp/scripts/phase6_preshard_shm/save_sharded_ckpt.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=lustre-loading-exp/results/phase6_preshard_shm/%x-%j.out

set -euo pipefail

SRC="${SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
OUT="${OUT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded}"

# Only affects how this one-time job READS the source; the shards it writes
# are identical either way (load format only changes how params get filled).
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"

EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp/scripts/phase6_preshard_shm"
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
TP_SIZE=4
MAX_MODEL_LEN=32768
MEM_FRACTION_STATIC=0.85

NODE="$(hostname)"
echo "job=${SLURM_JOB_ID} node=${NODE}"
echo "src=${SRC}"
echo "out=${OUT}"
echo "started=$(date --iso-8601=seconds)"

mkdir -p "${OUT}"

INNER_CMD=$(cat <<EOF
set -euo pipefail

# Not in the image -- every fastsafetensors job in this repo installs it first.
if [ "${LOAD_FORMAT}" = "fastsafetensors" ]; then
  pip install fastsafetensors -q
fi

SAVE_PY="${EXP_DIR}/save_sharded_state_fixed.py"
echo "using save script: \${SAVE_PY}"

python "\${SAVE_PY}" \
  --model-path "${SRC}" \
  --output "${OUT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --load-format "${LOAD_FORMAT}" \
  --trust-remote-code \
  --context-length "${MAX_MODEL_LEN}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result ==="
echo "shard files per rank:"
for r in 0 1 2 3; do
  n=$(ls "${OUT}"/model-rank-${r}-part-*.safetensors 2>/dev/null | wc -l)
  sz=$(du -csh "${OUT}"/model-rank-${r}-part-*.safetensors 2>/dev/null | tail -1 | cut -f1)
  echo "  rank ${r}: ${n} parts, ${sz:-0}"
done
echo "total: $(du -sh "${OUT}" 2>/dev/null | cut -f1)"
echo "metadata files: $(ls "${OUT}" | grep -vc 'model-rank-' || true) (config/tokenizer etc.)"
echo
echo "next: sbatch lustre-loading-exp/scripts/phase6_preshard_shm/phase6_preshard_shm.sbatch"
echo "finished=$(date --iso-8601=seconds)"
