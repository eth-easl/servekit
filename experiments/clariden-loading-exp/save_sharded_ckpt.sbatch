#!/bin/bash
# ONE-OFF, OFFLINE - build the TP=4 presharded checkpoint that
# preshard_shm_overlap.sbatch consumes. Not part of any measurement.
#
# The TP de-stride is done ONCE here and frozen into each rank's file, so at
# serve time every rank reads only its own slice, already contiguous and in
# final layout -- no strided narrow, no TP-fold redundancy, no cross-rank
# collective. That is the whole reason the preshard config loads fast.
#
# NOTE the output is TP-size-locked: these shards load only at
# --tensor-parallel-size 4. Change TP_SIZE in models.sh and you must rebuild.
#
# The llama70b preset does NOT need this -- its checkpoint already exists from
# the bristen run and was shown to be architecture-portable. Only run it for a
# model that has no sharded checkpoint yet.
#
#   cd experiments/clariden-loading-exp && ./submit.sh apertus8b save
#
# This job also doubles as the per-model gate: it stands up a full SGLang
# Engine, so if the model does not load under this image at all, it fails here
# for ~pennies instead of inside a measured run.
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/models.sh"

# Default loader, not fastsafetensors: this is a one-off offline job where load
# speed is irrelevant, and it avoids depending on an aarch64 wheel that the
# image does not ship.
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

EDF="${SLURM_SUBMIT_DIR}/sglang-clariden.toml"
SAVE_PY="${SLURM_SUBMIT_DIR}/save_sharded_state_fixed.py"

NODE="$(hostname)"
echo "job=${SLURM_JOB_ID} node=${NODE} cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-llama70b}"
echo "src=${MODEL}"
echo "out=${SHARDED_SRC}"
echo "started=$(date --iso-8601=seconds)"

if [ -e "${SHARDED_SRC}/model-rank-0-part-0.safetensors" ]; then
  echo "ERROR: ${SHARDED_SRC} already holds a sharded checkpoint." >&2
  echo "       Refusing to overwrite. Remove it by hand if that is what you want." >&2
  exit 1
fi
mkdir -p "${SHARDED_SRC}"

INNER_CMD=$(cat <<EOF
set -euo pipefail
python "${SAVE_PY}" \
  --model-path "${MODEL}" \
  --output "${SHARDED_SRC}" \
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
echo "=== result (save_sharded_ckpt) ==="
total_parts=0
for r in $(seq 0 $((TP_SIZE - 1))); do
  n=$(ls "${SHARDED_SRC}"/model-rank-${r}-part-*.safetensors 2>/dev/null | wc -l)
  sz=$(du -csh "${SHARDED_SRC}"/model-rank-${r}-part-*.safetensors 2>/dev/null | tail -1 | cut -f1)
  echo "  rank ${r}: ${n} parts, ${sz:-0}"
  total_parts=$((total_parts + n))
done
echo "  total: $(du -sh "${SHARDED_SRC}" 2>/dev/null | cut -f1), ${total_parts} shard files"
echo "  metadata files: $(ls "${SHARDED_SRC}" | grep -vc 'model-rank-' || true) (config/tokenizer etc.)"

# Every rank must have produced at least one part, or the preshard config would
# fail obscurely at load time instead of here.
missing=0
for r in $(seq 0 $((TP_SIZE - 1))); do
  ls "${SHARDED_SRC}"/model-rank-${r}-part-0.safetensors >/dev/null 2>&1 || missing=1
done
if [ "${missing}" -ne 0 ]; then
  echo "  FAILED: at least one rank produced no shards" >&2
  exit 1
fi
echo "  OK: all ${TP_SIZE} ranks present"
echo
echo "NEXT: ./submit.sh ${MODEL_PRESET:-llama70b} default"
echo "finished=$(date --iso-8601=seconds)"
