#!/bin/bash
# ONE-OFF, OFFLINE (vllm) - build the TP=4 presharded checkpoint that
# scripts/vllm/preshard_shm_overlap.sbatch consumes. Not part of any measurement.
#
# This exists because the SGLang-written checkpoint does NOT load in vLLM: vLLM's
# model has _k_scale/_v_scale/_q_scale/_prob_scale per layer, which SGLang never
# writes, and ShardedStateLoader hard-fails on the 320 missing keys (job 2918412).
# The shards are portable across architectures -- the SGLang arm proved that by
# reusing bristen's x86-written checkpoint on GH200 -- but not across engines.
#
# The TP de-stride is done ONCE here and frozen into each rank's file, so at
# serve time every rank reads only its own slice, already contiguous and in
# final layout.
#
# NOTE the output is TP-size-locked: these shards load only at
# --tensor-parallel-size 4. Change TP_SIZE in scripts/shared/models.sh and you
# must rebuild.
#
#   cd experiments/clariden-loading-exp && ./submit.sh vllm llama70b save
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/vllm/%x-%j.out

set -euo pipefail

ENGINE=vllm
source "${SLURM_SUBMIT_DIR}/scripts/shared/models.sh"

EDF="${SLURM_SUBMIT_DIR}/scripts/vllm/vllm-clariden.toml"
SAVE_PY="${SLURM_SUBMIT_DIR}/scripts/vllm/save_sharded_state.py"

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
# python3, not python: the vllm image has no \`python\` alias, unlike sglang's.
python3 "${SAVE_PY}" \
  --model-path "${MODEL}" \
  --output "${SHARDED_SRC}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${MEM_FRACTION_STATIC}"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (vllm save_sharded_ckpt) ==="
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

# The scale params are the whole reason this job exists, so check they are in
# there rather than trusting that vLLM's saver wrote what vLLM's loader wants.
python3 - "${SHARDED_SRC}/model-rank-0-part-0.safetensors" <<'PY'
import json, struct, sys
with open(sys.argv[1], "rb") as f:      # safetensors: 8-byte length, then JSON
    n_hdr = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n_hdr))
keys = [k for k in hdr if k != "__metadata__"]
scales = [k for k in keys if k.endswith(("_k_scale", "_v_scale", "_q_scale", "_prob_scale"))]
print("  rank0 part0: %d tensors, %d attention scale params" % (len(keys), len(scales)))
print("  MISSING SCALE PARAMS -- vllm will reject this" if not scales else "  OK: scale params present")
PY
echo
echo "NEXT: ./submit.sh vllm ${MODEL_PRESET:-llama70b} default"
echo "finished=$(date --iso-8601=seconds)"
