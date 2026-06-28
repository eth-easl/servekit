#!/bin/bash
#SBATCH --job-name=restripe-model
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/bench-loader/logs/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/bench-loader/logs/%x-%j.err
#
# restripe_model.sbatch — copy a canonical /capstor/store model into a
# wide-striped scratch dir (so reads aren't capped by stripe_count=1) and
# re-benchmark the new location. Default layout (-c 8 -S 4m) is inherited by
# every file created in the dest dir, so no per-file setstripe is needed.
#
# Scope:
#   MODEL_NAME=GLM-4.7-Flash   (59 GB,  ~2 min to copy)  [default — quick proof]
#   MODEL_NAME=GLM-5.2-FP8     (704 GB, ~20-30 min to copy)
#   PARALLEL_CP=8              concurrent dd copiers (one per source shard)
#
# After it runs, point the deploy at the new path, e.g.:
#   MODEL=/capstor/scratch/cscs/xyao/models/GLM-4.7-Flash

set -uo pipefail
HERE="/capstor/scratch/cscs/xyao/bench-loader"
BC="$HERE/bench_core.sh"
MODEL_NAME="${MODEL_NAME:-GLM-4.7-Flash}"
PARALLEL_CP="${PARALLEL_CP:-8}"
SRC="/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/${MODEL_NAME}"
DSTROOT="/capstor/scratch/cscs/xyao/models"
DST="${DSTROOT}/${MODEL_NAME}"

echo "############################################################"
echo "# RESTRIPE ${MODEL_NAME} ($(date --iso-8601=seconds))"
echo "############################################################"
echo "job=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "src=${SRC}"
echo "dst=${DST}"
echo "parallel_copiers=${PARALLEL_CP}"
echo

# 1. ensure dest dir + its default stripe layout (-c 8 -S 4m)
mkdir -p "${DSTROOT}"
# (re-)assert layout on the root; idempotent.
lfs setstripe -c 8 -S 4m "${DSTROOT}" 2>/dev/null || true
mkdir -p "${DST}"
lfs setstripe -c 8 -S 4m "${DST}" 2>&1
echo "dest default stripe:"; lfs getstripe "${DST}" | grep -E 'stripe_count|stripe_size'

# 2. copy NON-safetensors (config, tokenizer, etc.) first — small, fast
echo; echo "=== copying metadata files ==="
for f in config.json tokenizer* special_tokens* model.safetensors.index.json \
         *.json *.txt *.model; do
  [ -f "${SRC}/$f" ] && cp -n "${SRC}/$f" "${DST}/" && echo "  copied $f"
done

# 3. parallel-copy the safetensors shards. Source is stripe_count=1 but each
#    shard lives on a DIFFERENT OST, so N concurrent copiers read N distinct
#    OSTs at once -> fast despite the per-file single-stripe cap.
SHARDS=($(ls "${SRC}"/model-*.safetensors 2>/dev/null | sort -V))
echo; echo "=== copying ${#SHARDS[@]} shards, ${PARALLEL_CP} parallel dd copiers ==="
SRC_BYTES=0; for f in "${SHARDS[@]}"; do SRC_BYTES=$((SRC_BYTES + $(stat -c %s "$f"))); done
echo "src total = $((SRC_BYTES/1024/1024/1024)) GiB"

cp_one(){ # $1=src $2=dst  (dd 4M blocks; dest inherits -c8 -S4m from dir)
  local s="$1" d="$2" name
  name=$(basename "$s")
  if [ -f "$d" ] && [ "$(stat -c %s "$d")" = "$(stat -c %s "$s")" ]; then
    echo "  skip (exists+size-ok): $name"; return 0
  fi
  dd if="$s" of="$d" bs=4M oflag=direct status=none 2>/dev/null \
    || dd if="$s" of="$d" bs=4M status=none
}
export -f cp_one

t0=$(date +%s)
printf '%s\n' "${SHARDS[@]}" | xargs -P "${PARALLEL_CP}" -I{} \
  bash -c 'cp_one "$0" "'"${DST}"'/$(basename "$0")"' {}
t1=$(date +%s)
echo "copy wall = $((t1-t0))s  (~$((SRC_BYTES/1024/1024/(t1-t0>0?t1-t0:1))) MB/s)"

# 4. verify sizes match
echo; echo "=== verify ==="
DST_BYTES=0; for f in "${SHARDS[@]}"; do
  b=$(basename "$f"); s=$(stat -c %s "$f"); d=$(stat -c %s "${DST}/$b" 2>/dev/null || echo 0)
  DST_BYTES=$((DST_BYTES + d))
  [ "$s" != "$d" ] && echo "  MISMATCH $b: src=$s dst=$d"
done
echo "src_bytes=${SRC_BYTES}  dst_bytes=${DST_BYTES}  $([ "$SRC_BYTES" = "$DST_BYTES" ] && echo SIZE_OK || echo SIZE_MISMATCH)"
echo "sample dst stripe:"; lfs getstripe "${DST}/$(basename "${SHARDS[0]}")" | grep -E 'stripe_count|stripe_size'

# 5. re-benchmark the NEW location vs the numbers we already have for the source
echo; echo "=== bench NEW (wide-striped) location ==="
bash "$BC" "RESTRIPED-${MODEL_NAME}" "${DST}"/model-*.safetensors

echo; echo "############################################################"
echo "# DONE ($(date --iso-8601=seconds))"
echo "# To use: MODEL=${DST}"
echo "############################################################"
