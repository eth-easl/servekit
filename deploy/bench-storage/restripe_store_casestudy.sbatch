#!/bin/bash
#SBATCH --job-name=casestudy-restripe
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/bench-loader/logs/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/bench-loader/logs/%x-%j.err
#
# restripe_store_casestudy.sbatch
#
# Case study: copy GLM-4.7-Flash from the canonical /capstor/store model dir
# (stripe_count=1 → single-OST cap) into a NEW wide-striped sibling
# `restriped/GLM-4.7-Flash/` (-c 8 -S 4m), then benchmark BOTH side-by-side
# in the same job so the comparison is fair (same node, same OST load window).
#
# Source: /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
# Dest:   /capstor/store/cscs/swissai/infra01/hf_models/models/restriped/GLM-4.7-Flash
#
# Why store (not scratch): store is the persistent tier; a `restriped/` sibling
# here is canonical, shareable, and won't be purged. xyao is in group infra01
# (setgid on the parent), so new files inherit infra01 group ownership.

set -uo pipefail
HERE="/capstor/scratch/cscs/xyao/bench-loader"
BC="$HERE/bench_core.sh"
MODEL=GLM-4.7-Flash
SRC="/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/${MODEL}"
DSTROOT="/capstor/store/cscs/swissai/infra01/hf_models/models/restriped"
DST="${DSTROOT}/${MODEL}"
PARALLEL_CP="${PARALLEL_CP:-8}"

echo "############################################################"
echo "# CASE STUDY: restripe ${MODEL} into store/restriped (-c 8 -S 4m)"
echo "# ($(date --iso-8601=seconds))  job=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "############################################################"
echo "src=${SRC}"
echo "dst=${DST}"
echo

# 1. create dest + assert wide default layout (parent defaults to stripe_count=1!)
mkdir -p "${DSTROOT}"
lfs setstripe -c 8 -S 4m "${DSTROOT}" 2>&1
mkdir -p "${DST}"
lfs setstripe -c 8 -S 4m "${DST}" 2>&1
echo "dest dir default stripe:"; lfs getstripe "${DST}" | grep -E 'stripe_count|stripe_size|stripe_offset'
echo

# 2. metadata files (small)
echo "=== metadata ==="
copied=0
for f in "${SRC}"/*; do
  case "$(basename "$f")" in
    model-*.safetensors) continue;;   # handled in bulk below
  esac
  cp -n "$f" "${DST}/" && copied=$((copied+1))
done
echo "copied ${copied} non-shard files"
echo

# 3. parallel-copy shards (each source shard on a distinct OST → N copiers = N OSTs)
SHARDS=($(ls "${SRC}"/model-*.safetensors 2>/dev/null | sort -V))
SRC_BYTES=0; for f in "${SHARDS[@]}"; do SRC_BYTES=$((SRC_BYTES + $(stat -c %s "$f"))); done
echo "=== copying ${#SHARDS[@]} shards, ${PARALLEL_CP} parallel dd copiers ==="
echo "src total = $((SRC_BYTES/1024/1024/1024)) GiB"

cp_one(){
  local s="$1" d="$2" name; name=$(basename "$s")
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
echo

# 4. verify byte sizes + confirm new stripe
echo "=== verify ==="
DST_BYTES=0; mism=0
for f in "${SHARDS[@]}"; do
  b=$(basename "$f"); s=$(stat -c %s "$f"); d=$(stat -c %s "${DST}/$b" 2>/dev/null || echo 0)
  DST_BYTES=$((DST_BYTES + d))
  [ "$s" != "$d" ] && { echo "  MISMATCH $b: src=$s dst=$d"; mism=$((mism+1)); }
done
echo "src_bytes=${SRC_BYTES}  dst_bytes=${DST_BYTES}  $([ "$SRC_BYTES" = "$DST_BYTES" ] && echo SIZE_OK || echo SIZE_MISMATCH)  mismatches=${mism}"
echo "sample dst stripe:"; lfs getstripe "${DST}/$(basename "${SHARDS[0]}")" | grep -E 'lmm_stripe_count|lmm_stripe_size'
echo

# 5. SIDE-BY-SIDE benchmark: source (stripe_count=1) vs restriped (-c8 -S4m)
echo "############################################################"
echo "# BENCHMARK (same node, back-to-back → fair comparison)"
echo "############################################################"
echo; echo "########## A. SOURCE — stripe_count=1 (canonical) ##########"
# Cap source sweep at 16 streams: the 32-stream source read wedged on a busy
# OST in job 71669, and the clean 32-stream source number already exists from
# the original characterization (job 71661: 0.72 GB/s, collapsed).
STREAMS="1 2 4 8 16" bash "$BC" "SOURCE-stripe1" "${SRC}"/model-*.safetensors
echo; echo "########## B. RESTRIPED — stripe_count=8, stripe_size=4m ##########"
bash "$BC" "RESTRIPED-stripe8" "${DST}"/model-*.safetensors

echo; echo "############################################################"
echo "# CASE STUDY DONE ($(date --iso-8601=seconds))"
echo "# To use the restriped model: MODEL=${DST}"
echo "############################################################"
