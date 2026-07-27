#!/bin/bash
# NUMA-aware sliced stager: Lustre -> /dev/shm, with each rank's shard placed
# on the NUMA node local to the GPU that will read it.
#
# Copied from lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh
# (11.6 GB/s, O_DIRECT, concurrent slices per file) with exactly one change:
# every writer for `model-rank-R-part-*.safetensors` runs under
# `numactl --membind=<node local to GPU R>`, so the tmpfs pages it allocates
# land on that node.
#
# Why this matters (job 76361): 4 concurrent ranks reading tmpfs pages on a
# remote NUMA node get 2.9-4.9 GB/s each; reading node-local pages they get
# 9.9-10.0 GB/s. On an EPYC 7713 in NPS4 mode that gap is the entire
# difference between an 11 s weight_loading phase and a sub-5 s one.
#
# Placement is a property of the WRITER, so it must be fixed here, at stage
# time. Binding the TP rank later does not move pages that already exist.
#
# Usage: stage_to_shm_numa.sh <src_model_dir> <dest_dir> [slices] [bs] [read_mode]
set -euo pipefail

SRC="${1:?src model dir}"; DEST="${2:?dest dir, e.g. /dev/shm/llama70b}"
SLICES="${3:-64}"
BS="${4:-16M}"
READ_MODE="${5:-${READ_MODE:-direct}}"
BS_BYTES=$(( $(echo "$BS" | sed 's/M$//') * 1024 * 1024 ))
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$READ_MODE" in
  direct)   IFLAG="iflag=direct" ;;
  buffered) IFLAG="" ;;
  *) echo "read_mode must be 'direct' or 'buffered', got: $READ_MODE" >&2; exit 1 ;;
esac

[[ -d "$SRC" ]] || { echo "src not a dir: $SRC" >&2; exit 1; }
mkdir -p "$DEST"

# GPU -> NUMA node. Empty (or no numactl) means stage unbound, exactly like the
# original script -- a missing map must never silently bind everything to 0.
declare -A GPU_NODE=()
if command -v numactl >/dev/null 2>&1; then
  while read -r gpu node; do
    [[ -n "${gpu:-}" ]] && GPU_NODE[$gpu]=$node
  done < <(python3 "${HERE}/numa_map.py" 2>/dev/null || true)
fi
if [[ ${#GPU_NODE[@]} -eq 0 ]]; then
  echo "WARNING: no GPU->NUMA map (numactl missing or query failed); staging unbound" >&2
else
  echo "gpu->numa map: $(for g in $(echo "${!GPU_NODE[@]}" | tr ' ' '\n' | sort -n); do printf '%s->%s ' "$g" "${GPU_NODE[$g]}"; done)"
fi

mapfile -t FILES < <(find "$SRC" -maxdepth 1 -type f -printf '%f\n' | sort)
src_bytes="$(find "$SRC" -maxdepth 1 -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
avail_bytes="$(df --output=avail -B1 "$(dirname "$DEST")" | tail -1 | tr -d ' ')"
headroom=$((10*1024**3))
if (( avail_bytes < src_bytes + headroom )); then
  echo "ABORT: $(dirname "$DEST") has ${avail_bytes} bytes free, need ~${src_bytes} + ${headroom} headroom" >&2
  exit 1
fi

echo "staging(numa): $SRC -> $DEST (files=${#FILES[@]}, slices/file=$SLICES, bs=$BS, ${READ_MODE} read)"

# Pre-create every dest file at full size before any writer starts. NOTE: this
# only sets the size; tmpfs allocates a page on first WRITE, so the numactl
# binding on the writer is what decides placement, not this truncate.
for f in "${FILES[@]}"; do
  truncate -s "$(stat -c%s "$SRC/$f")" "$DEST/$f"
done

# Rank R's file -> node local to GPU R. Files with no rank in the name (config,
# tokenizer) are tiny and read by every rank, so leave them unbound.
node_for_file() {
  local f="$1" rank
  if [[ "$f" =~ model-rank-([0-9]+)-part- ]]; then
    rank="${BASH_REMATCH[1]}"
    echo "${GPU_NODE[$rank]:--}"
  else
    echo "-"
  fi
}

stage_slice() { # <src> <dst> <skip_blocks> <count_blocks> <node|->
  if [[ "$5" == "-" ]]; then
    dd if="$1" of="$2" bs="$BS" skip="$3" seek="$3" count="$4" \
       $IFLAG conv=notrunc status=none
  else
    numactl --membind="$5" --cpunodebind="$5" -- \
      dd if="$1" of="$2" bs="$BS" skip="$3" seek="$3" count="$4" \
         $IFLAG conv=notrunc status=none
  fi
}
export -f stage_slice
export BS IFLAG

emit() {
  local f size blocks per i skip cnt node
  for f in "${FILES[@]}"; do
    size=$(stat -c%s "$SRC/$f")
    node="$(node_for_file "$f")"
    blocks=$(( (size + BS_BYTES - 1) / BS_BYTES ))
    (( blocks == 0 )) && blocks=1
    per=$(( (blocks + SLICES - 1) / SLICES ))
    for (( i=0; i<SLICES; i++ )); do
      skip=$(( i * per )); (( skip >= blocks )) && break
      cnt=$per; (( skip + cnt > blocks )) && cnt=$(( blocks - skip ))
      echo "$SRC/$f $DEST/$f $skip $cnt $node"
    done
  done
}

NTASK=$(emit | wc -l)
start="$(date +%s.%N)"
emit | xargs -P "$NTASK" -n 5 bash -c 'stage_slice "$@"' _
end="$(date +%s.%N)"

dst_bytes="$(find "$DEST" -maxdepth 1 -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
[[ "$src_bytes" == "$dst_bytes" ]] || { echo "SIZE MISMATCH src=$src_bytes dst=$dst_bytes" >&2; exit 2; }

wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
gbps="$(awk -v b="$dst_bytes" -v w="$wall" 'BEGIN{printf "%.2f", b/w/1e9}')"
echo "staged_files=$(find "$DEST" -maxdepth 1 -type f | wc -l) staged_bytes=$dst_bytes slices=$SLICES read_mode=$READ_MODE readers=$NTASK stage_wall_s=$wall stage_GBps=$gbps"
