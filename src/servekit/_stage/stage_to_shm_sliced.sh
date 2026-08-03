#!/bin/bash
# Stage a model from Lustre into node-local /dev/shm, fanning out WITHIN each
# file as well as across files: each file is cut into SLICES contiguous ranges
# read concurrently, so every OST sees SLICES requests in flight instead of one.
# See experiments/lustre-loading-exp/ for the measurements.
#
# Usage: stage_to_shm_sliced.sh <src_model_dir> <dest_dir> [slices] [bs] [read_mode]
#   read_mode: direct (default, O_DIRECT -- honestly cold, and does not evict the
#     tmpfs copy it is creating) | buffered (page cache + readahead, which may
#     hide OST latency but holds every byte TWICE in RAM until reclaim).
#   FILE_PATTERN=<glob>  stage only matching files (default '*').
set -euo pipefail

SRC="${1:?src model dir}"; DEST="${2:?dest dir, e.g. /dev/shm/llama70b}"
SLICES="${3:-64}"
BS="${4:-16M}"
READ_MODE="${5:-${READ_MODE:-direct}}"
# The slice tiling counts in blocks, so BS_BYTES must agree with the block size
# dd is handed. Reject what we cannot convert rather than tile against the wrong
# unit: KB=1000, block lists and `x` products would all be silently wrong.
case "$BS" in
  *[0-9])   BS_BYTES=$(( BS )) ;;
  *[kK])    BS_BYTES=$(( ${BS%?} * 1024 )) ;;
  *[mM])    BS_BYTES=$(( ${BS%?} * 1024 * 1024 )) ;;
  *[gG])    BS_BYTES=$(( ${BS%?} * 1024 * 1024 * 1024 )) ;;
  *) echo "bs must be a byte count with an optional K/M/G suffix, got: $BS" >&2; exit 1 ;;
esac
(( BS_BYTES > 0 )) || { echo "bs must be positive, got: $BS" >&2; exit 1; }

# A plain string, not an array: stage_slice runs under `xargs bash -c` and only
# environment strings survive that boundary.
case "$READ_MODE" in
  direct)   IFLAG="iflag=direct" ;;
  buffered) IFLAG="" ;;
  *) echo "read_mode must be 'direct' or 'buffered', got: $READ_MODE" >&2; exit 1 ;;
esac

[[ -d "$SRC" ]] || { echo "src not a dir: $SRC" >&2; exit 1; }
mkdir -p "$DEST"

FILE_PATTERN="${FILE_PATTERN:-*}"
mapfile -t FILES < <(find "$SRC" -maxdepth 1 -type f -name "$FILE_PATTERN" -printf '%f\n' | sort)
(( ${#FILES[@]} > 0 )) || { echo "no files matching '$FILE_PATTERN' in $SRC" >&2; exit 1; }
src_bytes="$(find "$SRC" -maxdepth 1 -type f -name "$FILE_PATTERN" -printf '%s\n' | awk '{s+=$1} END{print s}')"
avail_bytes="$(df --output=avail -B1 "$(dirname "$DEST")" | tail -1 | tr -d ' ')"
headroom=$((10*1024**3))
if (( avail_bytes < src_bytes + headroom )); then
  echo "ABORT: $(dirname "$DEST") has ${avail_bytes} bytes free, need ~${src_bytes} + ${headroom} headroom for the server" >&2
  exit 1
fi

echo "staging(sliced): $SRC -> $DEST (files=${#FILES[@]}, slices/file=$SLICES, bs=$BS, ${READ_MODE} read, buffered write)"

# Full size up front, before any writer runs: N dd's racing to create the same
# path would be a bug. Writers then use conv=notrunc so none can shorten the
# file under its peers, and seek= so each writes only its own range.
for f in "${FILES[@]}"; do
  truncate -s "$(stat -c%s "$SRC/$f")" "$DEST/$f"
done

stage_slice() { # <src> <dst> <skip_blocks> <count_blocks>
  dd if="$1" of="$2" bs="$BS" skip="$3" seek="$3" count="$4" \
     $IFLAG conv=notrunc status=none
}
export -f stage_slice
export BS IFLAG

# "<src> <dst> <skip> <count>" per slice, tiling each file exactly. A file
# smaller than one block yields a single slice, so small config files just work.
emit() {
  local f size blocks per i skip cnt
  for f in "${FILES[@]}"; do
    size=$(stat -c%s "$SRC/$f")
    blocks=$(( (size + BS_BYTES - 1) / BS_BYTES ))
    (( blocks == 0 )) && blocks=1
    per=$(( (blocks + SLICES - 1) / SLICES ))
    for (( i=0; i<SLICES; i++ )); do
      skip=$(( i * per )); (( skip >= blocks )) && break
      cnt=$per; (( skip + cnt > blocks )) && cnt=$(( blocks - skip ))
      echo "$SRC/$f $DEST/$f $skip $cnt"
    done
  done
}

NTASK=$(emit | wc -l)
start="$(date +%s.%N)"
emit | xargs -P "$NTASK" -n 4 bash -c 'stage_slice "$@"' _
end="$(date +%s.%N)"

# Necessary but not sufficient: truncate already fixed the size, so this catches
# a partial stage but never wrong content. Only a checksum proves that -- see
# test_stage_is_bitwise_exact_across_concurrent_writers.
dst_bytes="$(find "$DEST" -maxdepth 1 -type f -name "$FILE_PATTERN" -printf '%s\n' | awk '{s+=$1} END{print s}')"
[[ "$src_bytes" == "$dst_bytes" ]] || { echo "SIZE MISMATCH src=$src_bytes dst=$dst_bytes" >&2; exit 2; }

wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
gbps="$(awk -v b="$dst_bytes" -v w="$wall" 'BEGIN{printf "%.2f", b/w/1e9}')"
echo "staged_files=$(find "$DEST" -maxdepth 1 -type f -name "$FILE_PATTERN" | wc -l) staged_bytes=$dst_bytes slices=$SLICES read_mode=$READ_MODE readers=$NTASK stage_wall_s=$wall stage_GBps=$gbps"
