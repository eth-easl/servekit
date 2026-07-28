#!/bin/bash
# VERBATIM COPY of experiments/lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh
# (only this header is added). Copied rather than referenced so this experiment
# dir is self-contained; it is a copy, not a fork -- do not diverge it. Nothing
# below is bristen-specific, so it runs unchanged on clariden/aarch64.
#
# Phase 4 (sliced) - stage a model from Lustre into node-local /dev/shm, with
# fan-out WITHIN each file as well as across files.
#
# Sibling of stage_to_shm.sh, which is left untouched as the control. That one
# runs `xargs -P 60` over the file list with one `dd` per file. Shards are
# stripe_count=1, so that is ONE reader per OST -- queue depth 1 everywhere. With
# O_DIRECT (no readahead) each reader keeps exactly one 16 MiB RPC in flight, so
# a single high-latency OST caps the whole stage: OST 8 currently answers in
# ~480 ms/RPC => ~25 MB/s => ~200 s for its one 5 GB shard, and `wall` is a max
# over the pool. See experiments/lustre-contention-exp/DD_VS_FASTSAFETENSORS.md.
#
# Here each file is cut into SLICES contiguous ranges read concurrently
# (`dd skip=/seek=/count=`), so every OST sees SLICES requests in flight. Raw
# reads scaled 0.72 -> 18.9 GB/s this way (30x32) on the same node and minute.
#
# Read mode is selectable. O_DIRECT (default) bypasses the page cache, so the
# stage is honestly cold and does not evict the tmpfs copy it is creating.
# READ_MODE=buffered drops iflag=direct: the client then caches and reads ahead,
# which may hide OST latency, but every byte is now held TWICE in RAM (page cache
# + tmpfs) until the kernel reclaims. The page cache is reclaimable so this should
# not OOM, but it does mean a re-read of the same source on the same node is no
# longer cold -- see feedback on fresh nodes per cold-start run.
# tmpfs writes are plain buffered writes either way, as in the control.
#
# CONCURRENT WRITERS INTO ONE FILE is the delicate part, so:
#   * every dest file is pre-created at full size with `truncate` BEFORE any
#     writer starts -- otherwise N dd's racing to create the same path is a bug
#   * every writer uses `conv=notrunc` so it cannot shorten the file under its
#     peers, and `seek=` so it writes only its own range
#   * slices tile each file exactly (no gaps, no overlap) -- verified by the
#     caller's checksum gate, not assumed
#
# Usage: stage_to_shm_sliced.sh <src_model_dir> <dest_dir> [slices] [bs] [read_mode]
#   read_mode: direct (default, O_DIRECT) | buffered (page cache + readahead)
#   FILE_PATTERN=<glob>  stage only matching files (default '*', i.e. all of them).
#     Phase 7 uses '*.safetensors' so it can copy the small config/tokenizer files
#     synchronously first and leave only the big ones in the overlapped window.
set -euo pipefail

SRC="${1:?src model dir}"; DEST="${2:?dest dir, e.g. /dev/shm/llama70b}"
SLICES="${3:-64}"
BS="${4:-16M}"
READ_MODE="${5:-${READ_MODE:-direct}}"
BS_BYTES=$(( $(echo "$BS" | sed 's/M$//') * 1024 * 1024 ))

# Kept as a plain string, not an array: stage_slice runs under `xargs bash -c`
# and only environment strings survive that boundary.
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

# Pre-create every dest file at full size. Must complete before any writer runs.
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

# Size parity, so a partial stage cannot look "done". This is necessary but NOT
# sufficient with concurrent writers -- truncate already fixed the size, so only
# a checksum proves the CONTENT. The caller must run that gate.
dst_bytes="$(find "$DEST" -maxdepth 1 -type f -name "$FILE_PATTERN" -printf '%s\n' | awk '{s+=$1} END{print s}')"
[[ "$src_bytes" == "$dst_bytes" ]] || { echo "SIZE MISMATCH src=$src_bytes dst=$dst_bytes" >&2; exit 2; }

wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
gbps="$(awk -v b="$dst_bytes" -v w="$wall" 'BEGIN{printf "%.2f", b/w/1e9}')"
echo "staged_files=$(find "$DEST" -maxdepth 1 -type f -name "$FILE_PATTERN" | wc -l) staged_bytes=$dst_bytes slices=$SLICES read_mode=$READ_MODE readers=$NTASK stage_wall_s=$wall stage_GBps=$gbps"
