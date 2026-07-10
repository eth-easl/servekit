#!/bin/bash
# Phase 4 - stage a model from Lustre into node-local /dev/shm (tmpfs).
#
# Read side matches the fastest raw dd config found in the Phase 2 stripe
# sweep (c8_s16M layout, N=60 concurrent O_DIRECT readers -> 7.80 GB/s, see
# results/phase2_stripe_sweep/phase2_c8_s16M-73645-nid002317.csv). Source
# reads stay O_DIRECT (bypass page cache); tmpfs writes are plain buffered
# writes (O_DIRECT to tmpfs is unsupported/pointless).
#
# Copies only top-level files (skips any original/ pth checkpoint subdir),
# same convention as make_striped_copy.sh.
#
# Usage: stage_to_shm.sh <src_model_dir> <dest_dir> [workers]
set -euo pipefail

SRC="${1:?src model dir}"; DEST="${2:?dest dir, e.g. /dev/shm/llama70b}"
WORKERS="${3:-60}"

[[ -d "$SRC" ]] || { echo "src not a dir: $SRC" >&2; exit 1; }
mkdir -p "$DEST"

src_bytes="$(find "$SRC" -maxdepth 1 -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
avail_bytes="$(df --output=avail -B1 "$(dirname "$DEST")" | tail -1 | tr -d ' ')"
headroom=$((10*1024**3))
if (( avail_bytes < src_bytes + headroom )); then
  echo "ABORT: $(dirname "$DEST") has ${avail_bytes} bytes free, need ~${src_bytes} + ${headroom} headroom for the server" >&2
  exit 1
fi

echo "staging: $SRC -> $DEST (workers=$WORKERS, O_DIRECT read, buffered write)"
stage_one() { dd if="$1" of="$2" bs=16M iflag=direct status=none; }
export -f stage_one

start="$(date +%s.%N)"
find "$SRC" -maxdepth 1 -type f -printf '%f\n' \
  | xargs -P "$WORKERS" -I{} bash -c 'stage_one "$1/$2" "$3/$2"' _ "$SRC" {} "$DEST"
end="$(date +%s.%N)"

# Verify byte-for-byte size parity so a partial stage can't look "done".
dst_bytes="$(find "$DEST" -maxdepth 1 -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
[[ "$src_bytes" == "$dst_bytes" ]] || { echo "SIZE MISMATCH src=$src_bytes dst=$dst_bytes" >&2; exit 2; }

wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
gbps="$(awk -v b="$dst_bytes" -v w="$wall" 'BEGIN{printf "%.2f", b/w/1e9}')"
echo "staged_files=$(find "$DEST" -maxdepth 1 -type f | wc -l) staged_bytes=$dst_bytes stage_wall_s=$wall stage_GBps=$gbps"
