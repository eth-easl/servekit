#!/bin/bash
# Phase 2 helper - create a Lustre-striped copy of a model.
#
# You cannot restripe a file in place: the stripe layout is fixed at creation.
# So we set a DEFAULT layout on a fresh directory and copy the model in; every
# new file inherits that layout (stripe_count C across OSTs, stripe_size S).
#
# Copies only top-level files (*.safetensors + config/tokenizer), i.e. skips any
# `original/` pth checkpoint subdir - that's all SGLang loads and keeps the copy
# to ~132 GB.
#
# Usage: make_striped_copy.sh <src_model_dir> <dest_dir> <stripe_count> <stripe_size> [copy_workers]
set -euo pipefail

SRC="${1:?src model dir}"; DEST="${2:?dest dir}"
C="${3:?stripe_count}";    S="${4:?stripe_size e.g. 4M}"
WORKERS="${5:-16}"

[[ -d "$SRC" ]] || { echo "src not a dir: $SRC" >&2; exit 1; }
[[ "$DEST" == "$SRC" ]] && { echo "refuse: dest == src" >&2; exit 1; }

echo "striped copy: $SRC -> $DEST  (stripe_count=$C stripe_size=$S workers=$WORKERS)"
mkdir -p "$DEST"
# Set the directory default layout BEFORE copying so files inherit it.
lfs setstripe -c "$C" -S "$S" "$DEST"
echo "dest dir layout:"; lfs getstripe -d "$DEST"

# Parallel copy of top-level files only (find -maxdepth 1). cp creates new files
# that inherit DEST's default stripe layout. Timed: copy wall-time + throughput
# is the Strategy-A provisioning cost (and doubles as a write-bandwidth number).
start="$(date +%s.%N)"
find "$SRC" -maxdepth 1 -type f -printf '%f\n' \
  | xargs -P "$WORKERS" -I{} cp -p "$SRC/{}" "$DEST/{}"
end="$(date +%s.%N)"

# Verify byte-for-byte size parity so a partial copy can't look "done".
src_bytes="$(find "$SRC"  -maxdepth 1 -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
dst_bytes="$(find "$DEST" -maxdepth 1 -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
[[ "$src_bytes" == "$dst_bytes" ]] || { echo "SIZE MISMATCH src=$src_bytes dst=$dst_bytes" >&2; exit 2; }

wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
gbps="$(awk -v b="$dst_bytes" -v w="$wall" 'BEGIN{printf "%.2f", b/w/1e9}')"
echo "verify - layout of one copied shard (expect stripe_count=$C):"
lfs getstripe -c "$DEST"/model-00001-of-*.safetensors 2>/dev/null || true
echo "copied_files=$(find "$DEST" -maxdepth 1 -type f | wc -l) copied_bytes=$dst_bytes copy_wall_s=$wall copy_GBps=$gbps"
