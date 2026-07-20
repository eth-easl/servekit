#!/bin/bash
# Full-model O_DIRECT read with BOTH kinds of parallelism, swept.
#
# `dd_read_sweep.sh` fans out across FILES only (`xargs -P N` over the shard
# list). Shards are stripe_count=1, so that is one reader per OST -- queue depth
# 1 on every OST, including a sick one. Its wall_s is a max over the pool, so one
# high-latency OST becomes the whole reported number (see NOTES.md: OST 8,
# ~480 ms/RPC, made capstor look like 0.6 GB/s while it was serving 6-8).
#
# This probe adds the second axis: each shard is cut into T contiguous slices
# read concurrently (`dd skip=/count=`), so every OST sees T requests in flight
# instead of 1. Total reader processes = files_in_flight x T.
#
# Identical bytes read at every T -- the slices of a file exactly tile it, so
# `wall_s` stays "time to read the whole model" and rows are comparable.
# T=1 with -P 30 reduces to `dd_read_sweep.sh` exactly, which is the control.
#
# Usage: dd_read_sliced.sh <model_dir> [bs] [csv_out] [slices_list] [pool]
#   pool: readers in flight. "auto" (default) = 30 x T, i.e. hold ~all files
#         open with T readers each. A number pins it for a fair-concurrency
#         comparison against the old probe.
set -euo pipefail

MODEL="${1:?usage: dd_read_sliced.sh <model_dir> [bs] [csv_out] [slices_list] [pool]}"
BS="${2:-16M}"
CSV="${3:-/dev/stdout}"
SLICES="${4:-1 2 4 8 16}"
POOL="${5:-auto}"

BS_BYTES=$(( $(echo "$BS" | sed 's/M$//') * 1024 * 1024 ))

mapfile -t SHARDS < <(ls "$MODEL"/*.safetensors | sort)
NSHARDS="${#SHARDS[@]}"
TOTAL=0
for f in "${SHARDS[@]}"; do TOTAL=$(( TOTAL + $(stat -c%s "$f") )); done
echo "host=$(hostname) shards=$NSHARDS total=$(awk -v t=$TOTAL 'BEGIN{printf "%.1fGB",t/1e9}') bs=$BS (O_DIRECT, sliced)" >&2

# One contiguous slice of one shard. dd reads a short final block itself, so the
# last slice of a file needs no special case beyond clamping count.
read_slice() { dd if="$1" of=/dev/null bs="$BS" skip="$2" count="$3" iflag=direct status=none; }
export -f read_slice
export BS

# Emit "<file> <skip_blocks> <count_blocks>" tasks tiling every shard into T
# slices. Ordered file-major, so the pool's active window walks the shard list
# rather than starting all 30 files at once.
emit_tasks() {
  local T="$1" f size blocks per i skip cnt
  for f in "${SHARDS[@]}"; do
    size=$(stat -c%s "$f")
    blocks=$(( (size + BS_BYTES - 1) / BS_BYTES ))
    per=$(( (blocks + T - 1) / T ))
    for (( i=0; i<T; i++ )); do
      skip=$(( i * per ))
      (( skip >= blocks )) && break
      cnt=$per
      (( skip + cnt > blocks )) && cnt=$(( blocks - skip ))
      echo "$f $skip $cnt"
    done
  done
}

echo "slices_per_file,pool,total_bytes,wall_s,agg_GBps" | tee "$CSV"
for T in $SLICES; do
  if [ "$POOL" = "auto" ]; then P=$(( NSHARDS * T )); else P="$POOL"; fi
  start="$(date +%s.%N)"
  emit_tasks "$T" | xargs -P "$P" -n 3 bash -c 'read_slice "$@"' _
  end="$(date +%s.%N)"
  wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
  agg="$(awk -v t="$TOTAL" -v w="$wall" 'BEGIN{printf "%.2f", t/w/1e9}')"
  echo "$T,$P,$TOTAL,$wall,$agg" | tee -a "$CSV"
done
