#!/bin/bash
# Phase 1.2 - full-model read time vs reader concurrency, page-cache bypassed
# (O_DIRECT via `dd iflag=direct`). fio is unavailable on these nodes; dd gives
# an honest cold-read number and is dependency-free.
#
# Design: fixed workload = ALL shards (the whole model, ~132 GB). For each
# concurrency N we drain the shard list with a pool of N parallel dd workers
# (`xargs -P N`), so every row reads the identical full model and `wall_s` is
# literally "time to load the whole model at parallelism N". N may exceed the
# shard count (workers pick up the next shard as they finish, exercising hot
# OSTs that host >1 shard).
#
# Usage: dd_read_sweep.sh <model_dir> [bs] [csv_out] [concurrency_list]
set -euo pipefail

MODEL="${1:?usage: dd_read_sweep.sh <model_dir> [bs] [csv_out] [concurrency_list]}"
BS="${2:-16M}"
CSV="${3:-/dev/stdout}"
CONC="${4:-1 4 8 16 32 64}"

mapfile -t SHARDS < <(ls "$MODEL"/*.safetensors | sort)
NSHARDS="${#SHARDS[@]}"
TOTAL=0
for f in "${SHARDS[@]}"; do TOTAL=$(( TOTAL + $(stat -c%s "$f") )); done
echo "host=$(hostname) shards=$NSHARDS total=$(awk -v t=$TOTAL 'BEGIN{printf "%.1fGB",t/1e9}') bs=$BS (O_DIRECT)" >&2

# Read one shard fully, O_DIRECT, discard. Alignment-safe (dd handles partial
# final block). Errors are fatal so a silent short read can't fake a fast time.
read_one() { dd if="$1" of=/dev/null bs="$BS" iflag=direct status=none; }
export -f read_one
export BS

echo "concurrency,total_bytes,wall_s,agg_GBps" | tee "$CSV"
for N in $CONC; do
  start="$(date +%s.%N)"
  printf '%s\n' "${SHARDS[@]}" | xargs -P "$N" -I{} bash -c 'read_one "$@"' _ {}
  end="$(date +%s.%N)"
  wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
  agg="$(awk -v t="$TOTAL" -v w="$wall" 'BEGIN{printf "%.2f", t/w/1e9}')"
  echo "$N,$TOTAL,$wall,$agg" | tee -a "$CSV"
done
