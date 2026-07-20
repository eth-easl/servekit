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
# `wall_s` is the max over the worker pool, so the aggregate alone cannot tell a
# uniformly slow filesystem from 29 fast shards plus one straggler -- the two
# produce the same number. Set PERSHARD_CSV to also record every shard's own
# read time and its OST. Every shard on capstor is stripe_count=1 (phase 1.1),
# i.e. one shard == one OST, so those rows are a per-OST bandwidth map.
#
# Usage: dd_read_sweep.sh <model_dir> [bs] [csv_out] [concurrency_list]
#        PERSHARD_CSV=<path> dd_read_sweep.sh ...   # + per-shard/per-OST rows
set -euo pipefail

MODEL="${1:?usage: dd_read_sweep.sh <model_dir> [bs] [csv_out] [concurrency_list]}"
BS="${2:-16M}"
CSV="${3:-/dev/stdout}"
CONC="${4:-1 4 8 16 32 64}"
PERSHARD="${PERSHARD_CSV:-}"

mapfile -t SHARDS < <(ls "$MODEL"/*.safetensors | sort)
NSHARDS="${#SHARDS[@]}"
TOTAL=0
for f in "${SHARDS[@]}"; do TOTAL=$(( TOTAL + $(stat -c%s "$f") )); done
echo "host=$(hostname) shards=$NSHARDS total=$(awk -v t=$TOTAL 'BEGIN{printf "%.1fGB",t/1e9}') bs=$BS (O_DIRECT)" >&2

# Read one shard fully, O_DIRECT, discard. Alignment-safe (dd handles partial
# final block). Errors are fatal so a silent short read can't fake a fast time.
# The row goes to stdout, which the caller below appends to PERSHARD (or drops);
# it is one short line written to an O_APPEND fd, so concurrent workers cannot
# interleave it. Both `date` calls bracket the dd and nothing else -- the OST
# lookup is a lookup in an already-built map, never an `lfs` call in the timed
# region.
read_one() {
  local f="$1" bytes start end wall
  [ -z "${PERSHARD:-}" ] && { dd if="$f" of=/dev/null bs="$BS" iflag=direct status=none; return; }
  bytes="$(stat -c%s "$f")"
  start="$(date +%s.%N)"
  dd if="$f" of=/dev/null bs="$BS" iflag=direct status=none
  end="$(date +%s.%N)"
  wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$N_CUR" "$(basename "$f")" \
    "$(awk -v n="$(basename "$f")" '$1==n{print $2; exit}' <<<"$OSTMAP")" \
    "$bytes" "$wall" \
    "$(awk -v b="$bytes" -v w="$wall" 'BEGIN{printf "%.3f", b/w/1e9}')"
}
export -f read_one
export BS PERSHARD

if [ -n "$PERSHARD" ]; then
  # Built once, up front: `lfs getstripe` is an MDS round-trip and must not land
  # inside a timed read. `lfs` is absent from some containers -> ost_idx=-1, and
  # the timings stay useful without it.
  OSTMAP=""
  for f in "${SHARDS[@]}"; do
    ost="$(lfs getstripe -i "$f" 2>/dev/null | tr -dc '0-9')"
    OSTMAP+="$(basename "$f") ${ost:--1}"$'\n'
  done
  export OSTMAP
  echo "concurrency,shard,ost_idx,bytes,wall_s,GBps" >"$PERSHARD"
fi

echo "concurrency,total_bytes,wall_s,agg_GBps" | tee "$CSV"
for N in $CONC; do
  export N_CUR="$N"
  start="$(date +%s.%N)"
  printf '%s\n' "${SHARDS[@]}" | xargs -P "$N" -I{} bash -c 'read_one "$@"' _ {} >>"${PERSHARD:-/dev/null}"
  end="$(date +%s.%N)"
  wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
  agg="$(awk -v t="$TOTAL" -v w="$wall" 'BEGIN{printf "%.2f", t/w/1e9}')"
  echo "$N,$TOTAL,$wall,$agg" | tee -a "$CSV"
done
