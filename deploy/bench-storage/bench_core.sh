#!/bin/bash
# bench_core.sh — measure single + parallel read bandwidth of a set of files.
#
# Maps each result onto the failure modes in docs/fast-loader.md:
#   * single O_DIRECT stream ............ §1 per-stream / single-TCP / one-OST cap
#   * parallel O_DIRECT scaling ......... §1 (scales linearly) vs §5 (saturates = server wall)
#   * cached warm read .................. local DRAM ceiling (the best case after staging)
#
# Usage: bench_core.sh <label> <file1> [file2 ...]
set -uo pipefail

LABEL="$1"; shift
FILES=("$@")
N=${#FILES[@]}
[ "$N" -eq 0 ] && { echo "no files for $LABEL"; exit 1; }

say(){ printf '\n===== %s =====\n' "$*"; }

F0="${FILES[0]}"
DIR=$(dirname "$F0")
say "TIER: $LABEL  (files=$N  dir=$DIR)"

# --- FS / mount / stripe topology ------------------------------------------
echo "fs_type=$(stat -f -c %T "$F0" 2>/dev/null || echo unknown)"
MP=$(df --output=target "$F0" 2>/dev/null | tail -1 | tr -d ' ')
echo "mountpoint=$MP"
grep -m1 " ${MP} " /proc/mounts 2>/dev/null \
  | awk -F' ' '{print "mount_opts="$4}' \
  | cut -d' ' -f1
if command -v lfs >/dev/null 2>&1; then
  printf 'lustre_stripe(dir): '; lfs getstripe -c "$DIR" 2>/dev/null | sed 's/^/stripe_count=/'
  printf 'lustre_stripe(file0):\n'; lfs getstripe "$F0" 2>/dev/null | grep -E 'stripe_count|stripe_size|stripe_offset' | sed 's/^/  /'
fi
if command -v nfsstat >/dev/null 2>&1; then
  grep -m1 " ${MP} " /proc/mounts 2>/dev/null | grep -qi nfs && echo "(nfs mount)"
fi

total_bytes(){ local t=0 b; for f in "${FILES[@]}"; do b=$(stat -c %s "$f"); t=$((t+b)); done; echo "$t"; }

TB=$(total_bytes)
echo "total_bytes=$((TB/1024/1024))MiB"

# --- timing helper ---------------------------------------------------------
# PER_STREAM_MB: cap bytes read per stream (dd count=), so a sweep completes
# fast even under heavy OST contention (default 512 MB). Whole-file when unset.
measure(){
  local k=$1; local iflags=$2; local tag=$3
  local start end countarg=()
  [ -n "${PER_STREAM_MB:-}" ] && countarg=(count=$((PER_STREAM_MB)))
  start=$(date +%s.%N)
  local pids=()
  local PER=${PER_STREAM_TIMEOUT:-180}   # kill a wedged stream so one bad OST can't hang the sweep
  for ((j=0;j<k;j++)); do
    ( timeout "$PER" dd if="${FILES[$((j % N))]}" of=/dev/null bs=1M $iflags "${countarg[@]}" 2>/dev/null ) &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  end=$(date +%s.%N)
  [ "$rc" -ne 0 ] && { echo "streams=$k  $tag  FAILED (timeout/iflags)"; return 1; }
  python3 - "$start" "$end" "$k" "${PER_STREAM_MB:-}" "${FILES[@]}" <<'PY'
import sys, os
start, end, k = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
cap = sys.argv[4]            # '' or '<MB>'
files = sys.argv[5:]
def fsize(j):
    s = os.path.getsize(files[j % len(files)])
    return min(s, int(cap)*1024*1024) if cap else s
tot = sum(fsize(j) for j in range(k))
dt = end - start
agg = tot/1e9/dt
per = agg/k*1000
print(f"streams={k:3d}  bytes={tot/1e9:6.2f}GB  wall={dt:6.2f}s  agg={agg:6.2f}GB/s  per_stream={per:5.0f}MB/s")
PY
}

# --- cached warm read (page-cache / local DRAM ceiling) --------------------
say "$LABEL — single-stream CACHED (warm, page cache = DRAM ceiling)"
dd if="$F0" of=/dev/null bs=1M 2>/dev/null >/dev/null   # prime
measure 1 "" "cached"

# --- single + parallel O_DIRECT (the real wire/streaming number) -----------
say "$LABEL — O_DIRECT streaming (cold-equivalent; bypasses page cache)"
if ! measure 1 "iflag=direct" "direct"; then
  echo "(O_DIRECT not honored here; falling back to cached reads for scaling)"
  for k in 1 2 4 8 16 32; do
    [ "$k" -gt 64 ] && break
    measure "$k" "" "cached"
  done
  exit 0
fi
say "$LABEL — parallel scaling (O_DIRECT)  [per_stream dropping => §5 server wall]"
for k in ${STREAMS:-2 4 8 16 32}; do
  measure "$k" "iflag=direct" "direct" || break
done
