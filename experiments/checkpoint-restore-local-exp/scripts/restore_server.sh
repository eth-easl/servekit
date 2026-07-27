#!/usr/bin/env bash
# Restore a checkpointed SGLang server from a CRIU snapshot (+ cuda-checkpoint), and
# verify it serves. This is the "cold-start replacement": instead of launching and
# re-initializing (~59s cold on this box), we thaw a warmed process from disk.
#
# Pairs with checkpoint_server.sh, which wrote the snapshot + ckpt_meta.txt (the GPU
# pids to reattach). Flow:
#   [--cold: evict snapshot from page cache]  ->  criu restore  ->  cuda-checkpoint
#   restore+unlock the GPU pids  ->  servekit bench (waits for the first /generate,
#   then runs baseline.sh's correctness + throughput workload).
#
# A SNAPSHOT IS SINGLE-USE. Restoring it once destroys it: --link-remap hard-links the
# /dev/shm POSIX semaphores that were unlinked at dump time, and the restored process
# deletes them when it exits, so the next restore dies with
#   Can't link dev/shm/link_remap.NNN -> dev/shm/sem.XXXXXX: No such file or directory
# Re-run `checkpoint_server.sh checkpoint` for each restore you want to measure (~22 s).
# Likewise, anything that appends to results/gate3b_server_checkpoint_restore/ckpt_server.log
# (the checkpointed server's fd 1) breaks the image -- criu refuses a file whose size
# changed since the dump.
#
# Usage:
#   scripts/restore_server.sh [--cold] [snapshot_dir]
#     --cold   evict the snapshot from page cache first (honest cold-node timing);
#              omit to measure a warm (cache-hot) restore.
# Default snapshot_dir: results/gate3b_server_checkpoint_restore/gate3b-run/img
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${CKPT_VENV:-$HOME/ckpt-venv}"
PYBIN="$VENV/bin/python"
SERVEKIT="$VENV/bin/servekit"
CRIU="$(command -v criu || echo /usr/sbin/criu)"
CCKPT="$HERE/bin/cuda-checkpoint"
PORT="${PORT:-8080}"
BASELINE_COLD_S="${BASELINE_COLD_S:-59.3}"   # cold-launch reference (baseline.sh --cold)

COLD=0; [ "${1:-}" = "--cold" ] && { COLD=1; shift; }
IMG="${1:-$HERE/results/gate3b_server_checkpoint_restore/gate3b-run/img}"
META="$IMG/ckpt_meta.txt"

[ -d "$IMG" ]   || { echo "no snapshot dir: $IMG (run checkpoint_server.sh first)"; exit 2; }
[ -f "$META" ]  || { echo "no ckpt_meta.txt in $IMG (need the GPU pids from checkpoint time)"; exit 2; }
GPIDS="$(grep -E '^gpu_pids=' "$META" | cut -d= -f2)"
[ -n "$GPIDS" ] || { echo "ckpt_meta.txt has no gpu_pids"; exit 2; }

now_ms(){ date +%s%3N; }
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/criu-xdg}"; mkdir -p "$XDG_RUNTIME_DIR"

echo "==== restore SGLang server from CRIU snapshot ===="
echo "snapshot: $IMG ($(du -sh "$IMG"|cut -f1)) | gpu pids: $GPIDS | mode: $([ $COLD = 1 ] && echo COLD || echo warm)"

if [ "$COLD" = 1 ]; then
  echo "-- evict snapshot from page cache (honest cold restore) --"
  "$PYBIN" "$HERE/scripts/cache_tools.py" evict --verify "$IMG" | sed 's/^/   /'
fi

echo "-- criu restore --"
t0=$(now_ms)
"$CRIU" --unprivileged restore -D "$IMG" --shell-job --tcp-close -d -o restore.log
rrc=$?; t1=$(now_ms)
if [ $rrc -ne 0 ]; then
  echo "RESTORE FAILED (criu rc=$rrc):"; grep -iE "Error" "$IMG/restore.log" | tail -10; exit 1
fi
echo "   criu restore: $((t1-t0)) ms"

echo "-- cuda-checkpoint restore + unlock (GPU reattach) --"
for p in $GPIDS; do "$CCKPT" --action restore --pid "$p" >/dev/null 2>&1 || echo "   [$p] gpu-restore FAILED"; done
for p in $GPIDS; do "$CCKPT" --action unlock  --pid "$p" >/dev/null 2>&1 || echo "   [$p] unlock FAILED"; done
t2=$(now_ms)
echo "   cuda-checkpoint restore: $((t2-t1)) ms | gpu: $($CCKPT --get-state --pid ${GPIDS%% *} 2>&1), $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# `servekit bench` replaces the old hand-rolled "did it answer once?" probe: it polls
# /generate until the restored server serves (recording ready_wait_s), then runs the SAME
# correctness + throughput workload as baseline.sh. That upgrades the gate from "answers
# a request" to "serves at full rate with coherent output" -- directly comparable to the
# baseline report. No launch log exists here, which is exactly why bench is standalone.
echo "-- verify: servekit bench on the restored server --"
BENCH_OUT="$HERE/results/gate3b_server_checkpoint_restore/restore-$(date -u +%Y%m%dT%H%M%SZ)-bench.json"
tb0=$(now_ms)
"$SERVEKIT" bench --url "http://127.0.0.1:$PORT" --out "$BENCH_OUT" \
  --wait-ready 120 --requests 64 --concurrency 16 --input-len 512 --output-len 128
vrc=$?

# restore-to-served must stop at the first served request, NOT at the end of the ~1 min
# load test -- so take it from the bench report's own ready_wait_s rather than the clock.
read -r READY_MS TPUT <<<"$("$PYBIN" -c "
import json
try:
    r = json.load(open('$BENCH_OUT'))
    print(int(r['ready_wait_s'] * 1000), (r.get('throughput') or {}).get('output_tok_per_s', -1))
except Exception:
    print(-1, -1)
" 2>/dev/null)"
if [ "${READY_MS:--1}" -ge 0 ]; then t3=$((tb0 + READY_MS)); else t3=$(now_ms); fi

echo
echo "==== RESULT ===="
printf "  restore-to-GPU-ready : %6d ms  (criu restore + cuda reattach)\n" $((t2-t0))
printf "  restore-to-served    : %6d ms  (incl. first real request)\n" $((t3-t0))
printf "  cold-launch baseline : %6.0f ms  (baseline.sh --cold)\n" "$(echo "$BASELINE_COLD_S*1000"|bc 2>/dev/null || echo $((59300)))"
awk -v r=$((t3-t0)) -v b="$BASELINE_COLD_S" 'BEGIN{printf "  speedup vs cold      : %.1fx\n", (b*1000)/r}'
printf "  post-restore tput    : %6s tok/s (compare against the baseline report)\n" "${TPUT:--1}"
echo "  bench report         : $BENCH_OUT"
[ $vrc -eq 0 ] && echo "  GATE: PASS (restored server serves at full rate, correct output)" || echo "  GATE: FAIL (see bench report)"
echo "==== end ===="
exit $vrc
