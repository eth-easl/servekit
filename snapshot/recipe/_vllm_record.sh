#!/bin/bash
# M3b — record a REAL vLLM captured graph end-to-end via the interposer, then
# validate (a) how many nodes resolve to a known identity and (b) whether one
# graph rebuilds + instantiates without HIP error.
#
# Evolved from the M3a.3/M3a.4 scaffold. The M3b additions that made it actually
# reach a FLUSH on real vLLM (the earlier runs never idled / crashed the namer):
#   * --cudagraph-capture-sizes clamped (default 1) so vLLM reaches /health fast;
#   * name resolution is sentinel-gated (IDLE mode): the recorder does not call
#     hipKernelNameRef until this script touches the sentinel at /health, so it
#     never races a capture;
#   * naming runs INLINE on the inference thread (a post-/health completion
#     probe drives hipLaunchKernel), with host-launched nodes named from the
#     __hipRegisterFunction map and device-handle nodes via a capped
#     hipKernelNameRef (see SNAPSHOT_RECORD_NAMEREF_LIMIT).
#
# Load order LD_PRELOAD="libsnapshot_redirect.so libsnapshot_record.so":
#   * redirect (loaded first) serves hipMalloc from the deterministic arena
#     (M2.4) so vLLM's addresses are reproducible;
#   * record (loaded second) observes module-load / get-function / capture /
#     launch identity — redirect does not interpose those symbols, so they
#     resolve to the recorder unmodified.
#
# Region/arena base is NOT passed to the recorder: this run does not restore
# (no bit-identical oracle for a real vLLM graph yet — see RESULTS.md M3b), so
# relocation is not exercised here.
set -uo pipefail

cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
REDIRECT="$PWD/snapshot/build/libsnapshot_redirect.so"
RECORDER="$PWD/snapshot/build/libsnapshot_record.so"
SNAP=snapshot/build/snapshot
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash

OUT=snapshot/record-vllm
rm -rf "$OUT"; mkdir -p "$OUT"

export LD_PRELOAD="$REDIRECT $RECORDER"
export SNAPSHOT_RECORD_OUT_DIR="$OUT"
export SNAPSHOT_RECORD_MAX_GRAPHS="${MAX_GRAPHS:-4}"
export SNAPSHOT_RECORD_VERBOSE=1
# M3b: hipKernelNameRef segfaults if a stream capture starts while it runs, so
# the off-path namer must NOT drain during init's back-to-back captures. IDLE
# mode makes it wait for a sentinel file (touched below once /health is ready)
# before resolving any names. AUTO mode (no sentinel) is for the synthetic test.
export SNAPSHOT_RECORD_DRAIN_MODE="${DRAIN_MODE:-idle}"
# Launch-time kernarg capture (off by default — must be isolated first).
export SNAPSHOT_RECORD_CAPTURE_ARGS="${CAPTURE_KERNARG:-0}"
SENTINEL="$OUT/drain.now"
rm -f "$SENTINEL"
export SNAPSHOT_RECORD_DRAIN_SENTINEL="$SENTINEL"
export SNAPSHOT_REDIRECT_ARENA=1
export SNAPSHOT_REDIRECT_REGION_GIB="${REGION_GIB:-72}"
export SNAPSHOT_REDIRECT_ALLOC_DIR="$OUT"
export SNAPSHOT_REDIRECT_VERBOSE=0
export VLLM_LOGGING_LEVEL=INFO
TP="${TP:-1}"
GMU="${GMU:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# M3b: vLLM's default cuda-graph capture sizes ([1,2,4] + range(8,256,8) ...) is
# ~34 sizes -> ~528 driver-level piecewise sub-graphs, so the capture phase runs
# minutes and vLLM never reaches the idle window the off-path namer needs to
# drain hipKernelNameRef. Restrict capture to a tiny explicit list (default a
# single batch size) so capture finishes in seconds and the model goes idle
# fast. CAPTURE_SIZES="" -> vLLM default (the M2.x behavior). Space-separated.
CAPTURE_SIZES="${CAPTURE_SIZES:-1}"
PORT="${PORT:-8811}"
DEADLINE="${RUN_SECS:-600}"
LOG="$OUT/vllm.log"

# Build the optional --cudagraph-capture-sizes tail (only when overridden).
CAPTURE_ARGS=()
if [ -n "$CAPTURE_SIZES" ]; then
  # shellcheck disable=SC2206
  CAPTURE_ARGS=(--cudagraph-capture-sizes $CAPTURE_SIZES)
fi

echo "[record-vllm] TP=$TP gmu=$GMU region=${SNAPSHOT_REDIRECT_REGION_GIB}GiB max_graphs=$SNAPSHOT_RECORD_MAX_GRAPHS capture_sizes='${CAPTURE_SIZES:-<vllm-default>}' start=$(date +%T)"

t0=$(date +%s)
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name rec \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" "${CAPTURE_ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

# Wait for vLLM to finish capturing + warming up and reach /health, THEN release
# the off-path namer (touch the sentinel). The namer will not call
# hipKernelNameRef until the sentinel exists, so it cannot race the next capture
# and segfault. Once the requested graphs are FLUSHed (named -> written), or the
# server exits / deadline passes, we stop. With --cudagraph-capture-sizes
# clamped, capture finishes in seconds so /health arrives fast.
REACHED_READY=0
SENT_PROBE=0
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  # Release the namer the instant vLLM is serving (capture phase is over).
  if [ "$REACHED_READY" -eq 0 ] && kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    REACHED_READY=1
    : > "$SENTINEL"
    echo "[record-vllm] READY at ${elapsed}s — sentinel touched, namer can drain on inference"
  fi
  # Naming now happens INLINE on vLLM's inference thread (hipLaunchKernel), so
  # send one completion to drive it. The first launch after the sentinel drains
  # the whole queue + FLUSHes. (Not before /health: that would race capture.)
  if [ "$REACHED_READY" -eq 1 ] && [ "$SENT_PROBE" -eq 0 ]; then
    SENT_PROBE=1
    echo "[record-vllm] sending completion probe to trigger main-thread naming..."
    curl -sS "http://127.0.0.1:${PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"rec","prompt":"The capital of France is","max_tokens":4,"temperature":0}' \
      >/dev/null 2>&1 || echo "[record-vllm] probe request failed (non-fatal)"
  fi
  n=$(ls "$OUT"/graph-*.snap 2>/dev/null | wc -l)
  if [ "$n" -ge "$SNAPSHOT_RECORD_MAX_GRAPHS" ]; then
    echo "[record-vllm] captured $n graph(s) at ${elapsed}s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[record-vllm] server exited at ${elapsed}s (captured $n graph(s))"; break
  fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then
    echo "[record-vllm] DEADLINE ${DEADLINE}s (captured $n graph(s))"; break
  fi
  sleep 3
done

echo "[record-vllm] grace ${NAMER_GRACE:-90}s for off-path name resolution"
sleep "${NAMER_GRACE:-90}"

# Kill aggressively: once we have the captures we want, there is no reason to
# wait for vLLM's graceful shutdown (which can hang mid-startup). SIGKILL the
# serve parent, then reap every EngineCore / multiprocessing child by pattern.
# (The recorded .snap files are already written at hipStreamEndCapture, so a
# hard kill cannot lose data.)
kill -9 "$SERVER_PID" 2>/dev/null
sleep 1
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'from multiprocessing' 2>/dev/null
pkill -9 -f 'VLLM_RPC' 2>/dev/null
# Do NOT `wait` on the parent — a mid-startup SIGTERM can block in D state.
sleep 5

# Self-diagnosis: dump the tail of vLLM's log + any traceback into the SLURM
# output unless we got every requested graph. Covers both failure modes seen:
# (a) vLLM crashed before /health, and (b) it reached /health but died during
# name-draining (the hipKernelNameRef thread-safety crash). No-op on success.
N_FINAL=$(ls "$OUT"/graph-*.snap 2>/dev/null | wc -l)
if [ "${REACHED_READY:-0}" != 1 ] || [ "${N_FINAL:-0}" -lt "${SNAPSHOT_RECORD_MAX_GRAPHS:-1}" ]; then
  echo
  echo "================= DIAGNOSTICS (ready=${REACHED_READY:-0} graphs=${N_FINAL:-0}/${SNAPSHOT_RECORD_MAX_GRAPHS:-1}) ================="
  echo "----- vllm.log tail (last 60 lines) -----"
  tail -60 "$LOG" 2>/dev/null
  echo "----- vllm.log errors / tracebacks -----"
  grep -anE "Traceback|Error|Exception|FAILED|failed|Aborted|abort|core dump|HIP error|RuntimeError|ValueError|AssertionError|out of memory|OOM|Segfault|killed" "$LOG" 2>/dev/null | tail -40
  echo "----- capture / namer markers -----"
  grep -aE "FIRST |NAMER|introspect|FLUSH|sentinel|READY at" "$LOG" 2>/dev/null | tail -25
  echo "----------------------------------------------------------------------------------------------------"
fi

echo
echo "=== [redirect] summary (arena served the engine) ==="
grep -E "\[redirect\] (pid=.* ARENA|SUMMARY)" "$LOG" | tail -8
echo "=== [record] summary (graphs the interposer serialized) ==="
grep -E "\[record\]" "$LOG" | tail -20
echo "=== recorded snapshots ==="
ls -lh "$OUT"/graph-*.snap 2>/dev/null || { echo "(no graphs recorded)"; exit 0; }

echo
echo "================= M3a.3 IDENTITY GATE (host-side, per graph) ================="
for f in "$OUT"/graph-*.snap; do
  echo "----- $(basename "$f") -----"
  "$SNAP" inspect "$f" || echo "  (inspect failed for $f)"
done

echo
echo "================= M3a.4 REBUILD GATE (one graph, GPU) ================="
FIRST=$(ls "$OUT"/graph-*.snap 2>/dev/null | head -1)
if [ -n "$FIRST" ]; then
  SNAPSHOT_REBUILD_DEBUG="${REBUILD_DEBUG:-1}" "$SNAP" rebuild-check "$FIRST" || echo "  (rebuild-check failed for $FIRST — see message above)"
fi

echo
echo "[record-vllm] DONE $(date +%T)"
