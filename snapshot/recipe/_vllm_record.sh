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

OUT="${OUT:-snapshot/record-vllm}"
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
# FIXED_BASE=1 backs the arena with a pinned VMM region
# (hipMemAddressReserve + one set_access over the whole region) instead of a
# driver-chosen hipMalloc. Pins base 0x600000000000 so record and restore land
# at the SAME base (Δ=0) and every device pointer is valid unmodified.
export SNAPSHOT_REDIRECT_FIXED_BASE="${SNAPSHOT_REDIRECT_FIXED_BASE:-0}"
export SNAPSHOT_REDIRECT_REGION_GIB="${REGION_GIB:-72}"
export SNAPSHOT_REDIRECT_ALLOC_DIR="$OUT"
export SNAPSHOT_REDIRECT_VERBOSE=0
export VLLM_LOGGING_LEVEL=INFO
# vLLM manages BOTH the Triton and inductor caches itself: it derives a
# cache dir from VLLM_CACHE_ROOT ($VLLM_CACHE_ROOT/torch_compile_cache/<hash>/)
# and OVERRIDES os.environ["TRITON_CACHE_DIR"] and ["TORCHINDUCTOR_CACHE_DIR"]
# to subdirs of that. So per-cache env vars are useless here; the ONLY knob
# that matters is VLLM_CACHE_ROOT. We point it at a dedicated frozen dir so no
# other (production) serve job can mutate the compiled artifacts between record
# and restore -> record/live compile bit-identically -> snapshot pointers
# relocate onto the correct live buffers. See _probe_cache2.sh for the override.
export VLLM_CACHE_ROOT="${SNAPSHOT_VLLM_CACHE_ROOT:-${VLLM_CACHE_ROOT:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm/cache/vllm}}"
# ALSO freeze the EAGER Triton kernels (flash-attn _fwd_kernel_stage2, MoE
# _fwd_grouped_kernel_stage1) which JIT-compile during model import BEFORE
# vLLM's inductor override takes effect. vLLM's override covers only the
# inductor-fused kernels; the eager ones still read this env var directly. Both
# record and restore must share the SAME dedicated dir for their HSACOs to
# match byte-for-byte.
export TRITON_CACHE_DIR="${SNAPSHOT_TRITON_CACHE_DIR:-${TRITON_CACHE_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm/cache/triton}}"
# Freeze Python's hash seed. The inductor's fusion passes iterate over Python
# dicts/sets; with the default random per-process seed the iteration ORDER
# differs across cold starts -> different op fusion -> different generated
# Triton source -> different HSACOs for triton_poi_fused_* kernels. Raw
# hand-written Triton kernels are unaffected (fixed source), which is why the
# M3g spike showed 16/16 deterministic but the real pipeline drifts. Setting
# this for BOTH record and restore makes the inductor generate byte-identical
# Triton source -> identical HSACOs -> matching module_hash.
export PYTHONHASHSEED="${SNAPSHOT_PYTHONHASHSEED:-0}"
TP="${TP:-1}"
GMU="${GMU:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# M3b: vLLM's default cuda-graph capture sizes ([1,2,4] + range(8,256,8) ...) is
# ~34 sizes -> ~528 driver-level piecewise sub-graphs. The off-path namer is
# sentinel-gated (drains AFTER /health on inference probes), so it no longer
# needs a tiny capture list to find an idle window -- default capture works.
# Default "1" is the fast dev/CI config; export CAPTURE_SIZES="" (empty) for the
# vLLM default list (the production workload whose ~264s capture the snapshot
# targets). Space-separated for an explicit list. Note: use ${VAR-1} (not :-1)
# so an explicitly-empty value is honored as vLLM default.
CAPTURE_SIZES="${CAPTURE_SIZES-1}"
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

# cg_meta record hook: observe each capture's entry.output and write a JSON
# (offset/shape/dtype per capture index) so lazy-restore can reconstruct
# entry.output without the forward. Defaults to alongside the snapshots.
if [ -n "${VLLM_CG_RECORD_META:-}" ]; then
  case "$VLLM_CG_RECORD_META" in
    1|yes|on|true) VLLM_CG_RECORD_META="$OUT/restore_meta.json" ;;
  esac
  export VLLM_CG_RECORD_META
  export PYTHONPATH="/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe/cginst_skip:${PYTHONPATH:-}"
  echo "[record-vllm] cg_meta RECORD active -> $VLLM_CG_RECORD_META"
fi

# cg_skip hook (record_pw / measure): load the cginst_skip sitecustomize so
# cg_skip.py activates when VLLM_CG_SKIP_CAPTURE is set. record_pw skips FULL
# captures during record, producing a PIECEWISE-only snapshot dir.
if [ -n "${VLLM_CG_SKIP_CAPTURE:-}" ]; then
  export PYTHONPATH="/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe/cginst_skip:${PYTHONPATH:-}"
  echo "[record-vllm] cg_skip active (VLLM_CG_SKIP_CAPTURE=$VLLM_CG_SKIP_CAPTURE)"
fi

t0=$(date +%s)
# Optional --compilation-config override (e.g. disable runtime combo-kernel
# benchmarking for deterministic HSACOs across cold starts).
CC_ARGS=()
if [ -n "${COMPILATION_CONFIG:-}" ]; then
  CC_ARGS=(--compilation-config "$COMPILATION_CONFIG")
fi

vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name rec \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" "${CAPTURE_ARGS[@]}" "${CC_ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

# Wait for vLLM to finish capturing + warming up and reach /health, THEN release
# the off-path namer (touch the sentinel). The namer will not call
# hipKernelNameRef until the sentinel exists, so it cannot race the next capture
# and segfault. Once the requested graphs are FLUSHed (named -> written), or the
# server exits / deadline passes, we stop. With --cudagraph-capture-sizes
# clamped, capture finishes in seconds so /health arrives fast.
REACHED_READY=0
SENT_PROBE=0
LASTN=0
STABLE=0
CONVERGE_S="${CONVERGE_S:-45}"   # seconds of no graph growth = converged
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  # Release the namer the instant vLLM is serving (capture phase is over).
  if [ "$REACHED_READY" -eq 0 ] && kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    REACHED_READY=1
    : > "$SENTINEL"
    echo "[record-vllm] READY at ${elapsed}s — sentinel touched, namer can drain on inference"
  fi
  # Naming happens INLINE on vLLM's inference thread (each hipLaunchKernel hook
  # call resolves a batch of pending funcs). Drive it with a steady stream of
  # completion probes after /health and break once the flush count converges
  # (no new .snap for CONVERGE_S seconds) -- meaning every capturable graph is
  # named + written. This is what makes default capture (~hundreds of graphs)
  # drain to completion; a fixed small probe burst stalls partway.
  if [ "$REACHED_READY" -eq 1 ]; then
    if [ "$SENT_PROBE" -eq 0 ]; then
      SENT_PROBE=1
      echo "[record-vllm] probing to drive inline naming (converge=${CONVERGE_S}s of no growth, max=${SNAPSHOT_RECORD_MAX_GRAPHS})..."
    fi
    curl -sS "http://127.0.0.1:${PORT}/v1/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"rec","prompt":"The capital of France is","max_tokens":1,"temperature":0}' \
      >/dev/null 2>&1 || true
    n=$(ls "$OUT"/graph-*.snap 2>/dev/null | wc -l)
    if [ "$n" -ge "$SNAPSHOT_RECORD_MAX_GRAPHS" ]; then
      echo "[record-vllm] captured $n graph(s) at ${elapsed}s"; break
    fi
    if [ "$n" -eq "$LASTN" ]; then
      STABLE=$((STABLE+1))
      if [ "$STABLE" -ge "$CONVERGE_S" ]; then
        echo "[record-vllm] flush converged at $n graph(s) (stable ${STABLE}s) at ${elapsed}s"; break
      fi
    else
      LASTN=$n; STABLE=0
    fi
  fi
  n=$(ls "$OUT"/graph-*.snap 2>/dev/null | wc -l)
  if [ "$n" -ge "$SNAPSHOT_RECORD_MAX_GRAPHS" ]; then
    :  # already handled above (convergence loop may break first)
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[record-vllm] server exited at ${elapsed}s (captured $n graph(s))"; break
  fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then
    echo "[record-vllm] DEADLINE ${DEADLINE}s (captured $n graph(s))"; break
  fi
  sleep "${PROBE_INTERVAL:-0}"
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
echo "================= M3a.3 IDENTITY GATE (host-side, sample) ================="
# At default capture (~hundreds of graphs) inspecting each would blow the time
# limit; sample the first few and report the aggregate count instead.
NG=$(ls "$OUT"/graph-*.snap 2>/dev/null | wc -l)
echo "recorded $NG graph(s); inspecting first 6:"
for f in $(ls "$OUT"/graph-*.snap 2>/dev/null | head -6); do
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
