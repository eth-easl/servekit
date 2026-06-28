#!/bin/bash
# M3i — full cold-start A/B/C measurement at vLLM DEFAULT capture sizes.
#
#   A. BASELINE : vanilla vLLM, live capture (the cost we're reducing)
#   B. RECORD   : capture all snapshots under FIXED_BASE=1 (one-time)
#   C. RESTORE  : skip-capture + restore snapshots under FIXED_BASE=1
#
# The capture phase (forward compute across ~34 sizes) is ~330s of the ~611s
# baseline. RESTORE skips the forward (shim) and rebuilds graphs from snapshot
# at EndCapture, so READY should drop to ~load+compile (~264s).
#
# NOTE on graph-48 (FULL decode graph) fault: the fault is at first
# hipGraphLaunch (post-startup inference), NOT during rebuild/READY. So the
# READY time is measurable cleanly. We separately report inference status.
#
# Usage:  bash _measure_coldstart.sh submit     # submit all 3 jobs
#         bash _measure_coldstart.sh report     # collect + print results
set -uo pipefail
RECIPE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe
RECORD_DIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-fb
TRITON=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VLLMC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4

# Common capture config: TP=1, gmu=0.60, DEFAULT capture sizes (empty = vLLM
# default ~34 sizes -> ~528+ piecewise sub-graphs + ~34 FULL graphs).
COMMON="TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0"

CMD="${1:-submit}"

submit() {
  echo "=== A. BASELINE (vanilla, default capture) ==="
  A=$(sbatch --time=00:40:00 --export=ALL,${COMMON},DEADLINE=2000 \
        "$RECIPE/vllm_coldstart.sbatch" | awk '{print $NF}')
  echo "BASELINE_JOB=$A"

  echo "=== B. RECORD (default capture, FIXED_BASE=1) ==="
  B=$(sbatch --time=01:00:00 \
        --export=ALL,${COMMON},SNAPSHOT_REDIRECT_FIXED_BASE=1,OUT=${RECORD_DIR},MAX_GRAPHS=3000,RUN_SECS=1800,NAMER_GRACE=120 \
        "$RECIPE/vllm_record.sbatch" | awk '{print $NF}')
  echo "RECORD_JOB=$B"

  echo "=== C. RESTORE (default capture, FIXED_BASE=1, skip-capture) ==="
  echo "(depends on RECORD $B — submit after it finishes)"
  echo "Restore command (run after record completes):"
  echo "  sbatch --time=00:40:00 \\"
  echo "    --export=ALL,${COMMON},SNAPSHOT_REDIRECT_FIXED_BASE=1,\\"
  echo "    SNAPSHOT_TRITON_CACHE_DIR=${TRITON},\\"
  echo "    SNAPSHOT_VLLM_CACHE_ROOT=${VLLMC},\\"
  echo "    SNAPSHOT_RESTORE_SNAP_MODULES=1,\\"
  echo "    RESTORE_DIR=${RECORD_DIR},VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=2000 \\"
  echo "    $RECIPE/vllm_restore.sbatch"
  echo
  echo "Jobs: BASELINE=$A RECORD=$B"
  echo "$A $B" > /tmp/measure_jobs.txt
}

report() {
  for J in $(cat /tmp/measure_jobs.txt 2>/dev/null); do
    echo "===== job $J ====="
    sacct -j "$J" --format=JobID,State,Elapsed,NodeList --noheader 2>/dev/null | head -3
    LOG=$(ls /capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/*${J}*.out \
          /capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/*${J}*.log 2>/dev/null | head -1)
    [ -n "$LOG" ] && grep -aE "READY at|SERVER_EXITED|DEADLINE|inference|rebuilt|reloc |fault|Paris" "$LOG" | head -10
  done
}

case "$CMD" in
  submit) submit ;;
  report) report ;;
  *) echo "usage: $0 {submit|report}"; exit 1 ;;
esac
