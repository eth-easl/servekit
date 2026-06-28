#!/bin/bash
# M3k++ -- Diagnose blocker #2 (queue exhaustion). Same as _m3k_restore_patch.sh
# but with SNAPSHOT_RESTORE_EMPTY_EXHAUSTED=1: when the snapshot queue exhausts,
# return an EMPTY graph (NO real hipStreamBeginCapture). This:
#   (a) avoids the hipErrorStreamCaptureUnsupported crash (no real capture), and
#   (b) logs "EndCapture -> EMPTY graph (exhausted at N)" for every deficit
#       graph -> gives the EXACT demand (how many graphs vLLM wants vs the 800
#       recorded). Confirms whether completing the recording fully fixes it.
# Inference on the deficit sizes (120, 128) is WRONG (empty graphs) — this run
# is for demand-counting + READY-time confirmation, NOT correctness.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/k25-vllm/snapshot/e2e-vllm-cache4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-fb

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3k-restore-empty] record dir has $NG graphs; EMPTY_EXHAUSTED=1 (demand probe)"

D=$(sbatch --time=00:25:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=1500,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_RESTORE_EMPTY_EXHAUSTED=1,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "M3K_RESTORE_EMPTY=$D"
echo "$D" >> /tmp/m3k_jobs.txt
