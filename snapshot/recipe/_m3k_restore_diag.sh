#!/bin/bash
# M3k++++ -- Diagnose the FULL-decode-graph replay fault (nil-deref at first
# hipGraphLaunch). Runs the functional FULL restore from record-default-c +
# HSACO patch, with rebuild + relocation diagnostics ON:
#   SNAPSHOT_REBUILD_DEBUG=1   -> per-node name/func/args dump (hip_graph.cpp)
#   SNAPSHOT_RESTORE_AUDIT=1   -> unpatched (stale) pointer report
#   AMD_SERIALIZE_KERNEL=3     -> make the fault synchronous (accurate PC)
# Goal: find WHICH kernel node holds a NULL/zero arg (the (nil) fault address),
# and whether any recorded pointer escaped relocation. Output is large
# (923 graphs x N nodes); we grep the FULL-phase tail after the run.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-c

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3k-restore-diag] record dir has $NG graphs; REBUILD_DEBUG + AUDIT + SERIALIZE on"

D=$(sbatch --time=00:40:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=2000,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_REBUILD_DEBUG=1,SNAPSHOT_RESTORE_AUDIT=1,AMD_SERIALIZE_KERNEL=3,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "M3K_RESTORE_DIAG=$D"
echo "$D" >> /tmp/m3k_jobs.txt
