#!/bin/bash
# M3k++ FINAL -- FULL functional restore from the COMPLETE recording
# (record-default-c, 923 graphs = all 19 default capture sizes). With the
# complete recording the snapshot queue no longer exhausts, so there is NO
# real-capture fallthrough (blocker #2 gone) and the HSACO patch makes every
# FULL graph addable (blocker from M3k gone). This is the first end-to-end
# test of functional FULL-graph restore: does it come up AND serve correct
# inference (Paris)? If the earlier nil-fault reappears, the rebuilt FULL
# graphs are broken at replay (a new, distinct blocker).
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-c

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3k-restore-final] record dir has $NG graphs (expect 923)"

D=$(sbatch --time=00:50:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=2400,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "M3K_RESTORE_FINAL=$D"
echo "$D" >> /tmp/m3k_jobs.txt
