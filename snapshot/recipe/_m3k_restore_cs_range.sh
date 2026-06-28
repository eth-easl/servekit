#!/bin/bash
# M3k++++++++++ -- SCALE BISECT (2): cs=1..64 (11 sizes, ~550 graphs).
# cs=1 (49 graphs) and cs=1,2 (~98) both gave Paris. Default 19 sizes faults.
# This test: do the first 11 sizes (1,2,4,8,16,24,32,40,48,56,64) work?
#  -> YES: break is in sizes 72-128 (the large decode sizes).
#  -> NO:  break is within 1-64 (smaller than expected).
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-c

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
CS="1,2,4,8,16,24,32,40,48,56,64"
echo "[m3k-restore-cs1-64] record-default-c ($NG graphs) + CAPTURE_SIZES='$CS' (~550 served)"

D=$(sbatch --time=00:35:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=$CS,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=1700,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "M3K_RESTORE_CS1_64=$D"
echo "$D" >> /tmp/m3k_jobs.txt
