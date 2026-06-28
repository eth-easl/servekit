#!/bin/bash
# M3k+++++++++ -- SCALE BISECT: restore from record-default-c at cs="1 2"
# (2 capture sizes, ~98 graphs served). Decides between:
#  (a) multi-size orchestration fault: even 2 sizes fault (slot mapping bug)
#  (b) scale/state-accumulation fault: 2 sizes work, only 19 sizes fault
# record-default-c has all sizes recorded, so cs=1,2 graphs are present.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-c

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3k-restore-cs12] record-default-c ($NG graphs) + CAPTURE_SIZES='1 2' (~98 served)"

D=$(sbatch --time=00:30:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=1,2,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=1500,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "M3K_RESTORE_CS12=$D"
echo "$D" >> /tmp/m3k_jobs.txt
