#!/bin/bash
# M3k+++++ -- ISOLATION TEST: shim_pw (PIECEWISE from record-default-c snapshot
# + FULL live-captured) to prove the FULL rebuilt graphs are the SOLE blocker.
# record-default-c (923 graphs) was recorded FULL_AND_PIECEWISE, so its
# PIECEWISE graphs are rebuildable. In shim_pw, cg_skip dummy's ONLY PIECEWISE
# captures (rebuilt from snapshot) and runs the REAL forward for FULL (live-
# captured, valid). If this gives Paris, the FULL rebuilt-from-snapshot graphs
# are definitively the entire remaining blocker (PIECEWISE rebuild+replay works).
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-c

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3k-restore-shimpw] record dir has $NG graphs; shim_pw (FULL live, PIECEWISE snapshot)"

D=$(sbatch --time=00:40:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim_pw,DEADLINE=2000,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "M3K_RESTORE_SHIMPW=$D"
echo "$D" >> /tmp/m3k_jobs.txt
