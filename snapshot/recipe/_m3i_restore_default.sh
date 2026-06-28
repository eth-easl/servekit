#!/bin/bash
# M3i restore at DEFAULT capture (run AFTER job B record-default-fb completes).
# Skip-capture (shim) + FIXED_BASE=1 → Δ=0, rebuild from snapshot, no forward.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-fb

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3i-restore-default] record dir has $NG graphs"

D=$(sbatch --time=00:50:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=2400 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "D_RESTORE_DEFAULT_SKIPCAP=$D"
echo "$D" >> /tmp/m3i_jobs.txt
