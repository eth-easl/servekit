#!/bin/bash
# M3k++++++ -- CONTROL: reproduce M3j Paris ✓ with record-default-pw (410
# PIECEWISE-only) + shim_pw. This is the CORRECT shim_pw config: the restore
# queue (410) exhausts after PIECEWISE captures, so FULL captures fall through
# to REAL (live) capture (valid). Confirms PIECEWISE rebuilt graphs (from a
# proper record_pw recording) replay correctly, isolating the fault to
# record-default-c's FULL rebuilt graphs.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-pw

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3k-restore-pw-ctrl] record dir has $NG PIECEWISE graphs; shim_pw (FULL live fallthrough)"

D=$(sbatch --time=00:40:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim_pw,DEADLINE=2000,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "M3K_RESTORE_PW_CTRL=$D"
echo "$D" >> /tmp/m3k_jobs.txt
