#!/bin/bash
# M3k scale-bisect runner. Usage: _m3k_restore_bisect.sh "1 2 4 8"
# FIX: CAPTURE_SIZES (space-separated) is EXPORTED in the shell and inherited
# via --export=ALL, NOT passed in the comma --export list (which mangles spaces
# via word-splitting, or commas via sbatch's own splitting).
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-c

CS="${1:?usage: $0 \"1 2 4\" [RDIR]}"
RDIR="${2:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-c}"
export CAPTURE_SIZES="$CS"

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[bisect] RDIR=$RDIR ($NG graphs) + CAPTURE_SIZES='$CS'"

# sbatch --export=ALL inherits CAPTURE_SIZES from the env (spaces preserved).
D=$(sbatch --time=00:35:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=1700,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "BISECT_CS(${CS// /_})=$D"
echo "$D" >> /tmp/m3k_jobs.txt
