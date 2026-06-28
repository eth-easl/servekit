#!/bin/bash
# M3k+ -- FULL-graph restore (shim mode) WITH the HSACO max_flat_workgroup_size
# patch (SNAPSHOT_HSACO_PATCH_MAXWG=1024). This is the previously-BROKEN path:
# the 766 FULL decode graphs in record-default-fb all failed rebuild at
# hipGraphAddKernelNode (block.x > MAX_THREADS_PER_BLOCK). The patch rewrites
# the over-restrictive metadata so graph-add accepts them. Measures cold-start
# vs the live-capture baseline and validates inference correctness (Paris).
#
# Mirrors recipe/_m3i_restore_default.sh; adds SNAPSHOT_HSACO_PATCH_MAXWG.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-fb

NG=$(ls "$RDIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3k-restore-patch] record dir has $NG graphs"

D=$(sbatch --time=00:50:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=$RDIR,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=2400,SNAPSHOT_HSACO_PATCH_MAXWG=1024,SNAPSHOT_RECORD_VERBOSE=1 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "M3K_RESTORE_PATCH=$D"
echo "$D" >> /tmp/m3k_jobs.txt
