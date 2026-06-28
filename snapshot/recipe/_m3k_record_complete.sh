#!/bin/bash
# M3k+++ -- Re-record a COMPLETE snapshot set (all 19 default capture sizes,
# ~923 graphs) to fix blocker #2 (record-default-fb has only 800 — sizes 120 &
# 128 were never recorded, forcing a real-capture fallthrough that crashes with
# hipErrorStreamCaptureUnsupported). Writes to a NEW dir (record-default-c) so
# record-default-fb is preserved.
#
# Budget: record-default-fb took ~66 min for 800 graphs (namer ~5s/graph). For
# ~923 graphs we allow RUN_SECS=5400 (90 min) + --time=02:00:00. CONVERGE_S=300
# (5 min of no growth) tolerates a slow naming tail without premature stop.
# MAX_GRAPHS=3000 > 923 so the run only stops on convergence (all graphs named)
# -- if it stalls near 800 again, that's the namer bottleneck (RESULTS.md #1).
set -uo pipefail
RECIPE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe
RECORD_DIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-c
COMMON="TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0"

NG=$(ls "$RECORD_DIR"/graph-*.snap 2>/dev/null | wc -l)
echo "[m3k-record-complete] OUT=$RECORD_DIR (currently $NG graphs); MAX_GRAPHS=3000 RUN_SECS=5400 CONVERGE_S=300"

D=$(sbatch --time=02:00:00 \
  --export=ALL,${COMMON},SNAPSHOT_REDIRECT_FIXED_BASE=1,OUT=${RECORD_DIR},MAX_GRAPHS=3000,RUN_SECS=5400,NAMER_GRACE=120,CONVERGE_S=300 \
  "$RECIPE/vllm_record.sbatch" | awk '{print $NF}')
echo "M3K_RECORD_COMPLETE=$D"
echo "$D" >> /tmp/m3k_jobs.txt
