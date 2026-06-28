#!/bin/bash
# M3k decisive test: re-record with ONLY cs=1,2,4 so record-order EXACTLY
# matches restore-order (no size-mismatch). Then restore from it at cs=1,2,4.
#  - If Paris  : record/restore size-mismatch WAS the root cause.
#  - If fault  : fundamental (exec-count / state), not ordering.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=$PWD/snapshot/e2e-triton-v4
VCC=$PWD/snapshot/e2e-vllm-cache4
OUT=$PWD/snapshot/record-cs124
rm -rf "$OUT"; mkdir -p "$OUT"

export CAPTURE_SIZES="1 2 4"
D=$(sbatch --time=01:00:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,OUT=$OUT,MAX_GRAPHS=2000,RUN_SECS=2400,NAMER_GRACE=180,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC \
  $RECIPE/vllm_record.sbatch | awk '{print $NF}')
echo "RERECORD_CS124=$D"
echo "$D" >> /tmp/m3k_jobs.txt
