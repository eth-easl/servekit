#!/bin/bash
# Submit the M3i cold-start measurement trio. Sourced env vars -> sbatch --export.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=snapshot/recipe
TRI=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4
VCC=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4
RDIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-fb

# ---- A. BASELINE: vanilla vLLM, default capture (19 sizes), warm caches ----
A=$(sbatch --time=00:40:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_TRITON_CACHE_DIR=$TRI,VLLM_CACHE_ROOT=$VCC,DEADLINE=2000,MODE=graph \
  $RECIPE/vllm_coldstart.sbatch | awk '{print $NF}')
echo "A_BASELINE_DEFAULT=$A"

# ---- B. RECORD: default capture under FIXED_BASE=1 (one-time, critical path) ----
B=$(sbatch --time=01:15:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,OUT=$RDIR,MAX_GRAPHS=5000,RUN_SECS=2400,NAMER_GRACE=180 \
  $RECIPE/vllm_record.sbatch | awk '{print $NF}')
echo "B_RECORD_DEFAULT=$B"

# ---- C. RESTORE cs=1 (skip-capture + FIXED_BASE): validate time-saving path ----
C=$(sbatch --time=00:30:00 \
  --export=ALL,TP=1,GMU=0.60,MAX_NUM_SEQS=64,MAX_MODEL_LEN=8192,CAPTURE_SIZES=1,REGION_GIB=72,PYTHONHASHSEED=0,SNAPSHOT_REDIRECT_FIXED_BASE=1,SNAPSHOT_TRITON_CACHE_DIR=$TRI,SNAPSHOT_VLLM_CACHE_ROOT=$VCC,SNAPSHOT_RESTORE_SNAP_MODULES=1,RESTORE_DIR=snapshot/record-fixedbase-all,VLLM_CG_SKIP_CAPTURE=shim,DEADLINE=1800 \
  $RECIPE/vllm_restore.sbatch | awk '{print $NF}')
echo "C_RESTORE_CS1_SKIPCAP=$C"

echo "$A $B $C" > /tmp/m3i_jobs.txt
echo "saved -> /tmp/m3i_jobs.txt"
