#!/bin/bash
#SBATCH --job-name=snap-fullfix-diag
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

# M3k diagnostic (phase 3): the DECISIVE trajectory test.
#
# Background (jobs 531792, 531829):
#   - All 766 FULL decode graphs fail rebuild at node#4, the first
#     `triton_poi_fused_mul_silu_slice_0` kernel: hipGraphAddKernelNode rc=-1
#     "unspecified launch failure". func_ok=1, argcnt=7==sigargc=7.
#   - The first 3 CK-GEMM nodes + 1 triton_red node (nodes 0-3) PASS.
#   - The piecewise (pw) graphs ALSO contain triton_poi_fused_mul_silu_slice_0
#     nodes and rebuild/restore CLEANLY (410/410, Paris ok).
#
# Hypothesis under test: the FULL graph's mul_silu_slice node carries a pointer
# arg that falls OUTSIDE the standalone-replay mapped region [region_base,
# region_base+size) -> unmapped -> ROCm's lazy Triton-descriptor build at
# AddKernelNode time derefs it -> hipErrorLaunchFailure. The pw instance of the
# SAME kernel points to an in-range buffer, so it succeeds.
#
# This job rebuilds build7 with the M3k per-arg pointer dump (hip_graph.cpp),
# then runs rebuild-check with SNAPSHOT_REBUILD_DEBUG=1 on:
#   (A) graph-66398-2.snap   (FULL decode, FAILS at mul_silu_slice)
#   (B) graph-94395-100.snap (pw, contains mul_silu_slice, SUCCEEDS)
# and prints each graph's region bounds + every node's arg pointer values, so we
# can compare FAIL-vs-OK pointer ranges directly.
#
# Decision rule after this job:
#   - If (A)'s mul_silu_slice pointers are OUTSIDE its region AND (B)'s are
#     INSIDE  -> trajectory/unmapped CONFIRMED -> implement option (a) replay.
#   - If both in-range -> fault is in the kernel/descriptor, NOT trajectory ->
#     option (a) won't help; pivot to (b) or kernel-descriptor investigation.

set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
REPO=/capstor/scratch/cscs/xyao/kimi-k25-vllm

echo "fullfix-diag3 start: $(date --iso-8601=seconds) node=$(hostname)"

echo "=== in-container build (ROCm 7) with M3k pointer-dump ==="
srun --environment=glm-47-flash-rocm bash -c "
  cd $REPO
  mkdir -p snapshot/build7 && cd snapshot/build7
  cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build . -j\${SLURM_CPUS_PER_TASK:-8} --target snapshot 2>&1 | grep -iE 'error:' | grep -v filesystem | head
  echo BUILD7_OK
"

run_one() {
  local TAG=$1; local DIR=$2; local FNAME=$3
  local F=$REPO/snapshot/$DIR/$FNAME
  echo
  echo "===================== ($TAG) $DIR/$FNAME ($(stat -c%s "$F") B) ====================="
  srun --environment=glm-47-flash-rocm bash -c "
    export HIP_VISIBLE_DEVICES=0
    export SNAPSHOT_REBUILD_DEBUG=1
    export SNAPSHOT_REBUILD_STICKY_DBG=1
    export SNAPSHOT_REBUILD_CLEAR_STICKY=1
    $REPO/snapshot/build7/snapshot rebuild-check '$F' 2>&1 \
      | grep -aE 'rebuild-check: allocator|rebuild-check:   map|rebuild-check: replay|rebuild\] node#|rebuild\] node#.*args:|backend_error'
  "
}

# (A) FULL decode graph that FAILS at the mul_silu_slice node (node#4).
run_one A-fail record-default-fb graph-66398-2.snap
# (B) pw graph that CONTAINS mul_silu_slice and rebuilds OK.
run_one B-ok   record-default-pw graph-94395-100.snap

echo
echo "fullfix-diag3 finished: $(date --iso-8601=seconds)"
