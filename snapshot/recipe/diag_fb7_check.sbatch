#!/bin/bash
#SBATCH --job-name=snap-fb7-check
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=00:12:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

# M3k — mechanism probe. Query each node kernel's resource footprint
# (hipFuncGetAttribute: regs, static/dynamic LDS, max threads, local mem) and
# print alongside the AddKernelNode result, to localize WHY pointwise@8-warp
# Triton kernels are rejected at add-time. Compare every node: OK (reduction@8,
# pointwise@4) vs FAIL (pointwise@8).
#
# Freshly-recorded ROCm-7 graph (graph-141716-2) fails at node#14
# (triton_poi add_index_select, block=512). Old ROCm-6 graph (graph-66398-2)
# fails at node#4 (triton_poi mul_silu_slice, block=512).

set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
REPO=/capstor/scratch/cscs/xyao/kimi-k25-vllm

echo "fb7-attr-probe start: $(date --iso-8601=seconds) node=$(hostname)"

# Rebuild build7 INSIDE the container so it has the current hip_graph.cpp with
# the hipFuncGetAttribute mechanism probe (the standalone sbatch does not
# rebuild, so a stale build7 would lack the attr lines).
echo "=== in-container rebuild build7 (attr probe) ==="
srun --environment=glm-47-flash-rocm bash -c "
  cd $REPO
  mkdir -p snapshot/build7 && cd snapshot/build7
  cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build . -j${SLURM_CPUS_PER_TASK:-8} --target snapshot 2>&1 | grep -iE 'error:' | grep -v filesystem | head
  echo BUILD7_OK
"

# Dump per-node: name (truncated), block, attr footprint, AddKernelNode rc.
# Truncate the long Cijk/CK names to a tag so the table is readable.
# $1 = snap file, $2 = HSACO patch target (empty/0 = patch OFF, control).
dump() {
  local F=$1
  local PATCH=$2
  srun --environment=glm-47-flash-rocm bash -c "
    export HIP_VISIBLE_DEVICES=0
    export SNAPSHOT_REBUILD_DEBUG=1
    export SNAPSHOT_DEBUG=1
    ${PATCH:+export SNAPSHOT_HSACO_PATCH_MAXWG=$PATCH;}
    $REPO/snapshot/build7/snapshot rebuild-check '$F' 2>&1 \
      | grep -aE 'hsaco-patch|rebuild\] node#[0-9]+ name|rebuild\] node#[0-9]+ attr|rebuild\] node#[0-9]+ hipGraphAddKernelNode rc|backend_error|rebuild-check: [0-9]+ nodes'
  " | awk '
    /rebuild-check: [0-9]+ nodes/ { print; next }
    /name=\x27Cijk/        { sub(/name=\x27Cijk[^\x27]*/, "name=<Cijk-GEMM>") }
    /name=\x27_ZN5aiter/    { sub(/name=\x27[^\x27]*/, "name=<aiter>") }
    /name=\x27_ZN7ck_tile/  { sub(/name=\x27[^\x27]*/, "name=<ck_tile>") }
    /name=\x27_ZN2ck/       { sub(/name=\x27[^\x27]*/, "name=<ck>") }
    { print }
  '
}

echo "===== FRESH ROCm-7 graph-141716-2 ====="
echo "--- patch OFF (control; expect FAIL at node#14) ---"
dump "$REPO/snapshot/record-vllm/graph-141716-2.snap" ""
echo "--- patch ON (maxwg=1024; expect OK) ---"
dump "$REPO/snapshot/record-vllm/graph-141716-2.snap" "1024"
echo
echo "===== OLD ROCm-6 graph-66398-2 ====="
echo "--- patch OFF (control; expect FAIL at node#4) ---"
dump "$REPO/snapshot/record-default-fb/graph-66398-2.snap" ""
echo "--- patch ON (maxwg=1024; expect OK) ---"
dump "$REPO/snapshot/record-default-fb/graph-66398-2.snap" "1024"

echo
echo "fb7-attr-probe finished: $(date --iso-8601=seconds)"
