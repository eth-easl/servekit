#!/bin/bash
#SBATCH --job-name=snap-rerecord-fb7
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

# M3k — decisive ROCm-7 re-record test (option 2).
#
# The record-default-fb FULL decode graphs fail rebuild at node#4
# (triton_poi_fused_mul_silu_slice_0, num_warps=8 / block=512). Hypothesis: that
# HSACO was compiled by ROCm-6 Triton and rejected by ROCm-7's
# hipGraphAddKernelNode. This job RE-RECORDS the FULL decode graphs from scratch
# inside the CURRENT ROCm-7.2.3 image (forcing Triton to recompile HSACO against
# ROCm 7 via fresh empty caches), then immediately rebuild-checks them.
#
#   * If a freshly-recorded FULL graph rebuilds  -> version skew CONFIRMED.
#       (the fix = re-record under ROCm 7; the >24% win is achievable)
#   * If FULL graphs STILL fail                  -> genuine ROCm-7 num_warps=8
#       bug; option 2 does not help, fall back to the working shim_pw path.
#
# CRITICAL: TRITON_CACHE_DIR / VLLM_CACHE_ROOT must point at FRESH EMPTY dirs,
# else Triton reuses the cached ROCm-6 HSACO and the test is meaningless.

set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
REPO=/capstor/scratch/cscs/xyao/kimi-k25-vllm
RECIPE=$REPO/snapshot/recipe

echo "rerecord-fb7 start: $(date --iso-8601=seconds) node=$(hostname)"

# --- 1. in-container build into build7/ (ROCm 7, .so.7) with current source ---
echo "=== in-container build (ROCm 7) -> build7/ ==="
srun --environment=glm-47-flash-rocm bash -c "
  cd $REPO
  mkdir -p snapshot/build7 && cd snapshot/build7
  cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build . -j\${SLURM_CPUS_PER_TASK:-8} --target snapshot snapshot_record snapshot_redirect 2>&1 \
    | grep -iE 'error:' | grep -v filesystem | head
  echo -n 'build7 linkage: '; readelf -d snapshot | grep -i 'amdhip'
  echo BUILD7_OK
"
# _vllm_record.sh hardcodes snapshot/build/libsnapshot_*.so; symlink the fresh
# .so.7-linked artifacts over the stale .so.6 ones so the recorder loads under
# ROCm 7.
ln -sf "$REPO/snapshot/build7/libsnapshot_record.so"   "$REPO/snapshot/build/libsnapshot_record.so"
ln -sf "$REPO/snapshot/build7/libsnapshot_redirect.so" "$REPO/snapshot/build/libsnapshot_redirect.so"

# --- 2. record FULL decode graphs under ROCm 7 with FRESH caches ---
OUT=$REPO/snapshot/record-default-fb7
rm -rf "$OUT"; mkdir -p "$OUT"
# Fresh empty caches -> Triton/inductor MUST recompile under ROCm 7 (no reuse of
# the ROCm-6 HSACO that lives in glm-47-flash-vllm/cache).
CACHE=$REPO/snapshot/cache7-rocm
rm -rf "$CACHE"; mkdir -p "$CACHE/triton" "$CACHE/vllm"

echo "=== record FULL decode graphs under ROCm 7 -> record-default-fb7 ==="
echo "(CAPTURE_SIZES empty = vLLM default; FIXED_BASE=1; MAX_GRAPHS=15; fresh caches)"
TP=1 GMU=0.60 REGION_GIB=72 \
  CAPTURE_SIZES= MAX_GRAPHS=15 RUN_SECS=1200 NAMER_GRACE=120 \
  SNAPSHOT_REDIRECT_FIXED_BASE=1 \
  SNAPSHOT_TRITON_CACHE_DIR="$CACHE/triton" \
  SNAPSHOT_VLLM_CACHE_ROOT="$CACHE/vllm" \
  PYTHONHASHSEED=0 \
  srun --environment=glm-47-flash-rocm bash "$RECIPE/_vllm_record.sh" || true

# --- 3. rebuild-check every captured FULL (17-node) graph ---
echo
echo "================= rebuild-check of freshly-recorded graphs ================="
NEW=$(ls "$OUT"/graph-*.snap 2>/dev/null | head -15)
if [ -z "$NEW" ]; then
  echo "NO GRAPHS RECORDED — record phase failed (see log above)"
  echo "rerecord-fb7 finished: $(date --iso-8601=seconds)"
  exit 1
fi
echo "recorded $(ls "$OUT"/graph-*.snap | wc -l) graph(s); rebuild-checking each:"
for f in $NEW; do
  NODES=$(grep -aoc "node#" <(LD_LIBRARY_PATH=/opt/rocm/lib "$REPO/snapshot/build7/snapshot" inspect "$f" 2>/dev/null) 2>/dev/null || echo "?")
  echo "----- $(basename "$f") -----"
  srun --environment=glm-47-flash-rocm bash -c "
    export HIP_VISIBLE_DEVICES=0
    export SNAPSHOT_REBUILD_DEBUG=1
    $REPO/snapshot/build7/snapshot rebuild-check '$f' 2>&1 \
      | grep -aE 'rebuild-check: [0-9]+ nodes|rebuild\] node#4 |hipGraphAddKernelNode rc=|backend_error|instantiate=|launch=ok'
  "
done

echo
echo "rerecord-fb7 finished: $(date --iso-8601=seconds)"
