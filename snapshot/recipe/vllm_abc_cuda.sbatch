#!/bin/bash
#SBATCH --job-name=vllm-abc-cuda-n5b
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=02:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/%x-%j.out

# N5b Task 8 (G5): A/B/C cold-start-to-ready + focused serving-overhead.
#   A = baseline graph mode (no interposer)
#   B = restore (interposer, skip capture, replay recorded graphs)
#   C = eager (--enforce-eager)
# One vllm serve per srun step (enroot holds CUDA IPC shmem). A warm-up serve
# precedes the measurement (warm weight/compile cache → isolates the capture
# phase). A record serve populates snap-n5b/ + meta-n5b/ before B.
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
SNAP_CUDA_DIR="${SNAP_CUDA_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR SNAP_CUDA_DIR
export EDF_PATH="${DEPLOY_DIR}/deploy/glm-47-flash-bristen-vllm:${SNAP_CUDA_DIR}/snapshot/recipe:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/measure-n5b"
export PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"; mkdir -p "${PIP_CACHE_DIR}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
TP_N="${TP:-4}"

echo "=== Step 1: build redirect + record .so (snapshot-cuda devel env) ==="
srun --environment=snapshot-cuda bash -lc '
  set -euo pipefail
  cd "${SNAP_CUDA_DIR}"
  CMAKE_PREFIX="${SNAP_CUDA_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
  if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi
  cmake -S snapshot -B snapshot/build-cuda -DSNAPSHOT_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80
  cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
    --target snapshot_record_cuda snapshot_redirect_cuda
  ls -l snapshot/build-cuda/libsnapshot_record_cuda.so \
        snapshot/build-cuda/libsnapshot_redirect_cuda.so
'

run_glm() {
  # $1 = bash snippet to run inside the glm env (pip + the snippet).
  srun --environment=glm-47-flash-vllm-cuda bash -lc "
    set -uo pipefail
    cd \"\${DEPLOY_DIR}\"
    pip install --quiet \"transformers=='${TRANSFORMERS_VERSION}'\"
    $1
  "
}

echo "=== Step 2: warm-up serve (populate triton/vllm cache; not measured) ==="
run_glm 'MODE=graph DEADLINE=1800 bash snapshot/recipe/_vllm_coldstart_cuda.sh || true'

echo "=== Step 3: A = baseline graph mode + serving probe ==="
run_glm 'PROFILE=baseline SERVE_PROBE_OUT="${DEPLOY_DIR}/measure-n5b/baseline.probe" \
           bash snapshot/recipe/_vllm_abc_cuda.sh'

echo "=== Step 4: record serve (populate snap-n5b/ + meta-n5b/ for B) ==="
run_glm 'MODE=record bash snapshot/recipe/_vllm_record_cuda.sh'

echo "=== Step 5: B = restore + serving probe ==="
run_glm 'PROFILE=restore SERVE_PROBE_OUT="${DEPLOY_DIR}/measure-n5b/restore.probe" \
           bash snapshot/recipe/_vllm_abc_cuda.sh'

echo "=== Step 6: C = eager ==="
run_glm 'PROFILE=eager bash snapshot/recipe/_vllm_abc_cuda.sh'

# ---------------------------------------------------------------------------
# Aggregate A/B/C + serving overhead.
# ---------------------------------------------------------------------------
echo "=== A/B/C aggregate ==="
A=$(cat "${DEPLOY_DIR}/measure-n5b/baseline.ready" 2>/dev/null || echo NONE)
B=$(cat "${DEPLOY_DIR}/measure-n5b/restore.ready"  2>/dev/null || echo NONE)
C=$(cat "${DEPLOY_DIR}/measure-n5b/eager.ready"    2>/dev/null || echo NONE)
echo "A_baseline_graph=${A}s   B_restore=${B}s   C_eager=${C}s"
ELIM="NA"
if [[ "$A" =~ ^[0-9]+$ ]] && [[ "$B" =~ ^[0-9]+$ ]] && [ "$A" -gt "$B" ]; then
  ELIM=$((A - B))
fi
# Serving overhead: restored throughput vs baseline throughput (identical graphs
# → expect ~equal). Honest accounting: report both, do not over-claim.
B_RPS=$(grep -oE "throughput_rps=[0-9.]+" "${DEPLOY_DIR}/measure-n5b/baseline.probe" 2>/dev/null | head -1 || echo "throughput_rps=NA")
R_RPS=$(grep -oE "throughput_rps=[0-9.]+" "${DEPLOY_DIR}/measure-n5b/restore.probe"  2>/dev/null | head -1 || echo "throughput_rps=NA")
echo "serving baseline ${B_RPS}   restored ${R_RPS}"
OVERHEAD_PCT="NA"
BV=$(echo "$B_RPS" | sed 's/throughput_rps=//')
RV=$(echo "$R_RPS" | sed 's/throughput_rps=//')
if [[ "$BV" =~ ^[0-9.]+$ ]] && [[ "$RV" =~ ^[0-9.]+$ ]] && [ "$(echo "$BV > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
  OVERHEAD_PCT=$(awk "BEGIN{printf \"%.1f\", (1.0 - ${RV}/${BV}) * 100.0}")
fi
echo "N5B_ABC: A=${A} B=${B} C=${C} elim=${ELIM}s overhead_pct=${OVERHEAD_PCT}"

echo "vllm-abc-cuda-n5b finished: $(date --iso-8601=seconds)"
