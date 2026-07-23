#!/bin/bash
# Apertus-8B cold start, DP=4 / TP=1. One script, three modes:
#
# NOTE: no --load-format fastsafetensors. It is incompatible with DP in v0.5.10:
# weight_utils.fastsafetensors_weights_iterator picks its device as
# cuda:{pg.rank()}, but each DP rank is its own process with tp_size=1, so
# pg.rank()==0 everywhere and all 4 ranks stage weights onto physical cuda:0 ->
# OOM. It works under TP only because there tp_rank happens to equal gpu_id.
#
#   sbatch --export=ALL,MODE=baseline foundry-tp1/apertus8b_dp4.sbatch   # no foundry
#   sbatch --export=ALL,MODE=save     foundry-tp1/apertus8b_dp4.sbatch   # capture + persist graphs
#   sbatch --export=ALL,MODE=load --exclude=<save node> foundry-tp1/apertus8b_dp4.sbatch
#
# TP=1/DP=4 because foundry does not support tensor parallel yet. Every mode runs
# the SAME image and the SAME engine flags; the only difference is whether
# --foundry-graph-extension-config-path is passed. LOAD must land on a different
# node than SAVE, or the archive and weights are warm in page cache and the
# cold-start number is optimistic.
#
#SBATCH --job-name=foundry-apertus8b-dp4
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=foundry-tp1/results/%x-%j.out

set -euo pipefail

MODE="${MODE:?set MODE=baseline|save|load  (sbatch --export=ALL,MODE=save ...)}"
case "${MODE}" in
  baseline|baseline_nopw|save|load) ;;
  *) echo "bad MODE='${MODE}' (want baseline|baseline_nopw|save|load)" >&2; exit 2 ;;
esac

FOUNDRY_DIR="${SLURM_SUBMIT_DIR}/foundry-tp1"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
EDF="${FOUNDRY_DIR}/apertus-8b-foundry.toml"
ARCHIVE="${FOUNDRY_DIR}/results/foundry_archive_apertus8b_dp4"

MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509
SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
SGLANG_PORT=8080
DP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${FOUNDRY_DIR}/results/apertus8b-dp4-${MODE}-${SLURM_JOB_ID}-${NODE}-profile.json"

# Per-mode deltas. Baseline gets neither the flag nor the NCCL overrides, so it
# is exactly the stock engine.
FOUNDRY_ARGS=""
EXTRA_ARGS=""
FOUNDRY_ENV=":"
case "${MODE}" in
  save|load)
    FOUNDRY_ARGS="--foundry-graph-extension-config-path ${FOUNDRY_DIR}/foundry_${MODE}.toml"
    # NVLS multicast / CUMEM P2P call cuMemMap with driver-capability flags that
    # foundry's VMM region does not carry -- pin NCCL to IPC/ring.
    FOUNDRY_ENV="export NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0"
    ;;
  baseline_nopw)
    # Control run. foundry's shim force-sets disable_piecewise_cuda_graph=True,
    # so the baseline's 39.95 s piecewise phase vanishes on SAVE/LOAD -- switched
    # off, not accelerated. This run turns it off WITHOUT foundry, so the
    # baseline->load delta can be split into "foundry restored graphs" vs
    # "piecewise was simply disabled". The flag is deliberately confined to this
    # labelled control; the main baseline keeps the script's original flags.
    EXTRA_ARGS="--disable-piecewise-cuda-graph"
    ;;
esac

if [ "${MODE}" = "save" ]; then
  echo "clearing archive: ${ARCHIVE}"
  rm -rf "${ARCHIVE}"
elif [ "${MODE}" = "load" ]; then
  [ -f "${ARCHIVE}/warmup_state.json" ] || { echo "no archive at ${ARCHIVE} - run MODE=save first" >&2; exit 3; }
fi

mkdir -p "${FOUNDRY_DIR}/results"

echo "job=${SLURM_JOB_ID} node=${NODE} mode=${MODE} model=${MODEL}"
echo "dp=${DP_SIZE} tp=1 mem_fraction_static=${MEM_FRACTION_STATIC} max_model_len=${MAX_MODEL_LEN}"
echo "archive=${ARCHIVE}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"
echo "NOTE: for MODE=load, submit with --exclude=${NODE} if this is the save run"

# servekit is not in the image and is under active development, so it is
# installed per run -- before the profiled launch, outside the measured window.
# Without --keep-alive, servekit tears the server down once the bench finishes,
# which is exactly the "wait for startup, then SIGTERM" the SAVE recipe wants.
INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
# Patch the image's sglang in the container's ephemeral overlay. Idempotent and
# sub-second, and it runs here -- before the profiled launch -- so it is outside
# the measured cold-start window. Kept out of the image so iterating on the patch
# does not cost a rebuild + 28 GB squashfs re-export.
python "${FOUNDRY_DIR}/apply_foundry_patch.py"
${FOUNDRY_ENV}
python -c "import foundry; print('foundry:', foundry.__file__)"
exec servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tp-size 1 \
    --dp-size "${DP_SIZE}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --enable-metrics \
    ${EXTRA_ARGS} ${FOUNDRY_ARGS}
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${MODE}) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"

if [ "${MODE}" = "save" ]; then
  echo "=== archive ==="
  du -sh "${ARCHIVE}" 2>/dev/null || true
  ls "${ARCHIVE}" 2>/dev/null || true
  echo "graphs per rank: $(ls "${ARCHIVE}"/rank_0/*.cugraph 2>/dev/null | wc -l)"
  echo "NEXT: sbatch --export=ALL,MODE=load --exclude=${NODE} foundry-tp1/apertus8b_dp4.sbatch"
fi
echo "finished=$(date --iso-8601=seconds)"
