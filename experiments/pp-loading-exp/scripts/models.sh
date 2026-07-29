# Model presets for the PP round. Sourced by every sbatch under scripts/.
#
# Deliberately parallel to clariden-loading-exp/scripts/shared/models.sh: same
# models, same context length, same mem-fraction, so the only intended
# differences from that round are the parallelism (TP=4 PP=1 -> TP=1 PP=4), the
# loader (sharded_state -> presharded) and the image (v0.5.10 -> a main nightly).
#
# The presharded dump is NOT the sharded_state checkpoint from the last round
# and cannot be reused: different loader, different layout, different
# filenames. PRESHARDED_ROOT is a fresh path on purpose.

case "${MODEL_PRESET:-apertus8b}" in
  apertus8b)
    MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509
    SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
    PRESHARDED_ROOT=/capstor/store/cscs/swissai/infra01/cold-start-experiments/apertus8b-pp4-presharded
    SHM_PRESHARDED=/dev/shm/apertus8b-presharded
    RESULTS_SUBDIR=apertus-8b
    ;;
  llama70b)
    MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
    SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
    PRESHARDED_ROOT=/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-pp4-presharded
    SHM_PRESHARDED=/dev/shm/llama70b-presharded
    RESULTS_SUBDIR=llama-3.1-70b
    ;;
  *)
    echo "unknown MODEL_PRESET: ${MODEL_PRESET} (want apertus8b or llama70b)" >&2
    exit 1
    ;;
esac

# The controlled variables of this round. TP=1 PP=4 puts one quarter of the
# layers on each of the node's 4 GPUs; TP=2 PP=2 exercises both rank fields at
# once, which is the case a mixed deployment would actually hit.
#
#   TP_SIZE_OVERRIDE=2 PP_SIZE_OVERRIDE=2 ./submit.sh apertus8b dump
#
# tp*pp must equal 4 (the node's GPU count). The dump is keyed by the full
# parallelism config, so the two live side by side and never collide.
TP_SIZE="${TP_SIZE_OVERRIDE:-1}"
PP_SIZE="${PP_SIZE_OVERRIDE:-4}"
PAR_TAG="tp${TP_SIZE}pp${PP_SIZE}"
PRESHARDED_ROOT="${PRESHARDED_ROOT}-${PAR_TAG}"
SHM_PRESHARDED="${SHM_PRESHARDED}-${PAR_TAG}"

# FORCED, not tuned, and arrived at by elimination:
#
#   fa3     the engine's own default, but the aarch64 sgl_kernel wheel ships no
#           FA3 in EITHER cu12 or cu13 build of commit 16a52bff
#           (has flash_ops: False). Dies on ImportError after weight loading.
#           Job 2922510.
#   triton  broken at PP>1. triton_backend.py:203 calls
#           get_value_buffer(0) -- a hardcoded layer 0 -- at construction. Only
#           pipeline stage 0 owns layer 0; every other stage maps it to a
#           negative local index and raises IndexError. Job 2922719, PP2/PP3.
#   flashinfer  no hardcoded layer id; every call site uses layer.layer_id,
#           which the KV pool maps to a local index correctly.
#
# Safe for what the round claims: both arms run the same image with the same
# backend, so the default-vs-presharded comparison is untouched. It does mean
# throughput here is not comparable to clariden-loading-exp -- already true
# across engine versions.
ATTENTION_BACKEND=flashinfer

MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256
SERVER_PORT=8080
READY_TIMEOUT=3000
BENCH_REPEATS="${BENCH_REPEATS:-1}"

MODEL="${MODEL_OVERRIDE:-${MODEL}}"
PRESHARDED_ROOT="${PRESHARDED_ROOT_OVERRIDE:-${PRESHARDED_ROOT}}"
SHM_PRESHARDED="${SHM_PRESHARDED_OVERRIDE:-${SHM_PRESHARDED}}"
