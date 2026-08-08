# Presets for the multi-node pipeline-parallel example. Sourced by both sbatch
# scripts, so prepare and serve cannot drift apart -- and they must not: the dump
# is keyed by the full parallel and quantization config, and a serve job whose
# flags differ from the dump's takes a cache miss and re-dumps the checkpoint.

case "${MODEL_PRESET:-llama70b}" in
  llama70b)
    MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
    SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
    PRESHARDED_ROOT=/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-pp-presharded
    EDF_NAME=sglang-nightly-clariden.toml
    NNODES=2
    TP_SIZE=4
    PP_SIZE=2
    EP_SIZE=1
    # fa3 is the engine's default and the aarch64 sgl_kernel wheel does not ship
    # it; triton hardcodes layer 0 at backend construction and so breaks every
    # pipeline stage but the first. flashinfer is what is left.
    ENGINE_EXTRA=(--attention-backend flashinfer)
    READY_TIMEOUT=3000
    ;;
  kimi-k3)
    MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K3
    SERVED_MODEL_NAME=moonshotai/Kimi-K3
    PRESHARDED_ROOT=/capstor/store/cscs/swissai/infra01/cold-start-experiments/kimi-k3-presharded
    # Not the nightly: KimiK3ForConditionalGeneration is in no released sglang,
    # and the nightly falls through to the transformers path and dies with
    # "Cannot find model module".
    EDF_NAME=sglang-kimi-k3.toml
    NNODES=8
    TP_SIZE=4
    PP_SIZE=8
    EP_SIZE=4
    ENGINE_EXTRA=(
      --moe-runner-backend marlin
      --prefill-attention-backend flashmla
      --decode-attention-backend flashmla
      --watchdog-timeout 3600
      --soft-watchdog-timeout 600
    )
    # A cold source load of 1.5 TB off Lustre ran ~105 min in the cookbook's
    # verified run; the prepare job also writes the dump on top of that.
    READY_TIMEOUT=9000
    ;;
  *)
    echo "unknown MODEL_PRESET: ${MODEL_PRESET} (want llama70b or kimi-k3)" >&2
    exit 1
    ;;
esac

NNODES="${NNODES_OVERRIDE:-${NNODES}}"
TP_SIZE="${TP_SIZE_OVERRIDE:-${TP_SIZE}}"
PP_SIZE="${PP_SIZE_OVERRIDE:-${PP_SIZE}}"
PAR_TAG="tp${TP_SIZE}pp${PP_SIZE}"
PRESHARDED_ROOT="${PRESHARDED_ROOT_OVERRIDE:-${PRESHARDED_ROOT}-${PAR_TAG}}"

DIST_PORT=20000
SERVE_PORT=8080
MAX_MODEL_LEN=32768
MEM_FRACTION_STATIC=0.85
MAX_RUNNING_REQUESTS=256
