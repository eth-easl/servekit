# Model presets. Sourced by every sbatch under scripts/ -- the single place
# where a model's paths live, so the scripts never disagree about them.
#
# Shared by BOTH engines: the sglang and vllm runs must differ only in the
# engine, so the model paths, TP size, context length and memory fraction are
# read from here by both and are not per-engine knobs. SHARDED_SRC is likewise
# shared: the vllm preshard run reuses the checkpoint SGLang wrote (both engines
# implement the same `sharded_state` format; scripts/vllm/preflight.sbatch gates
# that the filename pattern still matches).
#
# Select with MODEL_PRESET; `submit.sh` sets it for you. Everything below the
# case block is held CONSTANT across models on purpose: TP size, context length
# and mem-fraction are the controlled variables of the comparison, not knobs.
#
# TP_SIZE=4 matches profile/apertus-8b-bristen/serve_apertus_8b_sglang.sbatch,
# i.e. what the SwissAI platform actually deploys. NOTE the presharded
# checkpoints are TP-size-locked: they load only at --tensor-parallel-size 4.

case "${MODEL_PRESET:-llama70b}" in
  llama70b)
    MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
    SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
    SHARDED_SRC=/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded
    SHM_DEST=/dev/shm/llama70b
    RESULTS_SUBDIR=llama-3.1-70b
    ;;
  apertus8b)
    MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509
    SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
    SHARDED_SRC=/capstor/store/cscs/swissai/infra01/cold-start-experiments/apertus8b-tp4-sharded
    SHM_DEST=/dev/shm/apertus8b
    RESULTS_SUBDIR=apertus-8b
    ;;
  *)
    echo "unknown MODEL_PRESET: ${MODEL_PRESET} (want llama70b or apertus8b)" >&2
    exit 1
    ;;
esac

# The presharded checkpoint is PER ENGINE. vLLM's model carries four attention
# scale params per layer (_k_scale/_v_scale/_q_scale/_prob_scale) that SGLang
# never writes, and vLLM's ShardedStateLoader hard-fails on the missing keys
# (job 2918412, 320 of them for the 80-layer 70B). The shards are portable
# across ARCHITECTURES -- the SGLang arm reused a bristen x86 checkpoint on
# GH200 -- but not across engines. Set by each engine's sbatch, not by the
# caller.
if [ "${ENGINE:-sglang}" = vllm ]; then
  SHARDED_SRC="${SHARDED_SRC}-vllm"
fi

TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256
SERVER_PORT=8080
# Generous: the bristen mmap default reached 1123 s and the clariden capstor
# path is unmeasured. Still well inside the 1 h job limit.
READY_TIMEOUT=3000
# Benchmarks to run back-to-back against the SAME live server. 1 = the normal
# measurement. >1 diagnoses whether a throughput difference is a transient
# post-load effect (later repeats recover) or structural (they do not).
BENCH_REPEATS="${BENCH_REPEATS:-1}"

# Env overrides, so a one-off run needs no edit to this file.
MODEL="${MODEL_OVERRIDE:-${MODEL}}"
SHARDED_SRC="${SHARDED_SRC_OVERRIDE:-${SHARDED_SRC}}"
SHM_DEST="${SHM_DEST_OVERRIDE:-${SHM_DEST}}"
