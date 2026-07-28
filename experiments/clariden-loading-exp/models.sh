# Model presets. Sourced by every sbatch in this dir -- the single place where
# a model's paths live, so the three scripts never disagree about them.
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

TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256
SGLANG_PORT=8080
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
