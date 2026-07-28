#!/bin/bash
# submit.sh <sglang|vllm> <llama70b|apertus8b> <preflight|save|default|preshard> [sbatch args...]
#
# Exists only so the engine, the model preset, the job name and the results
# sub-directory cannot drift apart -- #SBATCH --output is static text and cannot
# read a runtime variable, so something has to pass them together. That
# something is this file rather than a 4-line sbatch invocation repeated by hand.
#
#   ./submit.sh vllm llama70b preflight                       # cheap gate, debug partition
#   ./submit.sh sglang apertus8b save                         # one-off: build the shards
#   ./submit.sh vllm llama70b default                         # note the node id
#   ./submit.sh vllm llama70b preshard --exclude=<node>       # must be a different node
set -euo pipefail

usage='usage: submit.sh <sglang|vllm> <llama70b|apertus8b> <preflight|save|default|preshard> [sbatch args]'
engine="${1:?$usage}"
preset="${2:?$usage}"
what="${3:?$usage}"
shift 3

case "$engine" in
  sglang|vllm) ;;
  *) echo "unknown engine: $engine (want sglang or vllm)" >&2; exit 1 ;;
esac

case "$preset" in
  llama70b)  sub=llama-3.1-70b ;;
  apertus8b) sub=apertus-8b ;;
  *) echo "unknown preset: $preset" >&2; exit 1 ;;
esac

case "$what" in
  preflight) script=preflight.sbatch ;;
  save)      script=save_sharded_ckpt.sbatch ;;
  default)   script=baseline_mmap.sbatch ;;
  preshard)  script=preshard_shm_overlap.sbatch ;;
  *) echo "unknown config: $what (want preflight|save|default|preshard)" >&2; exit 1 ;;
esac

# vllm has no save job on purpose: it reuses the checkpoint SGLang wrote.
if [ ! -f "scripts/${engine}/${script}" ]; then
  echo "no '${what}' script for ${engine} (scripts/${engine}/${script} does not exist)" >&2
  exit 1
fi

mkdir -p "results/${engine}/${sub}"
exec sbatch \
  --export=ALL,MODEL_PRESET="$preset" \
  --job-name="${engine}-${preset}-${what}" \
  --output="results/${engine}/${sub}/%x-%j.out" \
  "$@" "scripts/${engine}/${script}"
