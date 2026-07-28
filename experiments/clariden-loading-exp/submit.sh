#!/bin/bash
# submit.sh <llama70b|apertus8b> <save|default|preshard> [extra sbatch args...]
#
# Exists only so the model preset, the job name and the results sub-directory
# cannot drift apart -- #SBATCH --output is static text and cannot read a
# runtime variable, so something has to pass them together. That something is
# this file rather than a 3-line sbatch invocation repeated by hand.
#
#   ./submit.sh apertus8b save                        # one-off: build the shards
#   ./submit.sh apertus8b default                     # note the node id
#   ./submit.sh apertus8b preshard --exclude=<node>   # must be a different node
set -euo pipefail

preset="${1:?usage: submit.sh <llama70b|apertus8b> <save|default|preshard> [sbatch args]}"
what="${2:?usage: submit.sh <llama70b|apertus8b> <save|default|preshard> [sbatch args]}"
shift 2

case "$preset" in
  llama70b)  sub=llama-3.1-70b ;;
  apertus8b) sub=apertus-8b ;;
  *) echo "unknown preset: $preset" >&2; exit 1 ;;
esac

case "$what" in
  save)     script=save_sharded_ckpt.sbatch ;;
  default)  script=baseline_mmap.sbatch ;;
  preshard) script=preshard_shm_overlap.sbatch ;;
  *) echo "unknown config: $what (want save|default|preshard)" >&2; exit 1 ;;
esac

mkdir -p "results/${sub}"
exec sbatch \
  --export=ALL,MODEL_PRESET="$preset" \
  --job-name="${preset}-${what}" \
  --output="results/${sub}/%x-%j.out" \
  "$@" "$script"
