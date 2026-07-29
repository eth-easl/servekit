#!/bin/bash
# submit.sh <apertus8b|llama70b> <preflight|dump|default|presharded> [sbatch args...]
#
# Same reason as clariden-loading-exp/submit.sh: #SBATCH --output is static text
# and cannot read a runtime variable, so something has to pass the preset, job
# name and results sub-directory together.
#
#   ./submit.sh apertus8b preflight                       # the gate
#   ./submit.sh apertus8b dump                            # one-off, offline
#   ./submit.sh apertus8b default                         # note the node id
#   ./submit.sh apertus8b presharded --exclude=<node>     # must be a different node
set -euo pipefail

usage='usage: submit.sh <apertus8b|llama70b> <preflight|dump|default|presharded> [sbatch args]'
preset="${1:?$usage}"
what="${2:?$usage}"
shift 2

case "$preset" in
  apertus8b) sub=apertus-8b ;;
  llama70b)  sub=llama-3.1-70b ;;
  *) echo "unknown preset: $preset" >&2; exit 1 ;;
esac

case "$what" in
  preflight)  script=preflight.sbatch ;;
  dump)       script=dump_presharded.sbatch ;;
  default)    script=baseline_default.sbatch ;;
  presharded) script=presharded_shm_overlap.sbatch ;;
  *) echo "unknown config: $what (want preflight|dump|default|presharded)" >&2; exit 1 ;;
esac

mkdir -p "results/${sub}"
exec sbatch \
  --export=ALL,MODEL_PRESET="$preset" \
  --job-name="pp-${preset}-${what}" \
  --output="results/${sub}/%x-%j.out" \
  "$@" "scripts/${script}"
