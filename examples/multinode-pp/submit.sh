#!/bin/bash
# submit.sh <llama70b|kimi-k3> <prepare|run> [sbatch args...]
#
# #SBATCH --output and --nodes are static text and cannot read a runtime
# variable, so something has to pass the preset, node count and job name in
# together.
#
#   ./submit.sh llama70b prepare     # one-off, offline
#   ./submit.sh llama70b run
set -euo pipefail

usage='usage: submit.sh <llama70b|kimi-k3> <prepare|run> [sbatch args]'
preset="${1:?$usage}"
what="${2:?$usage}"
shift 2

case "$what" in
  prepare|run) script="${what}.sbatch" ;;
  *) echo "unknown job: $what (want prepare|run)" >&2; exit 1 ;;
esac

# Sourced here only to learn how many nodes to ask for.
MODEL_PRESET="$preset" source "$(dirname "$0")/models.sh"

mkdir -p logs
exec sbatch \
  --export=ALL,MODEL_PRESET="$preset" \
  --job-name="pp-${preset}-${what}" \
  --nodes="${NNODES}" \
  --output="logs/%x-%j.out" \
  "$@" "$(dirname "$0")/${script}"
