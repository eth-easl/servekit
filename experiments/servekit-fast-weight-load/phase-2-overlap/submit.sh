#!/bin/bash
# submit.sh <apertus8b|llama70b> <sharded|mmap> [sbatch args...]
#
# Exists so the job name, the model preset and the loader cannot drift apart --
# #SBATCH --output is static text and cannot read a runtime variable.
#
#   ./submit.sh apertus8b sharded              # cheap: iterate on the wrapper
#   ./submit.sh llama70b sharded --exclude=<node used before>
#   ./submit.sh llama70b mmap    --exclude=<...>   # the free-on-ready check
set -euo pipefail

usage='usage: submit.sh <apertus8b|llama70b> <seq|ovl> [sbatch args]'
preset="${1:?$usage}"
arm="${2:?$usage}"
shift 2

case "$preset" in apertus8b|llama70b) ;; *) echo "unknown preset: $preset" >&2; exit 1 ;; esac
case "$arm" in seq|ovl) ;; *) echo "unknown arm: $arm" >&2; exit 1 ;; esac

mkdir -p results
sbatch --job-name="phase2-${preset}-${arm}" \
  --export="ALL,MODEL_PRESET=${preset},ARM=${arm}" \
  "$@" scripts/launch.sbatch
