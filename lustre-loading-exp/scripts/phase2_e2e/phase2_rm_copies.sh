#!/bin/bash
# Delete the persistent striped copies made by phase2_make_copies.sbatch (~564 GB).
# Only ever removes llama70b_* dirs INSIDE $EXP_BASE - never the source model.
set -euo pipefail
EXP_BASE="${EXP_BASE:-/capstor/store/cscs/swissai/infra01/cold-start-experiments}"
shopt -s nullglob
copies=("${EXP_BASE}"/llama70b_*)
if (( ${#copies[@]} == 0 )); then echo "no copies under ${EXP_BASE}"; exit 0; fi
echo "about to delete:"; du -sh "${copies[@]}"
for d in "${copies[@]}"; do
  [[ "$d" == "${EXP_BASE}"/llama70b_* && -d "$d" ]] || { echo "refusing $d" >&2; exit 1; }
  echo "rm -rf $d"; rm -rf "$d"
done
echo "done. remaining:"; ls -la "${EXP_BASE}"
