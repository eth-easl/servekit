#!/bin/bash
# Phase 2 (e2e) step 1 - create PERSISTENT striped copies of the model, one per
# layout. Storage-only, no GPU/container. Copies are KEPT for the e2e jobs that
# follow; delete them with phase2_rm_copies.sh when done (~564 GB total).
#
# Why a separate job from the e2e runs (this is the point):
#   SGLang reads weights BUFFERED, so the OS page cache matters. The node that
#   writes a 141 GB copy has that copy hot in its page cache. If the same node
#   then served the model, weight_loading would be measured against RAM, not
#   Lustre. Copying here and serving on a DIFFERENT fresh node keeps every
#   e2e read genuinely cold.
#
# Submit from the repo root:
#   sbatch lustre-loading-exp/scripts/phase2_e2e/phase2_make_copies.sbatch
#
#SBATCH --job-name=phase2-make-copies
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=lustre-loading-exp/results/phase2_e2e/%x-%j.out

set -euo pipefail

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
EXP_BASE="${EXP_BASE:-/capstor/store/cscs/swissai/infra01/cold-start-experiments}"
LAYOUTS="${LAYOUTS:-4:4M 8:4M 16:4M 8:16M}"

EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
NODE="$(hostname)"
mkdir -p "$EXP_BASE"

# Each copy is ~141 GB; 4 layouts = ~564 GB.
avail_pct="$(df --output=pcent "$EXP_BASE" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ -n "$avail_pct" && "$avail_pct" -ge 92 ]]; then
  echo "ABORT: $EXP_BASE is ${avail_pct}% full; refusing to write ~564 GB of copies" >&2
  exit 1
fi

echo "job=${SLURM_JOB_ID} node=${NODE} model=${MODEL}"
echo "exp_base=${EXP_BASE} layouts='${LAYOUTS}'"
echo "started=$(date --iso-8601=seconds)"
lfs df -h "$EXP_BASE" 2>/dev/null | tail -1 || true

for pair in $LAYOUTS; do
  C="${pair%%:*}"; S="${pair##*:}"
  DEST="${EXP_BASE}/llama70b_c${C}_s${S}"
  echo "==================== layout stripe_count=$C stripe_size=$S ===================="
  if [[ -d "$DEST" ]]; then
    echo "already exists, skipping: $DEST"
    continue
  fi
  bash "${EXP_DIR}/scripts/lib/make_striped_copy.sh" "$MODEL" "$DEST" "$C" "$S" 16
done

echo "=== copies present ==="
du -sh "${EXP_BASE}"/llama70b_* 2>/dev/null || true
echo "finished=$(date --iso-8601=seconds)"
