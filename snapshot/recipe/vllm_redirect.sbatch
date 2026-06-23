#!/bin/bash
#SBATCH --job-name=snapshot-vllm-redirect
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail

GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
RECIPE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
mkdir -p /capstor/scratch/cscs/xyao/kimi-k25-vllm/logs

REDIR_BASE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/redirect-vllm

echo "vllm-under-redirect start: $(date --iso-8601=seconds) node $(hostname)"

# Two cold starts back to back: confirms the arena lets the engine run AND that
# the served addresses are reproducible across fresh processes (the precondition
# for snapshot/restore). TP=1 keeps the address space single-region.
for tag in run1 run2; do
  echo "================= COLD START $tag ================="
  TP="${TP:-1}" GMU="${GMU:-0.60}" REGION_GIB="${REGION_GIB:-64}" \
    ARENA="${ARENA:-1}" RUN_SECS="${RUN_SECS:-900}" RUN_TAG="$tag" \
    srun --environment=glm-47-flash-rocm bash "${RECIPE}/_vllm_redirect.sh" || true
done

echo "================= DETERMINISM DIFF ================="
# Compare the busiest worker log from each run. Identical addresses => the arena
# made torch's allocations reproducible across processes.
for f1 in "$REDIR_BASE"/run1/redir-*.log; do
  [ -e "$f1" ] || { echo "(no run1 alloc logs)"; break; }
  l1=$(wc -l < "$f1")
  echo "run1 $(basename "$f1"): $l1 allocs"
done
R1=$(ls -S "$REDIR_BASE"/run1/redir-*.log 2>/dev/null | head -1)
R2=$(ls -S "$REDIR_BASE"/run2/redir-*.log 2>/dev/null | head -1)
if [ -n "${R1:-}" ] && [ -n "${R2:-}" ]; then
  echo "busiest run1=$R1 ($(wc -l < "$R1") lines)  run2=$R2 ($(wc -l < "$R2") lines)"
  # Strip the absolute base by comparing op+size+rounded columns only first...
  echo "--- diff of (op size rounded va) first 5 hunks ---"
  diff <(cut -d' ' -f1-4 "$R1") <(cut -d' ' -f1-4 "$R2") | head -40
  echo "--- identical-address lines: $(comm -12 <(sort "$R1") <(sort "$R2") | wc -l) of $(wc -l < "$R1")/$(wc -l < "$R2") ---"
fi
echo "vllm-under-redirect end: $(date --iso-8601=seconds)"
