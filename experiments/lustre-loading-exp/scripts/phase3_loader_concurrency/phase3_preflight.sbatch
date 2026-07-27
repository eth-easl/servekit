#!/bin/bash
# Phase 3 preflight - prove the clone/verify/swap harness works BEFORE spending
# 2.5 h of GPU nodes on the sweep.
#
# Checks, in the real container, that: git exists, the compute node can reach
# GitHub, the pinned tag still resolves to the pinned SHA, the clone is
# byte-identical to the sglang installed in the image, the patch applies, and
# the swapped file both compiles and carries the knob.
#
# No GPUs, no model, ~2 min. The sweep chain hangs off this with
# --dependency=afterok, so a broken harness costs 2 minutes instead of
# producing 5 jobs' worth of numbers that mean nothing.
#
# Submit:
#   sbatch lustre-loading-exp/scripts/phase3_loader_concurrency/phase3_preflight.sbatch
#
#SBATCH --job-name=p3-preflight
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=lustre-loading-exp/results/phase3_loader_concurrency/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
S="${EXP_DIR}/scripts/phase3_loader_concurrency"

echo "job=${SLURM_JOB_ID} node=$(hostname) tag=preflight"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" "${S}/fst_files_per_rank.patch"

# The knob must be live in the file the engine will actually import, and must
# still default to upstream behavior when unset.
python - <<'PY'
import importlib.util, os, re
p = os.path.join(os.path.dirname(importlib.util.find_spec("sglang").origin),
                 "srt", "model_loader", "weight_utils.py")
src = open(p).read()
assert 'os.getenv("SGLANG_FST_FILES_PER_RANK", "1")' in src, "knob missing"
assert "rank_files = f_list[rank_idx :: pg.size()]" in src, "round-robin missing"
print("[preflight] knob present in the installed module, defaults to 1 (upstream)")
PY
echo "[preflight] ALL CHECKS PASSED"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

echo "finished=$(date --iso-8601=seconds)"
