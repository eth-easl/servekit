#!/bin/bash
# Phase 1.2 - raw full-model read bandwidth vs reader concurrency (O_DIRECT).
# Storage-only benchmark: reads on the host /capstor mount, no container/GPU.
# A fresh sbatch submission = a fresh exclusive node = a genuine cold read
# (page cache cannot carry over from a previous run).
#
# Submit from the repo root:
#   sbatch lustre-loading-exp/scripts/phase1.2_dd_sweep/phase1_dd_sweep.sbatch
# Optionally override the model:
#   sbatch lustre-loading-exp/scripts/phase1.2_dd_sweep/phase1_dd_sweep.sbatch /path/to/model_dir
#
#SBATCH --job-name=dd-read-sweep
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=lustre-loading-exp/results/phase1.2_dd_sweep/%x-%j.out

set -euo pipefail

MODEL="${1:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
BS="${BS:-16M}"
CONC="${CONC:-1 4 8 16 32 64}"

EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
NODE="$(hostname)"
CSV="${EXP_DIR}/results/phase1.2_dd_sweep/phase1_dd_sweep-${SLURM_JOB_ID}-${NODE}.csv"

echo "job_id=${SLURM_JOB_ID} node=${NODE} model=${MODEL} bs=${BS} conc='${CONC}'"
echo "csv=${CSV}"
echo "started=$(date --iso-8601=seconds)"

bash "${EXP_DIR}/scripts/lib/dd_read_sweep.sh" "${MODEL}" "${BS}" "${CSV}" "${CONC}"

echo "finished=$(date --iso-8601=seconds)"
echo "=== result ==="
cat "${CSV}"
