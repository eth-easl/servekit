#!/bin/bash
# Diagnostic, not a measurement. Stage BOTH presharded checkpoints with the same
# stager, back to back, on ONE node in ONE minute, and print each one's GB/s.
#
# It exists because the vllm preshard run (job 2918555) staged at 0.41 GB/s where
# the sglang runs hit 17.02 GB/s on the same node type, same stager, same slice
# count and identical striping (32 x 4 MiB). Two candidates, and they need
# different fixes:
#
#   capstor is slow right now      -> both checkpoints read slowly here, re-run
#                                     the measurement later, change nothing.
#   the new checkpoint reads slow  -> only the -vllm one is slow, i.e. its
#                                     objects landed on unlucky OSTs when it was
#                                     written, and it should be rewritten.
#
# Same node and adjacent minutes are the whole point: read order is fixed
# (sglang first) and /dev/shm is cleared between the two, so the only thing that
# differs between the numbers is which files were read.
#
# No GPUs: this touches storage only.
#
#   cd experiments/clariden-loading-exp
#   sbatch --partition=debug scripts/shared/stage_probe.sbatch
#
#SBATCH --job-name=stage-probe
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

STAGER="${SLURM_SUBMIT_DIR}/scripts/shared/stage_to_shm_sliced.sh"
BASE=/capstor/store/cscs/swissai/infra01/cold-start-experiments
SLICES="${SLICES:-64}"
CKPTS="${CKPTS:-llama70b-tp4-sharded llama70b-tp4-sharded-vllm}"

# EDF=<path> runs the stage INSIDE that container; unset runs it on the bare
# host. This matters: the first probe (job 2918617) ran on the host and reported
# 26-28 GB/s, which does NOT rule out the container being the slow part, because
# the measured runs stage from inside one.
EDF="${EDF:-}"

echo "job=${SLURM_JOB_ID} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
echo "edf=${EDF:-<none, bare host>}"
echo "started=$(date --iso-8601=seconds)"
df -h /dev/shm

run() {
  if [ -n "${EDF}" ]; then
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
      --environment="${EDF}" bash -c "$1"
  else
    bash -c "$1"
  fi
}

for ckpt in ${CKPTS}; do
  echo
  echo "=== ${ckpt} ==="
  # A fresh dest each time, removed after, so neither stage sees the other's
  # tmpfs pages and /dev/shm never has to hold both at once.
  dest="/dev/shm/probe-${ckpt}"
  run "FILE_PATTERN='*.safetensors' bash '${STAGER}' '${BASE}/${ckpt}' '${dest}' '${SLICES}'"
  run "rm -r '${dest}'"
done

echo
echo "finished=$(date --iso-8601=seconds)"
