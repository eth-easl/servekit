#!/bin/bash
# CUDA checkpoint/restore feasibility test on Bristen. Two srun steps on one node:
#
#   1. bundle : INSIDE the ubuntu/cuda container -- build criu, compile counter.
#               (the host has no nvcc / no criu build deps)
#   2. test   : on the BARE host -- run the actual checkpoint/restore gates.
#
# Why the test can't run in the container: pyxis/enroot installs a seccomp filter
# (Seccomp:2), which forces criu to call PTRACE_O_SUSPEND_SECCOMP -- that requires
# CAP_SYS_ADMIN in the *initial* userns, which no cluster user can get. On the bare
# host, `unshare -Urpf` gives us full caps in our own user+pid namespace with NO
# seccomp filter, which is exactly what criu needs.
#
#   sbatch experiments/cuda-checkpoint-exp/scripts/ckpt_restore.sbatch
#
#SBATCH --job-name=cuda-cr
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=experiments/cuda-checkpoint-exp/results/cuda-cr-%j.out

set -uo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR}/experiments/cuda-checkpoint-exp"
EDF="${EXP_DIR}/cuda-cr.toml"

echo "job=${SLURM_JOB_ID} node=$(hostname)"
echo "started=$(date --iso-8601=seconds)"

echo "##### STEP 1: bundle (in container) #####"
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     --environment="${EDF}" --container-remap-root \
     bash "${EXP_DIR}/scripts/bundle.sh"
echo "bundle_exit=$?"

echo
echo "##### STEP 2: checkpoint/restore test (bare host) #####"
# Bare host = the best case for criu: Seccomp:0, and `unshare -Urpf` gives full caps.
# If criu fails HERE, it cannot be blamed on the container.
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     bash "${EXP_DIR}/scripts/run_host.sh"
echo "test_exit=$?"

echo
echo "##### STEP 3: cuda-checkpoint INSIDE the container #####"
# This is where real SGLang serving runs, so this is the operationally relevant
# environment for cuda-checkpoint. (criu gates skipped: the container's seccomp
# filter rules criu out there no matter what.)
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     --environment="${EDF}" \
     bash -c "SKIP_CRIU=1 bash ${EXP_DIR}/scripts/run_host.sh"
echo "container_test_exit=$?"
echo "finished=$(date --iso-8601=seconds)"
