#!/bin/bash
# Why is the sliced stage 19-20 GB/s in stage_compare.sbatch but only 11.5-12.2
# GB/s inside the e2e job? Same stager, same 64 slices, same model.
#
# The cleanest existing evidence, SAME NODE (nid002297) five minutes apart, so
# node-to-node variation and Lustre drift are already ruled out:
#   75166 sliced leg, 13:20, batch step, no srun  -> 19.36 GB/s
#   75168 stage,      13:25, srun -c 64 + container -> 11.66 GB/s
#
# But TWO things differ between those, not one:
#   (a) CPU COUNT. stage_compare runs the stager directly in the batch step,
#       which under --exclusive gets all 128 CPUs of the node (it asks for 32,
#       but an exclusive batch step is not bound to that). The e2e runs it under
#       `srun --cpus-per-task=64`, i.e. 64 of 128. That is a 2x CPU ratio
#       against a 1.73x throughput ratio -- and the stage IS cpu-bound: O_DIRECT
#       reads DMA into the user buffer, but all 141 GB are then memcpy'd by the
#       CPU into tmpfs pages, ~40 GB/s of memory traffic across 1744 dd's.
#   (b) THE CONTAINER (--environment=$EDF), e.g. Lustre client tuning or
#       namespace overhead differing inside enroot.
#
# Four legs separate them. The job requests exactly like the real e2e run
# (--cpus-per-task=64 --gpus-per-node=4 --exclusive) and varies only how the
# stager is launched:
#   batch_bare    batch step, no srun, no container  -> should reproduce ~19-20
#   srun_bare_64  srun -c 64, no container           -> isolates srun/CPU binding
#   srun_ctr_64   srun -c 64 + container             -> the e2e config exactly
#   srun_ctr_128  srun -c 128 + container            -> container with all CPUs
#
# Reading the result:
#   batch_bare ~= srun_ctr_128 >> srun_bare_64 ~= srun_ctr_64
#       -> CPU COUNT. Container is innocent. Fix is one flag in the e2e sbatch.
#   batch_bare ~= srun_bare_64 >> srun_ctr_64
#       -> THE CONTAINER. Follow up on Lustre-client-in-container tuning.
#
# Each leg dumps its effective cpuset, nproc, and /dev/shm mount+size: the write
# half of the stage lands in tmpfs, so a different tmpfs mount inside the
# container (size, huge= option) would itself be a candidate explanation.
#
# Caveat, as in stage_compare.sbatch: O_DIRECT does not bypass the SERVER-side
# cache, so later legs may see warmer OSSs. batch_bare runs FIRST and therefore
# coldest -- if it is still the fastest, that conclusion is conservative. The
# measured warm-vs-cold effect is small anyway (75167 sliced-first cold 20.19
# vs 75166 sliced-second warm 19.36, ~4%).
#
#   sbatch lustre-loading-exp/scripts/phase4_shm/stage_isolate_container.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:12:00
#SBATCH --output=lustre-loading-exp/results/phase4_shm/%x-%j.out

set -uo pipefail

SRC="${SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SLICES="${SLICES:-64}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b}"

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
STAGER="${SLURM_SUBMIT_DIR}/lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh"
NODE="$(hostname)"

echo "job=${SLURM_JOB_ID} node=${NODE} slices=${SLICES}"
echo "started=$(date --iso-8601=seconds)"
echo "node CPUs: $(nproc) (batch step, unbound under --exclusive)"

RESULTS=""

# Printed inside every leg so the environment is described from within the leg,
# not assumed from outside.
PROBE='
  echo "  cpuset: $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null || echo n/a)"
  echo "  nproc:  $(nproc)"
  echo "  shm:    $(df -h /dev/shm | tail -1)"
  echo "  mount:  $(grep " /dev/shm " /proc/mounts || echo n/a)"
'

run_leg() { # <label> <launch: batch|srun> <container: yes|no> <cpus>
  local label="$1" launch="$2" container="$3" cpus="$4" out gbps
  echo
  echo "=== LEG: ${label} (launch=${launch} container=${container} cpus=${cpus}) ==="
  rm -rf "$SHM_DEST"

  if [ "$launch" = batch ]; then
    out=$( { eval "$PROBE"; bash "$STAGER" "$SRC" "$SHM_DEST" "$SLICES"; } 2>&1 )
  else
    local envflag=""
    [ "$container" = yes ] && envflag="--environment=${EDF}"
    out=$( srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
             --cpus-per-task="${cpus}" --gpus-per-task=4 ${envflag} \
             bash -c "${PROBE}"' bash "$0" "$1" "$2" "$3"' \
             "$STAGER" "$SRC" "$SHM_DEST" "$SLICES" 2>&1 )
  fi

  echo "$out" | sed 's/^/  /'
  gbps=$(echo "$out" | grep -o 'stage_GBps=[0-9.]*' | head -1 | cut -d= -f2)
  RESULTS="${RESULTS}${label}|${gbps:-FAILED}"$'\n'
}

run_leg batch_bare   batch no  0
run_leg srun_bare_64 srun  no  64
run_leg srun_ctr_64  srun  yes 64
run_leg srun_ctr_128 srun  yes 128

rm -rf "$SHM_DEST"

echo
echo "=== result (GB/s) ==="
echo "$RESULTS" | awk -F'|' 'NF==2{printf "  %-14s %s\n", $1, $2}'
echo
echo "  reference: 19.36-20.19 (batch step, no srun) vs 11.66-11.80 (srun -c 64 + container)"
echo "finished=$(date --iso-8601=seconds)"
