#!/bin/bash
# Does slicing each shard fix the probe? Full-model O_DIRECT read, sweeping
# slices-per-file, against the OLD probe as an in-job control.
#
# The old probe (dd_read_sweep.sh) fans out across files only: 30 shards,
# stripe_count=1, one reader each -> queue depth 1 per OST. OST 8 charges
# ~480 ms/RPC, so its shard alone takes ~200 s and, because wall_s is a max over
# the pool, becomes the entire reported aggregate (~0.6-0.9 GB/s) while capstor
# is actually serving 6-8 GB/s. Slicing gives every OST T requests in flight.
#
# Both probes run in the SAME job, minutes apart, on the same node -- the only
# comparison this repo trusts. Control runs FIRST so it cannot be flattered by
# anything the sweep warms (O_DIRECT anyway, but keep the habit).
#
#   sbatch experiments/lustre-contention-exp/scripts/slice_sweep.sbatch
#
#SBATCH --job-name=slice-sweep
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=experiments/lustre-contention-exp/results/%x-%j.out

set -uo pipefail
export PATH=$PATH:/usr/sbin

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
BS="${BS:-16M}"
SLICES="${SLICES:-1 2 4 8 16}"
EXP="${SLURM_SUBMIT_DIR}/experiments/lustre-contention-exp"
RES="${EXP}/results"
NODE="$(hostname)"
J="${SLURM_JOB_ID}"

echo "job=${J} node=${NODE} started=$(date --iso-8601=seconds)"
echo "model=${MODEL} bs=${BS} slices=${SLICES}"

# Is the OST-8 fault still present at all? If it has been fixed, the control
# will not be straggler-bound and the whole comparison means something else.
SICK="${MODEL}/model-00018-of-00030.safetensors"
CTRL_SHARD="${MODEL}/model-00017-of-00030.safetensors"
echo "=== is OST 8 still sick? (512 MiB, QD=1) ==="
for f in "${SICK}" "${CTRL_SHARD}"; do
  t0=$(date +%s.%N); dd if="$f" of=/dev/null bs=16M count=32 iflag=direct status=none; t1=$(date +%s.%N)
  awk -v s=$t0 -v e=$t1 -v n="$(basename $f)" -v o="$(lfs getstripe -i $f)" \
    'BEGIN{printf "  %s (ost %s): %6.1f MB/s\n", n, o, 512/(e-s)}'
done

echo
echo "=== CONTROL: old probe, fan-out across files only (N=30, 1 reader/file) ==="
bash "${EXP}/scripts/lib/dd_read_sweep.sh" "${MODEL}" "${BS}" \
     "${RES}/slice_ctl_oldprobe-${J}-${NODE}.csv" "30" \
     2>/dev/null || echo "control FAILED"

echo
echo "=== SWEEP: both parallelisms, 30 files x T slices ==="
bash "${EXP}/scripts/lib/dd_read_sliced.sh" "${MODEL}" "${BS}" \
     "${RES}/slice_sweep-${J}-${NODE}.csv" "${SLICES}" auto \
     2>/dev/null || echo "sweep FAILED"

echo
echo "=== fixed-pool control: is the win queue depth, or just more processes? ==="
echo "    (pool pinned to 30 readers total; T>1 then means FEWER files in"
echo "     flight but T readers each -- same process count as the old probe)"
bash "${EXP}/scripts/lib/dd_read_sliced.sh" "${MODEL}" "${BS}" \
     "${RES}/slice_fixedpool-${J}-${NODE}.csv" "1 2 4" 30 \
     2>/dev/null || echo "fixed-pool FAILED"

echo
echo "=== RESULTS ==="
echo "--- control (old probe):";  cat "${RES}/slice_ctl_oldprobe-${J}-${NODE}.csv" 2>/dev/null
echo "--- sliced sweep (pool=30xT):"; cat "${RES}/slice_sweep-${J}-${NODE}.csv" 2>/dev/null
echo "--- sliced, pool pinned to 30:"; cat "${RES}/slice_fixedpool-${J}-${NODE}.csv" 2>/dev/null
echo "finished=$(date --iso-8601=seconds)"
