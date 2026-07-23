#!/bin/bash
# Does the sliced probe keep scaling past T=16? (slice_sweep 75151 ended at
# T=16 = 9.59 GB/s and was still climbing steeply: 3.64 -> 9.59.)
#
# Two things get dangerous up here, so both are controlled:
#
# 1. SPAWN COST. Each task is `bash -c` + `dd` ~= 2 forks. At T=64 that is
#    ~1800 tasks, and if the read itself only takes ~7 s the fork storm is no
#    longer negligible -- it would show up as a fake plateau. So every T also
#    runs the IDENTICAL task list with `true` in place of the read: that is the
#    floor. A T is only believable if wall_s >> spawn_s.
# 2. THE MEASUREMENT GETTING TOO SHORT. 141 GB at 20 GB/s is ~7 s. Repeats x2
#    to see the noise rather than quote a single lucky number.
#
# Note T=64 does not actually give 64 slices: shards are ~298 x 16MiB blocks, so
# per = ceil(298/64) = 5 and only ~60 slices fit. At that point pool (1920)
# exceeds the task count (~1800) and EVERY slice of EVERY shard is in flight at
# once -- the probe's ceiling, not a tunable.
#
#   sbatch experiments/lustre-contention-exp/scripts/slice_sweep_deep.sbatch
#
#SBATCH --job-name=slice-deep
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
BS_BYTES=$(( $(echo "$BS" | sed 's/M$//') * 1024 * 1024 ))
EXP="${SLURM_SUBMIT_DIR}/experiments/lustre-contention-exp"
RES="${EXP}/results"
NODE="$(hostname)"
J="${SLURM_JOB_ID}"
CSV="${RES}/slice_deep-${J}-${NODE}.csv"

echo "job=${J} node=${NODE} started=$(date --iso-8601=seconds)"
echo "cpus=${SLURM_CPUS_PER_TASK} mem=$(awk '/MemTotal/{printf "%.0fGB",$2/1048576}' /proc/meminfo)"
echo "nic: $(lctl get_param -n osc.capstor-OST0008-osc-*.import 2>/dev/null | awk '/current_connection/{print $2; exit}')"

mapfile -t SHARDS < <(ls "$MODEL"/*.safetensors | sort)
NSHARDS="${#SHARDS[@]}"
TOTAL=0
for f in "${SHARDS[@]}"; do TOTAL=$(( TOTAL + $(stat -c%s "$f") )); done
echo "shards=${NSHARDS} total=$(awk -v t=$TOTAL 'BEGIN{printf "%.1fGB",t/1e9}') bs=${BS}"

read_slice() { dd if="$1" of=/dev/null bs="$BS" skip="$2" count="$3" iflag=direct status=none; }
export -f read_slice
export BS

emit_tasks() {
  local T="$1" f size blocks per i skip cnt
  for f in "${SHARDS[@]}"; do
    size=$(stat -c%s "$f")
    blocks=$(( (size + BS_BYTES - 1) / BS_BYTES ))
    per=$(( (blocks + T - 1) / T ))
    for (( i=0; i<T; i++ )); do
      skip=$(( i * per )); (( skip >= blocks )) && break
      cnt=$per; (( skip + cnt > blocks )) && cnt=$(( blocks - skip ))
      echo "$f $skip $cnt"
    done
  done
}

echo "slices_per_file,pool,tasks,rep,total_bytes,wall_s,agg_GBps,spawn_s" | tee "$CSV"
for T in 1 16 32 48 64; do
  P=$(( NSHARDS * T ))
  NTASK=$(emit_tasks "$T" | wc -l)

  # Floor: identical task list and pool, read replaced by `true`. Anything the
  # sweep reports that is close to this number is measuring fork(), not capstor.
  s0="$(date +%s.%N)"
  emit_tasks "$T" | xargs -P "$P" -n 3 bash -c 'true' _ >/dev/null 2>&1
  s1="$(date +%s.%N)"
  spawn="$(awk -v s="$s0" -v e="$s1" 'BEGIN{printf "%.2f", e-s}')"

  for rep in 1 2; do
    start="$(date +%s.%N)"
    emit_tasks "$T" | xargs -P "$P" -n 3 bash -c 'read_slice "$@"' _
    end="$(date +%s.%N)"
    wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
    agg="$(awk -v t="$TOTAL" -v w="$wall" 'BEGIN{printf "%.2f", t/w/1e9}')"
    echo "$T,$P,$NTASK,$rep,$TOTAL,$wall,$agg,$spawn" | tee -a "$CSV"
  done
done

echo
echo "=== ${CSV} ==="
cat "${CSV}"
echo "finished=$(date --iso-8601=seconds)"
