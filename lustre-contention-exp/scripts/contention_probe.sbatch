#!/bin/bash
# Lustre contention experiment - ONE 30-minute time-slot sample.
#
# Reads the whole 132 GB Llama-3.1-70B model from capstor Lustre, O_DIRECT, with
# a 32-worker pool, REPEATS times back to back, and records the aggregate read
# bandwidth (GiB/s) of each repeat with a wall-clock timestamp. The 24h driver
# (submit_24h.sh) fires 48 of these, one every 30 min, each on a FRESH node
# (whatever SLURM hands out) - so the resulting time series shows how capstor
# read bandwidth varies across a full day/night cycle.
#
# Storage-only: no GPU, no container, no server. Reads happen on the host
# /capstor mount, exactly like lustre-loading-exp phase1.2. O_DIRECT bypasses the
# CLIENT page cache, so a fresh node is a genuine cold read. (The capstor
# OSS-side cache is NOT bypassed; within a slot, repeats 2..N may run warmer than
# repeat 1 - that within-slot spread is exactly what the min/max whiskers show,
# which is why we keep all REPEATS rather than averaging in-job.)
#
# The dd probe (scripts/lib/dd_read_sweep.sh) is vendored into this experiment so
# it is fully self-contained; it originates from the loading experiment.
#
# Submit (normally via submit_24h.sh), from the repo root:
#   sbatch lustre-contention-exp/scripts/contention_probe.sbatch
# Overridable via env: MODEL, REPEATS, CONC, BS.
#
#SBATCH --job-name=lustre-contention
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=lustre-contention-exp/results/contention-%j.out

set -euo pipefail

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
REPEATS="${REPEATS:-5}"
CONC="${CONC:-32}"
BS="${BS:-16M}"

EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-contention-exp"
LIB="${EXP_DIR}/scripts/lib/dd_read_sweep.sh"
NODE="$(hostname)"
CSV="${EXP_DIR}/results/contention-${SLURM_JOB_ID}-${NODE}.csv"
SHARD_CSV="${EXP_DIR}/results/contention_pershard-${SLURM_JOB_ID}-${NODE}.csv"

echo "job=${SLURM_JOB_ID} node=${NODE} slot=${SLOT:-?} model=${MODEL}"
echo "repeats=${REPEATS} conc=${CONC} bs=${BS}"
echo "csv=${CSV}"
echo "pershard_csv=${SHARD_CSV}"
echo "started=$(date --iso-8601=seconds)"

# One data column extra vs the lib: iso_time (per repeat) and agg_GiBps (binary).
# The lib emits decimal agg_GBps (bytes/wall/1e9); we recompute GiB/s here from
# the raw total_bytes/wall_s so the shared lib stays untouched.
echo "iso_time,run_index,concurrency,total_bytes,wall_s,agg_GiBps" > "${CSV}"

tmp="$(mktemp)"
tmp_shard="$(mktemp)"
trap 'rm -f "$tmp" "$tmp_shard"' EXIT

# Per-shard rows, accumulated across repeats. agg_GiBps alone cannot separate
# "every OST is slow" from "one OST is a straggler and drags the pool's max" --
# both land on the same aggregate. Shards are stripe_count=1, so shard == OST.
echo "iso_time,run_index,concurrency,shard,ost_idx,bytes,wall_s,GBps" > "${SHARD_CSV}"

for run in $(seq 1 "${REPEATS}"); do
  ts="$(date --iso-8601=seconds)"
  # dd_read_sweep.sh <model_dir> <bs> <csv_out> <concurrency_list>; single N -> 1 data row.
  # PERSHARD_CSV is rewritten per invocation, so fold each repeat in below.
  PERSHARD_CSV="${tmp_shard}" bash "${LIB}" "${MODEL}" "${BS}" "${tmp}" "${CONC}"
  # tmp: header line, then one row "concurrency,total_bytes,wall_s,agg_GBps".
  row="$(tail -n 1 "${tmp}")"
  conc="$(echo "${row}" | cut -d, -f1)"
  total="$(echo "${row}" | cut -d, -f2)"
  wall="$(echo "${row}" | cut -d, -f3)"
  gib="$(awk -v t="${total}" -v w="${wall}" 'BEGIN{printf "%.3f", t/w/1073741824}')"
  echo "${ts},${run},${conc},${total},${wall},${gib}" | tee -a "${CSV}"
  # tmp_shard: header + one row per shard; stamp each with this repeat's identity.
  tail -n +2 "${tmp_shard}" | sed "s|^|${ts},${run},|" >> "${SHARD_CSV}"
done

echo "finished=$(date --iso-8601=seconds)"
echo "=== ${CSV} ==="
cat "${CSV}"

# The discriminator: if the pool is uniformly slow, min and max per-shard GB/s
# sit close together. If one OST is sick, the spread blows up and slowest_ost
# names it.
echo "=== per-shard spread by repeat (${SHARD_CSV}) ==="
awk -F, 'NR>1{
  n[$2]++; s[$2]+=$8
  if (mx[$2]=="" || $8>mx[$2]) mx[$2]=$8
  if (mn[$2]=="" || $8<mn[$2]) { mn[$2]=$8; slow[$2]=$5 }
}
END{
  printf "%4s %8s %8s %8s %8s %10s\n","run","min","med~avg","max","max/min","slowest_ost"
  for (r in n) printf "%4s %8.3f %8.3f %8.3f %8.1fx %10s\n", r, mn[r], s[r]/n[r], mx[r], mx[r]/mn[r], slow[r]
}' "${SHARD_CSV}" | sort -n
