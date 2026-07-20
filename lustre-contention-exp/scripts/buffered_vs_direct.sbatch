#!/bin/bash
# Whole model, 30 files x 1 reader each -- the OLD probe's exact shape -- with
# and without O_DIRECT. Nothing else changes.
#
# The claim under test: the old probe's "capstor is at 0.7 GB/s" is an artifact
# of `iflag=direct` alone. O_DIRECT disables Lustre client readahead, so each
# reader keeps exactly ONE 16 MiB RPC in flight; buffered readers ramp to
# max_read_ahead_per_file_mb=160 (~10 RPCs) and, on a latency-bound OST, earn
# proportionally more. On the sick OST-8 shard this was already measured on a
# single fd: O_DIRECT flat at ~25 MB/s, buffered ramping 30 -> 356.
#
# CACHE IS THE WHOLE DANGER HERE, so every pass is bracketed by /proc/meminfo:
#   * O_DIRECT populates NO page cache -> it runs FIRST and leaves the buffered
#     pass genuinely cold. Cached must be ~flat across it; if it is not, the
#     "O_DIRECT bypasses the cache" premise is wrong and the run is void.
#   * The buffered pass then caches ~141 GB (node has ~503 GB), so a REPEAT of
#     it is served from RAM. That repeat is run deliberately, as the positive
#     control: it must be absurdly fast, proving the cache really filled.
#   * The OSS-side cache is NOT bypassed by O_DIRECT and is not ours to flush,
#     so pass 1 may warm the servers for pass 2 -- a bias FAVOURING buffered.
#     ORDER=buffered_first inverts the whole sequence; run both and compare.
#
# Per-shard times are recorded, because the aggregate is a max over the pool and
# the interesting question is whether the OST-8 straggler SURVIVES readahead.
#
#   sbatch lustre-contention-exp/scripts/buffered_vs_direct.sbatch                       # direct first
#   sbatch --export=ALL,ORDER=buffered_first lustre-contention-exp/scripts/buffered_vs_direct.sbatch
#
#SBATCH --job-name=buf-vs-direct
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=lustre-contention-exp/results/%x-%j.out

set -uo pipefail
export PATH=$PATH:/usr/sbin

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
BS="${BS:-16M}"
POOL="${POOL:-30}"
ORDER="${ORDER:-direct_first}"
NODE="$(hostname)"
J="${SLURM_JOB_ID}"
RES="${SLURM_SUBMIT_DIR}/lustre-contention-exp/results"
CSV="${RES}/bufdirect_pershard-${J}-${NODE}.csv"

mapfile -t SHARDS < <(ls "$MODEL"/*.safetensors | sort)
TOTAL=0
for f in "${SHARDS[@]}"; do TOTAL=$(( TOTAL + $(stat -c%s "$f") )); done

echo "job=${J} node=${NODE} order=${ORDER} started=$(date --iso-8601=seconds)"
echo "shards=${#SHARDS[@]} total=$(awk -v t=$TOTAL 'BEGIN{printf "%.1fGB",t/1e9}') bs=${BS} pool=${POOL}"
echo "mem: $(awk '/MemTotal/{printf "%.0fGB total",$2/1048576}' /proc/meminfo), model is $(awk -v t=$TOTAL 'BEGIN{printf "%.0f",t/1e9}')GB"
echo "readahead: per_file=$(lctl get_param -n llite.capstor-*store*.max_read_ahead_per_file_mb 2>/dev/null | head -1)MB global=$(lctl get_param -n llite.capstor-*store*.max_read_ahead_mb 2>/dev/null | head -1)MB"

cached_gb() { awk '/^Cached:/{printf "%.1f", $2/1048576}' /proc/meminfo; }

# $1 = direct|buffered ; times each shard, appends per-shard rows, echoes agg GB/s
read_one() {
  local f="$1" flag="" t0 t1 w
  [ "$MODE" = "direct" ] && flag="iflag=direct"
  t0=$(date +%s.%N)
  dd if="$f" of=/dev/null bs="$BS" $flag status=none
  t1=$(date +%s.%N)
  w=$(awk -v s=$t0 -v e=$t1 'BEGIN{printf "%.2f", e-s}')
  printf '%s,%s,%s,%s,%s,%s\n' "$MODE" "$PASS" "$(basename "$f")" \
    "$(awk -v n="$(basename "$f")" '$1==n{print $2; exit}' <<<"$OSTMAP")" \
    "$w" "$(awk -v b="$(stat -c%s "$f")" -v w="$w" 'BEGIN{printf "%.1f", b/w/1e6}')"
}
export -f read_one
export BS

OSTMAP=""
for f in "${SHARDS[@]}"; do
  OSTMAP+="$(basename "$f") $(lfs getstripe -i "$f" 2>/dev/null | tr -dc '0-9')"$'\n'
done
export OSTMAP

echo "mode,pass,shard,ost_idx,wall_s,MBps" > "$CSV"

pass() {
  local mode="$1" label="$2" c0 c1 start end wall agg
  export MODE="$mode" PASS="$label"
  c0="$(cached_gb)"
  start="$(date +%s.%N)"
  printf '%s\n' "${SHARDS[@]}" | xargs -P "$POOL" -I{} bash -c 'read_one "$@"' _ {} >> "$CSV"
  end="$(date +%s.%N)"
  c1="$(cached_gb)"
  wall="$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')"
  agg="$(awk -v t="$TOTAL" -v w="$wall" 'BEGIN{printf "%.2f", t/w/1e9}')"
  printf '  %-22s wall=%8s s   agg=%6s GB/s   Cached %6s -> %6s GB\n' "$label" "$wall" "$agg" "$c0" "$c1"
}

echo
echo "=== whole model, ${POOL} files x 1 reader (the old probe's exact shape) ==="
printf '  %-22s %s\n' "(baseline)" "Cached now = $(cached_gb) GB"

if [ "$ORDER" = "buffered_first" ]; then
  pass buffered "buffered  (cold)"
  pass direct   "O_DIRECT"
  pass buffered "buffered  (WARM/RAM)"
else
  pass direct   "O_DIRECT  (cold)"
  pass buffered "buffered  (cold)"
  pass buffered "buffered  (WARM/RAM)"
fi

echo
echo "=== did the OST-8 straggler survive readahead? (per-shard, cold passes) ==="
python3 - "$CSV" <<'PY'
import csv, sys, statistics as st
rows = [r for r in csv.DictReader(open(sys.argv[1])) if "WARM" not in r["pass"]]
for mode in ("direct", "buffered"):
    rs = [r for r in rows if r["mode"] == mode]
    if not rs: continue
    v = sorted(float(r["MBps"]) for r in rs)
    slow = min(rs, key=lambda r: float(r["MBps"]))
    print(f"  {mode:9s} n={len(rs):2d}  slowest={v[0]:7.1f} MB/s (ost {slow['ost_idx']}, {slow['wall_s']}s)"
          f"  median={st.median(v):7.1f}  fastest={v[-1]:7.1f}  max/min={v[-1]/v[0]:5.1f}x")
PY
echo
echo "csv=${CSV}"
echo "finished=$(date --iso-8601=seconds)"
