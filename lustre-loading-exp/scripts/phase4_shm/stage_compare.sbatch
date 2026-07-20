#!/bin/bash
# Phase 4 staging: across-files fan-out (the control) vs across-files x
# within-file slicing. Same node, same job, minutes apart.
#
# stage_to_shm.sh runs one `dd ... iflag=direct` per file under `xargs -P 60`.
# Shards are stripe_count=1, so that is ONE reader per OST at queue depth 1, and
# O_DIRECT disables readahead -- so a single high-latency OST caps the stage and
# `wall` is a max over the pool. It measured 21.2 s @ 6.65 GB/s when every OST
# was healthy; OST 8 has since gone to ~480 ms/RPC (~25 MB/s), which should now
# put the control near ~200 s for its one 5 GB shard while the other 29 idle.
# stage_to_shm_sliced.sh cuts each file into SLICES concurrent ranges.
#
# THE GATE THAT MATTERS: the sliced stager has many dd's writing into ONE file at
# different offsets. Size parity proves nothing (truncate sets the size up
# front), so this job md5s EVERY staged file both ways and diffs the two
# manifests. Both hashings read from tmpfs (RAM) and run 30-way, so the gate is
# cheap. A speedup that corrupts weights is worth exactly nothing, and SGLang
# would happily serve garbage from it -- model-release-reload-exp already found
# that every cheap health signal stays green while output is destroyed.
#
# Caveat recorded rather than controlled: the control runs first, and O_DIRECT
# does not bypass the SERVER-side cache, so the sliced pass may see warmer OSSs.
# That bias favours the sliced pass. Run ORDER=sliced_first for the mirror image
# if the two land close; at the expected ~20x they will not.
#
#   sbatch lustre-loading-exp/scripts/phase4_shm/stage_compare.sbatch
#   sbatch --export=ALL,ORDER=sliced_first,SLICES=64 lustre-loading-exp/scripts/phase4_shm/stage_compare.sbatch
#
#SBATCH --job-name=stage-compare
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=lustre-loading-exp/results/phase4_shm/%x-%j.out

set -uo pipefail
export PATH=$PATH:/usr/sbin

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
DEST="${DEST:-/dev/shm/llama70b}"
SLICES="${SLICES:-64}"
WORKERS="${WORKERS:-60}"
ORDER="${ORDER:-control_first}"
S="${SLURM_SUBMIT_DIR}/lustre-loading-exp/scripts/phase4_shm"
NODE="$(hostname)"

echo "job=${SLURM_JOB_ID} node=${NODE} order=${ORDER} slices=${SLICES} workers=${WORKERS}"
echo "started=$(date --iso-8601=seconds)"
echo "shm: $(df -h /dev/shm | tail -1)"

echo
echo "=== is OST 8 still the straggler? (512 MiB, QD=1) ==="
for s in 00018 00017; do
  f="${MODEL}/model-${s}-of-00030.safetensors"
  t0=$(date +%s.%N); dd if="$f" of=/dev/null bs=16M count=32 iflag=direct status=none; t1=$(date +%s.%N)
  awk -v s=$t0 -v e=$t1 -v n="$s" -v o="$(lfs getstripe -i "$f" 2>/dev/null)" \
    'BEGIN{printf "  model-%s (ost %s): %6.1f MB/s\n", n, o, 512/(e-s)}'
done

# md5 every staged file, 30-way, from tmpfs. Sorted manifest -> diffable.
manifest() {
  ( cd "$DEST" && find . -maxdepth 1 -type f -printf '%f\n' \
      | xargs -P 30 -I{} md5sum {} | sort -k2 ) > "$1"
}

run_control() {
  echo
  echo "=== CONTROL: stage_to_shm.sh (${WORKERS} workers, 1 reader/file) ==="
  rm -rf "$DEST"
  bash "$S/stage_to_shm.sh" "$MODEL" "$DEST" "$WORKERS" 2>&1 | sed 's/^/  /'
  echo "  md5 manifest..."
  manifest /tmp/manifest_control.txt
  echo "  hashed $(wc -l < /tmp/manifest_control.txt) files"
}

run_sliced() {
  echo
  echo "=== SLICED: stage_to_shm_sliced.sh (30 files x ${SLICES} slices) ==="
  rm -rf "$DEST"
  bash "$S/stage_to_shm_sliced.sh" "$MODEL" "$DEST" "$SLICES" 2>&1 | sed 's/^/  /'
  echo "  md5 manifest..."
  manifest /tmp/manifest_sliced.txt
  echo "  hashed $(wc -l < /tmp/manifest_sliced.txt) files"
}

if [ "$ORDER" = "sliced_first" ]; then run_sliced; run_control; else run_control; run_sliced; fi

echo
echo "=== CORRECTNESS GATE: sliced output must be byte-identical to the control ==="
if diff -q /tmp/manifest_control.txt /tmp/manifest_sliced.txt >/dev/null 2>&1; then
  echo "  ✅ PASS - all $(wc -l < /tmp/manifest_sliced.txt) files byte-identical (md5)"
else
  echo "  ❌ FAIL - sliced stage CORRUPTED the model. Differing files:"
  diff /tmp/manifest_control.txt /tmp/manifest_sliced.txt | head -20
  echo "  The speedup below is meaningless. Do not use the sliced stager."
fi

rm -rf "$DEST"
echo
echo "finished=$(date --iso-8601=seconds)"
