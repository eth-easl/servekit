#!/bin/bash
# Phase 2 - does per-file striping help? For each (stripe_count, stripe_size)
# layout: make a striped copy of the model, run the full-model O_DIRECT read
# sweep on it, record the curve, then delete the copy to reclaim space.
# Compare against the native-layout Phase 1.2 baseline.
#
# Fresh exclusive node per submission (cold reads). Storage-only, no container.
#
# Submit from the repo root (creates the experiment base dir on first use):
#   sbatch lustre-loading-exp/scripts/phase2_stripe_sweep/phase2_stripe_sweep.sbatch
# Override layouts / location:
#   LAYOUTS="4:4M 8:4M 16:4M 8:16M" EXP_BASE=/capstor/.../infra01/cold-start-experiments \
#     sbatch lustre-loading-exp/scripts/phase2_stripe_sweep/phase2_stripe_sweep.sbatch
#
#SBATCH --job-name=stripe-sweep
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=lustre-loading-exp/results/phase2_stripe_sweep/%x-%j.out

set -euo pipefail

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
# Same Lustre FS as the model, writable by group infra01 (setgid). NOT created
# here silently - `mkdir -p` makes it on first real run; verify quota first.
EXP_BASE="${EXP_BASE:-/capstor/store/cscs/swissai/infra01/cold-start-experiments}"
# "stripe_count:stripe_size" pairs. Keep the list short - each copy is ~132 GB.
LAYOUTS="${LAYOUTS:-4:4M 8:4M 16:4M 8:16M}"
BS="${BS:-16M}"
# N=1 (single-stream ~0.4 GB/s, ~6 min on the full model) is dropped by default;
# we already have it from Phase 1.2. Add it back via CONC if wanted.
CONC="${CONC:-8 16 30 60}"
# Contention control: capstor is shared and swings 2-6x, so an absolute layout
# number is meaningless on its own. Before each layout we re-measure the NATIVE
# model at PROBE_N in the SAME time window; report layout relative to that.
PROBE_N="${PROBE_N:-16}"
KEEP="${KEEP:-0}"   # 1 = keep copies (e.g. best layout for Phase 3), 0 = delete

EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
NODE="$(hostname)"
mkdir -p "$EXP_BASE"

# Space guard: refuse to start if the FS is nearly full (each copy ~132 GB).
avail_pct="$(df --output=pcent "$EXP_BASE" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ -n "$avail_pct" && "$avail_pct" -ge 92 ]]; then
  echo "ABORT: $EXP_BASE is ${avail_pct}% full; refusing to write ~132 GB copies" >&2
  exit 1
fi

PROBE_CSV="${EXP_DIR}/results/phase2_stripe_sweep/phase2_nativeprobe-${SLURM_JOB_ID}-${NODE}.csv"
echo "job=${SLURM_JOB_ID} node=${NODE} model=${MODEL}"
echo "exp_base=${EXP_BASE} layouts='${LAYOUTS}' keep=${KEEP} probe_n=${PROBE_N}"
echo "native_probe_csv=${PROBE_CSV}"
echo "label,probe_GBps" | tee "$PROBE_CSV"
echo "capstor free before: $(lfs df -h "$EXP_BASE" 2>/dev/null | awk '/summary|filesystem/{print $0}' | tail -1)"

for pair in $LAYOUTS; do
  C="${pair%%:*}"; S="${pair##*:}"
  DEST="${EXP_BASE}/llama70b_c${C}_s${S}"
  CSV="${EXP_DIR}/results/phase2_stripe_sweep/phase2_c${C}_s${S}-${SLURM_JOB_ID}-${NODE}.csv"
  echo "==================== layout stripe_count=$C stripe_size=$S ===================="

  # Contemporaneous native-layout probe (single concurrency) for normalization.
  # Route through a temp file: dd_read_sweep uses `tee`, so /dev/stdout would
  # duplicate lines and break parsing.
  ptmp="$(mktemp)"
  bash "${EXP_DIR}/scripts/lib/dd_read_sweep.sh" "$MODEL" "$BS" "$ptmp" "$PROBE_N" >/dev/null
  pre="$(awk -F, 'NR==2{print $4}' "$ptmp")"
  echo "native_before_c${C}_s${S},$pre" | tee -a "$PROBE_CSV"

  bash "${EXP_DIR}/scripts/lib/make_striped_copy.sh" "$MODEL" "$DEST" "$C" "$S" 16
  echo "--- read sweep on striped copy ---"
  bash "${EXP_DIR}/scripts/lib/dd_read_sweep.sh" "$DEST" "$BS" "$CSV" "$CONC"
  cat "$CSV"

  # Native probe again after, so the layout sweep is bracketed by two native
  # samples (detects/averages contention drift across the ~minutes it took).
  bash "${EXP_DIR}/scripts/lib/dd_read_sweep.sh" "$MODEL" "$BS" "$ptmp" "$PROBE_N" >/dev/null
  post="$(awk -F, 'NR==2{print $4}' "$ptmp")"
  echo "native_after_c${C}_s${S},$post" | tee -a "$PROBE_CSV"
  rm -f "$ptmp"

  # Safety: only ever delete inside EXP_BASE, never the source model.
  if [[ "$KEEP" != "1" && "$DEST" == "$EXP_BASE"/* && "$DEST" != "$MODEL" ]]; then
    echo "--- deleting striped copy $DEST ---"; rm -rf "$DEST"
  else
    echo "--- keeping striped copy $DEST ---"
  fi
done

echo "finished=$(date --iso-8601=seconds)"
