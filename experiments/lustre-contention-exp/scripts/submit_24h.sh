#!/bin/bash
# Lustre contention experiment - 24h campaign driver.
#
# Pre-submits NSLOTS independent probe jobs in one shot, each with an ABSOLUTE
# --begin timestamp INTERVAL_MIN apart (default 48 x 30 min = 24 h). Fire and
# forget: nothing needs to stay running, and a login-node disconnect does not
# stop the campaign - SLURM releases each job at its begin time.
#
# Fresh node per slot BY DESIGN: no --exclude, no --dependency. Each job lands on
# whatever node is free at its begin time. dd is O_DIRECT so there is no
# cross-node page-cache warming to worry about; node variance is accepted noise.
# (This is the deliberate OPPOSITE of the loading experiment's submit chains,
# which grew an --exclude list to force a *different* cold node each time and
# serialized to avoid contention. Here contention IS the signal.)
#
# Run from the repo root so SLURM_SUBMIT_DIR resolves the relative paths inside
# the sbatch:
#   bash experiments/lustre-contention-exp/scripts/submit_24h.sh
# Env overrides: NSLOTS, INTERVAL_MIN, plus anything the probe reads (MODEL,
# REPEATS, CONC, BS) via --export=ALL.
set -euo pipefail

NSLOTS="${NSLOTS:-48}"          # 48 * 30 min = 24 h
INTERVAL_MIN="${INTERVAL_MIN:-30}"
SBATCH_SCRIPT="experiments/lustre-contention-exp/scripts/contention_probe.sbatch"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
  echo "ERROR: run from the repo root; ${SBATCH_SCRIPT} not found here ($(pwd))" >&2
  exit 1
fi

echo "submitting ${NSLOTS} slots, one every ${INTERVAL_MIN} min"
echo "campaign span: $(( (NSLOTS - 1) * INTERVAL_MIN )) min (~$(awk -v n=$((NSLOTS-1)) -v i=${INTERVAL_MIN} 'BEGIN{printf "%.1f", n*i/60}') h)"
echo

for k in $(seq 0 $((NSLOTS - 1))); do
  begin="$(date -d "+$((k * INTERVAL_MIN)) minutes" +%Y-%m-%dT%H:%M:%S)"
  slot="$(printf '%02d' "${k}")"
  jid="$(sbatch --parsable \
        --begin="${begin}" \
        --job-name="lc-slot${slot}" \
        --export=ALL,SLOT="${k}" \
        "${SBATCH_SCRIPT}")"
  echo "slot ${slot} -> job ${jid}  begin=${begin}"
done

echo
echo "all ${NSLOTS} jobs queued. watch with:  squeue --me --sort=S -o '%.10i %.12j %.9T %.20S %R'"
echo "results land in experiments/lustre-contention-exp/results/ ; plot with plot_contention.py after the window."
