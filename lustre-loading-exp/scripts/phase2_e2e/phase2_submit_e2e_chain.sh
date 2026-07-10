#!/bin/bash
# Phase 2 (e2e) driver: make copies, then run one e2e cold start per layout,
# each on its own fresh node, strictly serialized so no two runs contend on
# capstor at the same time.
#
# Order: copies -> native -> c4_s4M -> c8_s4M -> c16_s4M -> c8_s16M -> native
# The native baseline is run FIRST and LAST, bracketing the layouts, because
# capstor throughput drifts over tens of minutes (see NOTES Phase 1.2). Two
# native samples bound that drift; a single one cannot.
#
# Run from the repo root:  bash lustre-loading-exp/scripts/phase2_e2e/phase2_submit_e2e_chain.sh
set -euo pipefail

EXP_BASE="${EXP_BASE:-/capstor/store/cscs/swissai/infra01/cold-start-experiments}"
NATIVE="${NATIVE:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
S="lustre-loading-exp/scripts/phase2_e2e"

mkdir -p lustre-loading-exp/results/phase2_e2e

copy_jid="$(sbatch --parsable "$S/phase2_make_copies.sbatch")"
echo "make-copies      -> $copy_jid"

prev="$copy_jid"
submit() {  # submit <tag> <model_path>
  local tag="$1" path="$2" jid
  jid="$(sbatch --parsable --dependency=afterok:"$prev" \
        --job-name="p2e2e-$tag" --export=ALL,TAG="$tag",MODEL_PATH="$path" \
        "$S/phase2_e2e_layout.sbatch")"
  echo "e2e $tag  -> $jid  (after $prev)"
  prev="$jid"
}

submit native_first "$NATIVE"
submit c4_s4M       "$EXP_BASE/llama70b_c4_s4M"
submit c8_s4M       "$EXP_BASE/llama70b_c8_s4M"
submit c16_s4M      "$EXP_BASE/llama70b_c16_s4M"
submit c8_s16M      "$EXP_BASE/llama70b_c8_s16M"
submit native_last  "$NATIVE"

echo
echo "chain submitted. watch:  squeue -u \$USER"
echo "when finished, reclaim ~564 GB:  bash $S/phase2_rm_copies.sh"
