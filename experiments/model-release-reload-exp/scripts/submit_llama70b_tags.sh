#!/bin/bash
# Llama-70B: does releasing the KV cache change whether the reload survives?
#
# THE QUESTION. The 70B failed with `weights`-only (job 75115: reload reports
# success, model emits garbage). The Apertus-8B matrix then showed the tag set is
# NOT the trigger there - `weights`, `+kv_cache`, `+cuda_graph` and the no-release
# control were all 6/6 byte-identical. So either the 70B behaves differently, or the
# trigger is model-specific. This asks the tag question directly on the model that
# actually fails.
#
# WHY RE-RUN `w` TOO. We have seen the 70B garbage exactly ONCE. A one-off on a
# 20-minute job is not a finding, and a `w_kv` result is uninterpretable without a
# same-day `w` next to it. So `w` is a reproduction check, not a formality.
#
# CONCURRENT, NOT SERIALIZED - deliberately, and opposite to submit_chain.sh.
# That chain serialized and forced distinct nodes because it measured TIME, where
# capstor contention and page cache are confounds. This measures CORRECTNESS, which
# is binary: a warm page cache cannot turn garbage into Paris. So both points run at
# once and the answer arrives in ~20 min instead of ~45.
#   -> The COST: the two jobs read 141 GB each simultaneously, so their weight-load
#      and reload TIMINGS are mutually contended and NOT comparable to the numbers in
#      NOTES.md. Ignore the seconds here; read only the verdict.
#
# No dd probe (DD_PROBE=0): correctness question, and per lustre-contention-exp the
# dd aggregate is a max over workers, i.e. a measurement of the worst OST - not the
# instrument to reach for anyway.
#
# Run from the repo root:
#   bash experiments/model-release-reload-exp/scripts/submit_llama70b_tags.sh
set -euo pipefail

S="experiments/model-release-reload-exp/scripts/release_reload.sbatch"
[ -f "$S" ] || { echo "ERROR: run from the repo root; $S not found here ($(pwd))" >&2; exit 1; }
mkdir -p experiments/model-release-reload-exp/results

submit() {  # submit <tag> <release_tags>
  local tag="$1" tags="$2" jid
  # export + plain --export=ALL: `--export=ALL,RELEASE_TAGS=weights,kv_cache` splits
  # on commas and silently truncates to `weights`. That turned the first Apertus matrix
  # into duplicates of one point. release_reload.sbatch now also guards against it.
  export TAG="$tag" RUN_PREFIX=mrr70b MEMORY_SAVER=1 DO_CYCLE=1 DO_RELEASE=1 \
         RELEASE_TAGS="$tags" DD_PROBE=0
  jid="$(sbatch --parsable --job-name="mrr70b-$tag" "$S")"
  printf '%-6s -> %-7s tags=%s\n' "$tag" "$jid" "$tags"
}

echo "Llama-3.1-70B tag arms (TP=4, fastsafetensors, memory saver ON, concurrent)"
submit w    weights
submit w_kv weights,kv_cache
echo
echo "read: bash experiments/model-release-reload-exp/scripts/summarize_tags.sh   # (globs mrr8b-*)"
echo "      RES_GLOB=mrr70b bash experiments/model-release-reload-exp/scripts/summarize_tags.sh"
