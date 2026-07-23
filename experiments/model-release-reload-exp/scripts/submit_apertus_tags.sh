#!/bin/bash
# Apertus-8B: does releasing the KV cache change whether the reload survives?
#
# THE QUESTION. On Llama-70B we released `weights` ONLY and the reload produced
# garbage (see NOTES.md). Every upstream test releases ALL tags, so "weights alone"
# is an untested combination and a live suspect. This walks the tag set outward,
# one tag at a time, and brackets it with the control we never ran.
#
# WHY APERTUS-8B: identical image, identical EDF shape, and crucially **TP=4, the
# same as the 70B run** - the distributed path is a suspect, so it must not vary.
# Only the model changes. ~4 min/point instead of ~20, so the whole matrix costs
# less than one 70B job.
#
# THE MATRIX (each point differs from its neighbour by exactly one thing):
#
#   norelease   no release/resume, reload only    <- CONTROL: does the reload work AT ALL?
#   w           release weights                   <- reproduces the 70B failure on 8B
#   w_kv        release weights + kv_cache        <- THE QUESTION: does freeing KV matter?
#   all         release weights+kv_cache+cuda_graph  <- the only combo upstream tests
#
# HOW TO READ THE RESULT:
#   norelease garbage            -> update_weights_from_disk is broken on its own;
#                                   the memory saver was never involved. Everything
#                                   in NOTES.md about release/resume is a red herring.
#   norelease ok, w garbage      -> the release/resume cycle is the fault (matches
#                                   sglang#15246). Then w_kv/all say whether the tag
#                                   set is the trigger.
#   w garbage, w_kv/all ok       -> "weights-only" is the unsupported combination,
#                                   and we have a WORKAROUND, not just a bug.
#   all garbage                  -> even the upstream-tested tag set fails here ->
#                                   clean, high-value upstream report on v0.5.10.
#
# No dd probe (DD_PROBE=0): the question is correctness, not storage timing.
# Memory saver stays ON for every point, INCLUDING `norelease` - otherwise the
# control would differ from the others by two things instead of one.
#
# Fresh node per point is NOT needed here: this is a correctness question, and page
# cache only affects timing. We do NOT --exclude, so points can reuse nodes and the
# matrix finishes fast.
#
# Run from the repo root:
#   bash experiments/model-release-reload-exp/scripts/submit_apertus_tags.sh
set -euo pipefail

S="experiments/model-release-reload-exp/scripts/release_reload.sbatch"
RES="experiments/model-release-reload-exp/results"
[ -f "$S" ] || { echo "ERROR: run from the repo root; $S not found here ($(pwd))" >&2; exit 1; }
mkdir -p "$RES"

MODEL_8B=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509
export MODEL="${MODEL:-$MODEL_8B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-swiss-ai/Apertus-8B-Instruct-2509}"
export EDF_REL="profile/apertus-8b-bristen/apertus-8b-sglang.toml"

submit() {  # submit <tag> <do_release> <release_tags>
  local tag="$1" do_release="$2" tags="$3" jid
  # RELEASE_TAGS contains COMMAS, and `--export=ALL,K=V,...` treats commas as the
  # separator between variables -- `RELEASE_TAGS=weights,kv_cache` silently becomes
  # `RELEASE_TAGS=weights` plus two phantom variable names. That is not theoretical:
  # it is exactly how the first run of this matrix turned w_kv and all into duplicates
  # of w. So export into the environment and let plain `--export=ALL` carry it.
  export TAG="$tag" RUN_PREFIX=mrr8b MEMORY_SAVER=1 DO_CYCLE=1 \
         DO_RELEASE="$do_release" RELEASE_TAGS="$tags" DD_PROBE=0 \
         MODEL="$MODEL" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" EDF="$PWD/$EDF_REL"
  jid="$(sbatch --parsable --job-name="mrr8b-$tag" --time=00:25:00 --export=ALL "$S")"
  printf '%-10s -> %-7s do_release=%s tags=%s\n' "$tag" "$jid" "$do_release" "${tags:--}" >&2
  echo "$jid"
}

echo "Apertus-8B tag matrix (TP=4, memory saver ON everywhere, no dd probe)"
echo
J1=$(submit norelease 0 weights   | tail -1)
J2=$(submit w         1 weights   | tail -1)
J3=$(submit w_kv      1 weights,kv_cache | tail -1)
J4=$(submit all       1 weights,kv_cache,cuda_graph | tail -1)

echo
echo "submitted: $J1 $J2 $J3 $J4"
echo "watch:     squeue -u \$USER -n mrr8b-norelease,mrr8b-w,mrr8b-w_kv,mrr8b-all"
echo "read:      bash experiments/model-release-reload-exp/scripts/summarize_tags.sh"
