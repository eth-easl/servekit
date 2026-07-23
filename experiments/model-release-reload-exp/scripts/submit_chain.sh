#!/bin/bash
# Submit the two points, serially, on distinct nodes, after the capstor is quiet.
#
# WHY THIS EXISTS AT ALL (the two points are otherwise plain sbatch calls):
#
# 1. DON'T COLLIDE WITH THE CONTENTION CAMPAIGN. lustre-contention-exp pre-submits
#    48 slots that each read the whole 132 GB model with 32 O_DIRECT workers. That
#    is cross-node contention on the shared capstor OSTs - it would skew our
#    weight_loading, and our 141 GB read would skew their bandwidth sample right
#    back. The gaps between slots are ~10 min; our jobs need more. So we wait for
#    the last slot to finish. The begin time is computed FROM THE QUEUE, not
#    hardcoded, so this stays right if the campaign is extended or cancelled.
#
# 2. THE SECOND JOB MUST NOT REUSE THE FIRST JOB'S NODE. These nodes have ~515 GB
#    of RAM and the shards are ~141 GB, so after job 1 the whole model sits in the
#    page cache. Job 2 landing there would read its "cold" start out of RAM and
#    the ctl-vs-saver comparison would be garbage. `--exclusive` grants sole use
#    of a node, NOT a different node, and a --dependency chain hands the same one
#    straight back (this cost lustre-loading-exp an entire sweep). The only
#    reliable fix is to learn job 1's node and --exclude it - which can't be done
#    at submit time. Hence: submit, wait, learn, submit.
#
# Serialization is also required in its own right: two concurrent 141 GB reads
# would contend with each other exactly like the campaign would.
#
# Run from the repo root. Survives a disconnect:
#   cd <repo root> && nohup bash experiments/model-release-reload-exp/scripts/submit_chain.sh \
#        > experiments/model-release-reload-exp/results/chain.log 2>&1 &
set -euo pipefail

S="experiments/model-release-reload-exp/scripts/release_reload.sbatch"
RES="experiments/model-release-reload-exp/results"
[ -f "$S" ] || { echo "ERROR: run from the repo root; $S not found here ($(pwd))" >&2; exit 1; }
mkdir -p "$RES"

# Gate: start after the last queued contention slot ends (+ margin). If the
# campaign is already done, go now.
MARGIN_S="${MARGIN_S:-300}"
LAST=""
for j in $(squeue -u "$USER" -h -o "%i %j" | awk '$2 ~ /^lc-slot/ {print $1}'); do
  e="$(scontrol show job "$j" 2>/dev/null | grep -oP 'EndTime=\K\S+' || true)"
  [ -n "$e" ] && LAST="$(printf '%s\n%s\n' "$LAST" "$e" | sort | tail -1)"
done

BEGIN_ARG=()
if [ -n "$LAST" ]; then
  BEGIN="$(date -d "@$(( $(date -d "$LAST" +%s) + MARGIN_S ))" +%Y-%m-%dT%H:%M:%S)"
  BEGIN_ARG=(--begin="$BEGIN")
  echo "contention campaign runs until $LAST -> first job begins $BEGIN"
else
  echo "no lc-slot jobs queued -> starting now"
fi

USED="${EXCLUDE_NODES:-}"

submit_and_wait() {  # submit_and_wait <tag> <memory_saver> <do_cycle> [extra sbatch args...]
  local tag="$1" saver="$2" cycle="$3"; shift 3
  local jid node state exclude_arg=()
  [ -n "$USED" ] && exclude_arg=(--exclude="$USED")

  jid="$(sbatch --parsable "${exclude_arg[@]}" "$@" \
        --job-name="mrr-$tag" \
        --export=ALL,TAG="$tag",MEMORY_SAVER="$saver",DO_CYCLE="$cycle" \
        "$S")"
  printf '%-12s -> %-7s saver=%s cycle=%s %s\n' "$tag" "$jid" "$saver" "$cycle" "${exclude_arg[*]:-}"

  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 30
  done
  node="$(sacct -j "$jid" -X -n -o NodeList | head -1 | xargs)"
  state="$(sacct -j "$jid" -X -n -o State | head -1 | xargs)"
  echo "    -> finished on $node ($state)"
  [ -n "$node" ] && [ "$node" != "None" ] && USED="${USED:+$USED,}$node"
}

# Control first: it prices what --enable-memory-saver costs at cold start, and it
# must be measured here rather than remembered (the same config has measured
# 69.5-86.1 s across nodes).
submit_and_wait ctl_nosaver 0 0 "${BEGIN_ARG[@]}"
submit_and_wait saver_cycle 1 1

echo
echo "nodes used (must be distinct): $USED"
echo "results: $RES"
