#!/bin/bash
# Phase 1.1 - map each safetensors shard of a model to its Lustre OST(s).
# Usage: shard_ost_map.sh <model_dir>
# Prints, per shard: filename, stripe_count, stripe_size, OST index list.
# Then summarizes distinct OSTs used and any OST hosting >1 shard (hot OST).
set -euo pipefail

MODEL="${1:?usage: shard_ost_map.sh <model_dir>}"

printf '%-34s %6s %10s  %s\n' "shard" "count" "size" "ost_idx"
tmp="$(mktemp)"
for f in "$MODEL"/*.safetensors; do
  gs="$(lfs getstripe "$f" 2>/dev/null)"
  count="$(awk '/lmm_stripe_count/{print $2; exit}' <<<"$gs")"
  size="$(awk '/lmm_stripe_size/{print $2; exit}'  <<<"$gs")"
  # obdidx column holds the OST index for each stripe of the file
  osts="$(awk 'f{print $1} /obdidx/{f=1}' <<<"$gs" | paste -sd, -)"
  printf '%-34s %6s %10s  %s\n' "$(basename "$f")" "$count" "$size" "$osts"
  tr ',' '\n' <<<"$osts" >>"$tmp"
done

echo
echo "distinct OSTs used: $(sort -un "$tmp" | wc -l)"
echo "OSTs hosting >1 shard (hot):"
sort "$tmp" | uniq -c | awk '$1>1{printf "  OST %s: %s shards\n",$2,$1}'
rm -f "$tmp"
