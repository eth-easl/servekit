#!/bin/bash
# Does `dd iflag=direct` actually give a COLD read on capstor?
#
# O_DIRECT bypasses the CLIENT page cache. It says nothing about the Lustre
# OSS (server) read cache. If the OSS caches, then re-reading the same bytes
# gets faster even under O_DIRECT - and every "cold read" number in Phases
# 1.2 / 2 is really a warm read of unknown temperature.
#
# Test: single-stream O_DIRECT read of the SAME 4 GiB, 3 times back to back.
#   pass1 = cold (for a file nobody has touched)
#   pass2/3 = immediately re-read -> if much faster, OSS cache is live.
#
# Also reads a Llama-70B shard we have hammered for 2 days: if OSS cache
# persists across jobs/hours, its pass1 is already fast (no cold penalty).
#
# Usage: oss_cache_probe.sh [gib_per_pass] [passes]
set -euo pipefail

GIB="${1:-4}"
PASSES="${2:-3}"
BS=16M
COUNT=$(( GIB * 64 ))   # 16M * 64 = 1 GiB

M=/capstor/store/cscs/swissai/infra01/hf_models/models

# label:path  -- UNTOUCHED = never read by this project; HOT = read many times
FILES=(
  "UNTOUCHED_codellama:$M/codellama/CodeLlama-70b-hf/model-00023-of-00029.safetensors"
  "UNTOUCHED_tower72b:$M/Unbabel/Tower-Plus-72B/model-00013-of-00031.safetensors"
  "UNTOUCHED_emu3:$M/BAAI/Emu3-Chat-hf/model-00006-of-00008.safetensors"
  "HOT_llama70b:$M/meta-llama/Llama-3.1-70B-Instruct/model-00001-of-00030.safetensors"
)

echo "host=$(hostname) per_pass=${GIB}GiB bs=$BS passes=$PASSES (single stream, O_DIRECT)"
echo
printf '%-24s %8s %10s %10s\n' "file" "pass" "wall_s" "GB/s"
echo "------------------------------------------------------------"

for entry in "${FILES[@]}"; do
  label="${entry%%:*}"; path="${entry#*:}"
  [[ -r "$path" ]] || { printf '%-24s  UNREADABLE %s\n' "$label" "$path"; continue; }
  for p in $(seq 1 "$PASSES"); do
    start="$(date +%s.%N)"
    dd if="$path" of=/dev/null bs="$BS" count="$COUNT" iflag=direct status=none
    end="$(date +%s.%N)"
    read -r wall gbps < <(awk -v s="$start" -v e="$end" -v c="$COUNT" \
      'BEGIN{w=e-s; b=c*16*1048576; printf "%.2f %.2f\n", w, b/w/1e9}')
    printf '%-24s %8s %10s %10s\n' "$label" "$p" "$wall" "$gbps"
  done
  echo
done
