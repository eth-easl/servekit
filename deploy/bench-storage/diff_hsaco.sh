#!/bin/bash
# Analyze HSACO dumps from a recording: ELF sections, metadata, and code size.
# Usage: diff_hsaco.sh inspect <dump.co>
#        diff_hsaco.sh diff <dump1.co> <dump2.co>  (same-named kernel, two runs)
set -uo pipefail
D1="${1:-}"; D2="${2:-}"
mode="${0##*diff_hsaco.sh:}"
cmd="$1"; shift || true

if [ "$cmd" = "inspect" ]; then
  F="$1"
  echo "=== $F ($(stat -c%s "$F") bytes) ==="
  echo "--- ELF header ---"
  readelf -h "$F" 2>/dev/null | sed -n '1,12p'
  echo "--- sections ---"
  readelf -S "$F" 2>/dev/null | grep -E '\] \.(text|note|symtab|debug|rodata)|\.AMDGPU|Name|\[Nr\]' | head -30
  echo "--- NT_AMDGPU_METADATA note (first 400 chars) ---"
  readelf -n "$F" 2>/dev/null | head -c 400
  echo
  echo "--- .text section bytes ---"
  readelf -S "$F" 2>/dev/null | grep -A1 '\.text' | head -3
elif [ "$cmd" = "diff" ]; then
  A="$1"; B="$2"
  echo "=== diff $(basename "$A")  vs  $(basename "$B") ==="
  echo "A=$(stat -c%s "$A") bytes  B=$(stat -c%s "$B") bytes"
  if cmp -s "$A" "$B"; then
    echo "RESULT: BYTE-IDENTICAL (code+metadata stable)"
    exit 0
  fi
  echo "--- first byte difference ---"
  cmp "$A" "$B" | head -1
  echo "--- per-section hash comparison (md5 of each section's bytes) ---"
  for sec in .text .note .note.amdgpu .symtab .strtab .debug_* .rodata .comment; do
    ha=$(objcopy -O binary --only-section="$sec" "$A" /dev/stdout 2>/dev/null | md5sum | cut -d' ' -f1)
    hb=$(objcopy -O binary --only-section="$sec" "$B" /dev/stdout 2>/dev/null | md5sum | cut -d' ' -f1)
    [ -n "$ha" ] || [ -n "$hb" ] || continue
    tag="SAME"; [ "$ha" != "$hb" ] && tag="DIFFER"
    printf '  %-22s %s  %s  %s\n' "$sec" "${ha:-none}" "${hb:-none}" "$tag"
  done
  echo "--- disassembly diff (first kernel, .text) ---"
  diff <(objdump -d "$A" 2>/dev/null) <(objdump -d "$B" 2>/dev/null) | head -25
fi
