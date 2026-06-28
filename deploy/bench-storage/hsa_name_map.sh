#!/bin/bash
# Extract, for every HSACO dump in a directory:
#   kernel_name  text_size  text_md5  file_hash
# where:
#   kernel_name = FUNC symbol from .symtab (readelf -sW), fully demangled-ish
#   text_size   = .text section size (hex), from readelf -SW
#   text_md5    = md5 of the .text section bytes (dd at the section offset)
#   file_hash   = the 0x... hash in the filename (the recorder's hash_bytes)
#
# Two dumps with the same kernel_name and same text_md5 are byte-identical code;
# same name but different text_md5 = autotune picked a different config.
#
# Usage: hsa_name_map.sh <dump_dir> [label]
set -uo pipefail
DIR="${1:-}"
LABEL="${2:-$(basename "$DIR")}"
[ -d "$DIR" ] || { echo "no dir: $DIR"; exit 1; }

printf '# name\ttext_size\ttext_md5\tfile_hash\t%s\n' "$LABEL"
for f in "$DIR"/hsa-dump-*.co; do
  [ -e "$f" ] || continue
  fhash=$(basename "$f" | sed -E 's/hsa-dump-(0x[0-9a-f]+)\.co/\1/')
  # .text offset + size
  read ts_off ts_sz < <(readelf -SW "$f" 2>/dev/null | awk '/\.text /{print strtonum($5), strtonum($6); exit}')
  ts_off=${ts_off:-0}; ts_sz=${ts_sz:-0}
  if [ "$ts_sz" -gt 0 ]; then
    tmd5=$(dd if="$f" bs=1 skip="$ts_off" count="$ts_sz" 2>/dev/null | md5sum | cut -d' ' -f1)
  else
    tmd5="(no .text)"
  fi
  # FUNC symbol name (full width). Triton kernels appear as a single FUNC entry.
  kname=$(readelf -sW "$f" 2>/dev/null | awk '$4=="FUNC"{print $8; exit}')
  kname=${kname:-"(unknown)"}
  printf '%s\t0x%x\t%s\t%s\n' "$kname" "$ts_sz" "$tmd5" "$fhash"
done | sort
