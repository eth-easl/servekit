#!/bin/bash
# Robust per-section comparison of two AMDGPU HSACO ELF files using readelf offset+size + dd.
# Avoids objcopy (which silently produced empty extractions for these files).
# Usage: sec_diff.sh <A.co> <B.co>
set -uo pipefail
A="$1"; B="$2"
echo "=== $(basename "$A") ($(stat -c%s "$A") B)  vs  $(basename "$B") ($(stat -c%s "$B") B) ==="

# get_section_bytes <file> <section_name> -> prints "offset size" (hex offsets in decimal)
section_off() {
  local file="$1" name="$2"
  # readelf -S line for a section: "[ N] .name TYPE address offset size ..."
  # Columns are space-separated but the section-name field may contain brackets.
  readelf -S -W "$file" 2>/dev/null \
    | awk -v want="$name" '
        /^\s*\[ *[0-9]+\]/ {
          # $1="[N]", $2=".name", $3=type, $4=addr, $5=offset, $6=size ...
          sec=$2; off=$5; sz=$6;
          if (sec == want) { print off, sz; found=1; exit }
        }
        END { if (!found) print "MISSING 0" }
      '
}

hash_range() {
  # hash_range <file> <offset> <size>
  local file="$1" off="$2" sz="$3"
  if [ "$off" = "MISSING" ] || [ "$sz" = "0" ] || [ -z "$sz" ]; then
    echo "(section absent)"; return
  fi
  dd if="$file" bs=1 skip="$((16#$off))" count="$((16#$sz))" 2>/dev/null | md5sum | cut -d' ' -f1
}

echo "--- section offset/size/hash (offset/size in hex from readelf) ---"
printf "  %-16s %-10s %-10s %-34s %-34s %s\n" "section" "off" "size" "A.md5" "B.md5" "status"
for sec in .text .note .rodata .AMDGPU.csdata .symtab .strtab .debug_info .debug_line .debug_str .comment; do
  ra=$(section_off "$A" "$sec"); rb=$(section_off "$B" "$sec")
  oa=${ra% *}; sa=${ra#* }; ob=${rb% *}; sb=${rb#* }
  ha=$(hash_range "$A" "$oa" "$sa"); hb=$(hash_range "$B" "$ob" "$sb")
  tag="SAME"; { [ "$ha" != "$hb" ] || [ "$oa" != "$ob" ] || [ "$sa" != "$sb" ]; } && tag="DIFFER"
  printf "  %-16s %-10s %-10s %-34s %-34s %s\n" "$sec" "${oa}/${sa}" "${ob}/${sb}" "$ha" "$hb" "$tag"
done

echo "--- .text instruction count (disassembly) ---"
na=$(objdump -d "$A" 2>/dev/null | grep -cE '^\s+[0-9a-f]+:')
nb=$(objdump -d "$B" 2>/dev/null | grep -cE '^\s+[0-9a-f]+:')
echo "  A: $na instructions   B: $nb instructions"

echo "--- disassembly first divergence (if .text differs) ---"
diff <(objdump -d "$A" 2>/dev/null | grep -E '^\s+[0-9a-f]+:') \
     <(objdump -d "$B" 2>/dev/null | grep -E '^\s+[0-9a-f]+:') | head -20
