#!/bin/bash
# Compare two hsa_name_map.sh TSV outputs by kernel name, on .text size + md5.
# Usage: hsa_compare.sh <mapA.tsv> <mapB.tsv> [labelA] [labelB]
set -uo pipefail
A="${1:?mapA}"; B="${2:?mapB}"; LA="${3:-A}"; LB="${4:-B}"
TAB="$(printf '\t')"

echo "############ $LA vs $LB : Triton kernel .text stability ############"
join -t "$TAB" -1 1 -2 1 \
  <(grep "^triton_" "$A" | sort) \
  <(grep "^triton_" "$B" | sort) \
| awk -F"$TAB" -v la="$LA" -v lb="$LB" '
    BEGIN { same=0; diff=0;
      print "kernel                                                         " la "_size  " lb "_size  verdict";
    }
    {
      identical = ($3 == $6 && $2 == $5);
      if (identical) { same++; tag="IDENTICAL"; } else { diff++; tag="DIFFER"; }
      printf "%-62s %-8s %-8s %s\n", substr($1,1,62), $2, $5, tag;
    }
    END {
      print "---";
      printf "TOTAL: %d    IDENTICAL: %d    DIFFER: %d\n", same+diff, same, diff;
    }'
