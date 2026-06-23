#!/bin/bash
# Runs redirect_smoke three ways inside the container: baseline (real hipMalloc),
# then twice under libsnapshot_redirect.so. Redirected addresses should come from
# the deterministic region (0x6000...) and be identical across the two runs,
# while the computed result stays bit-identical in all three.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
SMOKE=snapshot/build/redirect_smoke
REDIR="$PWD/snapshot/build/libsnapshot_redirect.so"

echo "=== baseline (real hipMalloc) ==="
"$SMOKE"
echo "=== redirected run 1 ==="
LD_PRELOAD="$REDIR" SNAPSHOT_REDIRECT_VERBOSE=1 "$SMOKE"
echo "=== redirected run 2 (addresses should equal run 1) ==="
LD_PRELOAD="$REDIR" SNAPSHOT_REDIRECT_VERBOSE=1 "$SMOKE"
