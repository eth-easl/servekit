#!/bin/bash
# Tests whether the hipMalloc redirect transparently captures real PyTorch
# allocations: runs a tiny torch program baseline, then twice under the
# redirect lib. If torch's caching allocator uses hipMalloc, x.data_ptr() lands
# in the deterministic region and should match across the two redirected runs.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
REDIR="$PWD/snapshot/build/libsnapshot_redirect.so"
PY=snapshot/recipe/_torch_redirect_test.py

echo "=== torch baseline (real hipMalloc) ==="
python3 "$PY" || echo "(baseline failed)"
echo "=== torch redirected run 1 ==="
LD_PRELOAD="$REDIR" python3 "$PY" || echo "(redirected run 1 failed)"
echo "=== torch redirected run 2 (data_ptr should equal run 1) ==="
LD_PRELOAD="$REDIR" python3 "$PY" || echo "(redirected run 2 failed)"
