#!/bin/bash
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tr ':' '\n' | head -5
echo "---"
LIB=$PWD/snapshot/build
ls -la $LIB/libsnapshot_record.so $LIB/libsnapshot_redirect.so
echo "---"
ldd $LIB/libsnapshot_record.so 2>&1 | grep -i "amdhip\|not found"
echo "---"
export LD_PRELOAD="$LIB/libsnapshot_redirect.so $LIB/libsnapshot_record.so"
echo "testing date..."
date
echo "date OK"
echo "testing python..."
python -c "import hip" 2>&1 | head -3 || python --version 2>&1
echo "done"
