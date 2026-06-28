#!/bin/bash
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== compiler_interface.py:455-485 (base_cache_dir derivation) ==="
sed -n '455,485p' $VLLM/compilation/compiler_interface.py
echo ""
echo "=== backends.py:1048-1072 (cache_dir setup) ==="
sed -n '1048,1072p' $VLLM/compilation/backends.py
echo ""
echo "=== where is VLLM_CACHE_ROOT read into base_cache_dir? ==="
grep -rn "VLLM_CACHE_ROOT\|base_cache_dir\|cache_root" $VLLM/compilation/compiler_interface.py 2>/dev/null | head -8
