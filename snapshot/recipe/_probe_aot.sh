#!/bin/bash
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== decorators.py:300-330 (AOT load control) ==="
sed -n '300,330p' $VLLM/compilation/decorators.py
echo ""
echo "=== VLLM_FORCE_AOT_LOAD semantics ==="
grep -n "VLLM_FORCE_AOT_LOAD\|force_aot\|aot_load\|Directly load AOT" $VLLM/envs.py $VLLM/compilation/decorators.py 2>/dev/null | head -10
