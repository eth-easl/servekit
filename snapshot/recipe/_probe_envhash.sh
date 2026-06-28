#!/bin/bash
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== envs.compile_factors definition ==="
grep -n "def compile_factors" $VLLM/envs.py
grep -n "compile_factors" $VLLM/envs.py | head
echo "--- body ---"
START=$(grep -n "def compile_factors" $VLLM/envs.py | head -1 | cut -d: -f1)
sed -n "${START},$((START+30))p" $VLLM/envs.py
