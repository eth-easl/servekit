#!/bin/bash
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
echo "=== envs.py cache vars ==="
grep -niE "cache" $VLLM/envs.py 2>/dev/null | head -14
echo ""
echo "=== VLLM_CACHE_ROOT / inductor cache config in compilation ==="
grep -rn "VLLM_CACHE_ROOT\|TORCHINDUCTOR\|_inductor.config.cache\|cache_dir" $VLLM/compilation/*.py 2>/dev/null | head -14
echo ""
echo "=== torch inductor default cache_dir ==="
python3 -c "import torch._inductor.config as c; print('cache_dir=', repr(getattr(c,'cache_dir','NOATTR')))" 2>&1 | tail -1
echo "=== current TORCHINDUCTOR env ==="
echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-<unset>}"
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-<unset>}"
