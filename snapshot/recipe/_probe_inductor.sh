#!/bin/bash
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
TORCH=/usr/local/lib/python3.12/dist-packages/torch
echo "=== vLLM envs.py: cache vars ==="
grep -niE "cache" $VLLM/envs.py 2>/dev/null | head -10
echo ""
echo "=== vLLM sets inductor cache_dir? ==="
grep -rn "TORCHINDUCTOR\|inductor.*cache_dir\|config.cache_dir\|\.cache_dir =" $VLLM/compilation/*.py 2>/dev/null | head -10
echo ""
echo "=== vLLM compilation cache (save/load) ==="
grep -rn "torch_compiled\|compiled_artifact\|save_compiled\|compilation_cache\|use_inductor\|inductor_cache" $VLLM/compilation/*.py 2>/dev/null | head -10
echo ""
echo "=== torch inductor: how cache_dir is read ==="
grep -rn "TORCHINDUCTOR_CACHE_DIR\|cache_dir" $TORCH/_inductor/config.py 2>/dev/null | head -8
echo ""
echo "=== torch inductor: current cache_dir value ==="
python3 -c "
import os
print('env TORCHINDUCTOR_CACHE_DIR =', repr(os.environ.get('TORCHINDUCTOR_CACHE_DIR')))
import torch._inductor.config as c
for attr in dir(c):
    if 'cache' in attr.lower():
        print('config.'+attr, '=', repr(getattr(c, attr, '?')))
" 2>&1 | head -12
