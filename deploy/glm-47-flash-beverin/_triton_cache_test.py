#!/usr/bin/env python3
"""Definitive Triton cross-process cache reuse test (isolated from vLLM).

Process 1: compile a known @triton.jit kernel -> writes bucket to TRITON_CACHE_DIR.
Process 2: fresh process, compile the SAME kernel, instrument FileCacheManager
           to report hit vs miss for each artifact lookup.

If process 2 HITS  -> Triton cache works cross-process; the vLLM cold-start
                      problem is vLLM-specific (per-run kernel source/options).
If process 2 MISSES-> Triton cache itself is broken on this image/FS.
"""
import os, sys, json, glob

CACHE_DIR = os.environ.get("TRITON_CACHE_DIR", "")
REPORT = os.environ.get("TRITON_TEST_REPORT", "")

# ---- instrument FileCacheManager BEFORE compiling ----
import triton.runtime.cache as _tc
hits, misses = [], []
_orig_get_file = _tc.FileCacheManager.get_file
_orig_get_group = _tc.FileCacheManager.get_group

def _patched_get_file(self, filename):
    r = _orig_get_file(self, filename)
    (hits if r else misses).append((os.path.basename(self.cache_dir), filename))
    return r

def _patched_get_group(self, filename):
    r = _orig_get_group(self, filename)
    (hits if r else misses).append((os.path.basename(self.cache_dir), f"<grp:{filename}>"))
    return r

_tc.FileCacheManager.get_file = _patched_get_file
_tc.FileCacheManager.get_group = _patched_get_group

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x + y, mask=mask)

def run():
    # triton_key() is the dominant component of the cache key
    tk = _tc.triton_key()
    print(f"[{os.environ.get('TRITON_TEST_TAG','?')}] TRITON_CACHE_DIR={CACHE_DIR}")
    print(f"[{os.environ.get('TRITON_TEST_TAG','?')}] triton_key()={tk[:96]}...")
    N = 4096
    x = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1
    y = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1
    o = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _add_kernel[grid](x, y, o, N, BLOCK=128)
    torch.cuda.synchronize()
    print(f"[{os.environ.get('TRITON_TEST_TAG','?')}] compiled OK; result sum={o.sum().item():.4f}")
    print(f"[{os.environ.get('TRITON_TEST_TAG','?')}] HITS={len(hits)} MISSES={len(misses)}")
    if misses:
        print(f"[{os.environ.get('TRITON_TEST_TAG','?')}] first misses: {misses[:3]}")
    if hits:
        print(f"[{os.environ.get('TRITON_TEST_TAG','?')}] first hits: {hits[:3]}")
    # dump buckets present for this kernel
    buckets = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CACHE_DIR, "*")) if os.path.isdir(p))
    print(f"[{os.environ.get('TRITON_TEST_TAG','?')}] buckets_now={len(buckets)}")
    if REPORT:
        json.dump({"hits": len(hits), "misses": len(misses),
                   "triton_key_prefix": tk[:64]}, open(REPORT, "w"))

run()
