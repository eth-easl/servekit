#!/usr/bin/env python3
"""2x2 microbenchmark: {pageable, pinned-staged} x {contiguous, strided} tmpfs->GPU.

Isolates the two factors the py-spy profile implicates in the 19-27 s
weight_loading window (see ../NOTES.md), with no engine involved:

  * PINNING    - the loader copies from pageable mmap'd tmpfs, so the driver
                 must bounce every byte through its own small pinned buffer.
  * CONTIGUITY - RowParallelLinear narrows on input_dim, so a rank's down_proj
                 slice is 8192 separate ~14 KB fragments at stride 28672, while
                 the column-parallel slices are one contiguous 117 MB block.
                 Measured time/byte differs ~5.3x between the two.

Geometry is the real thing: Llama-3.1-70B down_proj is [8192, 28672] bf16
(470 MB). At TP=4 a rank's slice is 117 MB either way --
  contiguous: narrow(0, ...) -> [2048, 28672]
  strided   : narrow(1, ...) -> [8192,  7168]
so the two cases move identical bytes and differ ONLY in layout.

Faithfulness notes, both of which cost real accuracy if skipped:
  1. The backing file is WRITTEN, not sparse. A sparse tmpfs file reads back
     through the shared zero page, so every "copy" would hit one cached page
     and report a fantasy bandwidth.
  2. Each repetition uses a DIFFERENT tensor, over a FRESHLY mmap'd region.
     The real loader touches each weight page exactly once, paying a minor
     fault to map an already-resident tmpfs page into its address space.
     Re-copying one hot region would measure a steady state that never happens.

The sharp prediction under test: explicit staging should help a LOT on the
strided layout and BARELY AT ALL on the contiguous one. The driver already
bounces pageable copies through its own pinned buffer and pipelines them, so
for a contiguous source explicit staging just redoes the driver's work. The
strided source is different -- torch must first materialize a contiguous
*pageable* temp, which is then bounced again, so the bytes get walked ~3x
instead of 1x. If contiguous also shows a big win, this model is WRONG and
the cause is something else (e.g. driver staging-chunk size).

The `gather` and `dma_only` cells exist to decompose that: `gather` is the
CPU-side cost alone, `dma_only` is the wire-speed ceiling from
already-pinned memory. A staged mode can never beat `dma_only`, and can
never be faster than `gather` allows.

Also measures cudaHostRegister throughput, which decides the separate
"can we just pin the whole model in /dev/shm?" question: pinning 141 GB per
rank only beats a small reusable staging buffer if registration runs at
implausible speed (see NOTES.md for the break-even arithmetic).

Usage: h2d_microbench.py [--path /dev/shm/h2dbench] [--tensors 16] [--gpu 0]
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

ROWS, COLS = 8192, 28672  # Llama-3.1-70B down_proj
TP = 4
DTYPE = torch.bfloat16
ITEMSIZE = 2
TENSOR_BYTES = ROWS * COLS * ITEMSIZE          # 470 MB
SLICE_BYTES = TENSOR_BYTES // TP               # 117 MB
BATCH_TENSORS = 8                              # 8 x 117 MB = ~940 MB staging buffer


def build_file(path, n_tensors):
    """Write a real (non-sparse) backing file; see faithfulness note 1."""
    need = n_tensors * TENSOR_BYTES
    if os.path.exists(path) and os.path.getsize(path) == need:
        return need
    chunk = b"\xab" * (64 << 20)
    with open(path, "wb") as f:
        written = 0
        while written < need:
            n = min(len(chunk), need - written)
            f.write(chunk[:n])
            written += n
    return need


def fresh_map(path, n_tensors):
    """A NEW mapping each time -> empty PTEs -> honest minor faults on touch."""
    t = torch.from_file(
        path, shared=True, size=n_tensors * ROWS * COLS, dtype=DTYPE
    )
    return t.view(n_tensors, ROWS, COLS)


def make_slice(mm, idx, layout, rank=0):
    t = mm[idx]
    if layout == "contig":
        return t.narrow(0, rank * (ROWS // TP), ROWS // TP)
    return t.narrow(1, rank * (COLS // TP), COLS // TP)


def bench(path, n_tensors, layout, mode, dev, rank=0):
    mm = fresh_map(path, n_tensors)             # fresh mapping per cell
    shape = make_slice(mm, 0, layout, rank).shape
    dst = torch.empty(shape, dtype=DTYPE, device=dev)

    if mode == "pageable":
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_tensors):
            dst.copy_(make_slice(mm, i, layout, rank))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    elif mode == "pinned":
        buf = torch.empty(shape, dtype=DTYPE, pin_memory=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_tensors):
            buf.copy_(make_slice(mm, i, layout, rank))  # tmpfs -> pinned, de-strides
            dst.copy_(buf, non_blocking=True)      # pinned -> GPU, real DMA
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    elif mode == "pinned2x":
        # Double-buffered: stage tensor N+1 on the CPU while tensor N's DMA is
        # still in flight. This is the design NOTES.md actually recommends.
        bufs = [torch.empty(shape, dtype=DTYPE, pin_memory=True) for _ in range(2)]
        evs = [torch.cuda.Event() for _ in range(2)]
        for e in evs:
            e.record()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_tensors):
            k = i % 2
            evs[k].synchronize()                   # buffer k's last DMA done?
            bufs[k].copy_(make_slice(mm, i, layout, rank))
            dst.copy_(bufs[k], non_blocking=True)
            evs[k].record()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    elif mode == "gather":
        # CPU side ONLY: strided tmpfs -> contiguous pinned, no GPU involved.
        # This is the floor that any staging design has to pay.
        buf = torch.empty(shape, dtype=DTYPE, pin_memory=True)
        t0 = time.perf_counter()
        for i in range(n_tensors):
            buf.copy_(make_slice(mm, i, layout, rank))
        dt = time.perf_counter() - t0

    elif mode == "dma_only":
        # GPU side ONLY: source is ALREADY pinned and resident, so this is the
        # pure DMA ceiling with no host copy in the path at all. Nothing can
        # beat this number; how close the staged modes get to it says whether
        # the host side or the wire is the limit.
        src = torch.empty(shape, dtype=DTYPE, pin_memory=True)
        src.copy_(make_slice(mm, 0, layout, rank))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_tensors):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    elif mode in ("batch", "batch_mt"):
        # Stage K slices packed contiguously into ONE big pinned buffer, do a
        # single DMA, then split them apart on the GPU (D2D, ~1.5 TB/s, free).
        # The point is not fewer DMAs -- events already overlap those -- it is
        # that disjoint regions of one buffer can be gathered CONCURRENTLY,
        # which parallelizes the gather that the solo/4-proc runs identified
        # as the actual bottleneck.
        K = min(BATCH_TENSORS, n_tensors)
        n_elem = shape.numel()
        big = torch.empty(K * n_elem, dtype=DTYPE, pin_memory=True)
        gbuf = torch.empty(K * n_elem, dtype=DTYPE, device=dev)
        dsts = [torch.empty(shape, dtype=DTYPE, device=dev) for _ in range(K)]
        pool = ThreadPoolExecutor(max_workers=K) if mode == "batch_mt" else None

        def stage(j, i):
            big[j * n_elem:(j + 1) * n_elem].view(shape).copy_(
                make_slice(mm, i, layout, rank))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for base in range(0, n_tensors, K):
            idxs = list(range(base, min(base + K, n_tensors)))
            if pool is not None:
                list(pool.map(lambda p: stage(*p), enumerate(idxs)))
            else:
                for j, i in enumerate(idxs):
                    stage(j, i)
            gbuf.copy_(big, non_blocking=True)          # ONE big H2D
            for j in range(len(idxs)):                  # split on GPU (D2D)
                dsts[j].copy_(
                    gbuf[j * n_elem:(j + 1) * n_elem].view(shape),
                    non_blocking=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if pool is not None:
            pool.shutdown()

    moved = n_tensors * SLICE_BYTES
    del mm
    return moved / dt / 1e9, dt


def bench_register(path, n_tensors, gb=2.0):
    """cudaHostRegister throughput -> decides the pin-the-whole-model question."""
    mm = fresh_map(path, n_tensors)
    flat = mm.view(-1)
    nbytes = min(int(gb * 1e9), flat.numel() * ITEMSIZE)
    nbytes -= nbytes % 4096
    ptr = flat.data_ptr()
    cudart = torch.cuda.cudart()

    t0 = time.perf_counter()
    rc = cudart.cudaHostRegister(ptr, nbytes, 0)
    dt = time.perf_counter() - t0
    if int(rc) != 0:
        return None, f"cudaHostRegister failed rc={int(rc)}"
    cudart.cudaHostUnregister(ptr)
    del mm
    return nbytes / dt / 1e9, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/dev/shm/h2dbench.bin")
    ap.add_argument("--tensors", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="torch.set_num_threads; 0 = leave default. 4 procs x 64 "
                         "default threads on 64 cores thrashes -- see NOTES.md")
    a = ap.parse_args()

    if a.torch_threads:
        torch.set_num_threads(a.torch_threads)
    torch.cuda.set_device(a.gpu)
    dev = torch.device(f"cuda:{a.gpu}")
    torch.empty(1, device=dev)  # force context creation outside the timers

    total = build_file(a.path, a.tensors)
    pre = f"[{a.tag}] " if a.tag else ""
    print(f"{pre}gpu={a.gpu} threads={torch.get_num_threads()} file={a.path} {total/1e9:.1f} GB "
          f"({a.tensors} x {TENSOR_BYTES/1e6:.0f} MB tensors, "
          f"slice={SLICE_BYTES/1e6:.0f} MB)", flush=True)

    results = {}
    for layout in ("contig", "strided"):
        for mode in ("pageable", "pinned", "pinned2x", "gather", "dma_only",
                     "batch", "batch_mt"):
            gbps, dt = bench(a.path, a.tensors, layout, mode, dev, a.rank)
            results[(layout, mode)] = gbps
            print(f"{pre}  {layout:<8} {mode:<9} {gbps:6.2f} GB/s   ({dt:.2f}s)",
                  flush=True)

    reg, err = bench_register(a.path, a.tensors)
    if err:
        print(f"{pre}  cudaHostRegister: {err}", flush=True)
    else:
        # Break-even from NOTES.md: whole-model pinning needs ~40+ GB/s to be
        # interesting, and >8 GB/s merely to beat today's baseline.
        print(f"{pre}  cudaHostRegister {reg:6.2f} GB/s  "
              f"-> 141 GB would take {141/reg:6.1f}s", flush=True)

    base = results[("strided", "pageable")]
    best = max(results.values())
    print(f"{pre}SUMMARY strided/pageable (today) = {base:.2f} GB/s; "
          f"best = {best:.2f} GB/s; speedup = {best/base:.1f}x", flush=True)


if __name__ == "__main__":
    main()
