#!/usr/bin/env python3
"""Where does the gap between 4.37 s (projected) and 8.47 s (measured) go?

Job 76364 got `weight_loading` to 8.47 s with NUMA binding and NUMA-aware
staging, using the stock loader. The stream microbench (76361) said a
NUMA-bound pageable copy of the same bytes should take 4.37 s. The difference
has to be per-tensor work that a 512 MiB-chunk stream benchmark does not
model, and there are only a few candidates. This measures them on the REAL
staged checkpoint, one rank's real files, 4 ranks concurrently, no engine.

Strategies, each a strict superset of the previous one's savings:

  safe_open   `safe_open(path).get_tensor(key)` then a pageable `copy_`.
              Exactly what ShardedStateLoader does today.
  zerocopy    mmap the file once, build tensor views over it with
              torch.frombuffer, then the same pageable `copy_`. Identical
              bytes on the wire; the only difference is whether safetensors
              materialises a host-side copy of each tensor first.
  pinned      zerocopy views, plus a reusable pinned staging buffer and a
              double-buffered async DMA. Under NUMA binding the stream bench
              made this the best mode (9.94 vs 8.08 GB/s); unbound it was the
              worst, which is what made the old sharded_pin patch a
              regression.

Every strategy is bit-exact-verified against the source before its number is
reported, per SPEC.md §5.
"""

import argparse
import glob
import json
import mmap
import multiprocessing as mp
import os
import sys
import time


def bind(node):
    base = f"/sys/devices/system/node/node{node}/cpulist"
    try:
        with open(base) as fh:
            spec = fh.read().strip()
    except OSError:
        return None
    cpus = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        elif part:
            cpus.append(int(part))
    if cpus:
        os.sched_setaffinity(0, set(cpus))
    return cpus


def run_rank(rank, node, args, barrier, results):
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import shm_safetensors as sst

    bound = bind(node) if node is not None else None
    # Without this the `pinned` gather runs at torch's default thread count on
    # every rank at once and collapses (measured 3.67 GB/s vs 9.94 in the
    # stream bench, which did cap it). The pageable strategies are unaffected
    # because the driver's internal bounce does not use OMP -- which is
    # exactly why the effect hides.
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    files = sorted(glob.glob(os.path.join(args.dir, f"model-rank-{rank}-part-*.safetensors")))
    if not files:
        results[rank] = {"error": f"no files for rank {rank} in {args.dir}"}
        return

    # Allocate every destination up front. The engine already has the model
    # built on the GPU when load starts, so allocation is not part of the
    # phase and must not be timed here either.
    plan = []          # (path, key, dst)
    total_bytes = 0
    for path in files:
        with open(path, "rb") as fh:
            head = fh.read(1 << 20)
        header, data_start = sst.parse_header(head)
        for key, meta in header.items():
            start, end = meta["data_offsets"]
            nbytes = end - start
            total_bytes += nbytes
            plan.append((path, key, nbytes))

    dsts = {}
    for path, key, nbytes in plan:
        dsts[(path, key)] = torch.empty(nbytes, dtype=torch.uint8, device=dev)

    chunk = args.chunk_mib << 20

    # The gather (tmpfs -> pinned) is the entire phase once NUMA is right:
    # 7.36 GB/s per rank, against a 26.7 GB/s DMA that is already fully
    # hidden behind it. The bounce costs ~3x DRAM traffic per byte -- read
    # the source, read-for-ownership the destination line, write it -- and a
    # non-temporal store elides that middle read. torch's copy_ does not emit
    # NT stores; glibc's memcpy does above its non_temporal_threshold, which
    # a 512 MiB chunk is comfortably past. Hence the memmove variants.
    _pool = None
    if args.gather == "torch":
        def gather(buf, view, off, m):
            buf[:m].copy_(view[off:off + m])
    elif args.gather == "memmove":
        import ctypes as _c

        def gather(buf, view, off, m):
            _c.memmove(buf.data_ptr(), view.data_ptr() + off, m)
    else:  # mt: split one chunk across threads; ctypes releases the GIL
        import ctypes as _c
        from concurrent.futures import ThreadPoolExecutor

        _pool = ThreadPoolExecutor(max_workers=args.gather_threads)

        def gather(buf, view, off, m):
            step = (m + args.gather_threads - 1) // args.gather_threads
            dst_p, src_p = buf.data_ptr(), view.data_ptr() + off
            futs = [
                _pool.submit(_c.memmove, dst_p + i, src_p + i, min(step, m - i))
                for i in range(0, m, step)
            ]
            for f in futs:
                f.result()

    if args.strategy == "pinned":
        bufs = [torch.empty(chunk, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        evs = [torch.cuda.Event() for _ in range(2)]
        for e in evs:
            e.record()
        slot = 0

    torch.cuda.synchronize()
    barrier.wait()
    t0 = time.perf_counter()

    reg_s = dma_s = 0.0

    if args.strategy == "register":
        # The bounce costs 3x DRAM traffic per byte (read tmpfs, write pinned,
        # DMA reads pinned). Registering the tmpfs mapping itself makes it 1x:
        # the DMA engine reads the weight pages directly. Whether that pays
        # depends entirely on cudaHostRegister's rate on 4 KB tmpfs pages,
        # which has only ever been measured UNBOUND (2.59-4.56 GB/s). This
        # measures it NUMA-local, and reports registration and DMA separately
        # so a slow register cannot hide inside a good total.
        import ctypes

        import exp2_cudart as cu

        stream = torch.cuda.current_stream().cuda_stream
        # Hold every mapping alive for the whole run. Letting one be collected
        # frees the address range, the kernel hands the same address to the
        # next mmap, and cudaHostRegister then returns 712
        # (cudaErrorHostMemoryAlreadyRegistered) against the stale
        # registration -- observed on file 3 of 7, every rank, job 76368.
        keepalive = []
        registered = []
        mapped = []
        for path in files:
            fd = os.open(path, os.O_RDWR)
            try:
                mm = mmap.mmap(fd, 0)  # MAP_SHARED, PROT_READ|PROT_WRITE
            finally:
                os.close(fd)
            keepalive.append(mm)
            header, data_start = sst.parse_header(mm)
            src = torch.frombuffer(mm, dtype=torch.uint8)
            base = src.data_ptr()

            mapped.append((path, header, data_start, base, src.numel()))

        # Registration is the whole cost: 9.7 s/rank serial (3.6 GB/s) against
        # a 1.33 s DMA off the registered mapping. Whole files are registered
        # as single regions -- fewest calls, and no tensor can straddle two
        # separately-registered regions (cudaErrorInvalidValue).
        t = time.perf_counter()
        if args.reg_threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            def _reg(item):
                return cu.libcudart.cudaHostRegister(
                    ctypes.c_void_p(item[3]), ctypes.c_size_t(item[4]), 0
                )

            with ThreadPoolExecutor(max_workers=args.reg_threads) as ex:
                rcs = list(ex.map(_reg, mapped))
        else:
            rcs = [cu.libcudart.cudaHostRegister(
                ctypes.c_void_p(m[3]), ctypes.c_size_t(m[4]), 0) for m in mapped]
        reg_s = time.perf_counter() - t
        for m, rc in zip(mapped, rcs):
            if rc != 0:
                results[rank] = {"error": f"cudaHostRegister rc={rc} on {m[0]}"}
                return

        for path, header, data_start, base, _n in mapped:
            t = time.perf_counter()
            for key, meta in header.items():
                start, end = meta["data_offsets"]
                dst = dsts[(path, key)]
                rc = cu.libcudart.cudaMemcpyAsync(
                    ctypes.c_void_p(dst.data_ptr()),
                    ctypes.c_void_p(base + data_start + start),
                    ctypes.c_size_t(end - start),
                    1,  # cudaMemcpyHostToDevice
                    ctypes.c_void_p(stream),
                )
                if rc != 0:
                    results[rank] = {"error": f"cudaMemcpyAsync rc={rc} key={key}"}
                    return
            cu.libcudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
            dma_s += time.perf_counter() - t
            registered.append(base)

        for base in registered:
            cu.libcudart.cudaHostUnregister(ctypes.c_void_p(base))
    elif args.strategy == "safe_open":
        from safetensors import safe_open

        for path in files:
            with safe_open(path, framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    t = f.get_tensor(key)
                    dsts[(path, key)].copy_(t.contiguous().view(torch.uint8).flatten())
    else:
        for path in files:
            fd = os.open(path, os.O_RDONLY)
            try:
                mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
            finally:
                os.close(fd)
            header, data_start = sst.parse_header(mm)
            src = torch.frombuffer(mm, dtype=torch.uint8)
            for key, meta in header.items():
                start, end = meta["data_offsets"]
                view = src[data_start + start: data_start + end]
                dst = dsts[(path, key)]
                if args.strategy == "zerocopy":
                    dst.copy_(view)
                else:
                    n = view.numel()
                    for off in range(0, n, chunk):
                        m = min(chunk, n - off)
                        k = slot
                        slot ^= 1
                        evs[k].synchronize()
                        gather(bufs[k], view, off, m)
                        dst[off:off + m].copy_(bufs[k][:m], non_blocking=True)
                        evs[k].record()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # SPEC.md §5 gate 1, outside the timed region.
    bad = []
    from safetensors import safe_open

    for path in files:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                want = f.get_tensor(key).contiguous().view(torch.uint8).flatten()
                got = dsts[(path, key)].cpu()
                if not torch.equal(got, want):
                    bad.append(key)

    results[rank] = {
        "rank": rank, "numa_node": node, "cpus": len(bound) if bound else None,
        "files": len(files), "tensors": len(plan),
        "gb": total_bytes / 1e9, "elapsed_s": elapsed,
        "gb_s": total_bytes / 1e9 / elapsed,
        "register_s": reg_s, "dma_s": dma_s,
        "mismatches": bad[:5], "n_mismatch": len(bad),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",
                   choices=["safe_open", "zerocopy", "pinned", "register"],
                   required=True)
    p.add_argument("--dir", default="/dev/shm/llama70b-sharded")
    p.add_argument("--chunk-mib", type=int, default=512)
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--no-numa", action="store_true")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--reg-threads", type=int, default=1)
    p.add_argument("--gather", choices=["torch", "memmove", "mt"], default="torch")
    p.add_argument("--gather-threads", type=int, default=8)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    nodes = {}
    if not args.no_numa:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.popen(f"python3 {here}/numa_map.py").read()
        for line in out.strip().splitlines():
            g, n = line.split()
            nodes[int(g)] = int(n)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.ranks)
    results = ctx.Manager().dict()
    procs = [ctx.Process(target=run_rank,
                         args=(r, nodes.get(r), args, barrier, results))
             for r in range(args.ranks)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()

    rows = [results[r] for r in sorted(results.keys())]
    if any("error" in r for r in rows) or len(rows) != args.ranks:
        print(f"FAIL {args.strategy}: {rows}", file=sys.stderr)
        sys.exit(1)

    slowest = max(r["elapsed_s"] for r in rows)
    correct = all(r["n_mismatch"] == 0 for r in rows)
    tag = f"{args.strategy} numa={int(not args.no_numa)} chunk={args.chunk_mib}M thr={args.threads} g={args.gather}/{args.gather_threads}"
    secs = " ".join("%5.2f" % r["elapsed_s"] for r in rows)
    rates = " ".join("%5.2f" % r["gb_s"] for r in rows)
    extra = ""
    if any(r.get("register_s") for r in rows):
        extra = "  [reg %s | dma %s]" % (
            " ".join("%4.2f" % r["register_s"] for r in rows),
            " ".join("%4.2f" % r["dma_s"] for r in rows))
    print("%-34s per-rank s %s   GB/s %s   gated %5.2fs  correct=%s%s"
          % (tag, secs, rates, slowest, correct, extra))

    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps({"strategy": args.strategy, "numa": not args.no_numa,
                                 "chunk_mib": args.chunk_mib, "threads": args.threads, "reg_threads": args.reg_threads,
                                 "gather": args.gather, "gather_threads": args.gather_threads, "gated_s": slowest,
                                 "correct": correct, "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
