#!/usr/bin/env python3
"""How fast can 4 concurrent ranks move bytes /dev/shm -> GPU, really?

SPEC.md needs 7.06 GB/s sustained per rank (35.3 GB in 5 s). The wire does
21-24 GB/s per GPU with all 4 active, so the question is entirely what the
host can feed it. This measures that directly, with no engine in the way.

Modes, in increasing order of how much work we do ourselves:

  pageable   torch's own copy_ straight off the mmap'd tmpfs pages. This is
             what ShardedStateLoader does today; the driver bounces every
             byte through its private pinned buffer, single-threaded.
  pinned     gather into our own reusable pinned buffer, then an async DMA,
             double-buffered so the next gather overlaps the previous
             transfer. The gather is torch's copy_, which parallelises over
             OMP threads -- hence --threads.
  dma_only   from an already-pinned, already-filled buffer. The ceiling; no
             mode can beat it.

Two things this varies that nothing in exp1 ever did:

  --threads  4 ranks x 64 default OMP threads on 64 cores erased a 3.67x
             gather win in the exp1 microbench. The cap matters more than
             most of the design.
  --numa     bind each rank to the NUMA node local to its GPU, and write its
             source file while bound so the tmpfs pages first-touch there.
             A ~32 GB/s aggregate host-copy ceiling on a node whose DRAM
             should do several times that is the signature of cross-socket
             traffic, and it has never been tested here.

All 4 ranks pass a barrier before the timed region, so the numbers include
the contention that a real TP=4 load has and a solo microbench does not.

Correctness is checked, not assumed (SPEC.md §5): the source is a distinct
pseudorandom pattern per rank, and every mode verifies the GPU buffer
against it. A pipelined async copy that reuses a staging buffer too early
produces exactly the kind of corruption that a throughput number hides.
"""

import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

GIB = 1 << 30


def numa_topology():
    """{node: [cpu, ...]} from sysfs; {} if the node has no NUMA at all."""
    base = "/sys/devices/system/node"
    if not os.path.isdir(base):
        return {}
    topo = {}
    for name in sorted(os.listdir(base)):
        if not name.startswith("node") or not name[4:].isdigit():
            continue
        node = int(name[4:])
        try:
            with open(f"{base}/{name}/cpulist") as fh:
                topo[node] = parse_cpulist(fh.read().strip())
        except OSError:
            pass
    return {k: v for k, v in topo.items() if v}


def parse_cpulist(s):
    cpus = []
    for part in s.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def gpu_numa_node(gpu):
    """NUMA node of the GPU's PCI device, via sysfs. None if unavailable."""
    try:
        import pynvml  # noqa
    except ImportError:
        pynvml = None
    bdf = None
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            bdf = pynvml.nvmlDeviceGetPciInfo(h).busId
            if isinstance(bdf, bytes):
                bdf = bdf.decode()
            bdf = bdf.lower()
            if bdf.count(":") == 2 and len(bdf.split(":")[0]) == 8:
                bdf = bdf[4:]  # 00000000:07:00.0 -> 0000:07:00.0
        except Exception:
            bdf = None
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    if not bdf:
        return None
    path = f"/sys/bus/pci/devices/{bdf}/numa_node"
    try:
        with open(path) as fh:
            n = int(fh.read().strip())
        return n if n >= 0 else None
    except OSError:
        return None


def pattern(rank, nbytes):
    """Deterministic, rank-distinct, cheap to regenerate for verification."""
    seed = hashlib.sha256(f"exp2-rank-{rank}".encode()).digest()
    block = (seed * ((1 << 20) // len(seed) + 1))[: 1 << 20]
    reps, tail = divmod(nbytes, len(block))
    return block, reps, tail


def build_source(path, rank, nbytes):
    """Write a real, non-sparse file. Sparse tmpfs reads hit the shared zero
    page and report a fantasy bandwidth."""
    if os.path.exists(path) and os.path.getsize(path) == nbytes:
        return
    block, reps, tail = pattern(rank, nbytes)
    with open(path, "wb") as fh:
        for _ in range(reps):
            fh.write(block)
        if tail:
            fh.write(block[:tail])


def bind_numa(node, topo):
    cpus = topo.get(node)
    if not cpus:
        return None
    os.sched_setaffinity(0, set(cpus))
    return cpus


def run_rank(rank, args, topo, barrier, results):
    import numpy as np
    import torch

    gpu = rank
    node = gpu_numa_node(gpu) if args.numa else None
    if args.numa and node is None:
        node = rank % max(len(topo), 1) if topo else None
    bound = bind_numa(node, topo) if (args.numa and node is not None) else None

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    nbytes = args.size_gib * GIB
    path = os.path.join(args.dir, f"rank{rank}.bin")
    build_source(path, rank, nbytes)

    torch.cuda.set_device(gpu)
    dev = torch.device(f"cuda:{gpu}")

    # Fresh mmap: the real loader touches each weight page exactly once, so a
    # hot re-mapped region would measure a steady state that never happens.
    src_np = np.memmap(path, dtype=np.uint8, mode="r")
    src = torch.from_numpy(np.asarray(src_np))

    chunk = args.chunk_mib << 20
    dst = torch.empty(nbytes, dtype=torch.uint8, device=dev)

    if args.mode == "dma_only":
        staging = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
        staging.copy_(src[:chunk])
    elif args.mode == "pinned":
        bufs = [torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
                for _ in range(args.depth)]
        events = [torch.cuda.Event() for _ in range(args.depth)]
        for e in events:
            e.record()

    torch.cuda.synchronize()
    barrier.wait()
    t0 = time.perf_counter()

    if args.mode == "pageable":
        for off in range(0, nbytes, chunk):
            n = min(chunk, nbytes - off)
            dst[off:off + n].copy_(src[off:off + n])
    elif args.mode == "dma_only":
        for off in range(0, nbytes, chunk):
            n = min(chunk, nbytes - off)
            dst[off:off + n].copy_(staging[:n], non_blocking=True)
    else:
        for i, off in enumerate(range(0, nbytes, chunk)):
            n = min(chunk, nbytes - off)
            b = i % args.depth
            events[b].synchronize()   # previous DMA out of this buffer is done
            bufs[b][:n].copy_(src[off:off + n])
            dst[off:off + n].copy_(bufs[b][:n], non_blocking=True)
            events[b].record()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ok, detail = verify(dst, rank, nbytes, args.mode, torch)

    results[rank] = {
        "rank": rank,
        "gpu": gpu,
        "numa_node": node,
        "cpus_bound": len(bound) if bound else None,
        "elapsed_s": elapsed,
        "gib_s": nbytes / GIB / elapsed,
        "gb_s": nbytes / 1e9 / elapsed,
        "correct": ok,
        "verify": detail,
    }


def verify(dst, rank, nbytes, mode, torch):
    """dma_only replays one chunk, so only that chunk is meaningful."""
    if mode == "dma_only":
        return True, "skipped (dma_only replays a single chunk by design)"
    block, _, _ = pattern(rank, nbytes)
    probes = [0, nbytes // 3, nbytes // 2, nbytes - (1 << 20)]
    for off in probes:
        n = min(1 << 20, nbytes - off)
        got = dst[off:off + n].cpu().numpy().tobytes()
        want = bytes(block[(off % len(block)):(off % len(block)) + n])
        if len(want) < n:  # wraps the 1 MiB pattern block
            want += bytes(block[: n - len(want)])
        if got != want:
            return False, f"mismatch at offset {off}"
    return True, f"{len(probes)} x 1 MiB probes match"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pageable", "pinned", "dma_only"], required=True)
    p.add_argument("--dir", default="/dev/shm/exp2bench")
    p.add_argument("--size-gib", type=int, default=8)
    p.add_argument("--chunk-mib", type=int, default=512)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--numa", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    topo = numa_topology()

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.ranks)
    results = ctx.Manager().dict()
    procs = [ctx.Process(target=run_rank, args=(r, args, topo, barrier, results))
             for r in range(args.ranks)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()

    rows = [results[r] for r in sorted(results.keys())]
    if len(rows) != args.ranks:
        print(f"FAIL: only {len(rows)}/{args.ranks} ranks reported", file=sys.stderr)
        sys.exit(1)

    per_rank = [r["gb_s"] for r in rows]
    slowest = min(per_rank)
    # The gating rank is what cold start waits on, so that is the headline.
    projected = 35.3 / slowest

    summary = {
        "mode": args.mode, "threads": args.threads, "numa": args.numa,
        "chunk_mib": args.chunk_mib, "depth": args.depth,
        "size_gib": args.size_gib, "ranks": args.ranks,
        "numa_nodes": {str(k): len(v) for k, v in topo.items()},
        "per_rank_gb_s": per_rank,
        "aggregate_gb_s": sum(per_rank),
        "slowest_rank_gb_s": slowest,
        "projected_weight_loading_s": projected,
        "all_correct": all(r["correct"] for r in rows),
        "rows": rows,
    }

    tag = f"{args.mode} thr={args.threads} numa={int(args.numa)} chunk={args.chunk_mib}M depth={args.depth}"
    print(f"{tag:<52} per-rank {' '.join(f'{v:5.2f}' for v in per_rank)}"
          f"  agg {sum(per_rank):6.2f}  slowest {slowest:5.2f} GB/s"
          f"  -> {projected:5.2f}s  correct={summary['all_correct']}")

    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
