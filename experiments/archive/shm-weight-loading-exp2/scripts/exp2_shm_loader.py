"""exp2 replacement for ShardedStateLoader's inner load loop.

Two changes over the stock loop, both measured on the real checkpoint with
`shm_loader_bench.py` before being written into the engine:

1. **Zero-copy tensor views.** Stock calls `safe_open(path).get_tensor(key)`,
   which materialises a host-side copy of every tensor before it is sent to
   the GPU. Reading the safetensors header ourselves and building views with
   `torch.frombuffer` over one mmap of the file removes that copy entirely:
   7.08 s -> 5.51 s per rank on the real shards (job 76365), same bytes on
   the wire.

2. **Optional pinned double-buffered staging** (`SGLANG_EXP2_PINNED=1`), for
   the copy itself. This is the piece that only works with NUMA binding in
   place: unbound it is the slowest mode of all, which is why the old
   `sharded_pinned_h2d.patch` regressed to 14.37 s. It is also the piece that
   requires a thread cap, since the gather parallelises over OMP threads and
   4 ranks at torch's default count on one node collapses.

Both are env-gated and fall back to the stock `copy_` for anything they
cannot handle (non-contiguous destination, dtype/shape mismatch), so the
loader's behaviour is unchanged wherever the fast path does not apply.
"""

import ctypes
import json
import logging
import mmap
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import torch

logger = logging.getLogger(__name__)

_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _env_flag(name, default="0"):
    return os.environ.get(name, default).lower() not in ("0", "false", "")


def enabled():
    return _env_flag("SGLANG_EXP2_LOADER")


def _parse_header(buf):
    n = struct.unpack_from("<Q", buf, 0)[0]
    header = json.loads(bytes(memoryview(buf)[8:8 + n]))
    header.pop("__metadata__", None)
    return header, 8 + n


def _thread_cap():
    """cores/TP, so 4 ranks do not oversubscribe one node's cores.

    Under numactl --cpunodebind the affinity mask is already this rank's node,
    so its size is the right budget and nothing needs to know the TP degree.
    """
    override = os.environ.get("SGLANG_EXP2_THREADS")
    if override:
        return int(override)
    try:
        return max(1, len(os.sched_getaffinity(0)) // 2)  # physical, not SMT
    except OSError:
        return 16


def load(state_dict, filepaths):
    """Mirrors the stock loop's contract: copy every tensor found in
    `filepaths` into the matching entry of `state_dict`, popping as it goes."""
    use_pinned = _env_flag("SGLANG_EXP2_PINNED")
    chunk = int(os.environ.get("SGLANG_EXP2_CHUNK_MIB", "512")) << 20

    torch.set_num_threads(_thread_cap())
    logger.info(
        "exp2 loader: zero-copy views, pinned=%s, threads=%d, chunk=%d MiB",
        use_pinned, torch.get_num_threads(), chunk >> 20,
    )

    if use_pinned:
        bufs = [torch.empty(chunk, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        evs = [torch.cuda.Event() for _ in range(2)]
        for ev in evs:
            ev.record()
        slot = 0

        # Gather variant. Measured per-rank on the real checkpoint (job 76372,
        # 4 concurrent ranks): mt/4 7.95 GB/s, mt/8 7.88, torch 7.36,
        # single-threaded memmove 4.96. Splitting one chunk across a few
        # threads wins because ctypes releases the GIL and glibc's memcpy uses
        # non-temporal stores at this size, eliding the read-for-ownership
        # that torch's copy_ pays on every destination line. Past ~8 threads
        # it regresses -- the node's 2 memory channels are the wall, not cores.
        gather_threads = int(os.environ.get("SGLANG_EXP2_GATHER_THREADS", "4"))
        pool = ThreadPoolExecutor(max_workers=gather_threads) if gather_threads > 1 else None

        def gather(buf, view, off, m):
            if pool is None:
                ctypes.memmove(buf.data_ptr(), view.data_ptr() + off, m)
                return
            step = (m + gather_threads - 1) // gather_threads
            dst_p, src_p = buf.data_ptr(), view.data_ptr() + off
            futs = [pool.submit(ctypes.memmove, dst_p + i, src_p + i, min(step, m - i))
                    for i in range(0, m, step)]
            for f in futs:
                f.result()

    n_fast = n_slow = 0
    _t0 = time.perf_counter()

    for path in filepaths:
        fd = os.open(path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        finally:
            os.close(fd)

        header, data_start = _parse_header(mm)
        raw = torch.frombuffer(mm, dtype=torch.uint8)

        for key, meta in header.items():
            start, end = meta["data_offsets"]
            byte_view = raw[data_start + start: data_start + end]

            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            shape = meta["shape"]
            dtype = _DTYPE[meta["dtype"]]

            # Same narrowing the stock loop does for padded (LoRA) params.
            for dim, size in enumerate(shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if list(shape) != list(param_shape):
                logger.warning(
                    "loading tensor of shape %s into parameter '%s' of shape %s",
                    shape, key, param_shape,
                )

            can_fast = (
                use_pinned
                and param_data.is_cuda
                and param_data.is_contiguous()
                and param_data.dtype == dtype
            )
            if can_fast:
                flat = param_data.view(torch.uint8).flatten()
                n = byte_view.numel()
                for off in range(0, n, chunk):
                    m = min(chunk, n - off)
                    k = slot
                    slot ^= 1
                    # This slot's previous DMA must have drained before its
                    # buffer is overwritten; the other slot's may still be in
                    # flight, and that overlap is the point.
                    evs[k].synchronize()
                    gather(bufs[k], byte_view, off, m)
                    flat[off:off + m].copy_(bufs[k][:m], non_blocking=True)
                    evs[k].record()
                n_fast += 1
            else:
                typed = byte_view.view(dtype).view(shape) if shape else byte_view.view(dtype).view(())
                param_data.copy_(typed)
                n_slow += 1

            state_dict.pop(key)

    if use_pinned:
        if pool is not None:
            pool.shutdown(wait=True)
        # The pinned buffers are about to go out of scope. Freeing one while a
        # DMA is still reading it is the corruption mode the bit-exactness
        # gate exists to catch; do not rely on the gate to prevent it.
        torch.cuda.synchronize()

    # The copy alone, so it can be compared against the phase wall time. The
    # engine's "Load weight end" window also contains model construction and
    # state_dict filtering, which no copy optimisation can touch.
    logger.info("exp2 loader: %d tensors staged, %d plain, copy_elapsed=%.2f s",
                n_fast, n_slow, time.perf_counter() - _t0)
