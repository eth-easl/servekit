"""Client-side shim used by hugepage_sharded_h2d.patch, running inside a
TP rank's own process (spawned by SGLang).

Fetches the buffer staged by hugepage_stage_daemon.py over the broker,
filters it down to this rank's own model-rank-{rank}-part-*.safetensors
files, and drives a pipelined "register shard N+1 while shard N's copies
drain" sequence (STEP4_PLAN.md's Phase 2 design, applied per-rank -- see
hugepage-sharded-loading-exp plan). cudaHostRegister/cudaMemcpyAsync are
reached via ctypes against libcudart.so since torch.cuda.cudart() only
exposes cudaHostRegister/cudaHostUnregister, not cudaMemcpyAsync.
"""
import ctypes
import ctypes.util
import fnmatch
import os
import time

import hugepage_fd_broker as broker
import hugepage_stager as stager

_RETRY_ATTEMPTS = 30
_RETRY_DELAY_S = 1.0

_CUDA_MEMCPY_HOST_TO_DEVICE = 1  # cudaMemcpyKind enum
_CUDA_HOST_REGISTER_PORTABLE = 0x01  # cudaHostRegisterPortable
_CUDA_HOST_REGISTER_READ_ONLY = 0x08  # cudaHostRegisterReadOnly

# Registration is the dominant cost on this path (the DMA off an already-
# registered buffer runs at line rate, ~1.33 s for a rank's 35.3 GB), so the
# registration FLAGS are worth a knob. cudaHostRegisterReadOnly tells the
# driver the range will never be written by the host, letting it skip some
# write-tracking bookkeeping. Correct here by construction: the staged memfd
# is read-only from the moment the daemon finishes filling it. Off by
# default, so every prior number stands unchanged.
_REGISTER_FLAGS = _CUDA_HOST_REGISTER_PORTABLE
if os.environ.get("SGLANG_HUGEPAGE_REGISTER_READONLY", "0") not in ("0", "", "false"):
    _REGISTER_FLAGS |= _CUDA_HOST_REGISTER_READ_ONLY


def _find_loaded_cudart():
    """Resolve the libcudart.so ALREADY mapped into this process (i.e. the
    one torch itself loaded), not just any libcudart the dynamic linker can
    find. torch ships its own bundled libcudart (from the nvidia-cuda-runtime
    pip wheel, e.g. .../dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12)
    which is a DIFFERENT FILE from the system CUDA toolkit's copy
    ctypes.util.find_library("cudart") resolves to (e.g.
    /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12) -- two separate
    loaded library instances. A CUDA stream handle created via torch's
    instance (torch.cuda.current_stream().cuda_stream) is opaque to a
    cudaMemcpyAsync call made through a DIFFERENT instance, which reproduced
    cudaErrorInvalidValue reliably regardless of any threading/ordering fix
    (see NOTES.md). dlopen returns a handle to an already-mapped library
    (not a second copy) when given the exact same path, so finding that path
    via /proc/self/maps and loading THAT fixes it.
    """
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart.so" in line:
                    path = line.rstrip("\n").split()[-1]
                    if os.path.exists(path):
                        return path
    except OSError:
        pass
    return ctypes.util.find_library("cudart") or "libcudart.so.12"


_LIBCUDART_PATH = _find_loaded_cudart()
libcudart = ctypes.CDLL(_LIBCUDART_PATH)
libcudart.cudaGetLastError.restype = ctypes.c_int
libcudart.cudaGetLastError.argtypes = []
libcudart.cudaPeekAtLastError.restype = ctypes.c_int
libcudart.cudaPeekAtLastError.argtypes = []
libcudart.cudaStreamSynchronize.restype = ctypes.c_int
libcudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
libcudart.cudaHostRegister.restype = ctypes.c_int
libcudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
libcudart.cudaHostUnregister.restype = ctypes.c_int
libcudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
libcudart.cudaMemcpyAsync.restype = ctypes.c_int
libcudart.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
]
libcudart.cudaSetDevice.restype = ctypes.c_int
libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
libcudart.cudaDeviceSynchronize.restype = ctypes.c_int
libcudart.cudaDeviceSynchronize.argtypes = []


def _file_span_len(base, header, data_start):
    """Real (unpadded) byte length of one staged file's content, covering
    every tensor's data -- not the file's on-disk hugepage-rounded size."""
    end = max(v["data_offsets"][1] for v in header.values())
    return (data_start - base) + end


def tensor_end_offset_in_file(f, name):
    """End byte offset of tensor `name`, relative to file f's own `base`
    (i.e. in the same coordinate space PipelinedRegistrar's watermark uses) --
    (data_start - base) covers the header, plus the tensor's own end offset
    within the data section."""
    end_in_data = f["header"][name]["data_offsets"][1]
    return (f["data_start"] - f["base"]) + end_in_data


def _log(tag, msg):
    if tag:
        print(f"[hugepage {tag} pid={os.getpid()}] {msg}", flush=True)


def get_rank_files_meta(sock_path, rank_pattern, tag=""):
    """Fetch the broker's combined files_meta, filter to this rank's own
    files (fnmatch against rank_pattern, e.g. "model-rank-0-part-*.safetensors"),
    sorted by filename (part index order). Returns (buf, ordered_rank_files)
    or None on any failure -- caller falls back to safe_open.

    `tag` (e.g. "TP0") enables timestamped logging of each step, so a
    multi-rank run can be diffed to see exactly where one rank falls behind
    another (connect wait, mmap, or something downstream in the registrar).
    """
    _log(tag, f"libcudart resolved to {_LIBCUDART_PATH}")
    t0 = time.perf_counter()
    fetched = None
    for _ in range(_RETRY_ATTEMPTS):
        try:
            fetched = broker.fetch(sock_path)
            break
        except (ConnectionRefusedError, FileNotFoundError, EOFError, OSError):
            time.sleep(_RETRY_DELAY_S)
    if fetched is None:
        _log(tag, f"broker.fetch FAILED after {time.perf_counter()-t0:.3f}s")
        return None
    fd, total_len, files_meta = fetched
    _log(tag, f"broker.fetch done in {time.perf_counter()-t0:.3f}s "
              f"(total_len={total_len/1e9:.2f} GB, {len(files_meta)} files)")

    t1 = time.perf_counter()
    try:
        buf = stager.mmap_from_fd(fd, total_len)
    except OSError:
        return None
    _log(tag, f"mmap_from_fd done in {time.perf_counter()-t1:.3f}s")

    matched = sorted(f for f in files_meta if fnmatch.fnmatch(f, rank_pattern))
    if not matched:
        return None

    ordered = []
    for fname in matched:
        base, header, data_start = files_meta[fname]
        ordered.append(dict(
            filename=fname, base=base, header=header, data_start=data_start,
            span_len=_file_span_len(base, header, data_start),
        ))
    _log(tag, f"total get_rank_files_meta {time.perf_counter()-t0:.3f}s, "
              f"{len(ordered)} files matched {rank_pattern}")
    return buf, ordered


# 1 GiB default, matching the reference design (see plan/NOTES.md). Override
# via SGLANG_HUGEPAGE_CHUNK_BYTES for sweeping chunk size (smaller = more
# calls but finer interleaving across the 4 concurrently-loading ranks,
# larger = fewer calls but each holds whatever's contended for longer).
_REGISTER_CHUNK = int(os.environ.get("SGLANG_HUGEPAGE_CHUNK_BYTES", 1 << 30))

# How many chunks to keep registered ahead of what the copy loop currently
# needs. A "register the whole next file" version (bigger lumps, only
# refilled at file boundaries) worked but over-commits at each boundary;
# this instead maintains a small, continuously-topped-up window -- refilled
# on every wait_for() call, not just at file boundaries -- so it adapts to
# any chunk size, including much smaller ones than a file.
_LOOKAHEAD_CHUNKS = int(os.environ.get("SGLANG_HUGEPAGE_LOOKAHEAD_CHUNKS", 3))


class PipelinedRegistrar:
    """Registers this rank's file set in 1 GiB steps, synchronously on the
    SAME thread that issues cudaMemcpyAsync -- not a background thread.

    An earlier version ran registration on a background thread racing ahead
    of the copy loop. That reproduced a real cudaMemcpyAsync failure
    (cudaErrorInvalidValue) on two independent runs, and web research turned
    up a corroborating NVIDIA forum thread describing exactly this class of
    issue: cudaHostRegister across concurrent host threads is documented as
    unreliable/driver-version-dependent, with NVIDIA staff's own guidance
    being "keep drivers current" rather than a code-level fix (see
    hugepage-sharded-loading-exp/NOTES.md). Rather than depend on that, this
    version never calls cudaHostRegister and cudaMemcpyAsync from different
    threads at all.

    This does NOT give up the overlap the design wants: cudaMemcpyAsync
    itself doesn't block the CPU -- once issued, the GPU drains it on its own
    copy engine regardless of what the CPU does next. So the sequence
    "register chunk N -> issue chunk N's copies (returns immediately) ->
    register chunk N+1 -> issue chunk N+1's copies -> ..." still lets
    chunk N's DMA drain in the background while the CPU (single thread)
    moves on to registering and issuing chunk N+1 -- the same overlap,
    without the cross-thread CUDA API concurrency.

    1 GiB steps (not whole-file, ~4-5 GB each) are still used deliberately:
    if cudaHostRegister contends across the 4 concurrently-registering TP
    ranks (see NOTES.md), smaller steps mean each rank holds whatever's
    contended for less time per call and hands off more often.
    """

    def __init__(self, buf, ordered_rank_files, tag="", device_id=None):
        self._files = ordered_rank_files
        self._base_addr = ctypes.addressof(buf)
        self._tag = tag
        if device_id is not None:
            rc = libcudart.cudaSetDevice(device_id)
            if rc != 0:
                _log(tag, f"cudaSetDevice({device_id}) failed rc={rc}")

        # Chunk boundaries must land exactly on a tensor boundary, never
        # inside one: cudaMemcpyAsync's source range apparently cannot span
        # two SEPARATELY cudaHostRegister'd regions, even when they cover
        # physically adjacent, contiguous bytes -- confirmed by a diagnostic
        # run where the exact same tensor (whichever one straddled the
        # chunk[0]/chunk[1] boundary) failed with cudaErrorInvalidValue on
        # every rank, every time, regardless of any threading/ordering/
        # library fix (see NOTES.md). So chunks are built by walking each
        # file's tensors in offset order and cutting once accumulated size
        # reaches ~1 GiB, landing exactly at that tensor's end -- not by
        # blindly slicing every _REGISTER_CHUNK bytes.
        self._chunks = []  # (file_idx, offset_in_file, length)
        for fi, f in enumerate(ordered_rank_files):
            ends = sorted(tensor_end_offset_in_file(f, name) for name in f["header"])
            off = 0
            for end in ends:
                if end - off >= _REGISTER_CHUNK:
                    self._chunks.append((fi, off, end - off))
                    off = end
            if off < f["span_len"]:
                self._chunks.append((fi, off, f["span_len"] - off))

        self._watermark = [0] * len(ordered_rank_files)
        self._next_chunk = 0

    def _register_next_chunk(self):
        # An earlier version put a cudaDeviceSynchronize() here, on the
        # theory that a fresh cudaHostRegister racing a PREVIOUS chunk's
        # still-draining cudaMemcpyAsync was unsafe. A diagnostic run proved
        # that theory wrong: the real bug was chunk boundaries landing
        # mid-tensor (see the class docstring and NOTES.md) -- a pure
        # correctness bug, unrelated to concurrency. Registering a NEW,
        # non-overlapping chunk while a PREVIOUS chunk's copies are still in
        # flight was never actually the problem, so the drain here was pure
        # lost overlap for no safety benefit. Removed -- register_ahead_one()
        # is what intentionally calls this while copies may still be
        # draining, on purpose, to get that overlap back.
        chunk_idx = self._next_chunk
        fi, off, length = self._chunks[chunk_idx]
        f = self._files[fi]
        ptr = self._base_addr + f["base"] + off
        # Per-chunk registration logging removed -- at small chunk sizes
        # (e.g. 100 MB) this was hundreds of print()+flush calls per rank,
        # real I/O overhead that was itself slowing the measured run down.
        # wait_for()'s blocking-time print (the one diagnostic that matters:
        # is the lookahead window actually keeping up) stays unconditional.
        rc = libcudart.cudaHostRegister(ptr, length, _REGISTER_FLAGS)
        if rc != 0:
            raise OSError(f"cudaHostRegister failed rc={rc} for chunk[{chunk_idx}]")
        self._watermark[fi] = off + length
        self._next_chunk += 1

    def wait_for(self, file_idx, end_offset):
        """Ensure file_idx's watermark covers end_offset (registering
        synchronously if the copy loop has genuinely caught up -- logged,
        since that IS blocked time), then top up the lookahead window by
        _LOOKAHEAD_CHUNKS more chunks.

        History: a first version only registered 1 chunk ahead at file
        boundaries -- a diagnostic run showed nearly every within-file chunk
        transition blocking (0.01-0.4s each, ~5/file, summing to most of the
        measured time). A second version registered the WHOLE next file
        ahead at each boundary -- that made things WORSE (measured 8.24s vs
        7.03s), because registering ~5GB in one lump doesn't overlap well
        and just adds sequential latency at the boundary. This version
        instead maintains a small, constant-size lookahead window,
        continuously topped up on every call (not just at file boundaries)
        -- adapts to any chunk size, including much smaller ones than a
        file, and never does a big lump of registration at once. See
        NOTES.md.
        """
        if self._watermark[file_idx] < end_offset:
            t0 = time.perf_counter()
            n = 0
            while self._watermark[file_idx] < end_offset and self._next_chunk < len(self._chunks):
                self._register_next_chunk()
                n += 1
            dt = time.perf_counter() - t0
            tag = self._tag or f"pid{os.getpid()}"
            print(f"[hugepage {tag}] wait_for(file={file_idx}, end={end_offset/1e9:.3f}GB) "
                  f"blocked {dt:.3f}s registering {n} chunk(s) the copy loop caught up to "
                  f"(lookahead={_LOOKAHEAD_CHUNKS} wasn't enough)", flush=True)
        target = self._next_chunk + _LOOKAHEAD_CHUNKS
        while self._next_chunk < min(target, len(self._chunks)):
            self._register_next_chunk()

    def unregister_file(self, file_idx):
        """Unregister every chunk belonging to one file, once the copy loop
        has fully consumed it. Registered chunks are never read again after
        their tensors are copied, so there's no reason to keep them pinned --
        and keeping the live-registration COUNT bounded (rather than growing
        across the whole ~35 GB/rank load, ~35 discrete registrations deep)
        avoids exhausting whatever fixed-size driver/hardware table tracks
        pinned-memory registrations (a plausible explanation for the
        cudaMemcpyAsync rc=1 failures seen with unbounded accumulation --
        see NOTES.md)."""
        for fi, off, length in self._chunks:
            if fi != file_idx:
                continue
            f = self._files[fi]
            ptr = self._base_addr + f["base"] + off
            rc = libcudart.cudaHostUnregister(ptr)
            if rc != 0:
                _log(self._tag, f"cudaHostUnregister file[{fi}]+{off/1e9:.2f}GB "
                                f"failed rc={rc}")

    def unregister_all(self):
        for fi, off, length in self._chunks:
            f = self._files[fi]
            ptr = self._base_addr + f["base"] + off
            libcudart.cudaHostUnregister(ptr)


def cuda_memcpy_h2d_async(dst_ptr, src_ptr, nbytes, stream_ptr, tag="", debug_ctx=""):
    # The pending/sticky-error check below is diagnostic-only (added to
    # track down the chunk-boundary bug documented on PipelinedRegistrar,
    # see NOTES.md) -- it costs an extra CUDA API call per tensor copy, so
    # it's skipped entirely unless a tag is given, keeping normal runs at
    # the same cost as a bare cudaMemcpyAsync call.
    if tag:
        pending = libcudart.cudaPeekAtLastError()
        if pending != 0:
            libcudart.cudaGetLastError()  # clear it
            _log(tag, f"NOTE: pending CUDA error {pending} was already set "
                      f"before this cudaMemcpyAsync call ({debug_ctx})")
    rc = libcudart.cudaMemcpyAsync(
        dst_ptr, src_ptr, nbytes, _CUDA_MEMCPY_HOST_TO_DEVICE, stream_ptr
    )
    if rc != 0:
        post = libcudart.cudaGetLastError()
        # Always printed, even with tag="" -- an error here is rare/fatal
        # and must never be silently dropped just because verbose logging
        # is off for a clean timing run.
        print(f"[hugepage pid={os.getpid()}] cudaMemcpyAsync FAILED rc={rc} "
              f"post_cudaGetLastError={post} dst=0x{dst_ptr:x} src=0x{src_ptr:x} "
              f"nbytes={nbytes} stream=0x{stream_ptr:x} ({debug_ctx})", flush=True)
        raise OSError(f"cudaMemcpyAsync failed rc={rc}")


def cuda_stream_synchronize(stream_ptr):
    """Drain only `stream_ptr`, not the whole device (torch.cuda.synchronize()
    waits for every stream/work on the device, which is broader than we
    need -- we only need THIS rank's copy stream drained before touching its
    memory)."""
    rc = libcudart.cudaStreamSynchronize(stream_ptr)
    if rc != 0:
        raise OSError(f"cudaStreamSynchronize failed rc={rc}")
