#!/usr/bin/env python3
"""Phase 1.5 of STEP4_PLAN.md: stage safetensors shards into ONE anonymous
memfd(MFD_HUGETLB)-backed buffer, no filesystem path involved.

Why memfd instead of the Phase 0 gate's plain mmap(MAP_HUGETLB): a bare
anonymous mapping has no fd to hand to another process. memfd_create gives an
fd that can be passed across a fork+exec (or explicitly via SCM_RIGHTS, see
hugepage_fd_broker.py) so the 4 separately-spawned TP ranks can all mmap the
SAME staged bytes -- matching what today's /dev/shm design gets for free via
a path, without needing a real hugetlbfs mount (which STEP4_PLAN.md's Phase
0/1 findings ruled out: `mount -t hugetlbfs` needs CAP_SYS_ADMIN and fails
here, and the one writable hugetlbfs mount that DOES exist only accepts
ftruncate sizes that are exact 2 MB multiples, which breaks safe_open).

Layout: each file gets a 2 MB-aligned region within the combined buffer
(HUGEPAGE-sized padding, tracked ourselves -- never relies on the kernel
reporting an exact file size, since it can't). Reads use O_DIRECT + preadv
straight into the destination memoryview, mirroring stage_to_hugetlbfs.py's
slicing (SLICES=64 contiguous 4 KB-aligned ranges per file).

After staging, each file's safetensors header is parsed in place
(hugepage_safetensors.parse_header) so the caller gets back exactly the
metadata needed to construct tensor views later -- no safe_open involved
anywhere in this path.

Usage as a library:
    fd, total_len, files_meta = stage(src_dir, filenames)
    # files_meta[name] = (offset_in_buffer, header_dict, data_start)
"""
import ctypes
import ctypes.util
import os
from concurrent.futures import ThreadPoolExecutor

import hugepage_safetensors as hsf

ALIGN = 4096
HUGEPAGE = 2 * 1024 * 1024
MFD_HUGETLB, MFD_CLOEXEC = 0x0004, 0x0001

libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_long]
libc.memfd_create.restype = ctypes.c_int
libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]

PROT_READ, PROT_WRITE, MAP_SHARED = 0x1, 0x2, 0x01


def _round_up(n, to):
    return -(-n // to) * to


def create_memfd_hugetlb(total_len, name=b"hugepage_stage"):
    """fd + writable ctypes buffer view over `total_len` bytes of hugepages.

    total_len must already be a HUGEPAGE multiple -- the memfd ftruncate has
    the same exact-multiple constraint the Phase 0/1 gate found on the
    mounted hugetlbfs (this is a hugetlbfs-inode property, not specific to
    the mount path).
    """
    assert total_len % HUGEPAGE == 0
    fd = libc.memfd_create(name, MFD_HUGETLB | MFD_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()), "memfd_create")
    os.ftruncate(fd, total_len)
    addr = libc.mmap(None, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
    if addr in (None, (1 << 64) - 1):
        err = ctypes.get_errno()
        os.close(fd)
        raise OSError(err, os.strerror(err), "mmap")
    buf = (ctypes.c_ubyte * total_len).from_address(addr)
    return fd, buf


def _plan_slices(size, slices):
    """Aligned (offset, length, use_o_direct) ranges tiling [0, size)."""
    aligned = size - (size % ALIGN)
    per = _round_up(aligned, slices) // slices if aligned else 0
    per -= per % ALIGN
    if per == 0 and aligned:
        per = ALIGN
    out, off = [], 0
    while off < aligned:
        length = min(per, aligned - off)
        out.append((off, length, True))
        off += length
    if size > aligned:
        out.append((aligned, size - aligned, False))
    return out


def _read_slice(src_path, mv, dest_off, src_off, length, o_direct):
    flags = os.O_RDONLY | (os.O_DIRECT if o_direct else 0)
    fd = os.open(src_path, flags)
    try:
        got = os.preadv(fd, [mv[dest_off:dest_off + length]], src_off)
        if got != length:
            raise IOError(f"short read {got}/{length} at {src_off} of {src_path}")
    finally:
        os.close(fd)


def mmap_from_fd(fd, total_len):
    """Map an already-sized hugetlbfs fd (e.g. received via SCM_RIGHTS) --
    no ftruncate, the creator already sized it."""
    addr = libc.mmap(None, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
    if addr in (None, (1 << 64) - 1):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), "mmap")
    return (ctypes.c_ubyte * total_len).from_address(addr)


def stage(src_dir, filenames, slices=64, max_workers=None):
    sizes = {f: os.path.getsize(os.path.join(src_dir, f)) for f in filenames}
    layout = {}
    running = 0
    for f in filenames:
        layout[f] = running
        running += _round_up(sizes[f], HUGEPAGE)
    total_len = running

    fd, buf = create_memfd_hugetlb(total_len)
    mv = memoryview(buf)

    tasks = []
    for f in filenames:
        base = layout[f]
        for off, length, o_direct in _plan_slices(sizes[f], slices):
            tasks.append((os.path.join(src_dir, f), mv, base + off, off, length, o_direct))

    with ThreadPoolExecutor(max_workers=max_workers or len(tasks)) as pool:
        futs = [pool.submit(_read_slice, *t) for t in tasks]
        for fut in futs:
            fut.result()

    files_meta = {}
    for f in filenames:
        base = layout[f]
        header, data_start = hsf.parse_header(buf, base_offset=base)
        files_meta[f] = (base, header, data_start)

    return fd, total_len, files_meta, buf


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--slices", type=int, default=64)
    a = ap.parse_args()

    t0 = time.perf_counter()
    fd, total_len, files_meta, buf = stage(a.src_dir, a.files, a.slices)
    dt = time.perf_counter() - t0
    print(f"staged {len(a.files)} file(s), {total_len/1e9:.2f} GB buffer, "
          f"{dt:.2f}s, {total_len/dt/1e9:.2f} GB/s")
    for f, (base, header, data_start) in files_meta.items():
        print(f"  {f}: base={base} data_start={data_start} n_tensors={len(header)}")
