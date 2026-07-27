#!/usr/bin/env python3
"""Unprivileged OS page-cache control for the checkpoint/restore measurements.

Why this exists
---------------
We compare two routes to a ready engine: a cold launch (reads ~16 GB of safetensors
weights) and a CRIU restore (reads a ~20 GB snapshot image). The OS page cache serves
any recently read/written file from RAM, so a naive dump-then-restore reads the snapshot
straight from cache and looks 5-10x too fast. To measure honestly we must evict the
relevant files from cache before each *cold* run -- and prove they were actually evicted.

We cannot use the usual tools here:
  * `echo 3 > /proc/sys/vm/drop_caches` needs root (we have no sudo);
  * cache-busting with a >RAM file needs >125 GB free disk (we have ~80 GB);
  * O_DIRECT (the repo's Lustre trick) only bypasses cache on reads *we* issue, not
    SGLang's weight loader or criu's restore I/O.

What works unprivileged is surgical, per-file eviction:
  * evict:        os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)  -- drops a file's
                  clean pages from cache; no privileges required.
  * verify:       mincore(2) via ctypes -- reports how many of a file's pages are
                  actually resident, so we can assert ~0% after eviction (fadvise is
                  advisory and must be verified, never trusted).

CLI
---
  cache_tools.py resident <path> [<path>...]   # report resident-page % (recurses dirs)
  cache_tools.py evict    <path> [<path>...]   # sync, then fadvise-DONTNEED (recurses)
  cache_tools.py evict --verify <path>...      # evict then re-report residency

Importable API: evict(path), resident_pct(path) -> (resident_pages, total_pages, pct).
"""
import ctypes
import ctypes.util
import os
import sys

PAGE = os.sysconf("SC_PAGE_SIZE")
_PROT_READ = 0x1
_MAP_SHARED = 0x01
_MAP_FAILED = ctypes.c_void_p(-1).value

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.restype = ctypes.c_int
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.restype = ctypes.c_int
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


def _iter_files(path):
    """Yield regular files under path (path itself if it is a file)."""
    if os.path.isfile(path):
        yield path
        return
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.isfile(fp) and not os.path.islink(fp):
                yield fp


def _file_residency(fp):
    """Return (resident_pages, total_pages) for one file via mincore(2)."""
    size = os.path.getsize(fp)
    if size == 0:
        return (0, 0)
    npages = (size + PAGE - 1) // PAGE
    fd = os.open(fp, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr == _MAP_FAILED:
            err = ctypes.get_errno()
            raise OSError(err, "mmap failed for %s: %s" % (fp, os.strerror(err)))
        try:
            vec = (ctypes.c_ubyte * npages)()
            if _libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                err = ctypes.get_errno()
                raise OSError(err, "mincore failed for %s: %s" % (fp, os.strerror(err)))
            resident = sum(1 for i in range(npages) if vec[i] & 1)
            return (resident, npages)
        finally:
            _libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def resident_pct(path):
    """Aggregate residency over path (file or dir).

    Returns (resident_pages, total_pages, pct). pct is 0.0 for empty input.
    """
    res_total = 0
    pg_total = 0
    for fp in _iter_files(path):
        try:
            r, n = _file_residency(fp)
        except OSError:
            # unreadable / special file -- skip, don't let one file break the report
            continue
        res_total += r
        pg_total += n
    pct = (100.0 * res_total / pg_total) if pg_total else 0.0
    return (res_total, pg_total, pct)


def evict(path):
    """Flush dirty pages, then advise the kernel to drop `path`'s pages from cache.

    Recurses directories. Calls os.sync() once up front so freshly-written files
    (e.g. a just-created CRIU image) are clean and therefore actually evictable --
    POSIX_FADV_DONTNEED only drops clean pages.
    """
    os.sync()
    for fp in _iter_files(path):
        try:
            fd = os.open(fp, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _fmt(path):
    r, n, pct = resident_pct(path)
    mib = n * PAGE / (1024 * 1024)
    return "%6.2f%% resident  (%d/%d pages, %.0f MiB total)  %s" % (pct, r, n, mib, path)


def main(argv):
    if len(argv) < 2 or argv[1] not in ("resident", "evict"):
        print(__doc__)
        return 2
    cmd = argv[1]
    rest = argv[2:]
    verify = False
    if cmd == "evict" and rest and rest[0] == "--verify":
        verify = True
        rest = rest[1:]
    if not rest:
        print("error: no paths given", file=sys.stderr)
        return 2

    if cmd == "resident":
        for p in rest:
            print(_fmt(p))
    else:  # evict
        for p in rest:
            if verify:
                print("before: " + _fmt(p))
            evict(p)
            if verify:
                print("after : " + _fmt(p))
            else:
                print("evicted: %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
