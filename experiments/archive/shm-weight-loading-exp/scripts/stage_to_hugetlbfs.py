#!/usr/bin/env python3
"""Phase 1 of STEP4_PLAN.md: mmap-based hugetlbfs stager.

Mirrors stage_to_shm_sliced.sh's proven tuning (SLICES=64 contiguous ranges
per file, one O_DIRECT reader per slice -> queue depth per OST) but writes
through an mmap instead of `dd`, because hugetlbfs supports mmap, NOT write()
(see STEP4_PLAN.md). Each worker reads its slice from Lustre with O_DIRECT
straight into a memoryview of the destination mmap via os.preadv -- no
intermediate buffer, no bounce copy.

Destination filesystem: `mount -t hugetlbfs` needs CAP_SYS_ADMIN and fails in
this container (confirmed in the Phase 0 follow-up probe). The only
filesystem-backed hugetlbfs route available is the pre-existing, world-
writable (sticky-bit) libhugetlbfs pool mount at
/var/lib/hugetlbfs/global/pagesize-2097152 -- anonymous mmap(MAP_HUGETLB) (the
route that passed the Phase 0 gate) has no path a --model-path can point at,
which is why this mount matters.

Known risk (STEP4_PLAN.md): hugetlbfs may round allocations to 2 MB. If the
reported file size ends up padded, safetensors' header-vs-length check will
reject the shard. Run with --check-alignment on a small file FIRST.

Usage:
  stage_to_hugetlbfs.py --check-alignment <one_file>
  stage_to_hugetlbfs.py <src_model_dir> <dest_dir> [--slices 64]
"""
import argparse
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor

HUGE_MOUNT_DEFAULT = "/var/lib/hugetlbfs/global/pagesize-2097152"
ALIGN = 4096          # O_DIRECT alignment requirement for offset + buffer
HUGEPAGE = 2 * 1024 * 1024
DIRECT_MIN = 1 << 20  # below this, plain copy -- not worth O_DIRECT/mmap machinery


def check_alignment(src_file, huge_mount):
    """Answers STEP4_PLAN.md's known risk: does ftruncate on hugetlbfs keep
    the exact byte size, or does the filesystem round it up to 2 MB?"""
    size = os.path.getsize(src_file)
    dest = os.path.join(huge_mount, "align_check_" + os.path.basename(src_file))
    fd = os.open(dest, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(fd, size)
        got = os.fstat(fd).st_size
        print(f"src_size={size} dest_size_after_ftruncate={got} "
              f"{'OK: exact' if got == size else 'PADDED -- safe_open will likely reject'}")
        return got == size
    finally:
        os.close(fd)
        os.unlink(dest)


def stage_slice(src_path, mv, off, length, o_direct):
    flags = os.O_RDONLY | (os.O_DIRECT if o_direct else 0)
    fd = os.open(src_path, flags)
    try:
        got = os.preadv(fd, [mv[off:off + length]], off)
        if got != length:
            raise IOError(f"short read {got}/{length} at off={off} of {src_path}")
    finally:
        os.close(fd)


def plan_slices(size, slices):
    """Aligned slices covering [0, size). The tail slice absorbs whatever is
    left after the last 4 KB-aligned boundary -- reads it without O_DIRECT
    since neither its offset nor length is guaranteed aligned."""
    aligned_size = size - (size % ALIGN)
    per = -(-aligned_size // slices)          # ceil
    per -= per % ALIGN
    if per == 0:
        per = ALIGN
    out = []
    off = 0
    while off < aligned_size:
        length = min(per, aligned_size - off)
        out.append((off, length, True))
        off += length
    if size > aligned_size:
        out.append((aligned_size, size - aligned_size, False))
    return out


def stage_file(src_path, dest_path, slices, pool):
    size = os.path.getsize(src_path)
    if size < DIRECT_MIN:
        with open(src_path, "rb") as fsrc, open(dest_path, "wb") as fdst:
            fdst.write(fsrc.read())
        return []

    mmap_len = -(-size // HUGEPAGE) * HUGEPAGE   # hugetlbfs mmap must be huge-page aligned
    fd = os.open(dest_path, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)                       # logical size stays exact; see check_alignment
    mm = mmap.mmap(fd, mmap_len, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    mv = memoryview(mm)

    futs = [pool.submit(stage_slice, src_path, mv, off, length, o_direct)
            for off, length, o_direct in plan_slices(size, slices)]
    for f in futs:
        f.result()
    mv.release()
    mm.close()
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="source model dir (Lustre)")
    ap.add_argument("dest", nargs="?", help="dest dir under a hugetlbfs mount")
    ap.add_argument("--slices", type=int, default=64)
    ap.add_argument("--huge-mount", default=HUGE_MOUNT_DEFAULT)
    ap.add_argument("--check-alignment", metavar="FILE",
                     help="run only the known-risk check on one file and exit")
    a = ap.parse_args()

    if a.check_alignment:
        ok = check_alignment(a.check_alignment, a.huge_mount)
        raise SystemExit(0 if ok else 1)

    if not a.src or not a.dest:
        ap.error("src and dest are required unless --check-alignment is given")

    files = sorted(f for f in os.listdir(a.src)
                    if os.path.isfile(os.path.join(a.src, f)))
    src_bytes = sum(os.path.getsize(os.path.join(a.src, f)) for f in files)
    os.makedirs(a.dest, exist_ok=True)

    print(f"staging(hugetlbfs): {a.src} -> {a.dest} (files={len(files)}, "
          f"slices/file={a.slices}, mount={a.huge_mount})", flush=True)

    ntask = a.slices * len(files)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(ntask, 1)) as pool:
        for f in files:
            stage_file(os.path.join(a.src, f), os.path.join(a.dest, f), a.slices, pool)
    dt = time.perf_counter() - t0

    dst_bytes = sum(os.path.getsize(os.path.join(a.dest, f)) for f in files)
    if dst_bytes != src_bytes:
        raise SystemExit(f"SIZE MISMATCH src={src_bytes} dst={dst_bytes}")

    gbps = dst_bytes / dt / 1e9
    print(f"staged_files={len(files)} staged_bytes={dst_bytes} slices={a.slices} "
          f"stage_wall_s={dt:.2f} stage_GBps={gbps:.2f}", flush=True)


if __name__ == "__main__":
    main()
