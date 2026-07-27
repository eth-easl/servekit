#!/usr/bin/env python3
"""Phase 0 gate for STEP4_PLAN.md.

Two independent questions, both must pass before Step 4 gets built:

  0a. Is hugetlbfs usable in the container? Try, in order, stop at first
      success: mmap(MAP_HUGETLB) -> memfd_create(MFD_HUGETLB) -> mount
      hugetlbfs -> /dev/hugepages. Verify the mapping is REALLY huge via
      /proc/self/smaps (KernelPageSize) and /proc/meminfo (HugePages_Surp),
      not just that the syscall returned success.

  0b. Is cudaHostRegister dramatically faster on 2 MB pages than the
      measured 4 KB tmpfs baseline (2.59-4.56 GB/s, see NOTES.md)? Whole-shard
      registration only pays off at >=40 GB/s (141 GB in <4s, amortized over
      ~989 tensors/shard).

Pass criteria: hugepages allocatable >=8 GB AND registration >=40 GB/s. If
registration stays single-digit GB/s even on hugepages, in-place registration
is dead regardless of which staging route wins 0a -- fall back to fixing
h2d_pinned_staging.patch's blocking copy instead (STEP4_PLAN.md Phase 0).

Usage: hugetlbfs_gate.py [--gb 8] [--gpu 0]
"""
import argparse
import ctypes
import ctypes.util
import os
import subprocess
import time

import torch

PROT_READ, PROT_WRITE = 0x1, 0x2
MAP_SHARED, MAP_ANONYMOUS, MAP_HUGETLB = 0x01, 0x20, 0x40000
MFD_HUGETLB, MFD_CLOEXEC = 0x0004, 0x0001
MAP_FAILED = (1 << 64) - 1  # ctypes c_void_p wraps -1 to this on 64-bit

libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def smaps_entry(addr):
    """Find the /proc/self/smaps block covering `addr`, return its fields."""
    with open("/proc/self/smaps") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        head = line.split()[0]
        if "-" not in head:
            continue
        try:
            start_s, end_s = head.split("-")
            start, end = int(start_s, 16), int(end_s, 16)
        except ValueError:
            continue
        if not (start <= addr < end):
            continue
        block = {}
        for l2 in lines[i + 1:]:
            h2 = l2.split()[0]
            if "-" in h2 and ":" not in h2:
                break
            if ":" in l2:
                k, v = l2.split(":", 1)
                block[k.strip()] = v.strip()
        return block
    return None


def route_mmap_hugetlb(nbytes):
    addr = libc.mmap(None, nbytes, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0)
    if addr is None or addr == MAP_FAILED:
        return None, None, os.strerror(ctypes.get_errno())
    return addr, None, None


def route_memfd_hugetlb(nbytes):
    if not hasattr(libc, "memfd_create"):
        return None, None, "memfd_create not exported by libc"
    libc.memfd_create.restype = ctypes.c_int
    libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    fd = libc.memfd_create(b"hugetlbfs_gate", MFD_HUGETLB | MFD_CLOEXEC)
    if fd < 0:
        return None, None, os.strerror(ctypes.get_errno())
    os.ftruncate(fd, nbytes)
    addr = libc.mmap(None, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
    if addr is None or addr == MAP_FAILED:
        err = os.strerror(ctypes.get_errno())
        os.close(fd)
        return None, None, err
    return addr, fd, None


def route_mount_hugetlbfs(mnt="/mnt/huge_gate"):
    os.makedirs(mnt, exist_ok=True)
    r = subprocess.run(["mount", "-t", "hugetlbfs", "none", mnt],
                        capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()
    return mnt, None


def route_dev_hugepages(path="/dev/hugepages"):
    if not os.path.isdir(path):
        return False, "no such directory"
    if not os.access(path, os.W_OK):
        return False, "not writable (root-owned 0755, as expected)"
    return True, None


def probe_0a(nbytes):
    print(f"=== 0a: hugetlbfs usability (requesting {nbytes/1e9:.2f} GB) ===", flush=True)
    meminfo_before = read_meminfo()

    print("-- route 1: mmap(MAP_SHARED|MAP_ANONYMOUS|MAP_HUGETLB) --", flush=True)
    addr, fd, err = route_mmap_hugetlb(nbytes)
    if addr is not None:
        route = "mmap_anon_hugetlb"
    else:
        print(f"   failed: {err}", flush=True)
        print("-- route 2: memfd_create(MFD_HUGETLB) + ftruncate + mmap --", flush=True)
        addr, fd, err = route_memfd_hugetlb(nbytes)
        route = "memfd_hugetlb" if addr is not None else None
        if addr is None:
            print(f"   failed: {err}", flush=True)

    if addr is None:
        print("-- route 3: mount -t hugetlbfs --", flush=True)
        mnt, err = route_mount_hugetlbfs()
        if mnt is not None:
            print(f"   mounted at {mnt} (no anonymous mapping tested -- would need a file "
                  "inside it, which changes Phase 1's design)", flush=True)
            route = "mount_hugetlbfs"
        else:
            print(f"   failed: {err}", flush=True)
            print("-- route 4: /dev/hugepages write access --", flush=True)
            ok, err = route_dev_hugepages()
            route = "dev_hugepages" if ok else None
            print(f"   {'writable' if ok else 'failed: ' + str(err)}", flush=True)

    if addr is None and route not in ("mount_hugetlbfs",):
        print("RESULT 0a: FAIL -- no route to hugetlbfs/hugepages worked", flush=True)
        return None, None, 0

    if addr is None:
        print("RESULT 0a: PARTIAL -- mount worked but no anonymous-mapping route did; "
              "Phase 1 needs a filesystem path anyway, so this is workable but untested here",
              flush=True)
        return None, route, 0

    # Touch every page so the kernel actually backs the mapping (lazy alloc otherwise).
    PAGE_2MB = 2 * 1024 * 1024
    t0 = time.perf_counter()
    ctypes.memset(addr, 0xAB, nbytes)
    touch_dt = time.perf_counter() - t0
    print(f"   touched (memset) {nbytes/1e9:.2f} GB in {touch_dt:.2f}s", flush=True)

    entry = smaps_entry(addr)
    kps = entry.get("KernelPageSize") if entry else None
    print(f"   /proc/self/smaps KernelPageSize: {kps}", flush=True)

    meminfo_after = read_meminfo()
    surp_before = meminfo_before.get("HugePages_Surp", "?")
    surp_after = meminfo_after.get("HugePages_Surp", "?")
    print(f"   /proc/meminfo HugePages_Surp: {surp_before} -> {surp_after}", flush=True)

    really_huge = bool(kps and kps.strip().startswith("2048"))
    print(f"RESULT 0a: route={route} allocated={nbytes/1e9:.2f} GB "
          f"really_huge={really_huge}", flush=True)

    return addr, route, (nbytes if really_huge else 0)


def bench_register(addr, nbytes, gpu):
    torch.cuda.set_device(gpu)
    dev = torch.device(f"cuda:{gpu}")
    torch.empty(1, device=dev)  # force context creation outside the timer
    cudart = torch.cuda.cudart()

    t0 = time.perf_counter()
    rc = cudart.cudaHostRegister(addr, nbytes, 0)
    dt = time.perf_counter() - t0
    if int(rc) != 0:
        return None, f"cudaHostRegister failed rc={int(rc)}"
    cudart.cudaHostUnregister(addr)
    return nbytes / dt / 1e9, None


def probe_0b(addr, nbytes, gpu):
    print(f"\n=== 0b: cudaHostRegister throughput on hugepage-backed memory ===", flush=True)
    if addr is None:
        print("SKIPPED -- 0a produced no anonymous mapping to register", flush=True)
        return None
    gbps, err = bench_register(addr, nbytes, gpu)
    if err:
        print(f"RESULT 0b: FAIL -- {err}", flush=True)
        return None
    print(f"   cudaHostRegister: {gbps:.2f} GB/s -> 141 GB would take {141/gbps:.1f}s "
          f"(baseline on 4 KB tmpfs: 2.59-4.56 GB/s, 31-54s)", flush=True)
    print(f"RESULT 0b: {'PASS' if gbps >= 40 else 'FAIL'} "
          f"(need >=40 GB/s, got {gbps:.2f})", flush=True)
    return gbps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=8.0)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    nbytes = int(a.gb * 1e9)
    nbytes -= nbytes % 4096

    addr, route, huge_bytes = probe_0a(nbytes)
    gbps = probe_0b(addr, huge_bytes, a.gpu)

    if addr is not None:
        libc.munmap(addr, nbytes)

    pass_0a = huge_bytes > 0 and huge_bytes >= min(nbytes, int(8e9))
    pass_0b = gbps is not None and gbps >= 40
    print(f"\n=== GATE: {'PASS' if (pass_0a and pass_0b) else 'FAIL'} "
          f"(0a={'pass' if pass_0a else 'fail'}, 0b={'pass' if pass_0b else 'fail'}) ===",
          flush=True)
    if not (pass_0a and pass_0b):
        print("Per STEP4_PLAN.md: stop here, fall back to fixing "
              "h2d_pinned_staging.patch's blocking copy instead.", flush=True)


if __name__ == "__main__":
    main()
