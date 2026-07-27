#!/usr/bin/env python3
"""Gate v2: mbind is not enough for surplus hugepages. What is?

Gate v1 established, on a full exclusive node:
  - mbind(MPOL_PREFERRED) returns 0 and /proc/self/numa_maps really does show
    `prefer:0/1/2/3` on the four VMAs, so the policy IS recorded; and
  - every page still landed on ONE node -- node0 when the job held 16 CPUs,
    node3 when it held all 128 -- i.e. placement followed the FAULTING
    THREAD's CPU, not the policy.

The persistent hugepage pool is empty here (HugePages_Total: 0,
nr_overcommit_hugepages: 257505), so every page is a *surplus* hugetlb
allocation. That path does not honour the vma mempolicy the way the
persistent-pool path does.

Two candidate fixes, tested side by side on the same buffer:
  A. MPOL_BIND instead of MPOL_PREFERRED -- a hard nodemask rather than a
     hint. May be honoured where the hint was dropped.
  B. First-touch from a thread pinned to the target node -- lean into the
     observed behaviour instead of fighting it. This is what exp2 did on
     tmpfs (`numactl --membind` around each writer), just expressed per
     thread because here there is one shared buffer, not one file per writer.

Reports the actual landing node per slice via move_pages(2) for both.
"""
import ctypes
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpnuma_stage_daemon as d  # noqa: E402
import hugepage_stager as stager  # noqa: E402

SLICE = 64 << 20
N_NODES = len([p for p in os.listdir("/sys/devices/system/node")
               if p.startswith("node") and p[4:].isdigit()])


def node_cpus(node):
    with open(f"/sys/devices/system/node/node{node}/cpulist") as fh:
        spec = fh.read().strip()
    cpus = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def mbind_mode(addr, length, node, mode):
    nodemask = ctypes.c_ulong(1 << node)
    rc = d._libc.syscall(237, ctypes.c_void_p(addr), ctypes.c_size_t(length),
                         ctypes.c_int(mode), ctypes.byref(nodemask),
                         ctypes.c_ulong(64), ctypes.c_uint(0))
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), f"mbind(mode={mode},node={node})")


def where(base, off, n=8):
    addrs = [base + off + i * stager.HUGEPAGE for i in range(n)]
    arr = (ctypes.c_void_p * n)(*addrs)
    status = (ctypes.c_int * n)(*([-1] * n))
    rc = d._libc.syscall(279, ctypes.c_long(0), ctypes.c_ulong(n), arr,
                         ctypes.c_void_p(0), status, ctypes.c_int(0))
    return rc, sorted(set(status))


def run(label, toucher):
    total = SLICE * N_NODES
    fd, buf = stager.create_memfd_hugetlb(total)
    base = ctypes.addressof(buf)
    d._libc.syscall.restype = ctypes.c_long
    fails = 0
    for node in range(N_NODES):
        toucher(buf, base, node)
    for node in range(N_NODES):
        rc, got = where(base, node * SLICE)
        ok = rc == 0 and got == [node]
        fails += not ok
        print(f"  [{label}] want node{node}: got={got} [{'OK' if ok else 'FAIL'}]",
              flush=True)
    os.close(fd)
    return fails


def touch_mpol_bind(buf, base, node):
    mbind_mode(base + node * SLICE, SLICE, node, d.MPOL_BIND)
    for off in range(node * SLICE, (node + 1) * SLICE, stager.HUGEPAGE):
        buf[off] = 1


def touch_pinned_thread(buf, base, node):
    """First-touch from a thread pinned to the target node's CPUs."""
    def work():
        os.sched_setaffinity(0, node_cpus(node))
        for off in range(node * SLICE, (node + 1) * SLICE, stager.HUGEPAGE):
            buf[off] = 1
    t = threading.Thread(target=work)
    t.start()
    t.join()


def touch_both(buf, base, node):
    mbind_mode(base + node * SLICE, SLICE, node, d.MPOL_BIND)
    touch_pinned_thread(buf, base, node)


print(f"nodes={N_NODES} cpus_available={len(os.sched_getaffinity(0))}")
results = {}
for label, fn in (("A mpol_bind", touch_mpol_bind),
                  ("B pinned_touch", touch_pinned_thread),
                  ("C both", touch_both)):
    try:
        results[label] = run(label, fn)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] raised {exc!r}", flush=True)
        results[label] = N_NODES

for label, fails in results.items():
    print(f"{label}: {'PASS' if fails == 0 else f'{fails}/{N_NODES} wrong'}")
raise SystemExit(0 if min(results.values()) == 0 else 1)
