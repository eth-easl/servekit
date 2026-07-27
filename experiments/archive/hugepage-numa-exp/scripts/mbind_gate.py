#!/usr/bin/env python3
"""Cheap gate on the mbind/move_pages plumbing before spending an e2e job.

Allocates a small memfd(MFD_HUGETLB) region, mbinds each 64 MB slice to a
different NUMA node, touches it, and asks move_pages(2) where the pages
actually landed. Wrong syscall marshalling shows up here in seconds instead
of as a null result at the end of a 6-minute run.
"""
import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpnuma_stage_daemon as d  # noqa: E402
import hugepage_stager as stager  # noqa: E402

SLICE = 64 << 20
n_nodes = len([p for p in os.listdir("/sys/devices/system/node")
               if p.startswith("node") and p[4:].isdigit()])
print(f"nodes={n_nodes}")

total = SLICE * n_nodes
fd, buf = stager.create_memfd_hugetlb(total)
base = ctypes.addressof(buf)

for node in range(n_nodes):
    d.mbind_bind(base + node * SLICE, SLICE, node)
    print(f"mbind ok node={node}")

# Touch every huge page so the policy is exercised by a real fault.
# (Assign through the ctypes array, not a memoryview of it -- a memoryview
# over a c_ubyte array carries format '<B' and refuses item assignment.)
for off in range(0, total, stager.HUGEPAGE):
    buf[off] = 1

d._libc.syscall.restype = ctypes.c_long
fail = 0
for node in range(n_nodes):
    addrs = [base + node * SLICE + i * stager.HUGEPAGE for i in range(8)]
    arr = (ctypes.c_void_p * len(addrs))(*addrs)
    status = (ctypes.c_int * len(addrs))(*([-1] * len(addrs)))
    rc = d._libc.syscall(279, ctypes.c_long(0), ctypes.c_ulong(len(addrs)),
                         arr, ctypes.c_void_p(0), status, ctypes.c_int(0))
    got = sorted(set(status))
    ok = rc == 0 and got == [node]
    fail += not ok
    print(f"want node{node}: rc={rc} got={got} [{'OK' if ok else 'FAIL'}]")

print("GATE PASS" if not fail else f"GATE FAIL ({fail} node(s))")
raise SystemExit(1 if fail else 0)
