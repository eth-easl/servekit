#!/usr/bin/env python3
"""Stage daemon with NUMA-placed hugepages -- exp2's "page half", ported to
the memfd path.

The parent daemon (hugepage-sharded-loading-exp/scripts/hugepage_stage_daemon.py)
stages all four ranks' files into one memfd from one process with one big
ThreadPoolExecutor, so every hugepage lands on whatever node the thread that
faulted it happened to sit on. The GPU that later DMAs those bytes had no say
in it, and on an NPS4 node with a REVERSED GPU->node map (GPU0->node3 ...
GPU3->node0) that means most of every rank's read crosses the fabric.

exp2 fixed the equivalent problem on tmpfs by running each rank's writer under
`numactl --membind=<node local to GPU R>`. That approach does not port here:

  - there is one shared buffer, not one file per writer, so a process-wide
    memory policy is the wrong granularity; and
  - hugetlbfs pages are drawn from per-node pools at fault time, so placement
    has to be attached to the address range BEFORE the first touch.

So this uses mbind(2) directly on each file's region of the mapping. Regions
are already 2 MB-aligned by the parent stager's layout (every file starts at a
HUGEPAGE multiple), which mbind requires for a hugetlb mapping.

MPOL_BIND, not MPOL_PREFERRED, and that was measured rather than chosen.
MPOL_PREFERRED was the first attempt -- it degrades gracefully to a remote
page under pressure, which is the right trade for a staging buffer, whereas
MPOL_BIND can SIGBUS. It does not work here. `mbind_gate.py` /
`mbind_gate2.py` (jobs 76395-76400) established:

  - MPOL_PREFERRED returns 0 and /proc/self/numa_maps really does record
    `prefer:0/1/2/3` on the four VMAs -- and then every page lands on ONE
    node anyway: node0 when the job held 16 CPUs, node3 when it held 128.
  - Pinning the faulting thread to the target node does not fix it either
    (3/4 slices wrong).
  - MPOL_BIND places all 4/4 correctly.

The reason is that the persistent hugepage pool is empty on these nodes
(`HugePages_Total: 0`, `nr_overcommit_hugepages: 257505`), so every page is a
*surplus* hugetlb allocation, and that path does not honour a soft policy.
Only the hard nodemask survives to it. SIGBUS risk is accepted knowingly:
each node is ~128 GB and holds one rank's ~35 GB.

Usage: hpnuma_stage_daemon.py <checkpoint_dir> --sock <path> [--ready-file <p>]
                             [--no-place]
"""
import argparse
import ctypes
import ctypes.util
import glob
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "hugepage-sharded-loading-exp",
        "scripts",
    ),
)
import hugepage_fd_broker as broker  # noqa: E402
import hugepage_safetensors as hsf  # noqa: E402
import hugepage_stager as stager  # noqa: E402

MPOL_PREFERRED = 1
MPOL_BIND = 2
MPOL_INTERLEAVE = 3
RANK_RE = re.compile(r"model-rank-(\d+)-part-")

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)


def gpu_numa_map():
    """GPU -> local NUMA node. Never assume identity; on Bristen it is
    reversed. Returns {} if it cannot be determined, which callers must treat
    as "do not place" rather than "place on 0"."""
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "shm-weight-loading-exp2",
            "scripts",
        ),
    )
    import numa_map

    for probe in (numa_map.via_nvml, numa_map.via_sysfs):
        try:
            mapping = probe()
        except Exception as exc:  # noqa: BLE001
            print(f"# {probe.__name__} failed: {exc}", file=sys.stderr, flush=True)
            continue
        if mapping:
            return mapping
    return {}


def mbind_interleave(addr, length, n_nodes):
    """MPOL_INTERLEAVE across all nodes for [addr, addr+length).

    The default (no policy) already spreads a file's pages across nodes, but
    only incidentally -- via wherever each of the 448 staging threads happened
    to be running -- so the spread is uneven and varies run to run. Jobs
    76390/76401 showed that spread BEATS per-rank locality on this path
    (7.81 s unplaced vs 8.52 s placed), which makes a deliberate, even spread
    worth measuring against an accidental one.
    """
    nodemask = ctypes.c_ulong((1 << n_nodes) - 1)
    rc = _libc.syscall(
        237, ctypes.c_void_p(addr), ctypes.c_size_t(length),
        ctypes.c_int(MPOL_INTERLEAVE), ctypes.byref(nodemask),
        ctypes.c_ulong(64), ctypes.c_uint(0),
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), "mbind(interleave)")


def mbind_bind(addr, length, node):
    """Attach an MPOL_BIND policy for `node` to [addr, addr+length).

    maxnode is the number of BITS in the mask, and the kernel wants it to
    cover the mask word it is handed; nodes here are always < 64.
    """
    nodemask = ctypes.c_ulong(1 << node)
    rc = _libc.syscall(
        237,  # __NR_mbind on x86_64
        ctypes.c_void_p(addr),
        ctypes.c_size_t(length),
        ctypes.c_int(MPOL_BIND),
        ctypes.byref(nodemask),
        ctypes.c_ulong(64),
        ctypes.c_uint(0),
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), f"mbind(node={node})")


def stage_placed(src_dir, filenames, slices=64, place=True, interleave=False):
    """hugepage_stager.stage(), plus a per-file mbind before any page is
    touched. Returns the same tuple, so the broker is unchanged."""
    sizes = {f: os.path.getsize(os.path.join(src_dir, f)) for f in filenames}
    layout, running = {}, 0
    for f in filenames:
        layout[f] = running
        running += stager._round_up(sizes[f], stager.HUGEPAGE)
    total_len = running

    fd, buf = stager.create_memfd_hugetlb(total_len)
    base_addr = ctypes.addressof(buf)

    n_nodes = len([p for p in os.listdir("/sys/devices/system/node")
                   if p.startswith("node") and p[4:].isdigit()])

    mapping = {}
    if interleave:
        # One policy over the whole buffer -- no per-file granularity needed,
        # since every file gets the same even spread.
        mbind_interleave(base_addr, total_len, n_nodes)
        print(f"interleaved across {n_nodes} node(s)", flush=True)
    elif place:
        mapping = gpu_numa_map()
        if not mapping:
            print("WARNING: no GPU->node map; staging UNPLACED", flush=True)
        else:
            placed = {}
            for f in filenames:
                m = RANK_RE.search(f)
                if not m:
                    continue  # not a per-rank shard: leave to default policy
                node = mapping.get(int(m.group(1)))
                if node is None:
                    continue
                span = stager._round_up(sizes[f], stager.HUGEPAGE)
                mbind_bind(base_addr + layout[f], span, node)
                placed.setdefault(node, []).append(f)
            for node in sorted(placed):
                print(f"placed node{node}: {len(placed[node])} file(s) "
                      f"({', '.join(sorted(placed[node]))})", flush=True)

    mv = memoryview(buf)
    tasks = []
    for f in filenames:
        base = layout[f]
        for off, length, o_direct in stager._plan_slices(sizes[f], slices):
            tasks.append(
                (os.path.join(src_dir, f), mv, base + off, off, length, o_direct)
            )
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futs = [pool.submit(stager._read_slice, *t) for t in tasks]
        for fut in futs:
            fut.result()

    if mapping:
        report_placement(base_addr, layout, sizes, mapping)

    files_meta = {}
    for f in filenames:
        base = layout[f]
        header, data_start = hsf.parse_header(buf, base_offset=base)
        files_meta[f] = (base, header, data_start)
    return fd, total_len, files_meta, buf


def report_placement(base_addr, layout, sizes, mapping):
    """Where the pages ACTUALLY landed, per file, via move_pages(2).

    An mbind that returns 0 has only set a policy; it does not prove the
    fault honoured it. That is not a hypothetical here -- MPOL_PREFERRED
    returned 0, recorded `prefer:N` in numa_maps, and placed nothing (see the
    module docstring). Sampling the real node of a few pages per file is the
    only honest confirmation, and it is what decides whether a disappointing
    result means "placement did not help" or "placement did not happen".
    """
    _libc.syscall.restype = ctypes.c_long
    n = 16
    print("placement check (sampled, actual node per file):", flush=True)
    for f in sorted(sizes):
        m = RANK_RE.search(f)
        want = mapping.get(int(m.group(1))) if m else None
        span = stager._round_up(sizes[f], stager.HUGEPAGE)
        step = max(stager.HUGEPAGE, (span // n) - (span // n) % stager.HUGEPAGE)
        addrs = [base_addr + layout[f] + i * step for i in range(n)
                 if i * step < span]
        arr = (ctypes.c_void_p * len(addrs))(*addrs)
        status = (ctypes.c_int * len(addrs))(*([-1] * len(addrs)))
        rc = _libc.syscall(
            279,  # __NR_move_pages on x86_64; NULL nodes[] = query only
            ctypes.c_long(0), ctypes.c_ulong(len(addrs)), arr,
            ctypes.c_void_p(0), status, ctypes.c_int(0),
        )
        if rc != 0:
            print(f"  {f}: move_pages query failed errno={ctypes.get_errno()}",
                  flush=True)
            continue
        hist = {}
        for s in status:
            hist[s] = hist.get(s, 0) + 1
        got = ", ".join(f"node{k}={v}" for k, v in sorted(hist.items()))
        ok = "OK" if want is not None and list(hist) == [want] else "MIXED/UNPLACED"
        print(f"  {f}: want=node{want} got {got}  [{ok}]", flush=True)


_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--sock", required=True)
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--slices", type=int, default=64)
    ap.add_argument("--no-place", action="store_true",
                    help="stage without mbind -- byte-identical to the parent daemon")
    ap.add_argument("--interleave", action="store_true",
                    help="MPOL_INTERLEAVE over the whole buffer instead of "
                         "per-rank locality")
    a = ap.parse_args()

    filenames = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(a.checkpoint_dir, "*.safetensors"))
    )
    if not filenames:
        print(f"FATAL: no *.safetensors under {a.checkpoint_dir}", file=sys.stderr)
        raise SystemExit(1)

    place = not a.no_place
    print(f"staging {len(filenames)} file(s) from {a.checkpoint_dir} "
          f"(numa_place={place} interleave={a.interleave}) ...", flush=True)
    t0 = time.perf_counter()
    fd, total_len, files_meta, buf = stage_placed(
        a.checkpoint_dir, filenames, a.slices, place=place,
        interleave=a.interleave,
    )
    dt = time.perf_counter() - t0
    print(f"staged {total_len/1e9:.2f} GB in {dt:.2f}s ({total_len/dt/1e9:.2f} GB/s)",
          flush=True)

    if os.path.exists(a.sock):
        os.unlink(a.sock)
    srv = broker.FdBrokerServer(a.sock, fd, total_len, files_meta)
    srv.start()
    print(f"broker listening on {a.sock}", flush=True)

    if a.ready_file:
        with open(a.ready_file, "w") as f:
            f.write("ready\n")
        print(f"wrote ready-file {a.ready_file}", flush=True)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    while not _stop:
        time.sleep(1)
    print("stopping", flush=True)
    srv.stop()


if __name__ == "__main__":
    main()
