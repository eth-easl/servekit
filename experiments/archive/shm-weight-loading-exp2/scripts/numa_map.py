#!/usr/bin/env python3
"""GPU -> local NUMA node map, one "<gpu> <node>" pair per line.

Used by both the stager (to place each rank's tmpfs pages on the node its GPU
will read them from) and by sanity checks. The mapping is not the identity on
this hardware -- Bristen's EPYC 7713 in NPS4 reports GPU0->node3, GPU1->node2,
GPU2->node1, GPU3->node0 -- so it must be queried, never assumed.

Falls back to scanning sysfs by PCI class if NVML is unavailable, and prints
nothing at all if neither works, which callers must treat as "do not bind"
rather than "bind to 0".
"""

import ctypes
import glob
import math
import sys


def via_nvml():
    import pynvml

    pynvml.nvmlInit()
    try:
        n_nodes = len(glob.glob("/sys/devices/system/node/node[0-9]*"))
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        setsize = max(1, math.ceil(n_nodes / bits))
        out = {}
        for gpu in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            mask = pynvml.nvmlDeviceGetMemoryAffinity(
                handle, setsize, pynvml.NVML_AFFINITY_SCOPE_NODE
            )
            nodes = [n for n in range(n_nodes) if mask[n // bits] & (1 << (n % bits))]
            if nodes:
                out[gpu] = nodes[0]
        return out
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def via_sysfs():
    """PCI-order fallback. Only correct when CUDA ordering is PCI_BUS_ID."""
    out, gpu = {}, 0
    for dev in sorted(glob.glob("/sys/bus/pci/devices/*/")):
        try:
            with open(dev + "class") as fh:
                if fh.read().strip() != "0x030200":  # 3D controller
                    continue
            with open(dev + "numa_node") as fh:
                node = int(fh.read().strip())
        except OSError:
            continue
        if node >= 0:
            out[gpu] = node
        gpu += 1
    return out


def main():
    mapping = {}
    for probe in (via_nvml, via_sysfs):
        try:
            mapping = probe()
        except Exception as exc:  # noqa: BLE001
            print(f"# {probe.__name__} failed: {exc}", file=sys.stderr)
        if mapping:
            break
    for gpu in sorted(mapping):
        print(f"{gpu} {mapping[gpu]}")


if __name__ == "__main__":
    main()
