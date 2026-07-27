import ctypes, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpnuma_stage_daemon as d
import hugepage_stager as stager
print("nr_overcommit_hugepages:", open("/proc/sys/vm/nr_overcommit_hugepages").read().strip())
SLICE = 64 << 20
total = SLICE * 4
fd, buf = stager.create_memfd_hugetlb(total)
base = ctypes.addressof(buf)
print("base=%x" % base)
for node in range(4):
    d.mbind_preferred(base + node*SLICE, SLICE, node)
for off in range(0, total, stager.HUGEPAGE):
    buf[off] = 1
print("--- numa_maps entries covering the memfd ---")
for line in open("/proc/self/numa_maps"):
    a = int(line.split()[0], 16)
    if base <= a < base + total:
        print(line.rstrip()[:250])
print("--- meminfo after touch ---")
print(open("/proc/meminfo").read().count("x") and "")
for l in open("/proc/meminfo"):
    if "Huge" in l: print("  " + l.rstrip())
