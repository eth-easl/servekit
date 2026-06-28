#!/usr/bin/env python3
# Scan recorded .snap files: count NCCL-contaminated vs rebuildable graphs,
# and tally node-type / blind-reason distributions.
#
# Format (record_cuda_format.hpp v2): 8B magic "SNAPCUD1", u32 version(=2),
# u32 node_count, then per node:
#   tag u8  (0=Kernel,1=Memcpy,2=Memset,3=Sync,255=Blind)
#   Kernel: kind i32, module_hash u64, name_len u32, name, grid u32x3,
#           block u32x3, smem u32, kernarg_form u8, kernarg_size u32, kernarg
#   Memcpy/Memset: blob_size u32, blob
#   Sync: (no payload)
#   Blind: reason_len u32, reason
#   then deps: dep_count u32, dep_indices u32[dep_count]
import struct, sys, glob, os

SYNC, BLIND = 3, 255

def parse_nodes(b):
    off = 8  # skip magic
    (ver,) = struct.unpack_from("<I", b, off); off += 4
    if ver != 2:
        raise ValueError(f"bad version {ver}")
    (nn,) = struct.unpack_from("<I", b, off); off += 4
    nodes = []
    for _ in range(nn):
        (tag,) = struct.unpack_from("<B", b, off); off += 1
        name = None
        if tag == 0:  # Kernel
            (kind,) = struct.unpack_from("<i", b, off); off += 4
            off += 8  # module_hash
            (nl,) = struct.unpack_from("<I", b, off); off += 4
            name = b[off:off+nl].decode("utf8", "replace"); off += nl
            off += 12 + 12 + 4  # grid + block + smem
            off += 1  # kernarg_form
            (ksz,) = struct.unpack_from("<I", b, off); off += 4; off += ksz
        elif tag in (1, 2):  # Memcpy / Memset
            (sz,) = struct.unpack_from("<I", b, off); off += 4; off += sz
        elif tag == SYNC:  # no payload
            pass
        else:  # Blind (255) or unknown
            (rl,) = struct.unpack_from("<I", b, off); off += 4; off += rl
        (nd,) = struct.unpack_from("<I", b, off); off += 4; off += 4 * nd
        nodes.append((tag, name))
    return nodes

def main(root):
    for rank in ["0", "1", "2", "3"]:
        files = sorted(glob.glob(os.path.join(root, f"rank{rank}", "*.snap")))
        total = nccl = clean = blindg = syncg = 0
        tagcount = {0: 0, 1: 0, 2: 0, SYNC: 0, BLIND: 0}
        allnames = {}
        for f in files:
            try:
                ns = parse_nodes(open(f, "rb").read())
            except Exception:
                continue
            total += 1
            names = [n for (t, n) in ns if t == 0 and n]
            for (t, _) in ns:
                tagcount[t] = tagcount.get(t, 0) + 1
            if any(t == BLIND for (t, _) in ns):
                blindg += 1
            if any(t == SYNC for (t, _) in ns):
                syncg += 1
            if any("nccl" in n.lower() for n in names):
                nccl += 1
            else:
                clean += 1
            for n in names:
                allnames[n] = allnames.get(n, 0) + 1
        pct = 100.0 * clean / max(total, 1)
        print(f"rank{rank}: graphs={total} "
              f"nccl_graphs={nccl} clean(rebuildable)={clean} ({pct:.1f}%) "
              f"blind_graphs={blindg} sync_graphs={syncg}")
        print(f"  nodes by tag: Kernel={tagcount[0]} Memcpy={tagcount[1]} "
              f"Memset={tagcount[2]} Sync={tagcount[SYNC]} "
              f"Blind={tagcount[BLIND]}")
        print(f"  unique kernels={len(allnames)} total kernel launches="
              f"{tagcount[0]}")
        ncclnames = sorted(set(n for n in allnames if "nccl" in n.lower()))
        if ncclnames:
            print(f"  unique nccl kernels: {len(ncclnames)}")
            for n in ncclnames[:4]:
                print(f"    {n[:90]}")
        nonnccl_sample = sorted(allnames, key=lambda k: -allnames[k])[:3]
        for n in nonnccl_sample:
            print(f"    top kernel ({allnames[n]}x): {n[:70]}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "snap-n5b")
