#!/usr/bin/env python3
# Parse a .snap file (record_cuda_format.hpp v2) and dump node tags + blind reasons.
import struct, sys

def parse(path):
    with open(path, "rb") as f:
        b = f.read()
    off = 0
    magic = b[off:off+8]; off += 8
    assert magic == b"SNAPCUD1", magic
    (ver,) = struct.unpack_from("<I", b, off); off += 4
    (nnodes,) = struct.unpack_from("<I", b, off); off += 4
    print(f"file={path} version={ver} nodes={nnodes}")
    tags = {0:"Kernel",1:"Memcpy",2:"Memset",3:"Blind"}
    reasons = {}
    for i in range(nnodes):
        (tag,) = struct.unpack_from("<B", b, off); off += 1
        t = tags.get(tag, f"?{tag}")
        if t == "Kernel":
            (kind,) = struct.unpack_from("<i", b, off); off += 4
            (mh,) = struct.unpack_from("<Q", b, off); off += 8
            (nl,) = struct.unpack_from("<I", b, off); off += 4
            name = b[off:off+nl].decode("utf8","replace"); off += nl
            grid = struct.unpack_from("<III", b, off); off += 12
            block = struct.unpack_from("<III", b, off); off += 12
            (smem,) = struct.unpack_from("<I", b, off); off += 4
            (kf,) = struct.unpack_from("<B", b, off); off += 1
            (ksz,) = struct.unpack_from("<I", b, off); off += 4
            off += ksz
            print(f"  [{i}] Kernel kind={kind} name={name!r} grid={grid} block={block}")
        elif t in ("Memcpy","Memset"):
            (sz,) = struct.unpack_from("<I", b, off); off += 4
            off += sz
            print(f"  [{i}] {t} blob={sz}B")
        else:
            (rl,) = struct.unpack_from("<I", b, off); off += 4
            reason = b[off:off+rl].decode("utf8","replace"); off += rl
            reasons[reason] = reasons.get(reason,0)+1
            print(f"  [{i}] Blind reason={reason!r}")
        (ndeps,) = struct.unpack_from("<I", b, off); off += 4
        off += 4*ndeps
    print("blind reason summary:", reasons)

if __name__ == "__main__":
    parse(sys.argv[1])
