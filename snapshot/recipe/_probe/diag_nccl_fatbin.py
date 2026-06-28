#!/usr/bin/env python3
# Diagnose the actual fatbin format in libnccl's .nv_fatbin section. Dumps the
# bytes around every magic hit and also probes for alternative container magics.
import sys, struct

NCCL = sys.argv[1] if len(sys.argv) > 1 else \
    "/usr/lib/x86_64-linux-gnu/libnccl.so.2.28.9"
raw = open(NCCL, "rb").read()
e_shoff = struct.unpack_from("<Q", raw, 0x28)[0]
e_shentsize = struct.unpack_from("<H", raw, 0x3a)[0]
e_shnum = struct.unpack_from("<H", raw, 0x3c)[0]
e_shstrndx = struct.unpack_from("<H", raw, 0x3e)[0]
def shdr(i):
    base = e_shoff + i * e_shentsize
    n, t, fl, ad, off, sz = struct.unpack_from("<IIQQQQ", raw, base)
    return dict(name=n, offset=off, size=sz)
strtab = shdr(e_shstrndx)
strdata = raw[strtab["offset"]:strtab["offset"]+strtab["size"]]
def secname(i):
    n = shdr(i)["name"]; end = strdata.find(b"\x00", n)
    return strdata[n:end].decode()
secs = {secname(i): shdr(i) for i in range(e_shnum)}
print("CUDA-related sections:")
for nm in secs:
    if "nv" in nm.lower() or "cuda" in nm.lower() or "fat" in nm.lower():
        print(f"  {nm}: off={secs[nm]['offset']:#x} size={secs[nm]['size']}")

b = raw[secs[".nv_fatbin"]["offset"]:secs[".nv_fatbin"]["offset"]+secs[".nv_fatbin"]["size"]]
print(f"\n.nv_fatbin first 64 bytes:\n{b[:64].hex(' ')}")
# Search for known magics
for label, magic in [("old-fatbin 0xBA55ED50", b"\x50\xed\x55\xba"),
                     ("new-fatbin 0xEF66AEEF", b"\xef\xae\x66\xef"),
                     ("cubin-ELF 7f454c46", b"\x7fELF"),
                     ("PTX-version", b"version")]:
    cnt = 0
    off = 0
    first = -1
    while True:
        i = b.find(magic, off)
        if i < 0: break
        if first < 0: first = i
        cnt += 1; off = i + 1
    print(f"{label}: {cnt} hits, first at section-off {first:#x}" if cnt else f"{label}: 0")

# Dump bytes around first old-fatbin magic
print("\nbytes around first 0xBA55ED50 magic:")
i = b.find(b"\x50\xed\x55\xba")
if i >= 0:
    chunk = b[max(0,i-8):i+48]
    print(f"  off={i:#x}: {chunk.hex(' ')}")
    if i+16 <= len(b):
        mg, ver, hs, fs = struct.unpack_from("<IIII", b, i)
        print(f"  mg={mg:#x} ver={ver} hdrSz={hs} fatSz={fs}")
        print(f"  hdrSz in [16,4096]? {16<=hs<=4096}; fatSz<=section? {fs<=len(b)}; fits? {i+fs<=len(b)}")

# Dump bytes around first new-fatbin magic
i2 = b.find(b"\xef\xae\x66\xef")
if i2 >= 0:
    print(f"\nbytes around first 0xEF66AEEF magic at {i2:#x}:")
    chunk = b[max(0,i2-8):i2+48]
    print(f"  {chunk.hex(' ')}")

# Look at .nvFatBinSegment (the pointer/entry table)
if ".nvFatBinSegment" in secs:
    seg = secs[".nvFatBinSegment"]
    sb = raw[seg["offset"]:seg["offset"]+seg["size"]]
    print(f"\n.nvFatBinSegment size={len(sb)} (entry table)")
    print(f"  first 64 bytes: {sb[:64].hex(' ')}")
