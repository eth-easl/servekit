#!/usr/bin/env python3
# Scan libnccl's embedded CUDA fatbinaries and test whether the NCCL AllReduce
# kernel resolves via cuModuleLoadFatBinary + cuModuleGetFunction. Parses the
# ELF64 section headers directly (no objcopy dependency). Run INSIDE the vllm
# container. Usage: scan_nccl_fatbin.py [libnccl.so]
import os, sys, struct, ctypes

NCCL = sys.argv[1] if len(sys.argv) > 1 else \
    "/usr/lib/x86_64-linux-gnu/libnccl.so.2.28.9"
KNAME = ("_Z40ncclDevKernel_AllReduce_Sum_bf16_RING_LL"
         "24ncclDevKernelArgsStorageILm4096EE")

# --- 1. Parse ELF64 to extract .nv_fatbin section bytes ---
raw = open(NCCL, "rb").read()
assert raw[:4] == b"\x7fELF", "not ELF"
is64 = raw[4] == 2
assert is64, "not ELF64"
# ELF64 header offsets
e_shoff = struct.unpack_from("<Q", raw, 0x28)[0]
e_shentsize = struct.unpack_from("<H", raw, 0x3a)[0]
e_shnum = struct.unpack_from("<H", raw, 0x3c)[0]
e_shstrndx = struct.unpack_from("<H", raw, 0x3e)[0]

def shdr(i):
    base = e_shoff + i * e_shentsize
    name, typ, flags, addr, offset, size, link, info, align, entsize = \
        struct.unpack_from("<IIQQQQIIQQ", raw, base)
    return dict(name=name, type=typ, offset=offset, size=size, addr=addr)

# section name string table
strtab = shdr(e_shstrndx)
strdata = raw[strtab["offset"]:strtab["offset"] + strtab["size"]]

def secname(i):
    n = shdr(i)["name"]
    end = strdata.find(b"\x00", n)
    return strdata[n:end].decode()

fatbin_secs = [i for i in range(e_shnum) if secname(i) == ".nv_fatbin"]
print(f".nv_fatbin sections: {len(fatbin_secs)}")
assert fatbin_secs, "no .nv_fatbin section"
fs = shdr(fatbin_secs[0])
b = raw[fs["offset"]:fs["offset"] + fs["size"]]
print(f".nv_fatbin size = {len(b)}")

# --- 2. Find every 0xBA55ED50 magic (the driver knows the real format; we
#     do NOT validate the header ourselves — modern fatbins have a large header). ---
magic_offs = []
off = 0
while True:
    i = b.find(b"\x50\xed\x55\xba", off)
    if i < 0:
        break
    magic_offs.append(i)
    off = i + 4
print(f"magic hits = {len(magic_offs)}")

# --- 3. Try cuModuleLoadFatBinary on EVERY magic hit; search for the kernel. ---
cuda = ctypes.CDLL("libcuda.so.1")
cudart = ctypes.CDLL("libcudart.so")
cudart.cudaSetDevice(0)
mod = ctypes.c_void_p()
found = False
loaded = 0
for i in magic_offs:
    # Try a range of candidate sizes (we don't know the exact extent); pass the
    # whole remaining section from this offset — the driver stops at end-of-fatbin.
    fz = len(b) - i
    try:
        addr = ctypes.addressof((ctypes.c_ubyte * fz).from_buffer(b, i))
    except Exception:
        continue
    rc = cuda.cuModuleLoadFatBinary(ctypes.byref(mod), addr)
    if rc != 0 or not mod.value:
        continue
    loaded += 1
    func = ctypes.c_void_p()
    rc2 = cuda.cuModuleGetFunction(ctypes.byref(func), mod, KNAME.encode())
    if rc2 == 0 and func.value:
        found = True
        print(f"  KNAME FOUND at magic-off={i} func={func.value:#x}")
        name = ctypes.c_char_p()
        cuda.cuFuncGetName(ctypes.byref(name), func)
        print(f"  cuFuncGetName = {name.value}")
        break
    cuda.cuModuleUnload(mod)
print(f"fatbins_loaded={loaded} KNAME_FOUND={found}")
print("RESULT:", "RESOLVABLE-VIA-FATBIN" if found else "NOT-IN-FATBINS")
