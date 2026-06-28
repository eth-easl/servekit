import ctypes, sys
import torch
torch.cuda.init()
lib = ctypes.CDLL("/tmp/hookfat.so")
lib.n5b_load_all.restype = ctypes.c_int
r = lib.n5b_load_all()
print("N5B_RESULT nccl_kernel_found =", r)
sys.exit(0)
