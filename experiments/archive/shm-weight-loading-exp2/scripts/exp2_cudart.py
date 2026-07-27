"""Minimal ctypes bindings for the cudart calls torch does not expose.

torch.cuda.cudart() offers cudaHostRegister/cudaHostUnregister but not
cudaMemcpyAsync, which the register strategy needs in order to DMA straight
out of a registered tmpfs mapping.

The one non-obvious part is _find_loaded_cudart, and it is load-bearing.
torch ships its own libcudart (from the nvidia-cuda-runtime wheel) while the
system CUDA toolkit has another copy that ctypes.util.find_library resolves
to -- two separately loaded instances. A stream handle created through
torch's instance is opaque to a cudaMemcpyAsync issued through the other one,
and fails with cudaErrorInvalidValue no matter what else is fixed. dlopen on
the exact same path returns a handle to the already-mapped library, so
finding that path in /proc/self/maps and loading THAT is what makes torch's
stream handles usable here. (Learned the expensive way in
hugepage-sharded-loading-exp; see its NOTES.md.)
"""

import ctypes
import ctypes.util
import os

cudaMemcpyHostToDevice = 1
cudaHostRegisterPortable = 0x01


def _find_loaded_cudart():
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                if "libcudart.so" in line:
                    path = line.rstrip("\n").split()[-1]
                    if os.path.exists(path):
                        return path
    except OSError:
        pass
    return ctypes.util.find_library("cudart") or "libcudart.so.12"


LIBCUDART_PATH = _find_loaded_cudart()
libcudart = ctypes.CDLL(LIBCUDART_PATH)

libcudart.cudaGetLastError.restype = ctypes.c_int
libcudart.cudaGetLastError.argtypes = []
libcudart.cudaPeekAtLastError.restype = ctypes.c_int
libcudart.cudaPeekAtLastError.argtypes = []
libcudart.cudaStreamSynchronize.restype = ctypes.c_int
libcudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
libcudart.cudaHostRegister.restype = ctypes.c_int
libcudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
libcudart.cudaHostUnregister.restype = ctypes.c_int
libcudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
libcudart.cudaMemcpyAsync.restype = ctypes.c_int
libcudart.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
]
libcudart.cudaDeviceSynchronize.restype = ctypes.c_int
libcudart.cudaDeviceSynchronize.argtypes = []
