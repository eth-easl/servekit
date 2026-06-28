// _probe/test_nccl_dlsym.cpp — verify dlsym + cudaGetFuncBySymbol resolves
// the NCCL AllReduce kernel. Compiled + run INSIDE the vllm container.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

/* The single NCCL kernel that blocks 100% of recorded vLLM graphs. */
#define KNAME \
  "_Z40ncclDevKernel_AllReduce_Sum_bf16_RING_LL24ncclDevKernelArgsStorageILm4096EE"

int main(void) {
  CUfunction cufunc = nullptr;
  /* Force libnccl to be loaded (Python/torch load it; here we dlopen it). */
  void* nccl = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
  if (!nccl) { printf("dlopen libnccl FAIL: %s\n", dlerror()); return 1; }
  printf("libnccl loaded at %p\n", nccl);

  dlerror();
  void* sym = dlsym(RTLD_DEFAULT, KNAME);
  const char* err = dlerror();
  if (err) { printf("dlsym RTLD_DEFAULT err: %s\n", err); }
  printf("dlsym(RTLD_DEFAULT, KNAME) = %p\n", sym);
  if (!sym) {
    /* Try the library-local handle (symbol may not be in global scope). */
    sym = dlsym(nccl, KNAME);
    printf("dlsym(nccl, KNAME) = %p\n", sym);
  }

  if (!sym) { printf("SYMBOL NOT FOUND — host stub not exported.\n"); return 2; }

  /* Initialize a CUDA context (cudaGetFuncBySymbol needs one). */
  int dev = 0; cudaSetDevice(dev);
  /* The real call: map host stub -> CUfunction. */
  cudaError_t ce = cudaGetFuncBySymbol((cudaFunction_t*)&cufunc, sym);
  printf("cudaGetFuncBySymbol -> err=%d cufunc=%p\n", (int)ce, (void*)cufunc);
  if (ce != cudaSuccess || !cufunc) {
    printf("cudaGetFuncBySymbol FAIL: %s\n", cudaGetErrorString(ce));
    return 3;
  }
  const char* dn = nullptr;
  CUresult r = cuFuncGetName(&dn, cufunc);
  printf("cuFuncGetName -> r=%d name=%s\n", (int)r, dn ? dn : "(null)");
  printf("RESULT: %s\n",
         (r == CUDA_SUCCESS && dn && dn[0]) ? "RESOLVABLE" : "NO-NAME");
  return 0;
}
