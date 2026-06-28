// Probe: dlsym NCCL's __global__ host stub from libnccl, then
// cudaGetFuncBySymbol -> CUfunction. If this works, the rebuild resolver can
// resolve NCCL kernels by name without any fatbin parsing.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
int main() {
  CUcontext ctx{}; CUdevice dev{};
  cuInit(0); cuDeviceGet(&dev,0); cuDevicePrimaryCtxRetain(&ctx,dev); cuCtxSetCurrent(ctx);
  void* nccl = dlopen("libnccl.so.2", RTLD_NOW|RTLD_GLOBAL);
  if (!nccl) nccl = dlopen("libnccl.so", RTLD_NOW|RTLD_GLOBAL);
  fprintf(stderr,"[probe] dlopen libnccl=%p\n", nccl);
  const char* names[] = {
    "_Z40ncclDevKernel_AllReduce_Sum_bf16_RING_LL24ncclDevKernelArgsStorageILm4096EE",
    "_Z38ncclDevKernel_AllReduce_Sum_u8_RING_LL24ncclDevKernelArgsStorageILm4096EE",
    nullptr };
  for (int i=0; names[i]; ++i) {
    void* host = dlsym(nccl, names[i]);
    fprintf(stderr,"[probe] dlsym(%s) = %p\n", names[i], host);
    if (!host) { fprintf(stderr,"[probe]   dlerror: %s\n", dlerror()); continue; }
    CUfunction f{};
    cudaError_t rc = cudaGetFuncBySymbol(&f, host);
    fprintf(stderr,"[probe]   cudaGetFuncBySymbol -> rc=%d f=%p\n", rc, (void*)f);
    if (f) {
      const char* nm=nullptr;
      if (cuFuncGetName(&nm, f)==CUDA_SUCCESS) fprintf(stderr,"[probe]   cuFuncGetName=%s\n", nm?nm:"?");
    }
  }
  return 0;
}
