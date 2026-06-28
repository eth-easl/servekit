// _probe/test_register_fatbin.cpp — capture __cudaRegisterFatBinary args at
// load time, then DEFER-load them (after torch.cuda.init creates a context).
// Exposes n5b_load_all() for Python/ctypes to call post-init.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>
#include <cuda.h>

typedef void* (*regfat_t)(const void*);
static regfat_t real_regfat = nullptr;
static std::vector<const void*> g_ptrs;

extern "C" {
void* __cudaRegisterFatBinary(const void* arg) {
  if (!real_regfat)
    real_regfat = (regfat_t)dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
  if (arg) g_ptrs.push_back(arg);
  return real_regfat ? real_regfat(arg) : nullptr;
}

// Call this from Python AFTER torch.cuda.init(). Returns number of modules that
// loaded AND contained the NCCL kernel.
int n5b_load_all() {
  CUcontext ctx = nullptr;
  if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) {
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != CUDA_SUCCESS || !ctx) {
      fprintf(stderr, "[n5b] failed to retain primary context\n");
      return -1;
    }
    cuCtxSetCurrent(ctx);
  }
  // DIAGNOSTIC ONLY (safe): print the byte layout of each captured fatBinEntry
  // to identify the binary-pointer offset, then load via the correct field.
  const char* KNAME = "_Z40ncclDevKernel_AllReduce_Sum_bf16_RING_LL"
                      "24ncclDevKernelArgsStorageILm4096EE";
  int found = 0, loaded = 0;
  for (size_t i = 0; i < g_ptrs.size(); ++i) {
    const char* arg = (const char*)g_ptrs[i];
    unsigned int magic = *(unsigned int*)(arg + 0);
    // fatBinEntry magic is 0x466243b1; binary ptr is at offset 8 (after
    // magic u32 + version u32). Validate before deref.
    if (magic != 0x466243b1u) {
      if (i < 6) fprintf(stderr, "[n5b] ptr#%zu magic=%08x (not fatBinEntry)\n",
                         i, magic);
      continue;
    }
    void* binary = *(void**)(arg + 8);
    if (!binary) continue;
    unsigned int bmagic = *(unsigned int*)binary;
    if (i < 10)
      fprintf(stderr, "[n5b] ptr#%zu fatBinEntry binary=%p bmagic=%08x\n",
              i, binary, bmagic);
    CUmodule mod = nullptr;
    CUresult rc = cuModuleLoadFatBinary(&mod, binary);
    if (rc != CUDA_SUCCESS || !mod) {
      if (i < 10) fprintf(stderr, "[n5b]   load FAIL(%d)\n", (int)rc);
      continue;
    }
    ++loaded;
    unsigned int fc = 0; cuModuleGetFunctionCount(&fc, mod);
    // Enumerate ALL functions and print any whose name contains nccl/allreduce.
    if (fc) {
      std::vector<CUfunction> funcs(fc);
      if (cuModuleEnumerateFunctions(funcs.data(), fc, mod) == CUDA_SUCCESS) {
        for (CUfunction ff : funcs) {
          const char* fnm = nullptr;
          if (cuFuncGetName(&fnm, ff) == CUDA_SUCCESS && fnm && *fnm) {
            std::string s(fnm);
            if (s.find("nccl") != std::string::npos ||
                s.find("AllReduce") != std::string::npos ||
                s.find("ncclDev") != std::string::npos)
              fprintf(stderr, "[n5b]   NCCL-LIKE func in ptr#%zu: %s\n", i, fnm);
          }
        }
      }
    }
    CUfunction f = nullptr;
    cuModuleGetFunction(&f, mod, KNAME);
    const char* nm = nullptr;
    if (f) cuFuncGetName(&nm, f);
    fprintf(stderr, "[n5b] ptr#%zu load=OK funcs=%u nccl=%s%s%s\n",
            i, fc, f ? "FOUND" : "no", nm ? " name=" : "", nm ? nm : "");
    if (f) found = 1;
    cuModuleUnload(mod);
  }
  fprintf(stderr, "[n5b] total ptrs=%zu loaded=%d nccl_kernel_found=%d\n",
          g_ptrs.size(), loaded, found);
  return found;
}
}
