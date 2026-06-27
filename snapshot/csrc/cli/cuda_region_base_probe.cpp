// snapshot/csrc/cli/cuda_region_base_probe.cpp
// N5b Task 3: prove snapshot_record_cuda_region_base() returns the redirect's
// fixed base (non-zero) once the redirect is engaged. Run under
// LD_PRELOAD=redirect:record. A tiny cudaMalloc triggers the redirect's lazy
// region reservation (ensure_init), which sets g_base; the export then reads it.
//
// Compiled as a CUDA TU (LANGUAGE CUDA) so cudaMalloc is available; links only
// libcudart + libdl (no snapshot_core, no nvrtc).

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>

int main(void) {
  // Trigger the redirect's lazy region reservation (ensure_init on first
  // cudaMalloc). If the redirect is not LD_PRELOAD'd, this is a plain malloc.
  void* p = nullptr;
  if (cudaMalloc(&p, 1024) != cudaSuccess) p = nullptr;

  using Fn = std::uint64_t (*)();
  void* sym = dlsym(RTLD_DEFAULT, "snapshot_record_cuda_region_base");
  if (sym == nullptr) {
    std::printf("REGION_BASE=0 (symbol not found)\n");
    return 1;
  }
  const std::uint64_t b = reinterpret_cast<Fn>(sym)();
  std::printf("REGION_BASE=0x%llx\n", static_cast<unsigned long long>(b));

  if (p) cudaFree(p);
  return (b != 0) ? 0 : 1;
}
