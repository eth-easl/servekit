// A standalone CUDA program that uses *raw* cudaMalloc (no snapshot allocator)
// and runs the synthetic 3-kernel chain via the driver API. Run normally it
// allocates wherever the driver chooses; run under libsnapshot_redirect_cuda.so
// its cudaMalloc is served from the deterministic fixed-base region, so device
// addresses become reproducible across runs. Either way it must compute the
// same bit-identical result, proving the redirect is transparent.
//
// Kernels are launched with the cuLaunchKernel kernelParams pointer-array form
// (one void* per argument) rather than the packed-buffer form, so the driver
// sizes each argument from the function signature — sidestepping the N1
// exact-kernarg-size validation entirely.

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "snapshot/workload_kernels.hpp"

namespace {

#define RK(call)                                                            \
  do {                                                                      \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      std::fprintf(stderr, "FAIL %s: %s\n", #call, cudaGetErrorString(_e)); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

#define DK(call)                                                            \
  do {                                                                      \
    CUresult _e = (call);                                                  \
    if (_e != CUDA_SUCCESS) {                                              \
      const char* _s = nullptr;                                           \
      cuGetErrorString(_e, &_s);                                          \
      std::fprintf(stderr, "FAIL %s: %s\n", #call, _s ? _s : "?");        \
      return 1;                                                            \
    }                                                                      \
  } while (0)

CUresult launch(CUfunction f, void** args, std::uint32_t grid) {
  return cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
}

}  // namespace

int main() {
  constexpr std::uint32_t kN = 1u << 20;  // 1M elements
  constexpr std::size_t kBytes = static_cast<std::size_t>(kN) * 4;
  constexpr std::int32_t kBias = 17;
  constexpr std::int32_t kOffset = 3;
  constexpr std::uint32_t kXor = 0x9e3779b9u;

  std::uint32_t *A = nullptr, *B = nullptr, *C = nullptr, *OUT = nullptr;
  RK(cudaMalloc(reinterpret_cast<void**>(&A), kBytes));
  RK(cudaMalloc(reinterpret_cast<void**>(&B), kBytes));
  RK(cudaMalloc(reinterpret_cast<void**>(&C), kBytes));
  RK(cudaMalloc(reinterpret_cast<void**>(&OUT), kBytes));
  std::printf("addrs A=%p B=%p C=%p OUT=%p\n", static_cast<void*>(A),
              static_cast<void*>(B), static_cast<void*>(C),
              static_cast<void*>(OUT));

  // compile_synthetic_module establishes the device-0 primary context
  // (ensure_cuda_context) and compiles the cubin via nvrtc.
  std::vector<std::byte> image;
  std::vector<std::string> entries;
  snapshot::Status s = snapshot::compile_synthetic_module(&image, &entries);
  if (!s.ok()) {
    std::fprintf(stderr, "compile_synthetic_module: %s\n", s.message().c_str());
    return 1;
  }
  CUmodule mod{};
  DK(cuModuleLoadData(&mod, image.data()));
  CUfunction mul{}, relu{}, inplace{};
  DK(cuModuleGetFunction(&mul, mod, "mul_bias"));
  DK(cuModuleGetFunction(&relu, mod, "relu_offset"));
  DK(cuModuleGetFunction(&inplace, mod, "in_place"));

  std::vector<std::uint32_t> a(kN), b(kN);
  for (std::uint32_t i = 0; i < kN; ++i) {
    a[i] = i;
    b[i] = 2u * i + 1u;
  }
  RK(cudaMemcpy(A, a.data(), kBytes, cudaMemcpyHostToDevice));
  RK(cudaMemcpy(B, b.data(), kBytes, cudaMemcpyHostToDevice));

  const std::uint32_t grid = (kN + 255) / 256;
  void *pA = A, *pB = B, *pC = C, *pOUT = OUT;
  std::int32_t bias = kBias, offset = kOffset;
  std::uint32_t n = kN;
  void* mul_args[] = {&pA, &pB, &pC, &bias, &n};
  DK(launch(mul, mul_args, grid));
  void* relu_args[] = {&pC, &pOUT, &offset, &n};
  DK(launch(relu, relu_args, grid));
  void* inplace_args[] = {&pOUT, &n};
  DK(launch(inplace, inplace_args, grid));
  RK(cudaDeviceSynchronize());

  std::vector<std::uint32_t> out(kN);
  RK(cudaMemcpy(out.data(), OUT, kBytes, cudaMemcpyDeviceToHost));

  int mismatches = 0;
  for (std::uint32_t i = 0; i < kN; ++i) {
    const std::uint32_t c =
        i * (2u * i + 1u) + static_cast<std::uint32_t>(kBias);
    const std::uint32_t expected =
        (c + static_cast<std::uint32_t>(kOffset)) ^ kXor;
    if (out[i] != expected) {
      if (mismatches < 3) {
        std::fprintf(stderr, "mismatch i=%u got=%u exp=%u\n", i, out[i],
                     expected);
      }
      ++mismatches;
    }
  }
  std::printf("verify %s (%d mismatches)\n", mismatches ? "FAIL" : "OK",
              mismatches);
  return mismatches ? 1 : 0;
}
