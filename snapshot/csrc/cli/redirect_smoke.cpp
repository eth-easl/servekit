// A standalone HIP program that uses *raw* hipMalloc (no snapshot allocator) and
// runs the synthetic 3-kernel chain. Run normally it allocates wherever the
// driver chooses; run under libsnapshot_redirect.so its hipMalloc is served
// from the deterministic VMM region, so addresses become reproducible across
// runs. Either way it must compute the same bit-identical result, proving the
// redirect is transparent.

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "snapshot/workload_kernels.hpp"

namespace {

#define CK(call)                                                            \
  do {                                                                      \
    hipError_t _e = (call);                                                 \
    if (_e != hipSuccess) {                                                 \
      std::fprintf(stderr, "FAIL %s: %s\n", #call, hipGetErrorString(_e));  \
      return 1;                                                             \
    }                                                                       \
  } while (0)

struct __attribute__((packed)) MulBiasArgs {
  std::uint64_t a, b, c;
  std::int32_t bias;
  std::uint32_t n;
};
struct __attribute__((packed)) ReluOffsetArgs {
  std::uint64_t c, out;
  std::int32_t offset;
  std::uint32_t n;
};
struct __attribute__((packed)) InPlaceArgs {
  std::uint64_t out;
  std::uint32_t n;
  std::uint32_t pad;
};

hipError_t launch(hipFunction_t f, void* blob, std::size_t blob_size,
                  std::uint32_t grid) {
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, blob,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &blob_size,
                    HIP_LAUNCH_PARAM_END};
  return hipModuleLaunchKernel(f, grid, 1, 1, 256, 1, 1, 0, nullptr, nullptr,
                               config);
}

}  // namespace

int main() {
  constexpr std::uint32_t kN = 1u << 20;  // 1M elements
  constexpr std::size_t kBytes = static_cast<std::size_t>(kN) * 4;
  constexpr std::int32_t kBias = 17;
  constexpr std::int32_t kOffset = 3;
  constexpr std::uint32_t kXor = 0x9e3779b9u;

  std::uint32_t *A = nullptr, *B = nullptr, *C = nullptr, *OUT = nullptr;
  CK(hipMalloc(reinterpret_cast<void**>(&A), kBytes));
  CK(hipMalloc(reinterpret_cast<void**>(&B), kBytes));
  CK(hipMalloc(reinterpret_cast<void**>(&C), kBytes));
  CK(hipMalloc(reinterpret_cast<void**>(&OUT), kBytes));
  std::printf("addrs A=%p B=%p C=%p OUT=%p\n", static_cast<void*>(A),
              static_cast<void*>(B), static_cast<void*>(C),
              static_cast<void*>(OUT));

  std::vector<std::byte> image;
  std::vector<std::string> entries;
  snapshot::Status s = snapshot::compile_synthetic_module(&image, &entries);
  if (!s.ok()) {
    std::fprintf(stderr, "compile_synthetic_module: %s\n", s.message().c_str());
    return 1;
  }
  hipModule_t mod{};
  CK(hipModuleLoadData(&mod, image.data()));
  hipFunction_t mul{}, relu{}, inplace{};
  CK(hipModuleGetFunction(&mul, mod, "mul_bias"));
  CK(hipModuleGetFunction(&relu, mod, "relu_offset"));
  CK(hipModuleGetFunction(&inplace, mod, "in_place"));

  std::vector<std::uint32_t> a(kN), b(kN);
  for (std::uint32_t i = 0; i < kN; ++i) {
    a[i] = i;
    b[i] = 2u * i + 1u;
  }
  CK(hipMemcpy(A, a.data(), kBytes, hipMemcpyHostToDevice));
  CK(hipMemcpy(B, b.data(), kBytes, hipMemcpyHostToDevice));

  const std::uint32_t grid = (kN + 255) / 256;
  MulBiasArgs p3{reinterpret_cast<std::uint64_t>(A),
                 reinterpret_cast<std::uint64_t>(B),
                 reinterpret_cast<std::uint64_t>(C), kBias, kN};
  CK(launch(mul, &p3, sizeof(p3), grid));
  ReluOffsetArgs p2{reinterpret_cast<std::uint64_t>(C),
                    reinterpret_cast<std::uint64_t>(OUT), kOffset, kN};
  CK(launch(relu, &p2, sizeof(p2), grid));
  InPlaceArgs p1{reinterpret_cast<std::uint64_t>(OUT), kN, 0};
  CK(launch(inplace, &p1, sizeof(p1), grid));
  CK(hipDeviceSynchronize());

  std::vector<std::uint32_t> out(kN);
  CK(hipMemcpy(out.data(), OUT, kBytes, hipMemcpyDeviceToHost));

  int mismatches = 0;
  for (std::uint32_t i = 0; i < kN; ++i) {
    const std::uint32_t c = i * (2u * i + 1u) + static_cast<std::uint32_t>(kBias);
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
