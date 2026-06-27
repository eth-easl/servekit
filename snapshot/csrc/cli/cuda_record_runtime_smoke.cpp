// snapshot/csrc/cli/cuda_record_runtime_smoke.cpp
// N5b Task 1: two-kernel CLI smoke that captures the graph via the RUNTIME
// cudaStream*Capture API — the path torch.cuda.graph() / vLLM use (the N5a
// cuda_record_smoke.cpp drove capture through the DRIVER cuStream*Capture).
//
// Same two kernels as the N5a smoke:
//   k_add  — __global__, compiled into this TU by nvcc (fatbin path).
//   k_mul  — compiled at runtime by nvrtc and loaded via cuModuleLoadData
//             (module/nvrtc path).
//
// The graph is captured with the RUNTIME API but the two launches inside the
// window are MIXED (k_add via the runtime <<<>>>, k_mul via the driver
// cuLaunchKernel) — exactly the mix the N5a interposer already suppresses.
// Expected result: out[i] = (i+1)*2.  FNV-1a checksum printed as
// "RT_CHECKSUM=<hex>".  This exercises the Task-1 runtime shims and the CUgraph
// record dedupe (the runtime end-capture may route through the interposed
// driver cuStreamEndCapture inside libcudart → walk twice for one CUgraph).

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Driver-API error check.
#define DK(call)                                                              \
  do {                                                                        \
    CUresult _e = (call);                                                     \
    if (_e != CUDA_SUCCESS) {                                                 \
      const char* _s = nullptr;                                               \
      cuGetErrorString(_e, &_s);                                              \
      std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #call, \
                   _s ? _s : "?");                                            \
      return 1;                                                               \
    }                                                                         \
  } while (0)

// nvrtc error check.
#define NK(call)                                                              \
  do {                                                                        \
    nvrtcResult _e = (call);                                                  \
    if (_e != NVRTC_SUCCESS) {                                                \
      std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #call, \
                   nvrtcGetErrorString(_e));                                  \
      return 1;                                                               \
    }                                                                         \
  } while (0)

// Runtime-API error check. As in the N5a smoke, device buffers are allocated
// via the RUNTIME cudaMalloc so snapshot_redirect_cuda pins them at its fixed
// base (Δ=0).
#define RK(call)                                                              \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(_e));                                   \
      return 1;                                                               \
    }                                                                         \
  } while (0)

// Static kernel: compiled into this TU by nvcc (fatbin path).
__global__ void k_add(int* a, int* out, int n) {
  int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + 1;
}

// FNV-1a 32-bit hash over raw bytes.
std::uint32_t fnv1a(const void* data, std::size_t len) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint32_t h = 0x811c9dc5u;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint32_t>(p[i]);
    h *= 0x01000193u;
  }
  return h;
}

// nvrtc source: multiply each element by 2 (second graph node).
const char* k_mul_src = R"(
extern "C" __global__ void k_mul(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] *= 2;
}
)";

}  // namespace

int main() {
  constexpr int kN = 1024;
  constexpr std::size_t kBytes = static_cast<std::size_t>(kN) * sizeof(int);
  constexpr int kBlock = 256;
  const int kGrid = (kN + kBlock - 1) / kBlock;

  // Device + context (required for the driver-API calls: cuModuleLoadData,
  // cuModuleGetFunction, cuLaunchKernel, cuMemcpyHtoD).
  DK(cuInit(0));
  CUdevice dev{};
  DK(cuDeviceGet(&dev, 0));
  CUcontext ctx{};
  DK(cuDevicePrimaryCtxRetain(&ctx, dev));
  DK(cuCtxPushCurrent(ctx));

  int cc_major = 0, cc_minor = 0;
  DK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                          dev));
  DK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                          dev));
  char arch_opt[32];
  std::snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d",
                cc_major, cc_minor);
  std::printf("device: sm_%d%d (runtime capture path)\n", cc_major, cc_minor);

  // Device buffers: a[i]=i (input), out (result). Allocated via the RUNTIME
  // cudaMalloc so the redirect pins them at its fixed base (Δ=0).
  CUdeviceptr d_a{}, d_out{};
  RK(cudaMalloc(reinterpret_cast<void**>(&d_a), kBytes));
  RK(cudaMalloc(reinterpret_cast<void**>(&d_out), kBytes));

  std::vector<int> h_a(kN);
  for (int i = 0; i < kN; ++i) h_a[i] = i;
  DK(cuMemcpyHtoD(d_a, h_a.data(), kBytes));

  // nvrtc: compile k_mul → PTX → cuModuleLoadData → cuModuleGetFunction.
  nvrtcProgram prog{};
  NK(nvrtcCreateProgram(&prog, k_mul_src, "k_mul.cu", 0, nullptr, nullptr));
  const char* opts[] = {arch_opt};
  const nvrtcResult compile_res = nvrtcCompileProgram(prog, 1, opts);
  if (compile_res != NVRTC_SUCCESS) {
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(prog, log.data());
    std::fprintf(stderr, "nvrtc compile error: %s\n", log.c_str());
    return 1;
  }
  std::size_t ptx_size = 0;
  NK(nvrtcGetPTXSize(prog, &ptx_size));
  std::vector<char> ptx(ptx_size);
  NK(nvrtcGetPTX(prog, ptx.data()));
  NK(nvrtcDestroyProgram(&prog));

  CUmodule mod{};
  DK(cuModuleLoadData(&mod, ptx.data()));
  CUfunction f_mul{};
  DK(cuModuleGetFunction(&f_mul, mod, "k_mul"));

  // Non-default stream (required for cudaStreamBeginCapture). RUNTIME create.
  cudaStream_t stream{};
  RK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // CUDA graph capture via the RUNTIME API: k_add → k_mul (dep edge).
  RK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

  int* pa   = reinterpret_cast<int*>(d_a);
  int* pout = reinterpret_cast<int*>(d_out);
  int  n    = kN;
  // k_add: runtime <<<>>> launch on the capture stream.
  k_add<<<kGrid, kBlock, 0, stream>>>(pa, pout, n);
  if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
    std::fprintf(stderr, "k_add launch failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  // k_mul: DRIVER-API launch on the same stream (captured after k_add) — the
  // mixed runtime/driver launch mix the N5a interposer already suppresses.
  void* mul_args[] = {&pout, &n};
  DK(cuLaunchKernel(f_mul, static_cast<unsigned>(kGrid), 1u, 1u,
                    static_cast<unsigned>(kBlock), 1u, 1u,
                    0u, reinterpret_cast<CUstream>(stream), mul_args, nullptr));

  cudaGraph_t graph{};
  RK(cudaStreamEndCapture(stream, &graph));

  // Instantiate + launch via the RUNTIME API.
  cudaGraphExec_t exec{};
  RK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
  RK(cudaGraphLaunch(exec, stream));
  RK(cudaStreamSynchronize(stream));

  // Read back, verify, and print RT_CHECKSUM.
  std::vector<int> h_out(kN);
  DK(cuMemcpyDtoH(h_out.data(), d_out, kBytes));

  int mismatches = 0;
  for (int i = 0; i < kN; ++i) {
    const int expected = (i + 1) * 2;
    if (h_out[i] != expected) {
      if (mismatches < 3)
        std::fprintf(stderr, "mismatch i=%d got=%d exp=%d\n", i, h_out[i],
                     expected);
      ++mismatches;
    }
  }
  std::printf("verify %s (%d mismatches)\n", mismatches ? "FAIL" : "OK",
              mismatches);

  const std::uint32_t checksum = fnv1a(h_out.data(), kBytes);
  std::printf("RT_CHECKSUM=%08x\n", checksum);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cuModuleUnload(mod);
  cudaFree(reinterpret_cast<void*>(d_out));
  cudaFree(reinterpret_cast<void*>(d_a));
  cuCtxPopCurrent(nullptr);
  cuDevicePrimaryCtxRelease(dev);

  return mismatches ? 1 : 0;
}
