// snapshot/csrc/cli/cuda_record_smoke.cpp
// N5a Task 1: two-kernel CLI smoke (static k_add + nvrtc k_mul, CUDA graph capture).
//
// Captures a CUDA graph on a non-default stream over two kernels:
//   k_add  — __global__, compiled into this TU by nvcc; registered via
//             __cudaRegisterFunction so Task 2's interposer can identify it
//             by mangled name without extra machinery.
//   k_mul  — compiled at runtime by nvrtc and loaded via cuModuleLoadData.
//
// Expected result: out[i] = (i+1)*2.  FNV-1a checksum over the output bytes
// is printed as "CHECKSUM=<hex>".  Running twice must produce the same hex
// (determinism precondition for the Task 5 bit-identical restore gate).
//
// This file is compiled as a CUDA translation unit (set_source_files_properties
// LANGUAGE CUDA in CMakeLists.txt), giving us __global__ / <<< >>> support.

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Driver-API error check — print location + error string and return 1.
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

// Static kernel: compiled into this TU by nvcc.
// __cudaRegisterFunction is emitted automatically, giving Task 2's interposer
// a stable identity without any additional registration logic.
__global__ void k_add(int* a, int* out, int n) {
  int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + 1;
}

// FNV-1a 32-bit hash over raw bytes (deterministic, endian-stable on one arch).
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

  // -----------------------------------------------------------------------
  // Device + context (mirrors cuda_redirect_smoke.cpp pattern).
  // -----------------------------------------------------------------------
  DK(cuInit(0));
  CUdevice dev{};
  DK(cuDeviceGet(&dev, 0));
  CUcontext ctx{};
  DK(cuDevicePrimaryCtxRetain(&ctx, dev));
  DK(cuCtxPushCurrent(ctx));

  // Query compute capability to build a correct nvrtc arch flag at runtime.
  int cc_major = 0, cc_minor = 0;
  DK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                          dev));
  DK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                          dev));
  char arch_opt[32];
  std::snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d",
                cc_major, cc_minor);
  std::printf("device: sm_%d%d\n", cc_major, cc_minor);

  // -----------------------------------------------------------------------
  // Device buffers: a[i]=i (input), out (result).
  // -----------------------------------------------------------------------
  CUdeviceptr d_a{}, d_out{};
  DK(cuMemAlloc(&d_a, kBytes));
  DK(cuMemAlloc(&d_out, kBytes));

  std::vector<int> h_a(kN);
  for (int i = 0; i < kN; ++i) h_a[i] = i;
  DK(cuMemcpyHtoD(d_a, h_a.data(), kBytes));

  // -----------------------------------------------------------------------
  // nvrtc: compile k_mul → PTX → cuModuleLoadData → cuModuleGetFunction.
  // -----------------------------------------------------------------------
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

  // -----------------------------------------------------------------------
  // Non-default stream (required for cuStreamBeginCapture).
  // -----------------------------------------------------------------------
  CUstream stream{};
  DK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  // -----------------------------------------------------------------------
  // CUDA graph capture: k_add → k_mul (both on same stream → dependency edge).
  // -----------------------------------------------------------------------
  DK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));

  // k_add: static kernel, runtime-launched on the capture stream.
  // The CUstream ↔ cudaStream_t cast is valid (both are CUstream_st*).
  int* pa   = reinterpret_cast<int*>(d_a);
  int* pout = reinterpret_cast<int*>(d_out);
  int  n    = kN;
  k_add<<<kGrid, kBlock, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      pa, pout, n);

  // k_mul: driver-API launch on the same stream (captured after k_add).
  void* mul_args[] = {&pout, &n};
  DK(cuLaunchKernel(f_mul, static_cast<unsigned>(kGrid), 1u, 1u,
                    static_cast<unsigned>(kBlock), 1u, 1u,
                    0u, stream, mul_args, nullptr));

  CUgraph graph{};
  DK(cuStreamEndCapture(stream, &graph));

  // -----------------------------------------------------------------------
  // Instantiate, launch, synchronize.
  // -----------------------------------------------------------------------
  CUgraphExec exec{};
  DK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  DK(cuGraphLaunch(exec, stream));
  DK(cuStreamSynchronize(stream));

  // -----------------------------------------------------------------------
  // Read back, verify, and print CHECKSUM.
  // -----------------------------------------------------------------------
  std::vector<int> h_out(kN);
  DK(cuMemcpyDtoH(h_out.data(), d_out, kBytes));

  // Logical correctness: out[i] = k_mul(k_add(a[i])) = (i+1)*2.
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
  std::printf("CHECKSUM=%08x\n", checksum);

  // Cleanup (best-effort; process exits shortly after).
  cuGraphExecDestroy(exec);
  cuGraphDestroy(graph);
  cuStreamDestroy(stream);
  cuModuleUnload(mod);
  cuMemFree(d_out);
  cuMemFree(d_a);
  cuCtxPopCurrent(nullptr);
  cuDevicePrimaryCtxRelease(dev);

  return mismatches ? 1 : 0;
}
