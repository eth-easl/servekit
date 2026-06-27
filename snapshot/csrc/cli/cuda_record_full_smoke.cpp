// snapshot/csrc/cli/cuda_record_full_smoke.cpp
// N5b Task 2 (G1): FULL-like CLI smoke. Builds a graph with the NODE-TYPE
// VARIETY and NON-LINEAR DEPENDENCIES a real vLLM FULL/PIECEWISE graph has, so
// it exercises every N5a gap the record/restore path must close:
//
//   • MEMCPY nodes (cudaMemcpyAsync D2D) — N5a was kernel-only.
//   • MEMSET nodes (cudaMemsetAsync)    — N5a was kernel-only.
//   • `extra`-buffer kernargs (CU_LAUNCH_PARAM_BUFFER_POINTER launch) — N5a was
//     kernelParams-only.
//   • a DIAMOND DAG (two independent branches merging) so the create-then-link
//     rebuild is exercised on non-linear deps, and kernel→memcpy/memset edges
//     resolve (the N5a non-kernel edge-drop).
//
// Captured via the RUNTIME cudaStream*Capture API across TWO streams + events
// (cudaStreamCaptureModeRelaxed). Deterministic output: out[i] = 3*i + 4.
// FNV-1a checksum printed as "FULL_CHECKSUM=<hex>".
//
// Kernels:
//   k_add  : o = a + 1            (static, fatbin path)
//   k_mul  : o = o * 2  (in-place) (nvrtc, module path; launched via `extra`)
//   k_acc  : o = o + b            (static, fatbin path) — the merge kernel

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

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

#define NK(call)                                                              \
  do {                                                                        \
    nvrtcResult _e = (call);                                                  \
    if (_e != NVRTC_SUCCESS) {                                                \
      std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #call, \
                   nvrtcGetErrorString(_e));                                  \
      return 1;                                                               \
    }                                                                         \
  } while (0)

#define RK(call)                                                              \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(_e));                                   \
      return 1;                                                               \
    }                                                                         \
  } while (0)

// o = a + 1
__global__ void k_add(int* a, int* o, int n) {
  int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) o[i] = a[i] + 1;
}

// o = o + b  (the merge kernel at the bottom of the diamond)
__global__ void k_acc(int* o, int* b, int n) {
  int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) o[i] = o[i] + b[i];
}

std::uint32_t fnv1a(const void* data, std::size_t len) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint32_t h = 0x811c9dc5u;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint32_t>(p[i]);
    h *= 0x01000193u;
  }
  return h;
}

// k_mul: in-place multiply by 2. Launched via the `extra` buffer form to
// exercise the CU_LAUNCH_PARAM_BUFFER_POINTER record/replay path.
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
  std::printf("device: sm_%d%d (FULL-like graph)\n", cc_major, cc_minor);

  // Device buffers, all via runtime cudaMalloc (redirect pins them, Δ=0):
  //   d_a   : input a[i]=i
  //   d_out : final result
  //   d_t   : transient on the main chain
  //   d_b   : transient on the branch chain
  CUdeviceptr d_a{}, d_out{}, d_t{}, d_b{};
  RK(cudaMalloc(reinterpret_cast<void**>(&d_a), kBytes));
  RK(cudaMalloc(reinterpret_cast<void**>(&d_out), kBytes));
  RK(cudaMalloc(reinterpret_cast<void**>(&d_t), kBytes));
  RK(cudaMalloc(reinterpret_cast<void**>(&d_b), kBytes));

  std::vector<int> h_a(kN);
  for (int i = 0; i < kN; ++i) h_a[i] = i;
  DK(cuMemcpyHtoD(d_a, h_a.data(), kBytes));

  // nvrtc k_mul → module → CUfunction (module path).
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

  // Two capture streams + two events → a diamond DAG. Relaxed mode lets the
  // capture span streams cross-linked by cudaStreamWaitEvent.
  cudaStream_t s{}, s2{};
  RK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  RK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  cudaEvent_t ev1{}, ev2{};
  RK(cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming));
  RK(cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming));

  // ---------- capture (runtime API, relaxed, two streams) ----------
  RK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));

  // (1) MEMSET node: d_out = 0.
  RK(cudaMemsetAsync(reinterpret_cast<void*>(d_out), 0, kBytes, s));
  // (2) KERNEL node (kernelParams form): d_t = a + 1.
  k_add<<<kGrid, kBlock, 0, s>>>(reinterpret_cast<int*>(d_a),
                                 reinterpret_cast<int*>(d_t), kN);
  if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
    std::fprintf(stderr, "k_add launch failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  // (3) MEMCPY node (D2D): d_out ← d_t   → out = a+1.
  RK(cudaMemcpyAsync(reinterpret_cast<void*>(d_out),
                     reinterpret_cast<const void*>(d_t), kBytes,
                     cudaMemcpyDeviceToDevice, s));
  // (4) KERNEL node (extra-buffer form): d_out *= 2 → out = 2(a+1). This is the
  //     launch the N5a kernelParams-only path would have marked blind.
  {
    struct KArg { int* o; int n; } karg;
    karg.o = reinterpret_cast<int*>(d_out);
    karg.n = kN;
    std::size_t ksize = sizeof(karg);
    void* extra[5] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &karg,
                      CU_LAUNCH_PARAM_BUFFER_SIZE, &ksize,
                      CU_LAUNCH_PARAM_END};
    DK(cuLaunchKernel(f_mul, static_cast<unsigned>(kGrid), 1u, 1u,
                      static_cast<unsigned>(kBlock), 1u, 1u,
                      0u, reinterpret_cast<CUstream>(s), nullptr, extra));
  }

  // (5) Branch: record ev1 (after node 4) and start an INDEPENDENT chain on s2.
  RK(cudaEventRecord(ev1, s));
  RK(cudaStreamWaitEvent(s2, ev1));
  // (6) KERNEL node on s2: d_b = a + 1.
  k_add<<<kGrid, kBlock, 0, s2>>>(reinterpret_cast<int*>(d_a),
                                  reinterpret_cast<int*>(d_b), kN);
  if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
    std::fprintf(stderr, "k_add (branch) launch failed: %s\n",
                 cudaGetErrorString(e));
    return 1;
  }
  // (7) KERNEL node on s2 (in-place): d_b = d_b + 1 = a + 2.
  k_add<<<kGrid, kBlock, 0, s2>>>(reinterpret_cast<int*>(d_b),
                                  reinterpret_cast<int*>(d_b), kN);
  if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
    std::fprintf(stderr, "k_add (branch2) launch failed: %s\n",
                 cudaGetErrorString(e));
    return 1;
  }
  RK(cudaEventRecord(ev2, s2));
  RK(cudaStreamWaitEvent(s, ev2));

  // (8) MERGE KERNEL node (depends on node 4 AND node 7): d_out += d_b
  //     → out = 2(a+1) + (a+2) = 3a + 4.
  k_acc<<<kGrid, kBlock, 0, s>>>(reinterpret_cast<int*>(d_out),
                                 reinterpret_cast<int*>(d_b), kN);
  if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
    std::fprintf(stderr, "k_acc launch failed: %s\n", cudaGetErrorString(e));
    return 1;
  }

  cudaGraph_t graph{};
  RK(cudaStreamEndCapture(s, &graph));

  cudaGraphExec_t exec{};
  RK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
  RK(cudaGraphLaunch(exec, s));
  RK(cudaStreamSynchronize(s));

  // ---------- readback + verify + checksum ----------
  std::vector<int> h_out(kN);
  DK(cuMemcpyDtoH(h_out.data(), d_out, kBytes));

  int mismatches = 0;
  for (int i = 0; i < kN; ++i) {
    const int expected = 3 * i + 4;
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
  std::printf("FULL_CHECKSUM=%08x\n", checksum);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(ev2);
  cudaEventDestroy(ev1);
  cudaStreamDestroy(s2);
  cudaStreamDestroy(s);
  cuModuleUnload(mod);
  cudaFree(reinterpret_cast<void*>(d_b));
  cudaFree(reinterpret_cast<void*>(d_t));
  cudaFree(reinterpret_cast<void*>(d_out));
  cudaFree(reinterpret_cast<void*>(d_a));
  cuCtxPopCurrent(nullptr);
  cuDevicePrimaryCtxRelease(dev);

  return mismatches ? 1 : 0;
}
