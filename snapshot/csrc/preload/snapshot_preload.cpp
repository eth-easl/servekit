// snapshot_preload — an LD_PRELOAD shim that interposes the HIP driver/runtime
// entry points relevant to graph snapshotting, to (a) prove that driver-level
// interception works on AMD/ROCm the same way Foundry relies on it for NVIDIA,
// and (b) quantify, in a real workload (e.g. vLLM cold start), how much wall
// time is spent inside hipStreamBeginCapture/EndCapture windows and how many
// kernel nodes those graphs contain — i.e. the cost a snapshot/restore would
// avoid.
//
// It does NOT yet serialize anything; it observes. Real symbols are resolved
// with dlsym(RTLD_NEXT) so each wrapper forwards to the genuine HIP call.
//
// Build: a standalone shared object; preload with
//   LD_PRELOAD=/path/libsnapshot_preload.so <app>

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace {

using Clock = std::chrono::steady_clock;

std::atomic<std::uint64_t> g_module_launches{0};
std::atomic<std::uint64_t> g_runtime_launches{0};
std::atomic<std::uint64_t> g_captures{0};
std::atomic<std::uint64_t> g_launches_in_capture{0};
std::atomic<int> g_capture_depth{0};
std::atomic<long long> g_capture_ns{0};

std::mutex g_mu;
std::map<void*, Clock::time_point> g_capture_start;

bool verbose() {
  static const bool v = [] {
    const char* e = std::getenv("SNAPSHOT_PRELOAD_VERBOSE");
    return e != nullptr && e[0] != '0';
  }();
  return v;
}

// Optional per-process allocation log, enabled by SNAPSHOT_PRELOAD_ALLOC_DIR.
// Records the ordered sequence of device allocations (op, size, returned ptr)
// so two cold starts can be diffed to judge address determinism. Forwarding is
// unchanged — this only observes.
std::mutex g_log_mu;
std::atomic<std::uint64_t> g_alloc_seq{0};

std::FILE* alloc_log() {
  static std::FILE* f = []() -> std::FILE* {
    const char* dir = std::getenv("SNAPSHOT_PRELOAD_ALLOC_DIR");
    if (dir == nullptr || dir[0] == '\0') {
      return nullptr;
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/alloc-%d.log", dir,
                  static_cast<int>(getpid()));
    return std::fopen(path, "w");
  }();
  return f;
}

void log_alloc(char op, std::size_t size, void* ptr) {
  std::FILE* f = alloc_log();
  if (f == nullptr) {
    return;
  }
  const std::uint64_t seq = g_alloc_seq.fetch_add(1);
  std::lock_guard<std::mutex> lock(g_log_mu);
  std::fprintf(f, "%llu %c %zu %p\n", static_cast<unsigned long long>(seq), op,
               size, ptr);
  std::fflush(f);
}

template <typename Fn>
Fn resolve(const char* name) {
  return reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
}

struct Summary {
  ~Summary() {
    std::fprintf(
        stderr,
        "[snapshot-preload] pid=%d SUMMARY captures=%llu "
        "capture_total_ms=%.2f module_launches=%llu runtime_launches=%llu "
        "launches_inside_capture=%llu\n",
        static_cast<int>(getpid()),
        static_cast<unsigned long long>(g_captures.load()),
        static_cast<double>(g_capture_ns.load()) / 1e6,
        static_cast<unsigned long long>(g_module_launches.load()),
        static_cast<unsigned long long>(g_runtime_launches.load()),
        static_cast<unsigned long long>(g_launches_in_capture.load()));
  }
};
Summary g_summary;  // dumps totals at process exit

}  // namespace

extern "C" {

hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  using Fn = hipError_t (*)(hipStream_t, hipStreamCaptureMode);
  static Fn real = resolve<Fn>("hipStreamBeginCapture");
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_capture_start[static_cast<void*>(stream)] = Clock::now();
  }
  g_capture_depth.fetch_add(1);
  return real(stream, mode);
}

hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  using Fn = hipError_t (*)(hipStream_t, hipGraph_t*);
  static Fn real = resolve<Fn>("hipStreamEndCapture");
  const hipError_t status = real(stream, pGraph);

  Clock::time_point start{};
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_capture_start.find(static_cast<void*>(stream));
    if (it != g_capture_start.end()) {
      start = it->second;
      found = true;
      g_capture_start.erase(it);
    }
  }
  if (g_capture_depth.load() > 0) {
    g_capture_depth.fetch_sub(1);
  }

  std::size_t nodes = 0;
  if (status == hipSuccess && pGraph != nullptr && *pGraph != nullptr) {
    using GetNodes = hipError_t (*)(hipGraph_t, hipGraphNode_t*, std::size_t*);
    static GetNodes real_get = resolve<GetNodes>("hipGraphGetNodes");
    if (real_get != nullptr) {
      static_cast<void>(real_get(*pGraph, nullptr, &nodes));
    }
  }

  if (found) {
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    g_capture_ns.fetch_add(
        static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   Clock::now() - start)
                                   .count()));
    const std::uint64_t n = g_captures.fetch_add(1) + 1;
    if (verbose()) {
      std::fprintf(stderr,
                   "[snapshot-preload] capture #%llu dur=%.2fms nodes=%zu "
                   "pid=%d\n",
                   static_cast<unsigned long long>(n), ms, nodes,
                   static_cast<int>(getpid()));
    }
  }
  return status;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gx,
                                 unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by,
                                 unsigned int bz, unsigned int shared,
                                 hipStream_t stream, void** kernelParams,
                                 void** extra) {
  using Fn = hipError_t (*)(hipFunction_t, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, hipStream_t, void**,
                            void**);
  static Fn real = resolve<Fn>("hipModuleLaunchKernel");
  g_module_launches.fetch_add(1);
  if (g_capture_depth.load() > 0) {
    g_launches_in_capture.fetch_add(1);
  }
  return real(f, gx, gy, gz, bx, by, bz, shared, stream, kernelParams, extra);
}

hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks,
                           dim3 dimBlocks, void** args, size_t sharedMemBytes,
                           hipStream_t stream) {
  using Fn = hipError_t (*)(const void*, dim3, dim3, void**, size_t,
                            hipStream_t);
  static Fn real = resolve<Fn>("hipLaunchKernel");
  g_runtime_launches.fetch_add(1);
  if (g_capture_depth.load() > 0) {
    g_launches_in_capture.fetch_add(1);
  }
  return real(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
              stream);
}

// ---- allocation tracing (for the cross-cold-start determinism spike) ----

hipError_t hipMalloc(void** ptr, size_t size) {
  using Fn = hipError_t (*)(void**, size_t);
  static Fn real = resolve<Fn>("hipMalloc");
  const hipError_t status = real(ptr, size);
  if (status == hipSuccess && ptr != nullptr) {
    log_alloc('M', size, *ptr);
  }
  return status;
}

hipError_t hipFree(void* ptr) {
  using Fn = hipError_t (*)(void*);
  static Fn real = resolve<Fn>("hipFree");
  log_alloc('F', 0, ptr);
  return real(ptr);
}

hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
  using Fn = hipError_t (*)(void**, size_t, hipStream_t);
  static Fn real = resolve<Fn>("hipMallocAsync");
  const hipError_t status = real(ptr, size, stream);
  if (status == hipSuccess && ptr != nullptr) {
    log_alloc('A', size, *ptr);
  }
  return status;
}

// expandable_segments path: PyTorch reserves VA via the VMM API.
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment,
                                void* addr, unsigned long long flags) {
  using Fn = hipError_t (*)(void**, size_t, size_t, void*, unsigned long long);
  static Fn real = resolve<Fn>("hipMemAddressReserve");
  const hipError_t status = real(ptr, size, alignment, addr, flags);
  if (status == hipSuccess && ptr != nullptr) {
    log_alloc('R', size, *ptr);
  }
  return status;
}

}  // extern "C"
