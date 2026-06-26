// snapshot_redirect_cuda — the CUDA sibling of snapshot_redirect.cpp. An
// LD_PRELOAD shim that redirects device allocations into one contiguous
// deterministic region so an unmodified CUDA app's device addresses become
// reproducible across cold starts. The HIP redirect is left untouched; this is
// a parallel file (per the dual-vendor design), not a refactor of it.
//
// Two backings are available:
//   * FIXED_VMM (default): one cuMemAddressReserve(0x600000000000) + one
//     cuMemCreate(whole) + one cuMemMap + ONE cuMemSetAccess over the WHOLE
//     region, then bump + free-list sub-allocation inside it. cuMemSetAccess is
//     the canonical CUDA path (reliable, unlike the gfx942 hipMemSetAccess that
//     forced the AMD arena), so the fixed base is the default here: Δ=0 across
//     cold starts, which both the skip-capture and snapshot/restore strategies
//     depend on. The driver calls are made INLINE so this .so links only
//     libcuda and stays portable from the build container into the torch one.
//   * ARENA (SNAPSHOT_REDIRECT_ARENA=1): one real cudaMalloc of the whole
//     region, bump + free-list inside it. Driver-chosen base (Δ≠0, recorded for
//     relocation). Kept for diagnosis.
//
// Hooks the CUDA runtime malloc family (cudaMalloc/cudaFree/cudaMallocAsync/
// cudaFreeAsync) — what torch's caching allocator uses — mirroring the AMD
// redirect's four hipMalloc-family hooks. A size-bucketed free list keeps the
// returned addresses reproducible while memory stays bounded under a real
// engine's repeated alloc/free.

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

#include "snapshot/allocator.hpp"  // snapshot::kDefaultRequestedBase only

namespace {

std::mutex g_mu;

// --- backing selection -------------------------------------------------------
// Default OFF => fixed-base VMM is the default; =1 selects the cudaMalloc arena.
bool arena_mode() {
  static const bool a = [] {
    const char* e = std::getenv("SNAPSHOT_REDIRECT_ARENA");
    return e != nullptr && e[0] == '1';
  }();
  return a;
}

bool verbose() {
  static const bool v = [] {
    const char* e = std::getenv("SNAPSHOT_REDIRECT_VERBOSE");
    return e != nullptr && e[0] != '0';
  }();
  return v;
}

// --- shared region state -----------------------------------------------------
bool g_init_attempted = false;
bool g_init_ok = false;
std::uint64_t g_base = 0;
std::uint64_t g_region = 0;
std::uint64_t g_cursor = 0;  // arena bump cursor (bytes from g_base)
std::uint64_t g_granularity = 512;
bool g_fixed_base = false;

// va -> rounded size, for live region allocations.
std::map<std::uint64_t, std::uint64_t> g_live;
// rounded size -> va, for freed-but-resident blocks available for reuse.
std::multimap<std::uint64_t, std::uint64_t> g_free;

std::uint64_t g_served = 0;
std::uint64_t g_reused = 0;
std::uint64_t g_passthrough = 0;
std::uint64_t g_passthrough_bytes = 0;

std::FILE* alloc_log() {
  static std::FILE* f = []() -> std::FILE* {
    const char* dir = std::getenv("SNAPSHOT_REDIRECT_ALLOC_DIR");
    if (dir == nullptr || dir[0] == '\0') {
      return nullptr;
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/redir-cuda-%d.log", dir,
                  static_cast<int>(getpid()));
    return std::fopen(path, "w");
  }();
  return f;
}

std::uint64_t region_bytes() {
  const char* e = std::getenv("SNAPSHOT_REDIRECT_REGION_GIB");
  std::uint64_t gib =
      (e != nullptr && e[0] != '\0') ? std::strtoull(e, nullptr, 10) : 8ULL;
  if (gib == 0) {
    gib = 8ULL;
  }
  return gib << 30;
}

std::uint64_t round_up(std::uint64_t v, std::uint64_t a) {
  if (a == 0) {
    return v;
  }
  const std::uint64_t r = v % a;
  return r == 0 ? v : v + (a - r);
}

using MallocFn = cudaError_t (*)(void**, size_t);
MallocFn real_cudaMalloc() {
  static MallocFn fn =
      reinterpret_cast<MallocFn>(dlsym(RTLD_NEXT, "cudaMalloc"));
  return fn;
}

const char* cu_err(CUresult rc) {
  const char* s = nullptr;
  cuGetErrorString(rc, &s);
  return s ? s : "unknown CUDA error";
}

// Publish the region VA to the recorder exactly like the AMD redirect, so a
// later record/restore (N5) can map/relocate the arena. Harmless for N2.
void publish_region_env() {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%llu",
                static_cast<unsigned long long>(g_base));
  setenv("SNAPSHOT_RECORD_REGION_BASE", buf, 1);
  std::snprintf(buf, sizeof(buf), "%llu",
                static_cast<unsigned long long>(g_region));
  setenv("SNAPSHOT_RECORD_REGION_SIZE", buf, 1);
}

// Caller must hold g_mu.
bool ensure_init() {
  if (g_init_attempted) {
    return g_init_ok;
  }
  g_init_attempted = true;

  if (!arena_mode()) {
    // --- Fixed-base VMM (inline driver calls; mirrors HIP fixed_vmm_mode) ----
    CUresult rc = cuInit(0);  // idempotent; torch may already have inited
    if (rc != CUDA_SUCCESS) {
      std::fprintf(stderr, "[redirect-cuda] cuInit failed (%s); pass through\n",
                   cu_err(rc));
      return false;
    }
    CUdevice dev = 0;
    rc = cuDeviceGet(&dev, 0);
    if (rc != CUDA_SUCCESS) {
      std::fprintf(stderr, "[redirect-cuda] cuDeviceGet failed (%s)\n",
                   cu_err(rc));
      return false;
    }
    CUcontext ctx{};
    rc = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (rc != CUDA_SUCCESS) {
      std::fprintf(stderr, "[redirect-cuda] cuDevicePrimaryCtxRetain (%s)\n",
                   cu_err(rc));
      return false;
    }
    cuCtxSetCurrent(ctx);

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<int>(dev);
    std::size_t gran = 0;
    cuMemGetAllocationGranularity(&gran, &prop,
                                  CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    if (gran == 0) {
      gran = 1ULL << 21;  // 2 MiB fallback
    }
    const std::uint64_t rsize = round_up(region_bytes(), gran);

    CUdeviceptr ptr = static_cast<CUdeviceptr>(snapshot::kDefaultRequestedBase);
    rc = cuMemAddressReserve(&ptr, rsize, gran,
                             static_cast<CUdeviceptr>(
                                 snapshot::kDefaultRequestedBase),
                             0);
    if (rc != CUDA_SUCCESS || ptr == 0) {
      std::fprintf(stderr,
                   "[redirect-cuda] FIXED_VMM cuMemAddressReserve(base=0x%llx, "
                   "%lluGiB) failed (%s); pass through\n",
                   static_cast<unsigned long long>(
                       snapshot::kDefaultRequestedBase),
                   static_cast<unsigned long long>(rsize >> 30), cu_err(rc));
      return false;
    }
    const std::uint64_t reserved_base = static_cast<std::uint64_t>(ptr);

    CUmemGenericAllocationHandle handle{};
    rc = cuMemCreate(&handle, rsize, &prop, 0);
    if (rc != CUDA_SUCCESS) {
      std::fprintf(stderr,
                   "[redirect-cuda] FIXED_VMM cuMemCreate(%lluGiB) failed (%s)\n",
                   static_cast<unsigned long long>(rsize >> 30), cu_err(rc));
      cuMemAddressFree(ptr, rsize);
      return false;
    }
    rc = cuMemMap(ptr, rsize, 0, handle, 0);
    if (rc != CUDA_SUCCESS) {
      std::fprintf(stderr, "[redirect-cuda] FIXED_VMM cuMemMap failed (%s)\n",
                   cu_err(rc));
      cuMemRelease(handle);
      cuMemAddressFree(ptr, rsize);
      return false;
    }
    CUmemAccessDesc desc{};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = static_cast<int>(dev);
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    rc = cuMemSetAccess(ptr, rsize, &desc, 1);
    if (rc != CUDA_SUCCESS) {
      std::fprintf(stderr,
                   "[redirect-cuda] FIXED_VMM cuMemSetAccess(whole) failed (%s)\n",
                   cu_err(rc));
      cuMemUnmap(ptr, rsize);
      cuMemRelease(handle);
      cuMemAddressFree(ptr, rsize);
      return false;
    }

    g_base = reserved_base;
    g_region = rsize;
    g_cursor = 0;
    g_granularity = 512;
    g_fixed_base = (reserved_base == snapshot::kDefaultRequestedBase);
    g_init_ok = true;
    publish_region_env();
    std::fprintf(stderr,
                 "[redirect-cuda] pid=%d FIXED_VMM base=0x%llx size=%lluGiB "
                 "fixed_base_honored=%d\n",
                 static_cast<int>(getpid()),
                 static_cast<unsigned long long>(g_base),
                 static_cast<unsigned long long>(g_region >> 30), g_fixed_base);
    return true;
  }

  // --- Arena fallback (one real cudaMalloc of the whole region) --------------
  void* base = nullptr;
  cudaError_t e = real_cudaMalloc()(&base, region_bytes());
  if (e != cudaSuccess || base == nullptr) {
    std::fprintf(stderr,
                 "[redirect-cuda] arena cudaMalloc(%lluGiB) failed (%d); "
                 "pass through\n",
                 static_cast<unsigned long long>(region_bytes() >> 30),
                 static_cast<int>(e));
    return false;
  }
  g_base = reinterpret_cast<std::uint64_t>(base);
  g_region = region_bytes();
  g_cursor = 0;
  g_granularity = 512;
  g_fixed_base = false;
  g_init_ok = true;
  publish_region_env();
  std::fprintf(stderr,
               "[redirect-cuda] pid=%d ARENA base=0x%llx size=%lluGiB\n",
               static_cast<int>(getpid()),
               static_cast<unsigned long long>(g_base),
               static_cast<unsigned long long>(g_region >> 30));
  return true;
}

bool in_region(const void* p) {
  const std::uint64_t v = reinterpret_cast<std::uint64_t>(p);
  return g_base != 0 && v >= g_base && v < g_base + g_region;
}

// Serve a request from the region. Caller must hold g_mu and have init'd.
bool serve(size_t size, std::uint64_t* out_va) {
  const std::uint64_t rounded = round_up(size == 0 ? 1 : size, g_granularity);
  std::uint64_t va = 0;
  char op = 'M';

  auto it = g_free.find(rounded);
  if (it != g_free.end()) {
    va = it->second;
    g_free.erase(it);
    op = 'R';  // reused
    ++g_reused;
  } else {
    if (g_cursor + rounded > g_region) {
      if (verbose()) {
        std::fprintf(stderr,
                     "[redirect-cuda] region exhausted (used=%lluMiB "
                     "req=%lluKiB); real cudaMalloc\n",
                     static_cast<unsigned long long>(g_cursor >> 20),
                     static_cast<unsigned long long>(rounded >> 10));
      }
      ++g_passthrough;
      g_passthrough_bytes += rounded;
      return false;
    }
    va = g_base + g_cursor;
    g_cursor += rounded;
    ++g_served;
  }

  g_live[va] = rounded;
  if (std::FILE* f = alloc_log()) {
    std::fprintf(f, "%c %zu %llu 0x%llx\n", op, size,
                 static_cast<unsigned long long>(rounded),
                 static_cast<unsigned long long>(va));
    std::fflush(f);
  }
  *out_va = va;
  return true;
}

// Return a region pointer to the free list. Caller must hold g_mu.
bool release(void* ptr) {
  if (!in_region(ptr)) {
    return false;
  }
  const std::uint64_t va = reinterpret_cast<std::uint64_t>(ptr);
  auto it = g_live.find(va);
  if (it != g_live.end()) {
    g_free.emplace(it->second, va);  // keep resident, available for reuse
    g_live.erase(it);
  }
  return true;
}

}  // namespace

extern "C" {

cudaError_t cudaMalloc(void** ptr, size_t size) {
  if (ptr == nullptr) {
    return cudaErrorInvalidValue;
  }
  std::uint64_t out_va = 0;
  bool served = false;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (ensure_init()) {
      served = serve(size, &out_va);
    }
  }
  if (served) {
    *ptr = reinterpret_cast<void*>(out_va);
    return cudaSuccess;
  }
  return real_cudaMalloc()(ptr, size);
}

cudaError_t cudaFree(void* ptr) {
  if (ptr != nullptr) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (release(ptr)) {
      return cudaSuccess;
    }
  }
  using Fn = cudaError_t (*)(void*);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaFree"));
  return real(ptr);
}

// torch's caching allocator can route through the stream-ordered API. Stream
// ordering is irrelevant for a resident arena, so serve the same way.
cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaStream_t stream) {
  if (ptr == nullptr) {
    return cudaErrorInvalidValue;
  }
  std::uint64_t out_va = 0;
  bool served = false;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (ensure_init()) {
      served = serve(size, &out_va);
    }
  }
  if (served) {
    *ptr = reinterpret_cast<void*>(out_va);
    return cudaSuccess;
  }
  using Fn = cudaError_t (*)(void**, size_t, cudaStream_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaMallocAsync"));
  return real(ptr, size, stream);
}

cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream) {
  if (ptr != nullptr) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (release(ptr)) {
      return cudaSuccess;
    }
  }
  using Fn = cudaError_t (*)(void*, cudaStream_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaFreeAsync"));
  return real(ptr, stream);
}

// Exported accessors for the N5 record/restore shim (a separate .so that
// resolves these via dlsym(RTLD_DEFAULT, ...) to compute the relocation delta).
__attribute__((visibility("default")))
std::uint64_t snapshot_redirect_region_base() { return g_base; }

__attribute__((visibility("default")))
std::uint64_t snapshot_redirect_region_size() { return g_region; }

}  // extern "C"

namespace {
struct RedirectSummary {
  ~RedirectSummary() {
    if (g_base == 0) {
      return;
    }
    std::fprintf(stderr,
                 "[redirect-cuda] pid=%d SUMMARY mode=%s base=0x%llx served=%llu "
                 "reused=%llu passthrough=%llu(%lluMiB) live=%zu free=%zu "
                 "region_used=%lluMiB\n",
                 static_cast<int>(getpid()), arena_mode() ? "arena" : "fixed_vmm",
                 static_cast<unsigned long long>(g_base),
                 static_cast<unsigned long long>(g_served),
                 static_cast<unsigned long long>(g_reused),
                 static_cast<unsigned long long>(g_passthrough),
                 static_cast<unsigned long long>(g_passthrough_bytes >> 20),
                 g_live.size(), g_free.size(),
                 static_cast<unsigned long long>(g_cursor >> 20));
  }
};
RedirectSummary g_redirect_summary;
}  // namespace
