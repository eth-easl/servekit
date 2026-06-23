// snapshot_redirect — an LD_PRELOAD shim that *redirects* device allocations
// into a single contiguous deterministic region so an unmodified application's
// device addresses become reproducible across cold starts.
//
// Two backings are available:
//   * ARENA (default): one real hipMalloc of the whole region at init, then
//     bump + free-list sub-allocation inside it. This needs NO VMM mapping and
//     therefore never touches hipMemSetAccess — which is the call that proved
//     fragile on gfx942 when torch interleaves its own HIP work (the M2.3
//     blocker). The region base is driver-chosen, so it is recorded for
//     relocation (constant Delta), exactly the mechanism M1 already proves.
//   * VMM (SNAPSHOT_REDIRECT_ARENA=0): the original DeterministicAllocator path
//     that pins the fixed base 0x600000000000 via reserve/create/map/set_access.
//     Kept for the standalone determinism demos (M2.2); blocked under full vLLM.
//
// To survive a real engine (which allocates and frees repeatedly), the shim
// keeps a size-bucketed free list over the region: hipFree returns a block to
// the free list (physical stays resident), and a later hipMalloc of the same
// rounded size reuses it. The allocation/free *order* is what the determinism
// spike showed to be reproducible, so reuse decisions — and therefore the
// returned addresses — stay reproducible too, while memory stays bounded.

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>

#include "snapshot/allocator.hpp"
#include "snapshot/gpu_backend.hpp"

namespace {

std::mutex g_mu;

// --- backing selection -------------------------------------------------------
bool arena_mode() {
  static const bool a = [] {
    const char* e = std::getenv("SNAPSHOT_REDIRECT_ARENA");
    return e == nullptr || e[0] != '0';  // default: arena ON
  }();
  return a;
}

// --- VMM backing (legacy) ----------------------------------------------------
std::unique_ptr<snapshot::GpuBackend> g_backend;
snapshot::DeterministicAllocator g_alloc;

// --- shared region state -----------------------------------------------------
bool g_init_attempted = false;
bool g_init_ok = false;
std::uint64_t g_base = 0;
std::uint64_t g_region = 0;
std::uint64_t g_cursor = 0;       // arena bump cursor (bytes from g_base)
std::uint64_t g_granularity = 4096;
bool g_fixed_base = false;

// va -> rounded size, for live region allocations.
std::map<std::uint64_t, std::uint64_t> g_live;
// rounded size -> va, for freed-but-resident blocks available for reuse.
std::multimap<std::uint64_t, std::uint64_t> g_free;

std::uint64_t g_served = 0;
std::uint64_t g_reused = 0;
std::uint64_t g_passthrough = 0;
std::uint64_t g_passthrough_bytes = 0;

bool verbose() {
  static const bool v = [] {
    const char* e = std::getenv("SNAPSHOT_REDIRECT_VERBOSE");
    return e != nullptr && e[0] != '0';
  }();
  return v;
}

std::FILE* alloc_log() {
  static std::FILE* f = []() -> std::FILE* {
    const char* dir = std::getenv("SNAPSHOT_REDIRECT_ALLOC_DIR");
    if (dir == nullptr || dir[0] == '\0') {
      return nullptr;
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/redir-%d.log", dir,
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

using MallocFn = hipError_t (*)(void**, size_t);
MallocFn real_hipMalloc() {
  static MallocFn fn = reinterpret_cast<MallocFn>(dlsym(RTLD_NEXT, "hipMalloc"));
  return fn;
}

// Caller must hold g_mu.
bool ensure_init() {
  if (g_init_attempted) {
    return g_init_ok;
  }
  g_init_attempted = true;

  if (arena_mode()) {
    // One real hipMalloc for the whole region. Device alignment of 512 B
    // packs tightly while staying >= torch's tensor alignment; reuse keys on
    // the rounded size so identical request sequences reuse identically.
    void* base = nullptr;
    hipError_t e = real_hipMalloc()(&base, region_bytes());
    if (e != hipSuccess || base == nullptr) {
      std::fprintf(stderr,
                   "[redirect] arena hipMalloc(%lluGiB) failed (%s); "
                   "passing through\n",
                   static_cast<unsigned long long>(region_bytes() >> 30),
                   hipGetErrorString(e));
      return false;
    }
    g_base = reinterpret_cast<std::uint64_t>(base);
    g_region = region_bytes();
    g_cursor = 0;
    g_granularity = 512;
    g_fixed_base = false;
    g_init_ok = true;
    // Publish the arena's device VA range to the recorder via env vars it
    // already reads at FLUSH (SNAPSHOT_RECORD_REGION_BASE/SIZE). The recorder
    // only learns the region from the VMM hipMemAddressReserve hook, which the
    // arena path never calls (it uses one hipMalloc), so without this the
    // snapshot ships region_base=0 and the rebuild cannot replay/map the arena
    // — captured device pointers then point at unmapped VA and fault at
    // hipGraphAddKernelNode.
    {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%llu",
                    static_cast<unsigned long long>(g_base));
      setenv("SNAPSHOT_RECORD_REGION_BASE", buf, 1);
      std::snprintf(buf, sizeof(buf), "%llu",
                    static_cast<unsigned long long>(g_region));
      setenv("SNAPSHOT_RECORD_REGION_SIZE", buf, 1);
    }
    std::fprintf(stderr,
                 "[redirect] pid=%d ARENA base=0x%llx size=%lluGiB gran=%llu\n",
                 static_cast<int>(getpid()),
                 static_cast<unsigned long long>(g_base),
                 static_cast<unsigned long long>(g_region >> 30),
                 static_cast<unsigned long long>(g_granularity));
    return true;
  }

  // Legacy VMM backing.
  g_backend = snapshot::make_backend();
  if (g_backend->vendor() == snapshot::Vendor::kStub) {
    std::fprintf(stderr, "[redirect] no GPU backend; passing through\n");
    return false;
  }
  snapshot::Status s =
      g_alloc.init(*g_backend, region_bytes(), snapshot::kDefaultRequestedBase);
  if (!s.ok()) {
    std::fprintf(stderr, "[redirect] region init failed: %s; passing through\n",
                 s.message().c_str());
    return false;
  }
  g_base = g_alloc.state().region_base;
  g_region = g_alloc.state().region_size;
  constexpr std::uint64_t kVmmChunk = 2ULL << 20;
  g_granularity = g_alloc.state().granularity > kVmmChunk
                      ? g_alloc.state().granularity
                      : kVmmChunk;
  g_fixed_base = g_alloc.state().fixed_base_honored;
  g_init_ok = true;
  std::fprintf(stderr,
               "[redirect] pid=%d VMM base=0x%llx size=%lluGiB gran=%llu "
               "fixed_base_honored=%d\n",
               static_cast<int>(getpid()),
               static_cast<unsigned long long>(g_base),
               static_cast<unsigned long long>(g_region >> 30),
               static_cast<unsigned long long>(g_granularity), g_fixed_base);
  return true;
}

bool in_region(const void* p) {
  const std::uint64_t v = reinterpret_cast<std::uint64_t>(p);
  return g_base != 0 && v >= g_base && v < g_base + g_region;
}

// Serve a request from the region. Caller must hold g_mu and have init'd.
// Returns true and sets *out_va on success; false means caller should fall
// back to the real allocator.
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
  } else if (arena_mode()) {
    if (g_cursor + rounded > g_region) {
      if (verbose()) {
        std::fprintf(stderr,
                     "[redirect] arena exhausted (used=%lluMiB req=%lluKiB); "
                     "real hipMalloc\n",
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
  } else {
    snapshot::Status s = g_alloc.alloc(*g_backend, rounded, "redir", &va);
    if (!s.ok()) {
      if (verbose()) {
        std::fprintf(stderr,
                     "[redirect] region alloc failed (%s); real hipMalloc\n",
                     s.message().c_str());
      }
      ++g_passthrough;
      g_passthrough_bytes += rounded;
      return false;
    }
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
// Returns true if the pointer belonged to the region.
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

hipError_t hipMalloc(void** ptr, size_t size) {
  if (ptr == nullptr) {
    return hipErrorInvalidValue;
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
    return hipSuccess;
  }
  return real_hipMalloc()(ptr, size);
}

hipError_t hipFree(void* ptr) {
  if (ptr != nullptr) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (release(ptr)) {
      return hipSuccess;
    }
  }
  using Fn = hipError_t (*)(void*);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipFree"));
  return real(ptr);
}

// Stream-ordered allocator (torch's HIP caching allocator can route through
// this). Stream ordering is irrelevant for a resident arena, so serve the same
// way; the free returns to the pool.
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
  if (ptr == nullptr) {
    return hipErrorInvalidValue;
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
    return hipSuccess;
  }
  using Fn = hipError_t (*)(void**, size_t, hipStream_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipMallocAsync"));
  return real(ptr, size, stream);
}

hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
  if (ptr != nullptr) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (release(ptr)) {
      return hipSuccess;
    }
  }
  using Fn = hipError_t (*)(void*, hipStream_t);
  static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "hipFreeAsync"));
  return real(ptr, stream);
}

}  // extern "C"

namespace {
struct RedirectSummary {
  ~RedirectSummary() {
    if (g_base == 0) {
      return;
    }
    std::fprintf(stderr,
                 "[redirect] pid=%d SUMMARY mode=%s base=0x%llx served=%llu "
                 "reused=%llu passthrough=%llu(%lluMiB) live=%zu free=%zu "
                 "region_used=%lluMiB\n",
                 static_cast<int>(getpid()), arena_mode() ? "arena" : "vmm",
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
