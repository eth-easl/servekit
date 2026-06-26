# N2 — CUDA Redirect Interposer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `snapshot_redirect_cuda.cpp` — an `LD_PRELOAD` shim that serves `cudaMalloc` from a single fixed-base (`0x600000000000`) deterministic VMM region — so an unmodified CUDA application's device addresses become **byte-identical across cold starts**, on an A100, with correct compute and < 10 s startup overhead. Mirrors the proven HIP `snapshot_redirect.cpp`, with the HIP path left byte-for-byte untouched.

**Architecture:** A standalone shared object that intercepts the CUDA runtime malloc family (`cudaMalloc`/`cudaFree`/`cudaMallocAsync`/`cudaFreeAsync`) and serves them from one contiguous device region. The default backing is a **fixed-base VMM arena**: one `cuMemAddressReserve(0x600000000000)` + one `cuMemCreate(whole)` + one `cuMemMap` + **one** `cuMemSetAccess(whole region)`, then bump + size-bucketed-free-list sub-allocation inside it (Δ=0 across cold starts). The driver VMM calls are made **inline** (mirroring HIP's `fixed_vmm_mode`), so the `.so` links only `libcuda` and is portable from the build container into the torch container. A `SNAPSHOT_REDIRECT_ARENA=1` toggle selects a plain-`cudaMalloc` arena fallback for diagnosis. A raw-`cudaMalloc` smoke program and a real-torch script validate transparency + determinism on the cluster.

**Tech Stack:** C++17, CUDA Driver API (`cu*`, `libcuda`) + CUDA Runtime headers (`cuda_runtime_api.h`, for the hooked symbols' signatures only — not linked), `dlfcn` (`dlsym(RTLD_NEXT, …)`), CMake ≥ 3.20, enroot/pyxis on the CSCS **bristen** A100 cluster (SLURM `-A a-infra02`, partition `normal`), `rcc --profile bristen-snapshot push` for code sync.

## Global Constraints

- **Fixed base:** `snapshot::kDefaultRequestedBase` = `0x600000000000` (vendor-neutral constant in `snapshot/allocator.hpp`). N1 proved `cuMemAddressReserve` honors it on the A100 (`honored=1`); the Δ=0 fast path depends on it.
- **Default backing = fixed-base VMM.** `SNAPSHOT_REDIRECT_ARENA=1` selects the plain-`cudaMalloc` arena fallback (driver-chosen base, Δ≠0). This **inverts** the HIP default (HIP defaults to the `cudaMalloc`-equivalent arena because `hipMemSetAccess` was fragile; `cuMemSetAccess` is the canonical reliable CUDA path, so fixed-base is default here).
- **One-shot whole-region `set_access`.** Reserve + create + map + `cuMemSetAccess` are done **once over the entire region**, never per sub-allocation. Sub-allocation is pure bump + free-list arithmetic inside the single mapping.
- **Hook the runtime malloc family only** (`cudaMalloc`, `cudaFree`, `cudaMallocAsync`, `cudaFreeAsync`) — exactly the four the AMD redirect mirrors (`hipMalloc`/`hipFree`/`hipMallocAsync`/`hipFreeAsync`). Driver-direct (`cuMemAlloc`) and `expandable_segments` (driver VMM) hooking are out of scope (deferred to a later milestone; the default torch allocator and the smoke use `cudaMalloc`).
- **Inline driver VMM, no `snapshot_core` link.** The interposer makes `cu*` driver calls directly and includes `snapshot/allocator.hpp` only for the `kDefaultRequestedBase` constant. It links **only** `libcuda` + `libdl`, with `-static-libstdc++ -static-libgcc`, because it is built in the CUDA devel container and `LD_PRELOAD`ed into the torch (sglang) container.
- **Do NOT modify (regression invariant):** `csrc/backends/hip/*`, `csrc/backends/cuda/*` (N1, frozen), `csrc/preload/snapshot_redirect.cpp`, `csrc/preload/snapshot_record.cpp`, `csrc/preload/snapshot_preload.cpp`, all `csrc/core/*`, all `include/snapshot/*`, and existing `csrc/cli/*`. The only edits are: **new** files `csrc/preload/snapshot_redirect_cuda.cpp` and `csrc/cli/cuda_redirect_smoke.cpp`, the **CUDA branch** of `CMakeLists.txt`, and **new** files under `snapshot/recipe/`. (Note: the working tree may carry unrelated uncommitted AMD changes; N2 is executed in a clean worktree off the N1 commit, so the diff is N2-only.)
- **Run on hardware, not the login node.** Bristen login nodes have no GPU. Every build/gate runs in a container on a compute node via `sbatch`/`srun`. Build/raw-smoke container = `snapshot-cuda` (`nvidia/cuda:12.6.3-devel-ubuntu24.04`, toolchain, no torch). Torch container = `snapshot-torch-cuda` (pinned `lmsysorg/sglang` digest, CUDA 12.9.1, torch). A prefetched standalone CMake 3.30.8 already lives at `${DEPLOY_DIR}/cmake/` on shared `/capstor` (from N1).
- **Account/partition:** `-A a-infra02`, partition `normal`. `DEPLOY_DIR` / rcc remote = `/capstor/scratch/cscs/xyao/snapshot-cuda` (the `bristen-snapshot` rcc profile, already committed).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `snapshot/csrc/preload/snapshot_redirect_cuda.cpp` | `LD_PRELOAD` redirect shim: fixed-base VMM (inline `cu*`) + `cudaMalloc` arena fallback; serve/release/free-list; exported region accessors | Create |
| `snapshot/csrc/cli/cuda_redirect_smoke.cpp` | Standalone raw-`cudaMalloc` + driver-launch synthetic-workload program; prints addresses, verifies bit-identical compute | Create |
| `snapshot/CMakeLists.txt` | CUDA branch: build `snapshot_redirect_cuda` (minimal-dep `.so`) + `cuda_redirect_smoke` | Modify (CUDA branch only) |
| `snapshot/recipe/redirect_cuda_smoke.sbatch` | Build + raw determinism gate (2 preloaded runs byte-identical, control differs, compute OK) in the devel container | Create |
| `snapshot/recipe/snapshot-torch-cuda.toml` | EDF for the torch (sglang) container, pinned by digest | Create |
| `snapshot/recipe/_redirect_torch_smoke.py` | torch script: alloc tensors, print `data_ptr()`, verify a compute (mirrors `_smoke_fixed_vmm.py`) | Create |
| `snapshot/recipe/redirect_cuda_torch.sbatch` | Cross-container gate: build `.so` in devel, run torch twice under `LD_PRELOAD` in sglang; determinism + compute + startup-overhead | Create |
| `snapshot/RESULTS.md` | Append an "N2 — CUDA redirect interposer on A100" section | Modify (append only) |

---

## Task 1: CUDA redirect interposer + raw-smoke determinism gate

Create the redirect shim, a raw-`cudaMalloc` smoke program, wire both into the CUDA branch of CMake, and prove on the A100 that two `LD_PRELOAD`ed runs return byte-identical device addresses and correct compute (while a no-preload control differs).

**Files:**
- Create: `snapshot/csrc/preload/snapshot_redirect_cuda.cpp`
- Create: `snapshot/csrc/cli/cuda_redirect_smoke.cpp`
- Modify: `snapshot/CMakeLists.txt` (CUDA branch, lines 70–77 region)
- Create: `snapshot/recipe/redirect_cuda_smoke.sbatch`

**Interfaces:**
- Consumes: `snapshot::kDefaultRequestedBase` (header constant), `snapshot::compile_synthetic_module` (smoke only, via `snapshot_core`), the N1-built `snapshot` CLI environment (CUDA devel container).
- Produces: `libsnapshot_redirect_cuda.so` exporting C symbols `cudaMalloc`/`cudaFree`/`cudaMallocAsync`/`cudaFreeAsync` (interposers) and `snapshot_redirect_region_base()` / `snapshot_redirect_region_size()` (accessors, `extern "C"`, default visibility, consumed by the N5 record shim later); executable `cuda_redirect_smoke`.

- [ ] **Step 1: Write `snapshot_redirect_cuda.cpp`**

Create `snapshot/csrc/preload/snapshot_redirect_cuda.cpp` with:
```cpp
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
```

- [ ] **Step 2: Write `cuda_redirect_smoke.cpp`**

Create `snapshot/csrc/cli/cuda_redirect_smoke.cpp` with:
```cpp
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
```

- [ ] **Step 3: Wire both into the CUDA branch of `CMakeLists.txt`**

In `snapshot/CMakeLists.txt`, the CUDA branch currently ends at `snapshot_configure_cuda(snapshot_core)` (the `elseif(SNAPSHOT_BACKEND STREQUAL "CUDA")` block, around lines 70–77). Insert the following **after** that `snapshot_configure_cuda(snapshot_core)` line and **before** the closing `endif()`:
```cmake
  # LD_PRELOAD redirector: serves cudaMalloc from the deterministic fixed-base
  # VMM region so an unmodified CUDA app's device addresses become reproducible.
  # Built in the CUDA devel container but LD_PRELOADed into the torch (sglang)
  # container for the real-engine gate, so it must be self-contained: it makes
  # cu* driver calls inline (no snapshot_core link, no nvrtc) and links ONLY
  # libcuda; -static-libstdc++/-static-libgcc removes the C++-runtime ABI
  # dependency on the torch image. SNAPSHOT_CUDA_INCLUDE_DIR /
  # SNAPSHOT_CUDA_DRIVER_LIBRARY were resolved by snapshot_configure_cuda above.
  add_library(snapshot_redirect_cuda SHARED csrc/preload/snapshot_redirect_cuda.cpp)
  target_include_directories(snapshot_redirect_cuda PRIVATE
    "${SNAPSHOT_CUDA_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(snapshot_redirect_cuda PRIVATE
    "${SNAPSHOT_CUDA_DRIVER_LIBRARY}" ${CMAKE_DL_LIBS})
  target_link_options(snapshot_redirect_cuda PRIVATE
    -static-libstdc++ -static-libgcc)

  # Standalone raw-cudaMalloc GPU program to demonstrate transparent redirect.
  add_executable(cuda_redirect_smoke csrc/cli/cuda_redirect_smoke.cpp)
  snapshot_configure_cuda(cuda_redirect_smoke)
  target_link_libraries(cuda_redirect_smoke PRIVATE snapshot_core)
```

- [ ] **Step 4: Write the raw-smoke gate sbatch**

Create `snapshot/recipe/redirect_cuda_smoke.sbatch` with:
```bash
#!/bin/bash
#SBATCH --job-name=snapshot-redirect-cuda-smoke
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/snapshot-cuda/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build-cuda"

echo "redirect-cuda smoke started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"

srun --environment=snapshot-cuda \
  bash -lc 'set -euo pipefail
    CMAKE_PREFIX="${DEPLOY_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
    if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi
    cmake -S snapshot -B snapshot/build-cuda -DSNAPSHOT_BACKEND=CUDA
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target cuda_redirect_smoke snapshot_redirect_cuda
    LIB="$PWD/snapshot/build-cuda/libsnapshot_redirect_cuda.so"
    SMOKE="$PWD/snapshot/build-cuda/cuda_redirect_smoke"
    ls -l "$LIB" "$SMOKE"
    echo "--- linkage (expect libcuda only; no libstdc++/libnvrtc DT_NEEDED) ---"
    ldd "$LIB" || true

    echo "--- control: no preload (driver-chosen addresses) ---"
    "$SMOKE" | tee /tmp/r_ctrl.txt

    echo "--- run 1: fixed-base redirect ---"
    LD_PRELOAD="$LIB" "$SMOKE" | tee /tmp/r1.txt
    echo "--- run 2: fixed-base redirect ---"
    LD_PRELOAD="$LIB" "$SMOKE" | tee /tmp/r2.txt

    echo "--- determinism check ---"
    A1=$(grep "^addrs" /tmp/r1.txt); A2=$(grep "^addrs" /tmp/r2.txt)
    AC=$(grep "^addrs" /tmp/r_ctrl.txt)
    echo "run1: $A1"; echo "run2: $A2"; echo "ctrl: $AC"
    if [ "$A1" = "$A2" ]; then echo "REDIRECT_DETERMINISTIC=1"; else echo "REDIRECT_DETERMINISTIC=0"; exit 1; fi
    if grep -q "verify OK" /tmp/r1.txt && grep -q "verify OK" /tmp/r2.txt; then echo "COMPUTE_OK=1"; else echo "COMPUTE_OK=0"; exit 1; fi
    if [ "$A1" = "$AC" ]; then echo "CONTROL_DIFFERS=0 (addresses coincided)"; else echo "CONTROL_DIFFERS=1"; fi
    case "$A1" in
      *"A=0x600000000000"*) echo "FIXED_BASE_HONORED=1" ;;
      *) echo "FIXED_BASE_HONORED=0" ;;
    esac
  '

echo "redirect-cuda smoke finished: $(date --iso-8601=seconds)"
```

- [ ] **Step 5: Run the raw-smoke gate**

Run (from the worktree root, with `<REMOTE>` = `/capstor/scratch/cscs/xyao/snapshot-cuda`):
```bash
rcc --profile bristen-snapshot push
ssh bristen 'cd /capstor/scratch/cscs/xyao/snapshot-cuda && sbatch snapshot/recipe/redirect_cuda_smoke.sbatch'
# then tail:
ssh bristen 'tail -n 40 /capstor/scratch/cscs/xyao/snapshot-cuda/logs/snapshot-redirect-cuda-smoke-*.out'
```
Expected: the build links `libsnapshot_redirect_cuda.so` (`ldd` shows `libcuda.so.1` and libc/libdl, **no** `libstdc++`/`libnvrtc`), then:
```
run1: addrs A=0x600000000000 B=0x... C=0x... OUT=0x...
run2: addrs A=0x600000000000 B=0x... C=0x... OUT=0x...
ctrl: addrs A=0x7f... B=0x7f... C=0x7f... OUT=0x7f...
REDIRECT_DETERMINISTIC=1
COMPUTE_OK=1
CONTROL_DIFFERS=1
FIXED_BASE_HONORED=1
```
`REDIRECT_DETERMINISTIC=1` + `COMPUTE_OK=1` is the core gate: two preloaded runs return byte-identical addresses and correct compute. `CONTROL_DIFFERS=1` proves the determinism comes from the shim, not luck. If `FIXED_BASE_HONORED=0` the VMM still works (relocation handles a shifted base) but flag it — N1 already showed `honored=1`, so a regression here is meaningful. If any check fails, debug with `superpowers:systematic-debugging` before proceeding.

- [ ] **Step 6: Commit**

```bash
git add snapshot/csrc/preload/snapshot_redirect_cuda.cpp \
  snapshot/csrc/cli/cuda_redirect_smoke.cpp \
  snapshot/CMakeLists.txt \
  snapshot/recipe/redirect_cuda_smoke.sbatch
git commit -m "snapshot(cuda): N2 Task 1 — redirect_cuda interposer + raw smoke (byte-identical addresses on A100)"
```

---

## Task 2: Real-torch determinism + startup-overhead gate (cross-container)

Prove the same `.so` makes a real PyTorch process's device addresses byte-identical across two cold starts, with correct compute and < 10 s startup overhead — loading the devel-built `.so` into the pinned sglang (CUDA 12.9.1, torch) container.

**Files:**
- Create: `snapshot/recipe/snapshot-torch-cuda.toml`
- Create: `snapshot/recipe/_redirect_torch_smoke.py`
- Create: `snapshot/recipe/redirect_cuda_torch.sbatch`

**Interfaces:**
- Consumes: `libsnapshot_redirect_cuda.so` (Task 1, built to `snapshot/build-cuda/` on shared `/capstor`); the pinned sglang image digest (copied from `deploy/glm-47-flash-bristen/glm-47-flash-sglang.toml`).
- Produces: a reproducible cross-container gate (`sbatch redirect_cuda_torch.sbatch`) asserting `TORCH_DETERMINISTIC=1`, `TORCH_COMPUTE_OK=1`, and `STARTUP_OVERHEAD_S < 10`.

- [ ] **Step 1: Write the torch EDF**

Create `snapshot/recipe/snapshot-torch-cuda.toml` with (the image digest is copied verbatim from `deploy/glm-47-flash-bristen/glm-47-flash-sglang.toml`; re-resolve only when intentionally upgrading):
```toml
# Torch container for the N2 redirect_cuda real-engine gate on bristen. Pinned to
# the same lmsysorg/sglang digest the GLM-4.7-Flash deploy uses (CUDA 12.9.1,
# A100/sm_80) so the torch the redirect is validated against is the one N3's
# vLLM/SGLang baseline will use. libcuda.so is injected by the CSCS NVIDIA hook;
# libnvrtc is NOT needed (the redirect .so links only libcuda).
image = "lmsysorg/sglang@sha256:e216b7dc4ac1938b599b982233ccf7eb2b11dd1f07fc2e00a7b9841052c553be"

mounts = [
  "/capstor:/capstor",
  "/users/xyao:/users/xyao",
  "/iopsstor:/iopsstor",
]

workdir = "/capstor/scratch/cscs/xyao/snapshot-cuda"

[env]
HOME = "/capstor/scratch/cscs/xyao/snapshot-cuda/home"
# Use the image's own torch; never pick up stale user venvs on /users.
PYTHONPATH = ""
PYTHONNOUSERSITE = "1"

[annotations]
com.pyxis.entrypoint_log = "true"
```

- [ ] **Step 2: Write the torch smoke script**

Create `snapshot/recipe/_redirect_torch_smoke.py` with (mirrors `_smoke_fixed_vmm.py`, but prints all `data_ptr()`s on one line for an easy cross-run diff and verifies a compute):
```python
import os, torch

print(f"[torch-smoke] ARENA={os.environ.get('SNAPSHOT_REDIRECT_ARENA')} "
      f"REGION_GIB={os.environ.get('SNAPSHOT_REDIRECT_REGION_GIB')} "
      f"torch={torch.__version__}", flush=True)

torch.cuda.init()
a = torch.zeros(1024, 1024, device='cuda')
b = torch.full((1024, 1024), 3.0, device='cuda')
s = (a + b).sum().item()
d = torch.zeros(8 << 20, device='cuda')   # 8 MiB
e = torch.zeros(16 << 20, device='cuda')  # 16 MiB
torch.cuda.synchronize()

print(f"[torch-smoke] PTRS a=0x{a.data_ptr():x} b=0x{b.data_ptr():x} "
      f"d=0x{d.data_ptr():x} e=0x{e.data_ptr():x}", flush=True)
expect = 3.0 * 1024 * 1024
ok = abs(s - expect) < 1.0
print(f"[torch-smoke] SUM={s} EXPECT={expect} COMPUTE={'OK' if ok else 'FAIL'}",
      flush=True)
raise SystemExit(0 if ok else 1)
```

- [ ] **Step 3: Write the cross-container gate sbatch**

Create `snapshot/recipe/redirect_cuda_torch.sbatch` with:
```bash
#!/bin/bash
#SBATCH --job-name=snapshot-redirect-cuda-torch
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=/capstor/scratch/cscs/xyao/snapshot-cuda/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build-cuda" \
  "${DEPLOY_DIR}/home"

echo "redirect-cuda torch gate started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"

# Step 1 (devel container): build the redirect .so on shared /capstor.
srun --environment=snapshot-cuda \
  bash -lc 'set -euo pipefail
    CMAKE_PREFIX="${DEPLOY_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
    if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi
    cmake -S snapshot -B snapshot/build-cuda -DSNAPSHOT_BACKEND=CUDA
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target snapshot_redirect_cuda
    ls -l snapshot/build-cuda/libsnapshot_redirect_cuda.so
  '

# Step 2 (torch container): load the .so under real torch, twice, + overhead.
srun --environment=snapshot-torch-cuda \
  bash -lc 'set -euo pipefail
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    LIB="$PWD/snapshot/build-cuda/libsnapshot_redirect_cuda.so"
    PY="$PWD/snapshot/recipe/_redirect_torch_smoke.py"
    echo "--- torch preflight ---"
    python -c "import torch; print(torch.__version__, torch.version.cuda)"
    echo "--- ldd .so under torch container (expect no \"not found\") ---"
    if ldd "$LIB" | grep -i "not found"; then echo "LDD_OK=0"; exit 1; else echo "LDD_OK=1"; fi

    echo "--- baseline (no preload) timing ---"
    t0=$(date +%s.%N); python "$PY"; t1=$(date +%s.%N)
    BASE_S=$(awk "BEGIN{print $t1-$t0}")

    echo "--- run 1: fixed-base redirect ---"
    t2=$(date +%s.%N)
    SNAPSHOT_REDIRECT_REGION_GIB=8 LD_PRELOAD="$LIB" python "$PY" | tee /tmp/t1.txt
    t3=$(date +%s.%N)
    PRELOAD_S=$(awk "BEGIN{print $t3-$t2}")
    echo "--- run 2: fixed-base redirect ---"
    SNAPSHOT_REDIRECT_REGION_GIB=8 LD_PRELOAD="$LIB" python "$PY" | tee /tmp/t2.txt

    echo "--- determinism + overhead ---"
    P1=$(grep "PTRS" /tmp/t1.txt); P2=$(grep "PTRS" /tmp/t2.txt)
    echo "run1: $P1"; echo "run2: $P2"
    if [ "$P1" = "$P2" ]; then echo "TORCH_DETERMINISTIC=1"; else echo "TORCH_DETERMINISTIC=0"; exit 1; fi
    if grep -q "COMPUTE=OK" /tmp/t1.txt && grep -q "COMPUTE=OK" /tmp/t2.txt; then echo "TORCH_COMPUTE_OK=1"; else echo "TORCH_COMPUTE_OK=0"; exit 1; fi
    OVH=$(awk "BEGIN{print $PRELOAD_S-$BASE_S}")
    echo "BASE_S=$BASE_S PRELOAD_S=$PRELOAD_S STARTUP_OVERHEAD_S=$OVH"
    awk "BEGIN{exit !($OVH < 10.0)}" && echo "STARTUP_OVERHEAD_OK=1" || { echo "STARTUP_OVERHEAD_OK=0"; exit 1; }
  '

echo "redirect-cuda torch gate finished: $(date --iso-8601=seconds)"
```

- [ ] **Step 4: Run the torch gate**

Run:
```bash
rcc --profile bristen-snapshot push
ssh bristen 'cd /capstor/scratch/cscs/xyao/snapshot-cuda && sbatch snapshot/recipe/redirect_cuda_torch.sbatch'
ssh bristen 'tail -n 50 /capstor/scratch/cscs/xyao/snapshot-cuda/logs/snapshot-redirect-cuda-torch-*.out'
```
Expected:
```
LDD_OK=1
run1: [torch-smoke] PTRS a=0x600000000000 b=0x... d=0x... e=0x...
run2: [torch-smoke] PTRS a=0x600000000000 b=0x... d=0x... e=0x...
TORCH_DETERMINISTIC=1
TORCH_COMPUTE_OK=1
BASE_S=... PRELOAD_S=... STARTUP_OVERHEAD_S=...
STARTUP_OVERHEAD_OK=1
```
`TORCH_DETERMINISTIC=1` + `TORCH_COMPUTE_OK=1` + `STARTUP_OVERHEAD_OK=1` is the full N2 gate on a real engine. The redirect does no fatbin/import scanning (it is pure malloc redirection), so the overhead — dominated by the one-time region reserve+map — should be a second or two, far under 10 s.

Likely failure modes and first checks (debug with `superpowers:systematic-debugging` if hit):
- `LDD_OK=0`: a `not found` line names the missing lib. If it is `libcuda.so.1`, the NVIDIA hook did not inject (confirm `--gpus-per-node`); if `libnvrtc`, the redirect picked up a stray nvrtc dependency (it should not — verify Task 1's `ldd` showed none).
- torch import error: the image's torch may be under a non-default `python`; try `python3` or the entrypoint's interpreter, and confirm the preflight line printed a version.
- `a.data_ptr()` not `0x600000000000`: torch's first allocation may be a small pool block; the determinism check (run1==run2) is the binding gate, the exact base is informational — but a non-fixed base means `fixed_base_honored` failed (cross-check the `[redirect-cuda] FIXED_VMM ...` stderr line).

- [ ] **Step 5: Commit**

```bash
git add snapshot/recipe/snapshot-torch-cuda.toml \
  snapshot/recipe/_redirect_torch_smoke.py \
  snapshot/recipe/redirect_cuda_torch.sbatch
git commit -m "snapshot(cuda): N2 Task 2 — real-torch determinism + startup-overhead gate (cross-container)"
```

---

## Task 3: RESULTS section + regression check

Document the N2 result and verify mechanically that the HIP/core/N1 paths are untouched.

**Files:**
- Modify: `snapshot/RESULTS.md` (append an "N2 — CUDA redirect interposer on A100" section)

**Interfaces:**
- Consumes: the passing Task 1 + Task 2 gate logs.
- Produces: a documented result; a verified-clean regression diff.

- [ ] **Step 1: Verify the HIP / core / N1 paths are untouched**

From the worktree root, with `BASE` = the N1 commit the branch started from (`git merge-base main HEAD`, i.e. `bb6ab99`):
```bash
git diff --stat "$(git merge-base main HEAD)" -- \
  snapshot/csrc/backends \
  snapshot/csrc/preload/snapshot_redirect.cpp \
  snapshot/csrc/preload/snapshot_record.cpp \
  snapshot/csrc/preload/snapshot_preload.cpp \
  snapshot/csrc/core snapshot/include \
  snapshot/csrc/cli/main.cpp snapshot/csrc/cli/workload.cpp \
  snapshot/csrc/cli/redirect_smoke.cpp
```
Expected: **no output** (N2 adds only `snapshot_redirect_cuda.cpp`, `cuda_redirect_smoke.cpp`, the CUDA branch of `CMakeLists.txt`, and recipe/RESULTS files; it touches no HIP backend, no CUDA N1 backend, no existing interposer, no core, no header, no existing CLI). If anything prints, an edit strayed outside the additive scope — revert it before continuing.

- [ ] **Step 2: Append the N2 RESULTS section**

Append to `snapshot/RESULTS.md` a section `# snapshot — N2: CUDA redirect interposer on bristen (A100 / sm_80)` documenting:
- environment (bristen node, sm_80, build container CUDA 12.6.x + torch container CUDA 12.9.1 / sglang digest, `-A a-infra02`);
- **what was built**: `snapshot_redirect_cuda.cpp` (fixed-base VMM default via inline `cu*`, `cudaMalloc` arena fallback, runtime malloc-family hooks), `cuda_redirect_smoke`, the two gates and the torch EDF;
- **raw gate** (job id): `REDIRECT_DETERMINISTIC=1`, `COMPUTE_OK=1`, `CONTROL_DIFFERS=1`, `FIXED_BASE_HONORED=1`, with the captured `addrs` lines;
- **torch gate** (job id): `LDD_OK=1`, `TORCH_DETERMINISTIC=1`, `TORCH_COMPUTE_OK=1`, the two `PTRS` lines, and `BASE_S`/`PRELOAD_S`/`STARTUP_OVERHEAD_S`;
- a CUDA↔HIP redirect equivalence note (HIP `fixed_vmm_mode` ⇄ CUDA inline fixed-base VMM; `hipMalloc`-family ⇄ `cudaMalloc`-family; default inverted: fixed-base is the CUDA default because `cuMemSetAccess` is reliable);
- the cross-container build/run split and the `-static-libstdc++` + libcuda-only linkage that makes it work;
- a closing note that the HIP path is unchanged and **N3** (vLLM-CUDA TP=4 deploy + baseline cold start) is the next milestone.

Keep the format consistent with the existing N1 section.

- [ ] **Step 3: Commit**

```bash
git add snapshot/RESULTS.md
git commit -m "snapshot(cuda): N2 complete — redirect_cuda deterministic addresses on A100, RESULTS documented, HIP untouched"
```

---

## Self-Review

**Spec coverage (against the design §5 `redirect_cuda` paragraph + the N2 milestone-table gate):**
- Hook `cudaMalloc`/`cudaFree` (+ async family) → Task 1 Step 1 ✓ (driver-direct `cuMemAlloc` / `expandable_segments` deferred, noted in Global Constraints — not exercised by the default torch allocator or the smoke).
- Fixed-base Δ=0 VMM from day one: one reserve + one create + one map + **one** `set_access` over the whole region → Task 1 Step 1 ✓.
- Fixed-base is the default; `SNAPSHOT_REDIRECT_ARENA` fallback toggle → Task 1 Step 1 + Global Constraints ✓.
- Gate: byte-identical device addresses across 2 cold runs, correct compute, startup overhead < 10 s, on raw program **and** real torch → Task 1 (raw) + Task 2 (torch) ✓.
- Dual-vendor, HIP untouched, additive-only → Global Constraints + Task 3 Step 1 ✓.
- Run on bristen compute nodes via sbatch → Tasks 1–2 ✓.

**Placeholder scan:** the only deliberate placeholder is `<REMOTE>`/`<REMOTE_DIR>`-style paths, which are resolved to the concrete `/capstor/scratch/cscs/xyao/snapshot-cuda` (the `bristen-snapshot` rcc profile) throughout — no `<...>` remains in any code or command. No "TBD/TODO/handle edge cases"; all code is complete and compilable.

**Type consistency:** the interposer's hooked signatures match the CUDA runtime ABI (`cudaError_t cudaMalloc(void**, size_t)`, `cudaFree(void*)`, `cudaMallocAsync(void**, size_t, cudaStream_t)`, `cudaFreeAsync(void*, cudaStream_t)`); the inline VMM uses the same `CUmemAllocationProp`/`CUmemGenericAllocationHandle`/`CUmemAccessDesc`/`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`/`cuMemSetAccess` types N1's `cuda_vmm.cpp` proved on this device. The smoke uses `cuLaunchKernel`'s `kernelParams` (pointer-array) form, avoiding the packed-buffer size validation N1 documented. The exported accessor names (`snapshot_redirect_region_base`/`_size`) match the HIP redirect's, so the future N5 record shim resolves them identically across vendors. `kDefaultRequestedBase` is the shared `0x600000000000` constant from `allocator.hpp`.

**Scope:** N2 only (the `redirect_cuda` interposer + its gates). The `record_cuda` identity/capture/restore interposer is N5; vLLM-CUDA deploy is N3; skip-capture is N4 — separate plans per the design's milestone table.

---

## Execution Handoff

This plan is intended for **subagent-driven execution in a clean git worktree off the N1 commit (`bb6ab99`)**, so the large unrelated uncommitted AMD changes in the main working tree stay isolated and every per-task diff + the Task 3 regression check are N2-only. Each gate is an `sbatch` round-trip on bristen (`rcc --profile bristen-snapshot push` → `sbatch -A a-infra02`); the reviewer checks the SLURM log before approving each task.
