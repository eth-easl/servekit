// snapshot_record_cuda — N5a Task 2: LD_PRELOAD identity interposer for CUDA.
//
// Interposes the CUDA fatbin-registration and module-load hooks to build an
// IdentityMap of all kernels that will appear as graph nodes.  No record or
// restore logic is implemented here — the Task 2 deliverable is proving that
// both the static (fatbin / __cudaRegisterFunction) and runtime (nvrtc /
// cuModuleLoadData + cuModuleGetFunction) kernel paths are observable,
// captured, and reported in the exit SUMMARY.
//
// Env contract:
//   SNAPSHOT_RECORD_CUDA_MODE=record|restore  (default: record)
//   SNAPSHOT_RECORD_CUDA_DIR=<dir>            (snapshot directory; default: .)
//
// At process exit, prints to stderr:
//   [record-cuda] pid=<N> SUMMARY mode=<mode> dir=<dir>
//       identity: <total> functions (<fatbin> fatbin, <module> module)
//       recorded=<N> restored=<N> fallthrough=<N>
//
// Patterns mirrored from snapshot_redirect_cuda.cpp (N2):
//   dlsym(RTLD_NEXT, ...) lazy real-symbol resolution, static-local singleton
//   env parsing, namespace-scope RAII summary, -static-libstdc++/-libgcc.
//
// Links: libcuda (driver, for cuFuncGetName + driver types) + libcudart
// (runtime, for __cudaRegister* ABI) + libdl.
// -static-libstdc++/-static-libgcc: portable into the vLLM-CUDA image (N5b).

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace {

std::mutex g_mu;

// ---------------------------------------------------------------------------
// Env
// ---------------------------------------------------------------------------

enum class Mode { kRecord, kRestore };

Mode g_mode() {
  static const Mode m = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_CUDA_MODE");
    if (e && std::strcmp(e, "restore") == 0) return Mode::kRestore;
    return Mode::kRecord;
  }();
  return m;
}

const char* g_snap_dir() {
  static const char* d = [] -> const char* {
    const char* e = std::getenv("SNAPSHOT_RECORD_CUDA_DIR");
    return e ? e : ".";
  }();
  return d;
}

// ---------------------------------------------------------------------------
// Identity tables
// ---------------------------------------------------------------------------

// Fatbin registry: void** handle -> void* fatCubin.
// Populated by __cudaRegisterFatBinary.
std::map<void**, void*> g_fatbin_registry;

// Fatbin kernel symbol table: hostFun pointer -> deviceName (mangled kernel
// name).  Populated by __cudaRegisterFunction.
//
// This is how k_add (static kernel, compiled by nvcc) becomes identifiable:
//   deviceName = mangled device function name  (e.g. "_Z5k_addPiS_i")
//   hostFun    = host stub pointer (stable key in the runtime's own table)
//
// Used by snapshot_identity_for() to resolve a CUfunction from a graph node
// by calling cuFuncGetName() (CUDA 12.3+) and matching the result here.
std::map<const void*, std::string> g_hostfun_table;

// Module hash registry: CUmodule -> FNV-1a64 hash of the PTX/cubin image.
// Populated by cuModuleLoadData.
std::map<CUmodule, std::uint64_t> g_module_hashes;

// Module kernel identity: CUfunction -> {name, module_hash}.
// Populated eagerly by cuModuleGetFunction on success.
// This is how k_mul (nvrtc kernel) becomes identifiable at the point its
// handle is first obtained.
struct ModuleKernelId {
  std::string name;
  std::uint64_t module_hash = 0;
};
std::map<CUfunction, ModuleKernelId> g_module_identity;

// Placeholder counters for Tasks 3–4.
std::uint64_t g_recorded    = 0;
std::uint64_t g_restored    = 0;
std::uint64_t g_fallthrough = 0;

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

std::uint64_t fnv1a_64(const void* data, std::size_t len) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint64_t>(p[i]);
    h *= 0x00000100000001b3ULL;
  }
  return h;
}

// Hash a PTX/cubin image.  PTX images are null-terminated C strings, so
// strnlen() is safe.  Bounded to 1 MiB to avoid pathological inputs.
std::uint64_t hash_image(const void* image) {
  if (!image) return 0ULL;
  const char* s = static_cast<const char*>(image);
  constexpr std::size_t kMaxBytes = 1ULL << 20;  // 1 MiB
  const std::size_t len = std::strnlen(s, kMaxBytes);
  return fnv1a_64(image, len);
}

// ---------------------------------------------------------------------------
// Lazy real-symbol resolution via dlsym(RTLD_NEXT)
// Mirrors snapshot_redirect_cuda.cpp: each real_*() is a memoised thunk.
// ---------------------------------------------------------------------------

using RegisterFatBinaryFn = void** (*)(void*);
RegisterFatBinaryFn real_registerFatBinary() {
  static const auto fn = reinterpret_cast<RegisterFatBinaryFn>(
      dlsym(RTLD_NEXT, "__cudaRegisterFatBinary"));
  return fn;
}

using RegisterFunctionFn = void (*)(void**, const char*, char*, const char*,
                                    int, uint3*, uint3*, dim3*, dim3*, int*);
RegisterFunctionFn real_registerFunction() {
  static const auto fn = reinterpret_cast<RegisterFunctionFn>(
      dlsym(RTLD_NEXT, "__cudaRegisterFunction"));
  return fn;
}

using RegisterFatBinaryEndFn = void (*)(void**);
RegisterFatBinaryEndFn real_registerFatBinaryEnd() {
  static const auto fn = reinterpret_cast<RegisterFatBinaryEndFn>(
      dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd"));
  return fn;
}

using ModuleLoadDataFn = CUresult (*)(CUmodule*, const void*);
ModuleLoadDataFn real_cuModuleLoadData() {
  static const auto fn = reinterpret_cast<ModuleLoadDataFn>(
      dlsym(RTLD_NEXT, "cuModuleLoadData"));
  return fn;
}

using ModuleGetFunctionFn = CUresult (*)(CUfunction*, CUmodule, const char*);
ModuleGetFunctionFn real_cuModuleGetFunction() {
  static const auto fn = reinterpret_cast<ModuleGetFunctionFn>(
      dlsym(RTLD_NEXT, "cuModuleGetFunction"));
  return fn;
}

// ---------------------------------------------------------------------------
// RAII exit summary (mirrors RedirectSummary in snapshot_redirect_cuda.cpp)
// The destructor runs as a static-object finaliser at process exit, after
// all hooks have fired.  Globals are read without the mutex — this matches
// the redirect pattern; at static-destructor time the mutex is still alive
// (declared before g_record_summary → destroyed after) and the app is
// single-threaded for all practical purposes.
// ---------------------------------------------------------------------------
struct RecordSummary {
  ~RecordSummary() {
    const std::size_t fatbin_count = g_hostfun_table.size();
    const std::size_t module_count = g_module_identity.size();
    std::fprintf(stderr,
                 "[record-cuda] pid=%d SUMMARY mode=%s dir=%s "
                 "identity: %zu functions (%zu fatbin, %zu module) "
                 "recorded=%llu restored=%llu fallthrough=%llu\n",
                 static_cast<int>(getpid()),
                 g_mode() == Mode::kRecord ? "record" : "restore",
                 g_snap_dir(),
                 fatbin_count + module_count,
                 fatbin_count,
                 module_count,
                 static_cast<unsigned long long>(g_recorded),
                 static_cast<unsigned long long>(g_restored),
                 static_cast<unsigned long long>(g_fallthrough));
  }
};
RecordSummary g_record_summary;

}  // namespace

// ---------------------------------------------------------------------------
// Interposed symbols
// All declared extern "C" so the dynamic linker can find them by name without
// C++ mangling.
// ---------------------------------------------------------------------------

extern "C" {

// ---------------------------------------------------------------------------
// Fatbin / static-kernel registration hooks
// Called at library-load time by the cuda runtime for every TU that contains
// __global__ functions — before main() runs.
// ---------------------------------------------------------------------------

// __cudaRegisterFatBinary: called once per TU with __global__ kernels.
// Call real; store handle→fatCubin in fatbin registry (lightweight).
void** __cudaRegisterFatBinary(void* fatCubin) {
  void** const handle = real_registerFatBinary()(fatCubin);
  if (handle) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_fatbin_registry[handle] = fatCubin;
  }
  return handle;
}

// __cudaRegisterFunction: called once per __global__ per fatbin.
// Call real; record hostFun → deviceName (the mangled device kernel name).
// Eager gate: stores only pointers and a short string; no cubin scanning.
void __cudaRegisterFunction(
    void**      fatCubinHandle,
    const char* hostFun,
    char*       deviceFun,
    const char* deviceName,
    int         thread_limit,
    uint3*      tid,
    uint3*      bid,
    dim3*       bDim,
    dim3*       gDim,
    int*        wSize
) {
  real_registerFunction()(fatCubinHandle, hostFun, deviceFun, deviceName,
                          thread_limit, tid, bid, bDim, gDim, wSize);
  if (hostFun && deviceName) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_hostfun_table[static_cast<const void*>(hostFun)] = deviceName;
  }
}

// __cudaRegisterFatBinaryEnd: called after all __cudaRegisterFunction calls
// for a fatbin.  Pass-through only.
void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
  real_registerFatBinaryEnd()(fatCubinHandle);
}

// ---------------------------------------------------------------------------
// Module / nvrtc-kernel hooks
// Called explicitly by the application when loading runtime-compiled kernels.
// ---------------------------------------------------------------------------

// cuModuleLoadData: call real; on success compute a size-bounded FNV-1a hash
// of the image (PTX string for nvrtc output) and store mod→hash.
// Eager gate: no symbol enumeration on the load path.
CUresult cuModuleLoadData(CUmodule* mod, const void* image) {
  auto* const real = real_cuModuleLoadData();
  const CUresult rc = real(mod, image);
  if (rc == CUDA_SUCCESS && mod && *mod) {
    const std::uint64_t h = hash_image(image);
    std::lock_guard<std::mutex> lock(g_mu);
    g_module_hashes[*mod] = h;
  }
  return rc;
}

// cuModuleGetFunction: call real; on success insert the kernel's identity
// into g_module_identity keyed by the returned CUfunction handle.
// This is how k_mul (nvrtc / cuModuleLoadData) becomes identifiable.
// Eager gate: stores name string + O(log n) hash lookup only.
CUresult cuModuleGetFunction(CUfunction* f, CUmodule mod, const char* name) {
  auto* const real = real_cuModuleGetFunction();
  const CUresult rc = real(f, mod, name);
  if (rc == CUDA_SUCCESS && f && *f && name) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto hit = g_module_hashes.find(mod);
    const std::uint64_t mhash =
        (hit != g_module_hashes.end()) ? hit->second : 0ULL;
    g_module_identity[*f] = {name, mhash};
  }
  return rc;
}

// ---------------------------------------------------------------------------
// snapshot_identity_for — lazy identity resolver for graph-node CUfunction
// handles.  Used by Task 3 (graph-capture interposers) to tag each graph
// node with a stable kernel identity before serialisation.
//
// For module-kind functions (k_mul): O(log n) lookup in g_module_identity,
// populated eagerly by cuModuleGetFunction.
// For fatbin-kind functions (k_add): calls cuFuncGetName (CUDA 12.3+) to
// get the mangled device name, then matches it against g_hostfun_table.
//
// Returns 1 if resolved, 0 if the function is unrecognised.
// out_kind: 0 = fatbin/static, 1 = module/nvrtc.
// out_name: pointer into an internal std::string (valid until process exit).
// out_module_hash: FNV-1a64 hash of the PTX image; 0 for fatbin-kind.
// ---------------------------------------------------------------------------
__attribute__((visibility("default")))
int snapshot_identity_for(CUfunction     func,
                           int*           out_kind,
                           const char**   out_name,
                           std::uint64_t* out_module_hash) {
  // --- module-kind: O(log n) lookup in g_module_identity ---
  {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_module_identity.find(func);
    if (it != g_module_identity.end()) {
      if (out_kind)        *out_kind        = 1;
      if (out_name)        *out_name        = it->second.name.c_str();
      if (out_module_hash) *out_module_hash = it->second.module_hash;
      return 1;
    }
  }

  // --- fatbin-kind: cuFuncGetName → deviceName → g_hostfun_table match ---
  // cuFuncGetName was added in CUDA 12.3 (CUDA_VERSION >= 12030).
#if CUDA_VERSION >= 12030
  const char* dev_name = nullptr;
  const CUresult rc = cuFuncGetName(&dev_name, func);
  if (rc == CUDA_SUCCESS && dev_name) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (const auto& kv : g_hostfun_table) {
      if (kv.second == dev_name) {
        if (out_kind)        *out_kind        = 0;
        if (out_name)        *out_name        = kv.second.c_str();
        if (out_module_hash) *out_module_hash = 0ULL;
        return 1;
      }
    }
  }
#endif

  return 0;
}

}  // extern "C"
