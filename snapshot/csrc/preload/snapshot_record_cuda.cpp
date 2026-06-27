// snapshot_record_cuda — N5a LD_PRELOAD record interposer for CUDA.
//
// Task 2 built the IdentityMap (fatbin / __cudaRegisterFunction + nvrtc /
// cuModuleLoadData + cuModuleGetFunction) so every kernel that can appear as a
// graph node is identifiable.
//
// Task 3 (this revision) adds the CAPTURE/RECORD path: it interposes the driver
// cuStreamEndCapture, calls the real end-capture first (so run #1 executes
// normally), then — in record mode — walks the captured CUgraph and persists
// each KERNEL node (function identity + VERBATIM kernarg blob + grid/block dims
// + dependency edges) to a `.snap` file (record_cuda_format.hpp). Because this
// runs under snapshot_redirect_cuda's fixed-base device pointers, the recorded
// kernarg blob is byte-identical across runs (Δ=0) — Task 4 restores it
// verbatim, no cubin/PTX parser, no pointer relocation.
//
// SCOPE (N5a): interpose ONLY the driver cuStreamEndCapture (the CLI smoke uses
// the driver path exclusively). The runtime cudaStreamEndCapture — and the
// double-walk dedupe-by-CUgraph it would need — is deferred to N5b. Only KERNEL
// nodes are recorded; a kernel node whose kernargs use the `extra`
// (CU_LAUNCH_PARAM_BUFFER_POINTER) path or whose identity cannot be resolved is
// marked BLIND and counted (it is not restorable verbatim) — N5b territory.
//
// Env contract:
//   SNAPSHOT_RECORD_CUDA_MODE=record|restore  (default: record)
//   SNAPSHOT_RECORD_CUDA_DIR=<dir>            (snapshot directory; default: .)
//
// At process exit, prints to stderr:
//   [record-cuda] pid=<N> SUMMARY mode=<mode> dir=<dir>
//       identity: <total> functions (<fatbin> fatbin, <module> module)
//       recorded: graphs=<G> nodes=<N> edges=<E> blind=<B>
//       restored=<N> fallthrough=<N>
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
#include <vector>

#include "record_cuda_format.hpp"

// Forward declaration of the exported identity resolver (defined at the bottom
// of this TU inside the extern "C" block). The record walk below calls it; the
// signature uses a caller-owned buffer (copy-out) so a stored name can never
// dangle past a future cuModuleUnload eviction.
extern "C" int snapshot_identity_for(CUfunction func, int* out_kind,
                                     char* out_name, std::size_t out_name_cap,
                                     std::uint64_t* out_module_hash);

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
  static const char* d = []() {
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

// Record counters (Task 3) + restore/fallthrough placeholders (Task 4).
std::uint64_t g_recorded    = 0;  // total kernel nodes serialized (legacy token)
std::uint64_t g_restored    = 0;
std::uint64_t g_fallthrough = 0;

// Task 3 record detail.
std::uint64_t g_graph_index     = 0;  // next `.snap` file index (graph-NNNN.snap)
std::uint64_t g_graphs_recorded = 0;
std::uint64_t g_nodes_recorded  = 0;
std::uint64_t g_edges_recorded  = 0;
std::uint64_t g_blind           = 0;  // G4 gate must end at 0

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

// Hash a PTX/cubin image.  PTX images are null-terminated C strings, so a
// bounded scan is safe.  Cap at 1 MiB to avoid pathological inputs.
// Note: strnlen is POSIX and not in std::; use a manual loop for portability.
std::uint64_t hash_image(const void* image) {
  if (!image) return 0ULL;
  const char* s = static_cast<const char*>(image);
  constexpr std::size_t kMaxBytes = 1ULL << 20;  // 1 MiB
  std::size_t len = 0;
  while (len < kMaxBytes && s[len] != '\0') ++len;
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

// cuStreamEndCapture is the ONLY graph API we interpose, so it is the only one
// that needs a dlsym(RTLD_NEXT) real-symbol resolver (to reach past our own
// override to the driver). Its signature is stable and unversioned.
//
// SNAPSHOT_STR expands its argument through the preprocessor before stringizing
// so the dlsym name matches whatever symbol cuda.h binds `cuStreamEndCapture`
// to (it is unversioned today, but this stays correct if a future header
// macro-remaps it — and is exactly the name our exported override defines).
#define SNAPSHOT_STR2(x) #x
#define SNAPSHOT_STR(x) SNAPSHOT_STR2(x)
using StreamEndCaptureFn = CUresult (*)(CUstream, CUgraph*);
StreamEndCaptureFn real_cuStreamEndCapture() {
  static const auto fn = reinterpret_cast<StreamEndCaptureFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuStreamEndCapture)));
  return fn;
}

// ---------------------------------------------------------------------------
// Record path (Task 3): walk a captured CUgraph and serialize its kernel nodes.
//
// The graph-walk driver calls (cuGraphGetNodes / cuGraphNodeGetType /
// cuGraphKernelNodeGetParams / cuGraphNodeGetDependencies / cuFuncGetParamInfo)
// are invoked DIRECTLY rather than through dlsym(RTLD_NEXT) thunks. This is
// deliberate: in CUDA 12 several of these names are macro-remapped by cuda.h to
// versioned symbols whose ABI differs (cuGraphKernelNodeGetParams ->
// _v2 with the v2 CUDA_KERNEL_NODE_PARAMS struct; cuGraphNodeGetDependencies
// gained an edge-data overload). Calling them directly lets the header pick the
// symbol AND the matching struct/signature consistently — exactly as N1's
// cuda_graph.cpp does — eliminating any v1/v2 mismatch a hand-written dlsym
// thunk would risk. We never interpose these, so a direct call resolves to the
// real libcuda implementation.
// ---------------------------------------------------------------------------

// Sum over params of (offset+size) = the kernel's declared kernarg-segment size.
// Mirrors N1 cuda_graph.cpp::kernarg_size_for. cuFuncGetParamInfo returns
// CUDA_ERROR_INVALID_VALUE once `i` is past the last parameter.
std::size_t cuda_kernarg_size(CUfunction f) {
  std::size_t total = 0;
  for (std::size_t i = 0;; ++i) {
    std::size_t off = 0;
    std::size_t sz = 0;
    if (cuFuncGetParamInfo(f, i, &off, &sz) != CUDA_SUCCESS) break;
    if (off + sz > total) total = off + sz;
  }
  return total;
}

// Reconstitute the contiguous kernarg buffer verbatim: kernelParams[j] points
// at one arg value; copy each back into the flat blob at its declared offset.
std::vector<std::uint8_t> pack_kernarg(CUfunction f, void** kernel_params,
                                       std::size_t ksize) {
  std::vector<std::uint8_t> blob(ksize, 0u);
  for (std::size_t j = 0;; ++j) {
    std::size_t off = 0;
    std::size_t sz = 0;
    if (cuFuncGetParamInfo(f, j, &off, &sz) != CUDA_SUCCESS) break;
    if (kernel_params == nullptr || kernel_params[j] == nullptr) continue;
    if (off + sz > blob.size()) continue;
    std::memcpy(blob.data() + off, kernel_params[j], sz);
  }
  return blob;
}

// Walk a captured graph and serialize its kernel nodes to graph-NNNN.snap.
// Called from the cuStreamEndCapture hook AFTER the real end-capture, only in
// record mode. Runs WITHOUT g_mu held (snapshot_identity_for takes g_mu
// internally; the non-recursive mutex would deadlock otherwise) — counter
// updates re-acquire g_mu briefly.
void record_captured_graph(CUgraph graph) {
  if (graph == nullptr) return;

  std::size_t n = 0;
  if (cuGraphGetNodes(graph, nullptr, &n) != CUDA_SUCCESS) return;
  std::vector<CUgraphNode> nodes(n);
  if (n > 0 && cuGraphGetNodes(graph, nodes.data(), &n) != CUDA_SUCCESS) return;

  // Pass 1: assign a record index to each KERNEL node, in cuGraphGetNodes
  // order. Two passes so dep handles resolve to record indices regardless of
  // the (driver-defined, not guaranteed topological) node ordering.
  std::map<CUgraphNode, std::uint32_t> rec_index;
  std::vector<CUgraphNode> kernel_nodes;
  kernel_nodes.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    CUgraphNodeType t{};
    if (cuGraphNodeGetType(nodes[i], &t) != CUDA_SUCCESS) continue;
    if (t != CU_GRAPH_NODE_TYPE_KERNEL) continue;  // N5a: kernel nodes only
    rec_index[nodes[i]] = static_cast<std::uint32_t>(kernel_nodes.size());
    kernel_nodes.push_back(nodes[i]);
  }

  // Pass 2: build one record per kernel node (record position == node index, so
  // rec_index stays aligned even for blind nodes).
  snapshot_cuda::RecordedGraph rec;
  rec.nodes.reserve(kernel_nodes.size());
  std::uint64_t local_blind = 0;
  std::uint64_t local_edges = 0;
  for (CUgraphNode node : kernel_nodes) {
    snapshot_cuda::RecordedNode rn;
    bool node_blind = false;

    CUDA_KERNEL_NODE_PARAMS p{};
    if (cuGraphKernelNodeGetParams(node, &p) == CUDA_SUCCESS) {
      rn.grid[0] = p.gridDimX;
      rn.grid[1] = p.gridDimY;
      rn.grid[2] = p.gridDimZ;
      rn.block[0] = p.blockDimX;
      rn.block[1] = p.blockDimY;
      rn.block[2] = p.blockDimZ;
      rn.shared_mem_bytes = p.sharedMemBytes;

      // Function identity (copy-out into a caller-owned buffer).
      int kind = 0;
      char name_buf[1024];
      name_buf[0] = '\0';
      std::uint64_t mhash = 0;
      if (snapshot_identity_for(p.func, &kind, name_buf, sizeof(name_buf),
                                &mhash)) {
        rn.kind = kind;
        rn.name = name_buf;
        rn.module_hash = mhash;
      } else {
        node_blind = true;  // unrecognised CUfunction
      }

      // Verbatim kernarg blob. Only the kernelParams (array-of-void*) path is
      // supported in N5a; an `extra`-only node (CU_LAUNCH_PARAM_BUFFER_POINTER)
      // is N5b — mark blind and leave kernarg empty.
      if (p.kernelParams != nullptr) {
        const std::size_t ksize = cuda_kernarg_size(p.func);
        rn.kernarg = pack_kernarg(p.func, p.kernelParams, ksize);
      } else {
        node_blind = true;
      }
    } else {
      node_blind = true;
    }

    // Dependency edges -> record indices of predecessors.
    std::size_t dn = 0;
    if (cuGraphNodeGetDependencies(node, nullptr, &dn) == CUDA_SUCCESS &&
        dn > 0) {
      std::vector<CUgraphNode> deps(dn);
      if (cuGraphNodeGetDependencies(node, deps.data(), &dn) == CUDA_SUCCESS) {
        for (CUgraphNode d : deps) {
          const auto it = rec_index.find(d);
          if (it != rec_index.end()) {
            rn.deps.push_back(it->second);
            ++local_edges;
          }
        }
      }
    }

    if (node_blind) ++local_blind;
    rec.nodes.push_back(std::move(rn));
  }

  // Reserve this graph's file index.
  std::uint64_t idx = 0;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    idx = g_graph_index++;
  }

  char path[1280];
  std::snprintf(path, sizeof(path), "%s/graph-%04llu.snap", g_snap_dir(),
                static_cast<unsigned long long>(idx));
  const bool ok = snapshot_cuda::serialize_graph(rec, path);

  if (ok) {
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_graphs_recorded;
    g_nodes_recorded += rec.nodes.size();
    g_edges_recorded += local_edges;
    g_blind += local_blind;
    g_recorded += rec.nodes.size();
  }

  std::fprintf(stderr,
               "[record-cuda] pid=%d graph=%llu nodes=%zu edges=%llu blind=%llu "
               "-> %s (%s)\n",
               static_cast<int>(getpid()),
               static_cast<unsigned long long>(idx), rec.nodes.size(),
               static_cast<unsigned long long>(local_edges),
               static_cast<unsigned long long>(local_blind), path,
               ok ? "ok" : "FAILED");
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
                 "recorded: graphs=%llu nodes=%llu edges=%llu blind=%llu "
                 "restored=%llu fallthrough=%llu\n",
                 static_cast<int>(getpid()),
                 g_mode() == Mode::kRecord ? "record" : "restore",
                 g_snap_dir(),
                 fatbin_count + module_count,
                 fatbin_count,
                 module_count,
                 static_cast<unsigned long long>(g_graphs_recorded),
                 static_cast<unsigned long long>(g_nodes_recorded),
                 static_cast<unsigned long long>(g_edges_recorded),
                 static_cast<unsigned long long>(g_blind),
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
// Capture-end hook (record path)
// ---------------------------------------------------------------------------

// cuStreamEndCapture: call the real end-capture FIRST so the caller gets a
// working CUgraph and run #1 executes normally; then, in record mode, walk the
// captured graph and serialize its kernel nodes. SCOPE (N5a): the runtime
// cudaStreamEndCapture is intentionally NOT interposed — the CLI smoke uses the
// driver path exclusively, and adding the runtime hook now would need a
// double-walk dedupe-by-CUgraph with zero gate coverage (deferred to N5b).
CUresult cuStreamEndCapture(CUstream stream, CUgraph* phGraph) {
  const CUresult rc = real_cuStreamEndCapture()(stream, phGraph);
  if (rc == CUDA_SUCCESS && phGraph != nullptr && *phGraph != nullptr &&
      g_mode() == Mode::kRecord) {
    record_captured_graph(*phGraph);
  }
  return rc;
}

// ---------------------------------------------------------------------------
// snapshot_identity_for — identity resolver for graph-node CUfunction handles.
// Used by the Task 3 record walk to tag each node with a stable kernel identity
// before serialisation.
//
// For module-kind functions (k_mul): O(log n) lookup in g_module_identity,
// populated eagerly by cuModuleGetFunction.
// For fatbin-kind functions (k_add): calls cuFuncGetName (CUDA 12.3+) to
// get the mangled device name, then matches it against g_hostfun_table.
//
// COPY-OUT semantics (a Task-2 review finding): the resolved name is copied
// into a caller-owned, bounded, NUL-terminated buffer rather than returned as a
// pointer into internal std::string storage — a future cuModuleUnload eviction
// (Task 4) must never be able to dangle a name the caller already stored.
//
// Returns 1 if resolved (out_name filled, NUL-terminated), 0 if unrecognised.
// out_kind: 0 = fatbin/static, 1 = module/nvrtc.
// out_name: caller-owned buffer of out_name_cap bytes.
// out_module_hash: FNV-1a64 hash of the PTX image; 0 for fatbin-kind.
// ---------------------------------------------------------------------------
__attribute__((visibility("default")))
int snapshot_identity_for(CUfunction     func,
                          int*           out_kind,
                          char*          out_name,
                          std::size_t    out_name_cap,
                          std::uint64_t* out_module_hash) {
  auto copy_name = [&](const std::string& s) {
    if (out_name && out_name_cap) {
      const std::size_t k =
          s.size() < out_name_cap - 1 ? s.size() : out_name_cap - 1;
      std::memcpy(out_name, s.data(), k);
      out_name[k] = '\0';
    }
  };

  // --- module-kind: O(log n) lookup in g_module_identity ---
  {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_module_identity.find(func);
    if (it != g_module_identity.end()) {
      if (out_kind)        *out_kind        = 1;
      copy_name(it->second.name);
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
        copy_name(kv.second);
        if (out_module_hash) *out_module_hash = 0ULL;
        return 1;
      }
    }
  }
#endif

  return 0;
}

}  // extern "C"
