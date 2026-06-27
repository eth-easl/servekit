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
// Task 4 (this revision) adds the RESTORE path. In restore mode the same .so
// loads the recorded `.snap`s into an ordered queue and SHIMS the capture APIs:
// cuStreamBeginCapture fakes begin-capture (no real driver capture → G2),
// cuStreamEndCapture pops the queue and REBUILDS the graph (verbatim kernarg
// replay via cuGraphAddKernelNode, robust create-then-link), and BOTH launch
// entry points (runtime cudaLaunchKernel + driver cuLaunchKernel) SUPPRESS the
// launches issued inside the shim window so kernels never run eagerly — the
// rebuilt graph is the sole execution (suppressed=2 proves it; see the launch-
// suppression block). Recorded identities are reverse-resolved to live
// CUfunctions (g_module_identity for nvrtc kernels, g_hostfun_table +
// cudaGetFuncBySymbol for fatbin kernels); any unresolved node is BLIND (G4=0).
//
// SCOPE (N5a): the driver capture/launch path is interposed (the CLI smoke uses
// it exclusively). Only KERNEL nodes were recorded; a kernel node whose kernargs
// use the `extra` (CU_LAUNCH_PARAM_BUFFER_POINTER) path or whose identity
// cannot be resolved was marked BLIND and counted (not restorable verbatim) —
// N5b territory.
//
// N5b Task 1 extends this with RUNTIME capture-API shims
// (cudaStream{Begin,End,IsCapturing,GetCaptureInfo}) — the path
// torch.cuda.graph()/vLLM drive — routed to the SAME record/restore logic, with
// dedupe-by-CUgraph (g_walked_graphs) so a graph walked by both the runtime
// hook and the driver hook it invokes internally is serialized exactly once.
// A SUMMARY `rt_capture=<n>` field counts the runtime-path windows.
//
// N5b Task 2 extends record/restore to ALL graph node types: MEMCPY and MEMSET
// nodes are recorded structurally (verbatim params struct) and rebuilt; the
// `extra`-buffer kernarg form is recorded verbatim; every node is indexed so
// non-kernel dependency edges resolve (fixing the N5a edge-drop); any
// still-unsupported node type is BLIND with an explicit reason (no silent
// drop). `.snap` format bumped to v2 (type-tagged nodes).
//
// Env contract:
//   SNAPSHOT_RECORD_CUDA_MODE=record|restore  (default: record)
//   SNAPSHOT_RECORD_CUDA_DIR=<dir>            (snapshot directory; default: .)
//
// At process exit, prints to stderr:
//   [record-cuda] pid=<N> SUMMARY mode=<mode> dir=<dir>
//       identity: <total> functions (<fatbin> fatbin, <module> module)
//       recorded: graphs=<G> nodes=<N> edges=<E> blind=<B>
//       restored=<N> fallthrough=<M> suppressed=<S> real_begin=<R>
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
#include <sys/stat.h>
#include <unistd.h>

// RTLD_DEFAULT (GNU extension) is exposed under the same _GNU_SOURCE guard as
// the RTLD_NEXT the existing thunks use. Defensive fallback so the build does
// not depend on a specific toolchain header configuration.
#ifndef RTLD_DEFAULT
#define RTLD_DEFAULT ((void*)0)
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
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
  // N5b Task 3: per-rank snapshot dirs. If SNAPSHOT_RECORD_CUDA_DIR contains
  // "%r", substitute the worker rank from the first set of {RANK, LOCAL_RANK,
  // VLLM_DP_RANK, SLURM_PROCID} so each TP worker records/restores its own
  // .snap set without collision. Resolved once (cached in a static string).
  static const std::string d = []() {
    const char* e = std::getenv("SNAPSHOT_RECORD_CUDA_DIR");
    if (!e) return std::string(".");
    std::string s = e;
    const std::size_t p = s.find("%r");
    if (p == std::string::npos) return s;
    std::string rank;
    // N5b (vLLM TP=4): vLLM's set_cuda_visible_devices restricts each TP worker
    // process to exactly ONE physical GPU and remaps it to device 0, so
    // cuCtxGetDevice returns 0 for ALL workers. CUDA_VISIBLE_DEVICES (a single
    // physical GPU id per worker) is the authoritative, deterministic rank.
    // It MUST be checked BEFORE the rank env vars: srun sets SLURM_PROCID=0 for
    // the single serve task, which all 4 spawned workers inherit, so the env
    // vars would route every worker to rank0.
    rank.clear();
    if (const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES")) {
      std::string s2(cvd);
      std::string first;
      int count = 0;
      for (std::size_t i = 0; i < s2.size();) {
        while (i < s2.size() && !((unsigned char)s2[i] >= '0' && s2[i] <= '9')) ++i;
        if (i >= s2.size()) break;
        std::string num;
        while (i < s2.size() && (unsigned char)s2[i] >= '0' && s2[i] <= '9') { num += s2[i]; ++i; }
        if (count == 0) first = num;
        ++count;
      }
      if (count == 1 && !first.empty()) rank = first;
    }
    if (rank.empty()) {
      // N5b (vLLM TP=4): CUDA_VISIBLE_DEVICES lists multiple GPUs (vLLM does
      // NOT isolate workers via device-remapping — all see 0,1,2,3 and each
      // pins itself via cudaSetDevice(local_rank)). The rank env vars are also
      // wrong here (srun's SLURM_PROCID=0 is inherited by all 4 workers). The
      // authoritative per-worker rank is the CURRENT device index: vLLM calls
      // cudaSetDevice(rank) before capture, so cuCtxGetDevice == rank at the
      // first record_captured_graph call. Must NOT fall through to SLURM_PROCID.
      CUdevice dev = -1;
      if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) rank = std::to_string(dev);
    }
    if (rank.empty()) {
      const char* rank_envs[] = {"RANK", "LOCAL_RANK", "VLLM_DP_RANK",
                                 "SLURM_PROCID"};
      for (const char* name : rank_envs) {
        if (const char* v = std::getenv(name)) {
          if (*v) { rank = v; break; }
        }
      }
    }
    if (rank.empty()) rank = "0";
    std::fprintf(stderr, "[record-cuda] pid=%d rank-resolve: "
                 "CUDA_VISIBLE_DEVICES=\"%s\" -> rank=%s\n",
                 static_cast<int>(getpid()),
                 std::getenv("CUDA_VISIBLE_DEVICES")
                     ? std::getenv("CUDA_VISIBLE_DEVICES") : "(unset)",
                 rank.c_str());
    std::string out;
    out.reserve(s.size() + rank.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (i + 1 < s.size() && s[i] == '%' && s[i + 1] == 'r') {
        out += rank;
        ++i;
      } else {
        out += s[i];
      }
    }
    return out;
  }();
  return d.c_str();
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
// Peak (high-water) size of g_module_identity. The cuModuleUnload hook (Task 3)
// evicts entries to bound growth from Triton re-JIT and drop stale handles, so
// g_module_identity.size() at process exit UNDERcounts. The record walk runs at
// cuStreamEndCapture (before any cuModuleUnload), so resolution is unaffected;
// this peak lets the exit SUMMARY report how many module kernels were actually
// identified during the run.
std::size_t g_module_identity_peak = 0;

// N5b Task 3: prebuilt reverse-identity maps for O(log n) resolution at
// rebuild time (the N5a path scanned g_module_identity / g_hostfun_table in
// O(N) per node — acceptable for the CLI smoke, but hundreds of graphs ×
// thousands of nodes want the index). Maintained incrementally by the
// registration hooks so they stay current under lazy module loads, and evicted
// by the cuModuleUnload hook (bounds process-lifetime growth).
std::map<std::pair<std::string, std::uint64_t>, CUfunction> g_func_by_key;
std::map<std::string, const void*> g_hostfun_by_devname;
// CUmodule → its CUfunctions, so cuModuleUnload can evict identity entries.
std::map<CUmodule, std::set<CUfunction>> g_module_to_functions;

// Record + restore counters.
std::uint64_t g_restored    = 0;  // graphs rebuilt+returned by the restore shim
std::uint64_t g_fallthrough = 0;  // capture windows that fell through to real capture

// Task 3 record detail.
std::uint64_t g_graph_index     = 0;  // next `.snap` file index (graph-NNNN.snap)
std::uint64_t g_graphs_recorded = 0;
std::uint64_t g_nodes_recorded  = 0;
std::uint64_t g_edges_recorded  = 0;
std::uint64_t g_blind           = 0;  // G4 gate must end at 0

// Task 4 restore detail.
// g_suppressed_launches: launches intercepted DURING a shim-capture window and
//   returned success WITHOUT executing (so the rebuilt graph is the sole
//   execution). The gate asserts this == 2 (k_add via cudaLaunchKernel + k_mul
//   via cuLaunchKernel) — the proof that no kernel ran eagerly.
// g_real_begin_capture: number of times we called the REAL cuStreamBeginCapture
//   (record mode pass-through, or restore fall-through). In a shimmed restore it
//   stays 0 — the G2 "real-begin-capture count = 0" evidence.
std::uint64_t g_suppressed_launches = 0;
std::uint64_t g_real_begin_capture  = 0;
// N5b: runtime-API capture windows observed (the path PyTorch/vLLM drive).
// 0 for the N5a driver-API smoke; 1 for the Task-1 runtime smoke; N for vLLM.
std::uint64_t g_rt_captures         = 0;

// ---------------------------------------------------------------------------
// Restore state (Task 4)
// ---------------------------------------------------------------------------
//
// The deserialized `.snap` graphs are loaded ONCE (lazily, at first capture in
// restore mode) into an ordered queue and never mutated afterward, so the
// kernarg byte vectors they own keep stable addresses for the process lifetime.
// The rebuild points each node's driver `extra` config at those bytes verbatim
// (Δ=0 → the device pointers inside are valid), so the queue MUST outlive every
// graph the caller later instantiates — hence process-lifetime static storage.
struct RestoreState {
  std::vector<snapshot_cuda::RecordedGraph> queue;  // in graph-NNNN.snap order
  std::size_t next   = 0;                            // next graph to pop+rebuild
  bool        loaded = false;
};
RestoreState g_restore;

// Streams currently inside a SHIMMED capture window (begin-capture was faked).
// Guarded by g_mu. g_shim_active mirrors its non-emptiness as a lock-free flag
// so the launch hooks can early-out with a single atomic load on the hot path.
std::set<CUstream> g_shim_streams;
std::atomic<int>   g_shim_active{0};

// N5b: record-walk dedupe. The RUNTIME cudaStreamEndCapture (PyTorch's path)
// may route its internal cuStreamEndCapture through this same interposer
// (libcudart resolves the driver symbol against the LD_PRELOAD global scope)
// → one real capture fires BOTH the runtime and driver hooks. Keyed by the
// CUgraph handle so each captured graph is serialized exactly once. (Handles
// are process-unique across the capture phase; vLLM does not destroy graphs
// mid-capture, so the set is not evicted — see Task 3 for scale hardening.)
std::set<CUgraph> g_walked_graphs;

// N5b: count each real begin-capture once per window. When the runtime hook
// calls the real runtime, libcudart may hit the interposed driver
// cuStreamBeginCapture → two begin calls for one window. Deduped by stream;
// the matching erase is in both EndCapture paths.
std::set<CUstream> g_real_begin_streams;

// Per-rebuilt-node launch config (the driver `extra` buffer-pointer form). The
// config[] array and the size it references must stay valid until the caller
// instantiates the graph — which the shim cannot observe — so these are kept
// alive for the whole process (never freed). For N5a (one tiny graph) trivial.
struct NodeConfig {
  void*       config[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  std::size_t blob_size = 0;
};
std::vector<std::unique_ptr<NodeConfig>> g_node_configs;

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

// SNAPSHOT_STR expands its argument through the preprocessor before stringizing
// so a dlsym(RTLD_NEXT, ...) name tracks whatever symbol cuda.h binds the API
// to (several CUDA 12 entry points are macro-remapped to versioned symbols).
// Defined here, before the first Task-3 use (cuModuleLoadDataEx / Unload).
#define SNAPSHOT_STR2(x) #x
#define SNAPSHOT_STR(x) SNAPSHOT_STR2(x)

// N5b Task 3: the extended module-load form (some nvrtc/Triton paths use
// cuModuleLoadDataEx instead of cuModuleLoadData) and module unload (eviction).
using ModuleLoadDataExFn = CUresult (*)(CUmodule*, const void*, unsigned int,
                                        CUjit_option*, void**);
ModuleLoadDataExFn real_cuModuleLoadDataEx() {
  static const auto fn = reinterpret_cast<ModuleLoadDataExFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuModuleLoadDataEx)));
  return fn;
}

using ModuleUnloadFn = CUresult (*)(CUmodule);
ModuleUnloadFn real_cuModuleUnload() {
  static const auto fn = reinterpret_cast<ModuleUnloadFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuModuleUnload)));
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
using StreamEndCaptureFn = CUresult (*)(CUstream, CUgraph*);
StreamEndCaptureFn real_cuStreamEndCapture() {
  static const auto fn = reinterpret_cast<StreamEndCaptureFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuStreamEndCapture)));
  return fn;
}

// Restore-shim composition (Task 4): we also interpose cuStreamBeginCapture,
// cuStreamIsCapturing and BOTH launch entry points (runtime cudaLaunchKernel +
// driver cuLaunchKernel). Each needs a dlsym(RTLD_NEXT) real resolver. The
// SNAPSHOT_STR() macro is used for the driver names so the dlsym string tracks
// whatever symbol cuda.h binds (e.g. cuStreamBeginCapture -> _v2, or a PTSZ
// variant) — exactly matching the name our own override defines after the same
// header macro-expansion, so override and real-thunk stay in lock-step.
using StreamBeginCaptureFn = CUresult (*)(CUstream, CUstreamCaptureMode);
StreamBeginCaptureFn real_cuStreamBeginCapture() {
  static const auto fn = reinterpret_cast<StreamBeginCaptureFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuStreamBeginCapture)));
  return fn;
}

using StreamIsCapturingFn = CUresult (*)(CUstream, CUstreamCaptureStatus*);
StreamIsCapturingFn real_cuStreamIsCapturing() {
  static const auto fn = reinterpret_cast<StreamIsCapturingFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuStreamIsCapturing)));
  return fn;
}

using LaunchKernelFn = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                    unsigned, unsigned, unsigned, unsigned,
                                    CUstream, void**, void**);
LaunchKernelFn real_cuLaunchKernel() {
  static const auto fn = reinterpret_cast<LaunchKernelFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuLaunchKernel)));
  return fn;
}

using CudaLaunchKernelFn = cudaError_t (*)(const void*, dim3, dim3, void**,
                                           std::size_t, cudaStream_t);
CudaLaunchKernelFn real_cudaLaunchKernel() {
  static const auto fn = reinterpret_cast<CudaLaunchKernelFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cudaLaunchKernel)));
  return fn;
}

// N5b Task 1: RUNTIME capture-API real-symbol resolvers. PyTorch/vLLM drive
// CUDA-graph capture through the runtime cudaStream*Capture API (the
// cudaStream_t / cudaGraph_t types from cuda_runtime_api.h), not the driver
// cuStream* API the N5a CLI smoke used. cudaGraph_t IS CUgraph (both are
// CUgraph_st*), so the runtime hooks route to the SAME record/restore logic.
//
// SNAPSHOT_STR is used for the runtime names too so the dlsym string tracks
// whatever symbol cuda_runtime_api.h binds (e.g. a PTSZ / _v2 variant),
// exactly matching the name our own override defines after the same header
// macro-expansion — override and real-thunk stay in lock-step.
using CudaStreamBeginCaptureFn =
    cudaError_t (*)(cudaStream_t, cudaStreamCaptureMode);
CudaStreamBeginCaptureFn real_cudaStreamBeginCapture() {
  static const auto fn = reinterpret_cast<CudaStreamBeginCaptureFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cudaStreamBeginCapture)));
  return fn;
}

using CudaStreamEndCaptureFn = cudaError_t (*)(cudaStream_t, cudaGraph_t*);
CudaStreamEndCaptureFn real_cudaStreamEndCapture() {
  static const auto fn = reinterpret_cast<CudaStreamEndCaptureFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cudaStreamEndCapture)));
  return fn;
}

using CudaStreamIsCapturingFn =
    cudaError_t (*)(cudaStream_t, cudaStreamCaptureStatus*);
CudaStreamIsCapturingFn real_cudaStreamIsCapturing() {
  static const auto fn = reinterpret_cast<CudaStreamIsCapturingFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cudaStreamIsCapturing)));
  return fn;
}

// cudaStreamGetCaptureInfo gained a `const cudaGraphEdgeData**` out-param in
// CUDA 13 (7 params); CUDA 12.x has 6. The override + real-resolver typedef
// must match the header so the extern "C" override is not a conflicting
// declaration (a hard compile error in either image).
#if CUDA_VERSION >= 13000
using CudaStreamGetCaptureInfoFn = cudaError_t (*)(
    cudaStream_t, cudaStreamCaptureStatus*, unsigned long long*,
    cudaGraph_t*, const cudaGraphNode_t**, const cudaGraphEdgeData**, size_t*);
#else
using CudaStreamGetCaptureInfoFn = cudaError_t (*)(
    cudaStream_t, cudaStreamCaptureStatus*, unsigned long long*,
    cudaGraph_t*, const cudaGraphNode_t**, size_t*);
#endif
CudaStreamGetCaptureInfoFn real_cudaStreamGetCaptureInfo() {
  static const auto fn = reinterpret_cast<CudaStreamGetCaptureInfoFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cudaStreamGetCaptureInfo)));
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

// N5b Task 2: extract the verbatim kernarg blob from the `extra` launch config
// (CU_LAUNCH_PARAM_BUFFER_POINTER / BUFFER_SIZE) used by some vLLM/Triton
// launches. The `extra` array is a sequence of {tag, value} pairs terminated
// by CU_LAUNCH_PARAM_END. Returns the blob (empty on failure) — Δ=0 makes it
// directly valid at restore, replayed via the same `extra` form.
std::vector<std::uint8_t> pack_extra_kernarg(void** extra) {
  if (extra == nullptr) return {};
  const void* buffer_ptr = nullptr;
  std::size_t  buffer_sz  = 0;
  for (std::size_t i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
    // CU_LAUNCH_PARAM_{END,BUFFER_POINTER,BUFFER_SIZE} are ((void*)0xNN)
    // sentinels in cuda.h, so compare the tag as a void* directly (a
    // static_cast<uintptr_t> would be an invalid void*->integer conversion).
    void* tag = extra[i];
    if (tag == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      buffer_ptr = extra[i + 1];
    } else if (tag == CU_LAUNCH_PARAM_BUFFER_SIZE) {
      buffer_sz = *static_cast<const std::size_t*>(extra[i + 1]);
    }
  }
  if (buffer_ptr == nullptr || buffer_sz == 0) return {};
  return std::vector<std::uint8_t>(static_cast<const std::uint8_t*>(buffer_ptr),
                                   static_cast<const std::uint8_t*>(buffer_ptr) +
                                       buffer_sz);
}

// CUDA 13 added a CUgraphEdgeData out/in param to the graph dependency APIs
// (cuGraphNodeGetDependencies and cuGraphAddDependencies gained an edgeData
// slot, becoming _v2 with an extra arg). These inline wrappers present one
// arity to the call sites so the same source builds under CUDA 12.x and 13.
#if CUDA_VERSION >= 13000
static inline CUresult snap_node_get_deps(CUgraphNode n, CUgraphNode* deps,
                                          size_t* num) {
  return cuGraphNodeGetDependencies(n, deps, nullptr, num);
}
static inline CUresult snap_graph_add_deps(CUgraph g, const CUgraphNode* from,
                                           const CUgraphNode* to, size_t n) {
  return cuGraphAddDependencies(g, from, to, nullptr, n);
}
#else
static inline CUresult snap_node_get_deps(CUgraphNode n, CUgraphNode* deps,
                                          size_t* num) {
  return cuGraphNodeGetDependencies(n, deps, num);
}
static inline CUresult snap_graph_add_deps(CUgraph g, const CUgraphNode* from,
                                           const CUgraphNode* to, size_t n) {
  return cuGraphAddDependencies(g, from, to, n);
}
#endif

// Walk a captured graph and serialize its kernel nodes to graph-NNNN.snap.
// Called from the cuStreamEndCapture hook AFTER the real end-capture, only in
// record mode. Runs WITHOUT g_mu held (snapshot_identity_for takes g_mu
// internally; the non-recursive mutex would deadlock otherwise) — counter
// updates re-acquire g_mu briefly.
void record_captured_graph(CUgraph graph) {
  if (graph == nullptr) return;

  // N5b: dedupe-by-CUgraph. If this graph was already walked (runtime
  // end-capture routed through the driver hook inside libcudart), skip —
  // otherwise the same graph would be serialized twice (graph-NNNN.snap +
  // graph-(NNNN+1).snap), corrupting the per-graph count and the restore
  // queue ordering.
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_walked_graphs.insert(graph).second) return;
  }

  std::size_t n = 0;
  if (cuGraphGetNodes(graph, nullptr, &n) != CUDA_SUCCESS) return;
  std::vector<CUgraphNode> nodes(n);
  if (n > 0 && cuGraphGetNodes(graph, nodes.data(), &n) != CUDA_SUCCESS) return;

  // Pass 1: assign a record index to EVERY node (kernel, memcpy, memset, and
  // any other type), in cuGraphGetNodes order. Indexing all nodes — not just
  // kernel nodes — is what fixes the N5a non-kernel edge-drop: a dependency
  // edge into a memcpy/memset node now resolves to a real record index.
  std::map<CUgraphNode, std::uint32_t> rec_index;
  std::vector<CUgraphNode> all_nodes;
  all_nodes.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    rec_index[nodes[i]] = static_cast<std::uint32_t>(all_nodes.size());
    all_nodes.push_back(nodes[i]);
  }

  // Pass 2: build one record per node, dispatching on its type. Record
  // position == node index, so rec_index stays aligned for every tag.
  snapshot_cuda::RecordedGraph rec;
  rec.nodes.reserve(all_nodes.size());
  std::uint64_t local_blind = 0;
  std::uint64_t local_edges = 0;
  for (CUgraphNode node : all_nodes) {
    snapshot_cuda::RecordedNode rn;
    bool node_blind = false;

    CUgraphNodeType nt{};
    const bool have_type = (cuGraphNodeGetType(node, &nt) == CUDA_SUCCESS);
    if (have_type && nt == CU_GRAPH_NODE_TYPE_KERNEL) {
      rn.tag = snapshot_cuda::NodeTag::Kernel;
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

        // Verbatim kernarg blob. N5b: support BOTH launch forms —
        //   kernelParams (array-of-void*)  → pack via cuFuncGetParamInfo
        //   extra (CU_LAUNCH_PARAM_BUFFER_POINTER) → verbatim buffer copy
        // (vLLM/Triton may use the extra form). Both replay via the driver
        // `extra` config; kernarg_form records which form was captured.
        if (p.kernelParams != nullptr) {
          const std::size_t ksize = cuda_kernarg_size(p.func);
          rn.kernarg = pack_kernarg(p.func, p.kernelParams, ksize);
          rn.kernarg_form = 0;
        } else if (p.extra != nullptr) {
          rn.kernarg = pack_extra_kernarg(p.extra);
          rn.kernarg_form = 1;
          if (rn.kernarg.empty()) node_blind = true;  // malformed extra config
        } else {
          node_blind = true;  // neither kernarg form present
        }
      } else {
        node_blind = true;
      }
    } else if (have_type && nt == CU_GRAPH_NODE_TYPE_MEMCPY) {
      rn.tag = snapshot_cuda::NodeTag::Memcpy;
      CUDA_MEMCPY3D p{};
      if (cuGraphMemcpyNodeGetParams(node, &p) == CUDA_SUCCESS) {
        // Verbatim struct blob: Δ=0 makes the device pointers inside valid at
        // restore. (Same image on record + restore ⟹ identical struct layout.)
        const auto* src = reinterpret_cast<const std::uint8_t*>(&p);
        rn.blob.assign(src, src + sizeof(p));
      } else {
        node_blind = true;
        rn.reason = "cuGraphMemcpyNodeGetParams failed";
      }
    } else if (have_type && nt == CU_GRAPH_NODE_TYPE_MEMSET) {
      rn.tag = snapshot_cuda::NodeTag::Memset;
      CUDA_MEMSET_NODE_PARAMS p{};
      if (cuGraphMemsetNodeGetParams(node, &p) == CUDA_SUCCESS) {
        const auto* src = reinterpret_cast<const std::uint8_t*>(&p);
        rn.blob.assign(src, src + sizeof(p));
      } else {
        node_blind = true;
        rn.reason = "cuGraphMemsetNodeGetParams failed";
      }
    } else {
      // N5b (vLLM): wait-event (6) / event-record (7) / empty (5) / host (3) /
      // child-graph (4) nodes. They carry no serializable kernel identity or
      // (for events) process-local CUevent handles. Record them as Sync and
      // rebuild as EMPTY (no-op) nodes — the dependency EDGES already encode
      // the ordering, so an empty node preserving the deps is functionally
      // equivalent for the graph's output (the canonical Δ=0-restore property).
      rn.tag = snapshot_cuda::NodeTag::Sync;
    }

    // Dependency edges -> record indices of predecessors (all node types are
    // now indexed, so kernel→memcpy/memset edges resolve correctly).
    std::size_t dn = 0;
    if (snap_node_get_deps(node, nullptr, &dn) == CUDA_SUCCESS &&
        dn > 0) {
      std::vector<CUgraphNode> deps(dn);
      if (snap_node_get_deps(node, deps.data(), &dn) == CUDA_SUCCESS) {
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
  // Ensure the (possibly per-rank, e.g. rank%r -> rank2/) snapshot dir exists
  // before writing — the vLLM recipe points SNAPSHOT_RECORD_CUDA_DIR at a
  // per-rank path it does NOT pre-create. POSIX mkdir -p walk (no <filesystem>
  // dependency, keeps the .so self-contained for the vLLM image).
  {
    std::string d;
    for (const char* c = g_snap_dir(); *c; ++c) {
      d += *c;
      if (*c == '/') ::mkdir(d.c_str(), 0777);  // EEXIST is harmless
    }
    if (!d.empty()) ::mkdir(d.c_str(), 0777);
  }
  const bool ok = snapshot_cuda::serialize_graph(rec, path);

  if (ok) {
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_graphs_recorded;
    g_nodes_recorded += rec.nodes.size();
    g_edges_recorded += local_edges;
    g_blind += local_blind;
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
// Restore path (Task 4): load `.snap`s, reverse-resolve identities, rebuild.
// ---------------------------------------------------------------------------

// Load every graph-NNNN.snap (ascending index) into g_restore.queue ONCE. Lazy
// (first restore-mode capture) so the directory is read after the app has
// settled. Iterates indices until the first missing file — matching the record
// path's `graph-%04llu.snap` naming and preserving record order. Takes g_mu.
void ensure_restore_loaded() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_restore.loaded) return;
  g_restore.loaded = true;
  for (std::uint64_t idx = 0;; ++idx) {
    char path[1280];
    std::snprintf(path, sizeof(path), "%s/graph-%04llu.snap", g_snap_dir(),
                  static_cast<unsigned long long>(idx));
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) break;  // first gap ends the sequence
    std::fclose(f);
    snapshot_cuda::RecordedGraph g;
    if (snapshot_cuda::deserialize_graph(path, &g)) {
      g_restore.queue.push_back(std::move(g));
    } else {
      std::fprintf(stderr, "[record-cuda] pid=%d restore: FAILED to parse %s\n",
                   static_cast<int>(getpid()), path);
    }
  }
  std::fprintf(stderr,
               "[record-cuda] pid=%d restore: loaded %zu graph(s) from %s\n",
               static_cast<int>(getpid()), g_restore.queue.size(), g_snap_dir());
}

// REVERSE identity resolution: a recorded {kind, name, module_hash} -> the live
// CUfunction in THIS process. The forward maps (g_module_identity for nvrtc/
// module kernels, g_hostfun_table for fatbin/static kernels) are populated by
// the load-time and module hooks BEFORE the first capture, so both kernels are
// resolvable by rebuild time. Returns nullptr (→ blind, G4) if unresolved.
CUfunction resolve_function(const snapshot_cuda::RecordedNode& nd) {
  // Cache (any resolved name → CUfunction, shared across all resolution paths).
  auto cached = [&]() -> CUfunction {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto it = g_func_by_key.find({nd.name, nd.module_hash});
    return it != g_func_by_key.end() ? it->second : nullptr;
  };
  if (CUfunction c = cached()) return c;

  if (nd.kind == 1) {
    // module/nvrtc: match BOTH the kernel name and the PTX image hash (Task 3
    // proved the hash is run-to-run deterministic, so this is exact). N5b Task 3:
    // O(log n) via the prebuilt reverse index (the N5a per-node scan is gone).
    if (CUfunction c = cached()) return c;
  }

  // fatbin/static: find the host stub whose registered deviceName == nd.name,
  // then map it to a CUfunction via cudaGetFuncBySymbol (CUDA 12.0+;
  // cudaFunction_t and CUfunction are the same CUfunc_st*). N5b Task 3: O(log n)
  // via the prebuilt deviceName → hostFun reverse index.
  if (nd.kind == 0) {
    const void* hostfun = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      const auto it = g_hostfun_by_devname.find(nd.name);
      if (it != g_hostfun_by_devname.end()) hostfun = it->second;
    }
    if (hostfun != nullptr) {
      CUfunction cufunc = nullptr;
      const cudaError_t e =
          cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t*>(&cufunc), hostfun);
      if (e == cudaSuccess && cufunc != nullptr) {
        std::lock_guard<std::mutex> lock(g_mu);
        g_func_by_key[{nd.name, nd.module_hash}] = cufunc;
        return cufunc;
      }
    }
  }

  // N5b (vLLM): kind=2 (name-only — Triton/Inductor/NCCL kernels whose device
  // name was captured via cuFuncGetName at record but which are not
  // fatbin-registered and bypass cuModuleGetFunction), AND a fallback for any
  // kind whose primary path missed. Resolve by scanning EVERY loaded module
  // (tracked by the cuModuleLoadData hook) via cuModuleGetFunction(module,
  // name). vLLM loads the same modules at restore (Δ=0), so the name resolves.
  // One-time O(names × modules) at restore cold start; cached after first hit.
  if (!nd.name.empty()) {
    std::vector<CUmodule> mods;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      mods.reserve(g_module_hashes.size());
      for (const auto& kv : g_module_hashes) mods.push_back(kv.first);
    }
    auto* real_get = real_cuModuleGetFunction();
    for (CUmodule m : mods) {
      CUfunction f = nullptr;
      if (real_get && real_get(&f, m, nd.name.c_str()) == CUDA_SUCCESS && f) {
        std::lock_guard<std::mutex> lock(g_mu);
        g_func_by_key[{nd.name, nd.module_hash}] = f;
        return f;
      }
    }
  }
  return nullptr;
}

// Rebuild a CUgraph from a recorded graph using the ROBUST create-then-link
// pattern (NOT N1's in-order link, which silently drops edges whose source
// isn't built yet — unsafe because cuGraphGetNodes order is not guaranteed
// topological). Pass 1 creates every node (kernel / memcpy / memset) with NO
// deps; pass 2 wires the edges, so any node ordering works. The kernarg / struct
// blobs are replayed verbatim via the driver `extra` buffer-pointer config
// Memset rebuild workaround (N5b Task 2, found on CUDA 12.6 / bristen A100):
// cuGraphAddMemsetNode REJECTS the redirect's cuMemMap'd (fixed-VMM) dst
// pointers (rc=1 invalid value), even though cudaMemsetAsync and kernel
// launches accept them and the runtime created the memset node fine during
// capture. cuGraphAddMemcpyNode is NOT affected. Workaround: rebuild the memset
// as a CHILD GRAPH NODE by capturing the equivalent runtime cudaMemsetAsync
// (the runtime capture path creates the node without the broken validator).
// `parent_subgraphs` keeps the captured sub-graphs alive for the process
// lifetime (cuGraphAddChildGraphNode copies the topology, but the sub-graph
// handle is retained defensively).
//
// Semantics: maps the recorded CUDA_MEMSET_NODE_PARAMS to a runtime byte-fill.
// Correct when the memset value has uniform bytes (value=0, or any 0xXYXYXYXY
// pattern) — which covers every practical vLLM memset (zero-fill). For an
// elementSize>1 value with non-uniform bytes (a true multi-byte-element fill
// the runtime byte-memset cannot represent) this returns non-zero and the
// caller marks the node BLIND.
CUresult add_memset_via_child(CUgraph parent, CUgraphNode* node_out,
                              const snapshot_cuda::RecordedNode& nd,
                              std::vector<cudaGraph_t>& parent_subgraphs) {
  if (nd.blob.size() != sizeof(CUDA_MEMSET_NODE_PARAMS))
      return static_cast<CUresult>(1);
  const CUDA_MEMSET_NODE_PARAMS& p =
      *reinterpret_cast<const CUDA_MEMSET_NODE_PARAMS*>(nd.blob.data());
  // Uniform-byte check: the value, replicated to elementSize bytes, must equal
  // the recorded value (else a byte-fill is not equivalent).
  unsigned int v32 = p.value;
  unsigned char vbyte = static_cast<unsigned char>(v32 & 0xFFu);
  unsigned int built = 0;
  for (unsigned b = 0; b < p.elementSize; ++b)
      built |= (static_cast<unsigned>(vbyte) << (8 * b));
  if (p.elementSize > 1 && built != v32) return static_cast<CUresult>(1);

  CUstream tmp = nullptr;
  if (cuStreamCreate(&tmp, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
      return static_cast<CUresult>(1);
  cudaStream_t rs = reinterpret_cast<cudaStream_t>(tmp);
  // Call the REAL runtime begin/end directly (bypass our restore shim): this is
  // an internal helper capture, NOT a restore window. Going through our hook
  // would see an empty restore queue (the main graph is already popped) and
  // fall through to real anyway — but it would also bump g_fallthrough /
  // g_real_begin_capture, corrupting the G2 counters.
  if (real_cudaStreamBeginCapture()(rs, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
      cuStreamDestroy(tmp); return static_cast<CUresult>(1);
  }
  if (p.height <= 1) {
      cudaMemsetAsync(reinterpret_cast<void*>(p.dst), vbyte,
                      p.width * p.elementSize, rs);
  } else {
      cudaMemset2DAsync(reinterpret_cast<void*>(p.dst), p.pitch, vbyte,
                        p.width * p.elementSize, p.height, rs);
  }
  cudaGraph_t sub = nullptr;
  cudaError_t er = real_cudaStreamEndCapture()(rs, &sub);
  cuStreamDestroy(tmp);
  if (er != cudaSuccess || sub == nullptr) return static_cast<CUresult>(1);
  CUresult rc = cuGraphAddChildGraphNode(node_out, parent, nullptr, 0,
                                         reinterpret_cast<CUgraph>(sub));
  parent_subgraphs.push_back(sub);  // keep alive defensively
  return rc;
}

// (kernels) or the verbatim params struct (memcpy/memset) — exactly N1's launch
// mechanism, extended to non-kernel nodes. On any failure the partial graph is
// destroyed and false is returned. Must be called WITHOUT g_mu held
// (resolve_function takes it).
bool rebuild_graph(const snapshot_cuda::RecordedGraph& rec, CUgraph* out) {
  CUgraph g = nullptr;
  if (cuGraphCreate(&g, 0) != CUDA_SUCCESS) return false;

  const std::size_t n = rec.nodes.size();
  std::vector<CUgraphNode> built(n, nullptr);

  // Current context — cuGraphAddMemcpyNode / cuGraphAddMemsetNode require it.
  CUcontext ctx = nullptr;
  static_cast<void>(cuCtxGetCurrent(&ctx));  // best-effort; null → primary ctx

  // Memset nodes are rebuilt as child sub-graphs (cuMemMap-dst workaround);
  // keep the captured sub-graph handles alive for the process lifetime.
  std::vector<cudaGraph_t> subgraphs;

  // Pass 1: create all nodes with NO deps, tag-dispatched (kernel / memcpy /
  // memset / blind). Create-then-link so any cuGraphGetNodes ordering works.
  // The kernarg / struct blobs are replayed VERBATIM (the restore queue owns
  // them with stable addresses for the process lifetime); Δ=0 makes every
  // embedded device pointer valid unmodified. On any failure the partial graph
  // is destroyed and false is returned. Must be called WITHOUT g_mu held
  // (resolve_function takes it).
  for (std::size_t i = 0; i < n; ++i) {
    const snapshot_cuda::RecordedNode& nd = rec.nodes[i];
    CUgraphNode node = nullptr;
    CUresult rc = CUDA_SUCCESS;
    const char* what = "?";

    if (nd.tag == snapshot_cuda::NodeTag::Kernel) {
      const CUfunction func = resolve_function(nd);
      if (func == nullptr) {
        {
          std::lock_guard<std::mutex> lock(g_mu);
          ++g_blind;
        }
        std::fprintf(stderr,
                     "[record-cuda] pid=%d restore: BLIND node %zu name=%s "
                     "kind=%d (unresolved) — cannot rebuild\n",
                     static_cast<int>(getpid()), i, nd.name.c_str(), nd.kind);
        static_cast<void>(cuGraphDestroy(g));
        return false;
      }
      // Per-node `extra` config pointing at the VERBATIM recorded kernarg bytes.
      // Kept alive for the process lifetime (caller instantiates later, unseen).
      auto cfg = std::make_unique<NodeConfig>();
      cfg->blob_size = nd.kernarg.size();
      cfg->config[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
      cfg->config[1] = const_cast<std::uint8_t*>(nd.kernarg.data());
      cfg->config[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
      cfg->config[3] = &cfg->blob_size;
      cfg->config[4] = CU_LAUNCH_PARAM_END;

      CUDA_KERNEL_NODE_PARAMS p{};
      p.func           = func;
      p.gridDimX       = nd.grid[0];
      p.gridDimY       = nd.grid[1];
      p.gridDimZ       = nd.grid[2];
      p.blockDimX      = nd.block[0];
      p.blockDimY      = nd.block[1];
      p.blockDimZ      = nd.block[2];
      p.sharedMemBytes = nd.shared_mem_bytes;
      p.kernelParams   = nullptr;
      p.extra          = cfg->config;

      rc   = cuGraphAddKernelNode(&node, g, nullptr, 0, &p);
      what = "cuGraphAddKernelNode";
      if (rc == CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mu);
        g_node_configs.push_back(std::move(cfg));
      }
    } else if (nd.tag == snapshot_cuda::NodeTag::Memcpy) {
      // Verbatim CUDA_MEMCPY3D replay (mutable local copy → non-const ptr).
      if (nd.blob.size() != sizeof(CUDA_MEMCPY3D)) {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_blind;
        std::fprintf(stderr,
                     "[record-cuda] pid=%d restore: BLIND memcpy node %zu "
                     "(blob %zu != CUDA_MEMCPY3D %zu)\n",
                     static_cast<int>(getpid()), i, nd.blob.size(),
                     sizeof(CUDA_MEMCPY3D));
        static_cast<void>(cuGraphDestroy(g));
        return false;
      }
      CUDA_MEMCPY3D params =
          *reinterpret_cast<const CUDA_MEMCPY3D*>(nd.blob.data());
      rc   = cuGraphAddMemcpyNode(&node, g, nullptr, 0, &params, ctx);
      what = "cuGraphAddMemcpyNode";
    } else if (nd.tag == snapshot_cuda::NodeTag::Memset) {
      // Verbatim CUDA_MEMSET_NODE_PARAMS replay.
      if (nd.blob.size() != sizeof(CUDA_MEMSET_NODE_PARAMS)) {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_blind;
        std::fprintf(stderr,
                     "[record-cuda] pid=%d restore: BLIND memset node %zu "
                     "(blob %zu != CUDA_MEMSET_NODE_PARAMS %zu)\n",
                     static_cast<int>(getpid()), i, nd.blob.size(),
                     sizeof(CUDA_MEMSET_NODE_PARAMS));
        static_cast<void>(cuGraphDestroy(g));
        return false;
      }
      CUDA_MEMSET_NODE_PARAMS params =
          *reinterpret_cast<const CUDA_MEMSET_NODE_PARAMS*>(nd.blob.data());
      if (std::getenv("SNAPSHOT_RESTORE_MEMSET_DBG")) {
        std::fprintf(stderr,
                     "[record-cuda] memset node %zu: dst=%p pitch=%zu "
                     "value=%u elementSize=%u width=%zu height=%zu ctx=%p\n",
                     i, (void*)params.dst, params.pitch, params.value,
                     params.elementSize, params.width, params.height,
                     (void*)ctx);
      }
      // cuGraphAddMemsetNode rejects cuMemMap'd (fixed-VMM) dst pointers on CUDA
      // 12.6 — rebuild via a runtime-captured child sub-graph instead (see
      // add_memset_via_child). Falls back to BLIND only if the value has
      // non-uniform bytes (not representable as a runtime byte-fill).
      rc = add_memset_via_child(g, &node, nd, subgraphs);
      if (rc != CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_blind;
        std::fprintf(stderr,
                     "[record-cuda] pid=%d restore: BLIND memset node %zu "
                     "(child-capture workaround failed rc=%d; "
                     "value=%u elementSize=%u)\n",
                     static_cast<int>(getpid()), i, (int)rc,
                     params.value, params.elementSize);
        static_cast<void>(cuGraphDestroy(g));
        return false;
      }
      what = "cuGraphAddMemsetNode(child-workaround)";
    } else if (nd.tag == snapshot_cuda::NodeTag::Sync) {
      // wait-event / event-record / empty / host / child — rebuilt as an EMPTY
      // (no-op) node; the dependency edges (wired in pass 2) encode the ordering.
      rc   = cuGraphAddEmptyNode(&node, g, nullptr, 0);
      what = "cuGraphAddEmptyNode(Sync)";
    } else {
      // Blind node (unsupported type recorded as blind, or unresolved).
      std::lock_guard<std::mutex> lock(g_mu);
      ++g_blind;
      std::fprintf(stderr,
                   "[record-cuda] pid=%d restore: BLIND node %zu tag=%d "
                   "reason=%s — cannot rebuild\n",
                   static_cast<int>(getpid()), i, static_cast<int>(nd.tag),
                   nd.reason.c_str());
      static_cast<void>(cuGraphDestroy(g));
      return false;
    }

    if (rc != CUDA_SUCCESS) {
      const char* es = nullptr;
      cuGetErrorString(rc, &es);
      std::fprintf(stderr,
                   "[record-cuda] pid=%d restore: %s node %zu failed: %s\n",
                   static_cast<int>(getpid()), what, i, es ? es : "?");
      static_cast<void>(cuGraphDestroy(g));
      return false;
    }
    built[i] = node;
  }

  // Pass 2: add dependency edges (dep d -> node i), tolerant of any node order.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::uint32_t d : rec.nodes[i].deps) {
      if (d >= n) continue;  // defensive: corrupt index
      const CUresult rc =
          snap_graph_add_deps(g, &built[d], &built[i], 1);
      if (rc != CUDA_SUCCESS) {
        const char* es = nullptr;
        cuGetErrorString(rc, &es);
        std::fprintf(stderr,
                     "[record-cuda] pid=%d restore: cuGraphAddDependencies "
                     "(%u->%zu) failed: %s\n",
                     static_cast<int>(getpid()), d, i, es ? es : "?");
        static_cast<void>(cuGraphDestroy(g));
        return false;
      }
    }
  }

  *out = g;
  return true;
}

// Pop the next queued graph and rebuild it. Returns false if the queue is
// exhausted (caller falls through to real capture) or rebuild failed.
bool restore_next_graph(CUgraph* out) {
  const snapshot_cuda::RecordedGraph* rec = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_restore.next >= g_restore.queue.size()) return false;
    rec = &g_restore.queue[g_restore.next];
    ++g_restore.next;
  }
  return rebuild_graph(*rec, out);  // g_mu released: rebuild re-locks as needed
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
    // Peak, not current size: cuModuleUnload eviction would otherwise undercount
    // at exit (the smoke unloads its nvrtc module before the SUMMARY destructor).
    const std::size_t module_count = g_module_identity_peak;
    std::fprintf(stderr,
                 "[record-cuda] pid=%d SUMMARY mode=%s dir=%s "
                 "identity: %zu functions (%zu fatbin, %zu module) "
                 "recorded: graphs=%llu nodes=%llu edges=%llu blind=%llu "
                 "restored=%llu fallthrough=%llu suppressed=%llu "
                 "real_begin=%llu rt_capture=%llu\n",
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
                 static_cast<unsigned long long>(g_fallthrough),
                 static_cast<unsigned long long>(g_suppressed_launches),
                 static_cast<unsigned long long>(g_real_begin_capture),
                 static_cast<unsigned long long>(g_rt_captures));
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
    // N5b Task 3: reverse index (deviceName → hostFun) for O(log n) resolve.
    g_hostfun_by_devname[deviceName] = static_cast<const void*>(hostFun);
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
    std::lock_guard<std::mutex> lock(g_mu);
    // N5b Task 3: dedupe — cuModuleLoadData may internally route through
    // cuModuleLoadDataEx (also interposed), so only hash a module once.
    if (g_module_hashes.find(*mod) == g_module_hashes.end()) {
      g_module_hashes[*mod] = hash_image(image);
    }
  }
  return rc;
}

// cuModuleLoadDataEx (N5b Task 3): the extended module-load form. Same hash
// capture as cuModuleLoadData (deduped), so nvrtc/Triton paths that use this
// entry point also populate g_module_hashes for kernel identity.
CUresult cuModuleLoadDataEx(CUmodule* mod, const void* image,
                            unsigned int numOptions, CUjit_option* options,
                            void** optValues) {
  auto* const real = real_cuModuleLoadDataEx();
  const CUresult rc = real(mod, image, numOptions, options, optValues);
  if (rc == CUDA_SUCCESS && mod && *mod) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_module_hashes.find(*mod) == g_module_hashes.end()) {
      g_module_hashes[*mod] = hash_image(image);
    }
  }
  return rc;
}

// cuModuleUnload (N5b Task 3): evict the module's identity entries so repeated
// module load/unload (e.g. Triton re-JIT) does not grow the maps unboundedly.
// The CUfunction handles become invalid after unload; keeping them would also
// risk a stale-handle resolve at rebuild time.
CUresult cuModuleUnload(CUmodule mod) {
  if (mod) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto mfit = g_module_to_functions.find(mod);
    if (mfit != g_module_to_functions.end()) {
      for (CUfunction f : mfit->second) {
        const auto iit = g_module_identity.find(f);
        if (iit != g_module_identity.end()) {
          g_func_by_key.erase({iit->second.name, iit->second.module_hash});
          g_module_identity.erase(iit);
        }
      }
      g_module_to_functions.erase(mfit);
    }
    g_module_hashes.erase(mod);
  }
  return real_cuModuleUnload()(mod);
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
    if (g_module_identity.size() > g_module_identity_peak) {
      g_module_identity_peak = g_module_identity.size();
    }
    // N5b Task 3: prebuilt reverse index + module→function set (for resolve +
    // cuModuleUnload eviction).
    g_func_by_key[{name, mhash}] = *f;
    g_module_to_functions[mod].insert(*f);
  }
  return rc;
}

// ---------------------------------------------------------------------------
// Launch-suppression hooks (Task 4 — REQUIRED for the restore gate)
//
// WHY: G2 forbids calling the real capture, so the kernel launches the smoke
// issues between begin- and end-capture are NOT absorbed by any real capture.
// If we let them through they execute EAGERLY, and then the rebuilt graph runs
// them AGAIN. The smoke happens to be idempotent (k_add overwrites out from the
// immutable input), so a naive shim would still print the right CHECKSUM —
// passing G1 by luck, not by correctness. To make "skip capture" actually skip
// (and make G1 a genuine single-execution proof) the shim SUPPRESSES every
// launch on a shim-capturing stream: return success without executing, and
// count it. The gate asserts suppressed=2 (k_add + k_mul).
//
// Hot-path discipline: the first thing each hook does is a single lock-free
// atomic load; if no stream is in a shim-capture window (always true in record
// mode and post-capture) it is a pure pass-through to the real function.
// ---------------------------------------------------------------------------

// Returns true (and counts it) iff `stream` is inside a shimmed capture window,
// i.e. the launch should be suppressed instead of executed. Internal linkage
// (static) so it is not exported as a public dynamic symbol.
static bool launch_is_suppressed(CUstream stream) {
  if (g_shim_active.load(std::memory_order_acquire) == 0) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_shim_streams.count(stream) == 0) return false;
  ++g_suppressed_launches;
  return true;
}

// N5b: count each real begin-capture once per window. When the runtime hook
// calls the real runtime, libcudart may route its internal cuStreamBeginCapture
// through our (interposed) driver hook → two begin calls for one window. The
// runtime/driver begin hooks both gate g_real_begin_capture on this; the
// matching erase happens in both EndCapture paths. Returns true the FIRST time
// `stream` is seen in a window.
static bool count_real_begin_once(CUstream stream) {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_real_begin_streams.insert(stream).second;
}

// Runtime cudaLaunchKernel — backs the static k_add<<<>>> launch.
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, std::size_t sharedMem,
                             cudaStream_t stream) {
  if (launch_is_suppressed(reinterpret_cast<CUstream>(stream))) {
    return cudaSuccess;
  }
  return real_cudaLaunchKernel()(func, gridDim, blockDim, args, sharedMem,
                                 stream);
}

// Driver cuLaunchKernel — backs the nvrtc k_mul launch.
CUresult cuLaunchKernel(CUfunction f, unsigned gridDimX, unsigned gridDimY,
                        unsigned gridDimZ, unsigned blockDimX,
                        unsigned blockDimY, unsigned blockDimZ,
                        unsigned sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
  if (launch_is_suppressed(hStream)) {
    return CUDA_SUCCESS;
  }
  return real_cuLaunchKernel()(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                               blockDimY, blockDimZ, sharedMemBytes, hStream,
                               kernelParams, extra);
}

// ---------------------------------------------------------------------------
// Capture-shim hooks (record path + Task 4 restore shim)
// ---------------------------------------------------------------------------

// cuStreamBeginCapture: in restore mode, decide shim-vs-real HERE so the whole
// window is consistent. If a graph remains in the restore queue, fake begin-
// capture (mark the stream, return success, do NOT touch the real driver) —
// this is what keeps the real-begin-capture count at 0 (G2). If the queue is
// exhausted, fall through to a real capture (count it) so a short recording
// degrades gracefully. In record mode it is a transparent pass-through.
CUresult cuStreamBeginCapture(CUstream stream, CUstreamCaptureMode mode) {
  if (g_mode() == Mode::kRestore) {
    ensure_restore_loaded();
    bool shim = false;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      if (g_restore.next < g_restore.queue.size()) {
        // Idempotent mark vs the runtime hook (only one of the two fires per
        // window in practice, since a shimmed begin never calls the real API).
        shim = g_shim_streams.insert(stream).second;
      }
    }
    if (shim) {
      g_shim_active.fetch_add(1, std::memory_order_release);
      return CUDA_SUCCESS;  // no real begin-capture
    }
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_fallthrough;  // queue empty → real capture for this window
  }
  if (count_real_begin_once(stream)) {
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_real_begin_capture;
  }
  return real_cuStreamBeginCapture()(stream, mode);
}

// cuStreamIsCapturing: report ACTIVE for a shim-capturing stream so any code
// that polls capture status sees a consistent window (the N5a smoke doesn't,
// but N5b will). Otherwise defer to the real driver.
CUresult cuStreamIsCapturing(CUstream stream, CUstreamCaptureStatus* status) {
  if (g_mode() == Mode::kRestore &&
      g_shim_active.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_shim_streams.count(stream) > 0) {
      if (status) *status = CU_STREAM_CAPTURE_STATUS_ACTIVE;
      return CUDA_SUCCESS;
    }
  }
  return real_cuStreamIsCapturing()(stream, status);
}

// cuStreamEndCapture:
//  - Restore + shim-capturing stream: pop the next `.snap`, rebuild it into a
//    fresh CUgraph, hand it back as *phGraph, end the window. No real capture
//    happened, so this is the sole construction of the graph. (G2/G4.)
//  - Otherwise (record mode, or restore fall-through): call the REAL end-
//    capture; in record mode walk+serialize the captured graph (Task 3 path,
//    UNCHANGED). N5b Task 1: the RUNTIME cudaStreamEndCapture is now
//    interposed too (PyTorch's path); record_captured_graph's dedupe-by-
//    CUgraph keeps the walk single if libcudart routes the runtime call back
//    through this driver hook.
CUresult cuStreamEndCapture(CUstream stream, CUgraph* phGraph) {
  if (g_mode() == Mode::kRestore) {
    bool shimmed = false;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      shimmed = (g_shim_streams.count(stream) > 0);
    }
    if (shimmed) {
      CUgraph rebuilt = nullptr;
      const bool ok = restore_next_graph(&rebuilt);
      {
        std::lock_guard<std::mutex> lock(g_mu);
        g_shim_streams.erase(stream);
        g_real_begin_streams.erase(stream);  // N5b: clear begin-dedupe token
      }
      g_shim_active.fetch_sub(1, std::memory_order_release);
      if (!ok || rebuilt == nullptr) {
        if (phGraph) *phGraph = nullptr;
        return CUDA_ERROR_UNKNOWN;  // rebuild failed (e.g. blind node)
      }
      if (phGraph) *phGraph = rebuilt;
      {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_restored;
      }
      return CUDA_SUCCESS;
    }
  }

  const CUresult rc = real_cuStreamEndCapture()(stream, phGraph);
  if (rc == CUDA_SUCCESS && phGraph != nullptr && *phGraph != nullptr &&
      g_mode() == Mode::kRecord) {
    record_captured_graph(*phGraph);
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_real_begin_streams.erase(stream);  // N5b: clear begin-dedupe token
  }
  return rc;
}

// ---------------------------------------------------------------------------
// N5b Task 1: RUNTIME capture-API interposers (the PyTorch / vLLM path).
// Mirror the driver hooks above; cudaGraph_t == CUgraph (CUgraph_st*), so the
// restore path hands the rebuilt CUgraph back as a cudaGraph_t. The CUgraph
// record dedupe (record_captured_graph) and the per-window begin dedupe
// (count_real_begin_once) keep each window's walk/count single even if
// libcudart routes the runtime call through the driver hook.
// ---------------------------------------------------------------------------

cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                   cudaStreamCaptureMode mode) {
  if (g_mode() == Mode::kRestore) {
    ensure_restore_loaded();
    bool shim = false;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      if (g_restore.next < g_restore.queue.size()) {
        shim = g_shim_streams.insert(stream).second;  // idempotent vs driver hook
      }
    }
    if (shim) {
      g_shim_active.fetch_add(1, std::memory_order_release);
      std::lock_guard<std::mutex> lock(g_mu);
      ++g_rt_captures;  // runtime-path window (authoritative for vLLM)
      return cudaSuccess;  // no real begin-capture
    }
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_fallthrough;  // queue empty → real capture for this window
  } else {
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_rt_captures;  // record-mode runtime capture observed
  }
  if (count_real_begin_once(stream)) {
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_real_begin_capture;
  }
  return real_cudaStreamBeginCapture()(stream, mode);
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* phGraph) {
  if (g_mode() == Mode::kRestore) {
    bool shimmed = false;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      shimmed = (g_shim_streams.count(stream) > 0);
    }
    if (shimmed) {
      CUgraph rebuilt = nullptr;
      const bool ok = restore_next_graph(&rebuilt);
      {
        std::lock_guard<std::mutex> lock(g_mu);
        g_shim_streams.erase(stream);
        g_real_begin_streams.erase(stream);
      }
      g_shim_active.fetch_sub(1, std::memory_order_release);
      if (!ok || rebuilt == nullptr) {
        if (phGraph) *phGraph = nullptr;
        return cudaErrorUnknown;
      }
      if (phGraph) *phGraph = reinterpret_cast<cudaGraph_t>(rebuilt);
      {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_restored;
      }
      return cudaSuccess;
    }
  }

  const cudaError_t rc = real_cudaStreamEndCapture()(stream, phGraph);
  if (rc == cudaSuccess && phGraph != nullptr && *phGraph != nullptr &&
      g_mode() == Mode::kRecord) {
    record_captured_graph(reinterpret_cast<CUgraph>(*phGraph));
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_real_begin_streams.erase(stream);
  }
  return rc;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                  cudaStreamCaptureStatus* status) {
  if (g_mode() == Mode::kRestore &&
      g_shim_active.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_shim_streams.count(stream) > 0) {
      if (status) *status = cudaStreamCaptureStatusActive;
      return cudaSuccess;
    }
  }
  return real_cudaStreamIsCapturing()(stream, status);
}

// PyTorch's capture path polls cudaStreamGetCaptureInfo; report ACTIVE for a
// shim-capturing stream so the window is consistent. graphId / captureInfo are
// left untouched (we do not synthesize a capture id); for a shimmed stream the
// only consumer that matters is the status poll. Signature is version-dependent
// (CUDA 13 added a const cudaGraphEdgeData** out-param).
#if CUDA_VERSION >= 13000
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream,
                                     cudaStreamCaptureStatus* captureStatus,
                                     unsigned long long* graphId,
                                     cudaGraph_t* graphOut,
                                     const cudaGraphNode_t** dependenciesOut,
                                     const cudaGraphEdgeData** edgeDataOut,
                                     size_t* numDependenciesOut) {
  if (g_mode() == Mode::kRestore &&
      g_shim_active.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_shim_streams.count(stream) > 0) {
      if (captureStatus) *captureStatus = cudaStreamCaptureStatusActive;
      return cudaSuccess;
    }
  }
  return real_cudaStreamGetCaptureInfo()(stream, captureStatus, graphId,
                                         graphOut, dependenciesOut,
                                         edgeDataOut, numDependenciesOut);
}
#else
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream,
                                     cudaStreamCaptureStatus* captureStatus,
                                     unsigned long long* graphId,
                                     cudaGraph_t* graphOut,
                                     const cudaGraphNode_t** dependenciesOut,
                                     size_t* numDependenciesOut) {
  if (g_mode() == Mode::kRestore &&
      g_shim_active.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_shim_streams.count(stream) > 0) {
      if (captureStatus) *captureStatus = cudaStreamCaptureStatusActive;
      return cudaSuccess;
    }
  }
  return real_cudaStreamGetCaptureInfo()(stream, captureStatus, graphId,
                                         graphOut, dependenciesOut,
                                         numDependenciesOut);
}
#endif

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
  if (const char* dbg = std::getenv("SNAPSHOT_RECORD_CUDA_IDENTITY_DBG"); dbg) {
    std::size_t modsz = 0, fatsz = 0;
    { std::lock_guard<std::mutex> lock(g_mu); modsz = g_module_identity.size(); fatsz = g_hostfun_table.size(); }
    std::fprintf(stderr,
        "[record-cuda] identity-dbg func=%p cuFuncGetName rc=%d dev_name=%s "
        "module_map=%zu fatbin_table=%zu\n",
        (void*)func, (int)rc, (rc==CUDA_SUCCESS && dev_name)?dev_name:"(null)",
        modsz, fatsz);
  }
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
    // N5b (vLLM): cuFuncGetName returned a valid device name but it is NOT
    // fatbin-registered. This is the common case for Triton/Inductor/NCCL
    // kernels, which load via cuModuleLoadData and bypass
    // __cudaRegisterFunction (and mostly cuModuleGetFunction too). Record the
    // name anyway as kind=2 (name-only); restore resolves it by scanning every
    // loaded module via cuModuleGetFunction(module, name).
    if (out_kind)        *out_kind        = 2;
    copy_name(dev_name);
    if (out_module_hash) *out_module_hash = 0ULL;
    return 1;
  }
#endif

  return 0;
}

// ---------------------------------------------------------------------------
// snapshot_record_cuda_region_base — N5b Task 3 export for the Python layer.
// Returns the fixed VMM base snapshot_redirect_cuda reserved (Δ=0), or 0 if the
// redirect .so is not active. The Python cg_meta_cuda layer resolves this
// symbol via ctypes to reconstruct entry.output device pointers at restore
// (region_base + recorded offset == recorded data_ptr under Δ=0). The redirect
// .so exports snapshot_redirect_region_base(); resolve it through RTLD_DEFAULT
// (the redirect is LD_PRELOAD'd ahead of this .so, so its export is visible).
// ---------------------------------------------------------------------------
__attribute__((visibility("default")))
std::uint64_t snapshot_record_cuda_region_base() {
  using Fn = std::uint64_t (*)();
  void* sym = dlsym(RTLD_DEFAULT, "snapshot_redirect_region_base");
  if (sym == nullptr) return 0ULL;
  return reinterpret_cast<Fn>(sym)();
}

}  // extern "C"
