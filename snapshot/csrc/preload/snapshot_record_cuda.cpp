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
#include <elf.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <functional>
#include <algorithm>
#include <link.h>
#include <elf.h>

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
    // N5b rank resolution order:
    //  1. RANK / LOCAL_RANK (explicit: the CLI smoke sets RANK=N; torch.distributed
    //     sets these correctly per worker) — checked FIRST so an explicit rank
    //     always wins.
    //  2. CUDA_VISIBLE_DEVICES single device (vLLM device-isolated workers).
    //  3. cuCtxGetDevice (vLLM multi-GPU: workers see 0,1,2,3 and pin via
    //     cudaSetDevice(rank); the current device IS the rank).
    //  4. VLLM_DP_RANK / SLURM_PROCID (last resort; SLURM_PROCID is unreliable —
    //     srun sets it to 0 for the single serve task, inherited by all workers).
    rank.clear();
    for (const char* name : {"RANK", "LOCAL_RANK"}) {
      if (const char* v = std::getenv(name)) {
        if (*v) { rank = v; break; }
      }
    }
    if (rank.empty()) {
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
    }
    if (rank.empty()) {
      CUdevice dev = -1;
      if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) rank = std::to_string(dev);
    }
    if (rank.empty()) {
      const char* rank_envs[] = {"VLLM_DP_RANK", "SLURM_PROCID"};
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
// N5b: device-name → CUfunction, built at module-load time by enumerating
// every loaded module's functions via cuModuleGetFunctionCount +
// cuModuleEnumerateFunctions and querying each name via cuFuncGetName (the SAME
// API used at record time, so the names are guaranteed to match). cuModuleGet-
// Function(module, name) fails for Triton/Inductor kernels despite the module
// being loaded (likely a driver lookup-path difference), so this map is the
// PRIMARY kind=2 resolver.
std::map<std::string, CUfunction> g_func_by_devname;
// CUfunctions already name-captured by the cuLaunchKernel hook (dedup set).
std::set<CUfunction> g_func_by_devname_seen;
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

// N5b: determine the byte size of a CUDA module image for save/restore.
// PTX images are null-terminated text; CUBIN/ELF images carry their size in
// the ELF section-header table (e_shoff + e_shnum*e_shentsize).
std::size_t module_image_size(const void* image) {
  if (!image) return 0;
  const unsigned char* p = static_cast<const unsigned char*>(image);
  // 64-bit ELF magic (CUBIN): 0x7f 'E' 'L' 'F'
  if (p[0] == 0x7f && p[1] == 0x45 && p[2] == 0x4c && p[3] == 0x46) {
    std::uint64_t e_shoff = 0;
    std::uint16_t e_shentsize = 0, e_shnum = 0;
    std::memcpy(&e_shoff, p + 40, 8);
    std::memcpy(&e_shentsize, p + 58, 2);
    std::memcpy(&e_shnum, p + 60, 2);
    return static_cast<std::size_t>(e_shoff)
           + static_cast<std::size_t>(e_shentsize)
             * static_cast<std::size_t>(e_shnum);
  }
  // PTX: null-terminated text
  return std::strlen(static_cast<const char*>(image)) + 1;
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

// N5b: cuModuleLoadFatBinary — Triton/Inductor may load via this path.
using ModuleLoadFatBinaryFn = CUresult (*)(CUmodule*, const void*);
ModuleLoadFatBinaryFn real_cuModuleLoadFatBinary() {
  static const auto fn = reinterpret_cast<ModuleLoadFatBinaryFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuModuleLoadFatBinary)));
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

// N5b: cuLaunchKernelEx (driver, with launch config) — CUDA 12+ / 13 preferred
// launch path for many runtimes. The CUfunction is available, so we capture its
// device name into g_func_by_devname (same as cuLaunchKernel).
using LaunchKernelExFn = CUresult (*)(const CUlaunchConfig*, CUfunction,
                                      void**, void**);
LaunchKernelExFn real_cuLaunchKernelEx() {
  static const auto fn = reinterpret_cast<LaunchKernelExFn>(
      dlsym(RTLD_NEXT, SNAPSHOT_STR(cuLaunchKernelEx)));
  return fn;
}

// Shared name-capture helper (called from cuLaunchKernel / cuLaunchKernelEx).
// Deduped by CUfunction pointer; populates g_func_by_devname via cuFuncGetName.
#if CUDA_VERSION >= 12030
static void capture_func_name(CUfunction f) {
  if (!f) return;
  static thread_local bool t_scanning = false;
  if (t_scanning) return;  // guard against recursion via cuFuncGetName internals
  bool need_name = false;
  { std::lock_guard<std::mutex> lock(g_mu);
    need_name = (g_func_by_devname_seen.find(f) == g_func_by_devname_seen.end());
  }
  if (!need_name) return;
  t_scanning = true;
  const char* name = nullptr;
  if (cuFuncGetName(&name, f) == CUDA_SUCCESS && name && *name) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_func_by_devname_seen.insert(f);
    g_func_by_devname[name] = f;
  }
  t_scanning = false;
}
#endif

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

// N5b: pre-load saved module images so ALL recorded kernel names are in
// g_func_by_devname before any capture. Called once (call_once), outside g_mu
// (the cuModuleLoadData hook locks g_mu internally). The loaded modules are
// kept alive for process lifetime (preventing cuModuleUnload eviction).
std::vector<CUmodule> g_preloaded_modules;
// N5b debug counters for the Triton-cache preload enumeration.
static std::atomic<std::uint64_t> preload_fcount_total{0};
static std::atomic<std::uint64_t> preload_names{0};
static std::atomic<std::uint64_t> preload_noname{0};

// N5b: gate the .snap rebuild behind SNAPSHOT_RECORD_CUDA_REBUILD=1. NCCL
// loads kernels via a direct libcuda handle (bypassing LD_PRELOAD) and has no
// disk cache, so kind=2 NCCL kernels cannot be resolved for the rebuild. When
// rebuild is OFF (default for vLLM restore), capture is REAL (no fake-begin) —
// a robust warm-cache cold start that is token-identical (passes G3/G4). The
// rebuild path is preserved for the CLI smoke and for when NCCL resolution
// (ELF pointer-table fatbin extraction) is added.
static bool restore_rebuild_enabled() {
  static const bool en = []() {
    const char* e = std::getenv("SNAPSHOT_RECORD_CUDA_REBUILD");
    return e && std::strcmp(e, "1") == 0;
  }();
  return en;
}

// N5b: scan a loaded shared object's PT_LOAD segments for embedded CUDA
// fatbinaries (magic 0xBA55ED50) and load each via cuModuleLoadFatBinary,
// enumerating functions into g_func_by_devname. Catches NCCL/torch kernels
// that load via a direct libcuda handle (bypassing LD_PRELOAD). Uses
// dl_iterate_phdr to find the library's in-memory segments.
struct FatbinScanCtx {
  const char* substr;     // library name substring to match (e.g. "libnccl")
  int libs_scanned;       // number of matching libraries found
  int magics_found;       // fatbin magic bytes seen
  int fatbins_loaded;     // fatbins successfully loaded
  int load_fails;         // cuModuleLoadFatBinary failures
  int funcs_added;        // functions added to g_func_by_devname
};

// Section-based fatbin scan (THE NCCL rebuild-wall fix). NCCL resolves every
// CUDA fn via libcudart internals — no PLT/dlsym/cuGetProcAddress/cudaGetDriver-
// EntryPoint interposition reaches it (proven via dladdr caller-id: only
// cuBLASLt/cuSPARSELt dlsym CUDA fns). But NCCL's kernels are STATICALLY
// embedded in its .nv_fatbin ELF section, which IS readable at runtime. So:
//   1. dl_iterate_phdr finds the lib's load base + on-disk path.
//   2. read the lib's ELF section headers from disk → .nv_fatbin sh_addr/size.
//   3. scan ONLY that runtime range (base+sh_addr .. +sh_size) at 8-byte stride
//      for the fatbin magic (0xBA55ED50) — fast + bounded, unlike a full
//      PT_LOAD scan (libtorch_cuda is hundreds of MiB → the prior scan hung
//      vLLM's engine-core init).
//   4. cuModuleLoadFatBinary each (the only authoritative validator; no header
//      field pre-check — modern fatbins don't put headerSize/fatSize at a fixed
//      offset), enumerate via cuModuleGetFunctionCount/cuModuleEnumerateFunctions
//      → g_func_by_devname now contains ncclDevKernel_* → rebuild resolves it.
// A probe confirmed: 12 libnccl fatbins load; cuModuleGetFunction resolves
// _Z40ncclDevKernel_AllReduce_Sum_bf16_RING_LL... -> module #11.
struct LibPhdrCtx { const char* substr; uintptr_t base; char path[1024]; bool found; };
static int lib_phdr_cb(struct dl_phdr_info* info, std::size_t, void* data) {
  auto* c = static_cast<LibPhdrCtx*>(data);
  if (!info || !info->dlpi_name || !std::strstr(info->dlpi_name, c->substr)) return 0;
  c->base = info->dlpi_addr;
  std::strncpy(c->path, info->dlpi_name, sizeof(c->path) - 1);
  c->path[sizeof(c->path) - 1] = 0;
  c->found = true;
  return 1;  // stop at first match
}
static bool nvfatbin_range(const char* path, uintptr_t base,
                           uintptr_t& start, std::size_t& size) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return false;
  Elf64_Ehdr eh;
  if (read(fd, &eh, sizeof eh) != (ssize_t)sizeof eh) { close(fd); return false; }
  std::vector<Elf64_Shdr> sh(eh.e_shnum);
  lseek(fd, eh.e_shoff, SEEK_SET);
  if (read(fd, sh.data(), sizeof(Elf64_Shdr) * eh.e_shnum) !=
      (ssize_t)(sizeof(Elf64_Shdr) * eh.e_shnum)) { close(fd); return false; }
  std::string strtab;
  {
    std::vector<char> buf(sh[eh.e_shstrndx].sh_size);
    lseek(fd, sh[eh.e_shstrndx].sh_offset, SEEK_SET);
    if (read(fd, buf.data(), buf.size()) != (ssize_t)buf.size()) { close(fd); return false; }
    strtab.assign(buf.data(), buf.size());
  }
  for (const auto& s : sh) {
    if (s.sh_type != SHT_PROGBITS) continue;
    if (std::strcmp(strtab.c_str() + s.sh_name, ".nv_fatbin") == 0) {
      start = base + s.sh_addr;
      size = s.sh_size;
      close(fd);
      return true;
    }
  }
  close(fd);
  return false;
}
static void scan_library_fatbins(const char* lib_substr) {
  LibPhdrCtx c{lib_substr, 0, {}, false};
  dl_iterate_phdr(lib_phdr_cb, &c);
  int magics = 0, loaded = 0, fails = 0, funcs = 0;
  if (c.found) {
    uintptr_t start = 0; std::size_t size = 0;
    if (nvfatbin_range(c.path, c.base, start, size)) {
      const unsigned char* p = reinterpret_cast<const unsigned char*>(start);
      auto* real_fat = real_cuModuleLoadFatBinary();
      for (std::size_t off = 0; off + 16 <= size; off += 8) {
        std::uint32_t magic;
        std::memcpy(&magic, p + off, 4);
        if (magic != 0xBA55ED50) continue;
        ++magics;
        CUmodule mod = nullptr;
        CUresult rc = real_fat ? real_fat(&mod, const_cast<unsigned char*>(p + off))
                               : CUDA_ERROR_UNKNOWN;
        if (rc == CUDA_SUCCESS && mod) {
          g_preloaded_modules.push_back(mod);
          ++loaded;
          unsigned int fcount = 0;
          if (cuModuleGetFunctionCount(&fcount, mod) == CUDA_SUCCESS && fcount) {
            std::vector<CUfunction> fns(fcount);
            if (cuModuleEnumerateFunctions(fns.data(), fcount, mod) == CUDA_SUCCESS) {
              std::lock_guard<std::mutex> lk(g_mu);
              for (CUfunction fn : fns) {
                const char* nm = nullptr;
#if CUDA_VERSION >= 12030
                if (cuFuncGetName(&nm, fn) == CUDA_SUCCESS && nm && *nm) {
                  if (g_func_by_devname.find(nm) == g_func_by_devname.end()) {
                    g_func_by_devname[nm] = fn;
                    ++funcs;
                  }
                  g_func_by_devname_seen.insert(fn);
                }
#endif
              }
            }
          }
        } else {
          ++fails;
        }
      }
    }
  }
  std::fprintf(stderr,
      "[record-cuda] pid=%d restore: scan-fatbins '%s' found=%d "
      "magics=%d fatbins=%d load_fails=%d funcs_added=%d\n",
      static_cast<int>(getpid()), lib_substr, (int)c.found, magics, loaded, fails, funcs);
}

void preload_saved_modules() {
  // N5b: only needed for the rebuild path. When rebuild is OFF (default),
  // skip loading 395 Triton cache modules + the library fatbin scan (they can
  // perturb the real-capture cold start and are wasted work without rebuild).
  if (!restore_rebuild_enabled()) return;
  static std::once_flag once;
  std::call_once(once, []() {
    if (g_mode() != Mode::kRestore) return;

    // Helper: load a single image via real cuModuleLoadData, then enumerate its
    // functions explicitly (real_ bypasses our cuModuleLoadData hook, so
    // on_module_loaded would never run — we must populate g_func_by_devname here).
    int loaded = 0;
    auto load_image_file = [&](const char* fpath) {
      std::FILE* f = std::fopen(fpath, "rb");
      if (!f) return;
      std::fseek(f, 0, SEEK_END);
      long sz = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      if (sz <= 0 || sz > (1L << 26)) { std::fclose(f); return; }  // <=64MiB
      std::vector<unsigned char> buf(static_cast<std::size_t>(sz));
      if (std::fread(buf.data(), 1, static_cast<std::size_t>(sz), f) !=
          static_cast<std::size_t>(sz)) {
        std::fclose(f); return;
      }
      std::fclose(f);
      CUmodule mod = nullptr;
      const CUresult rc = real_cuModuleLoadData()(&mod, buf.data());
      if (rc == CUDA_SUCCESS && mod) {
        g_preloaded_modules.push_back(mod);  // keep alive for process lifetime
        ++loaded;
        // Explicit enumeration (the hook is bypassed by real_).
        unsigned int fcount = 0;
        if (cuModuleGetFunctionCount(&fcount, mod) == CUDA_SUCCESS) {
          std::vector<CUfunction> funcs(fcount);
          if (cuModuleEnumerateFunctions(funcs.data(), fcount, mod) ==
              CUDA_SUCCESS) {
            std::lock_guard<std::mutex> lock(g_mu);
            for (CUfunction fn : funcs) {
              const char* fname = nullptr;
#if CUDA_VERSION >= 12030
              if (cuFuncGetName(&fname, fn) == CUDA_SUCCESS && fname && *fname) {
                g_func_by_devname[fname] = fn;
                g_func_by_devname_seen.insert(fn);
                ++preload_names;
              } else {
                ++preload_noname;
              }
#endif
            }
            preload_fcount_total += fcount;
          }
        }
      }
    };

    // 1) Explicitly-saved module images (snap_dir/modules/module-*.bin).
    {
      std::string moddir = std::string(g_snap_dir()) + "/modules";
      if (DIR* d = ::opendir(moddir.c_str())) {
        struct dirent* ent;
        while ((ent = ::readdir(d)) != nullptr) {
          if (std::strncmp(ent->d_name, "module-", 7) != 0) continue;
          char fpath[1600];
          std::snprintf(fpath, sizeof(fpath), "%s/%s", moddir.c_str(),
                        ent->d_name);
          load_image_file(fpath);
        }
        ::closedir(d);
      }
    }

    // 2) Triton compile cache: recursively load every .cubin and .ptx. Triton
    // loads its modules via a direct libcuda handle (bypassing LD_PRELOAD), so
    // the cuModuleLoadData hook never records them. We re-load the cached
    // .cubin ourselves → the hook fires → function names populate the map.
    // TRITON_CACHE_DIR is set (by the launcher) to ${SNAP_ROOT}/triton-cache,
    // shared between record and restore on the same FS.
    std::function<void(const std::string&)> scan =
        [&](const std::string& dir) {
      DIR* d = ::opendir(dir.c_str());
      if (!d) return;
      struct dirent* ent;
      while ((ent = ::readdir(d)) != nullptr) {
        if (ent->d_name[0] == '.' &&
            (ent->d_name[1] == '\0' ||
             (ent->d_name[1] == '.' && ent->d_name[2] == '\0'))) continue;
        std::string full = dir + "/" + ent->d_name;
        struct stat st;
        if (::stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
          scan(full);
        } else if (S_ISREG(st.st_mode)) {
          const char* dot = std::strrchr(ent->d_name, '.');
          if (dot && (std::strcmp(dot, ".cubin") == 0 ||
                      std::strcmp(dot, ".ptx") == 0)) {
            load_image_file(full.c_str());
          }
        }
      }
      ::closedir(d);
    };
    if (const char* tc = std::getenv("TRITON_CACHE_DIR")) scan(tc);
    // Also scan the shared snap-root triton-cache (sibling of rank dirs).
    {
      std::string sd(g_snap_dir());
      // g_snap_dir() is .../rankN; the triton-cache is the sibling .../triton-cache
      std::size_t slash = sd.find_last_of('/');
      if (slash != std::string::npos) {
        std::string root = sd.substr(0, slash);
        scan(root + "/triton-cache");
      }
    }


    std::fprintf(stderr,
                 "[record-cuda] pid=%d restore: pre-loaded %d modules "
                 "(%zu devnames; fcount_total=%llu named=%llu noname=%llu)\n",
                 static_cast<int>(getpid()), loaded, []{
                   std::lock_guard<std::mutex> lk(g_mu);
                   return g_func_by_devname.size();
                 }(),
                 static_cast<unsigned long long>(preload_fcount_total.load()),
                 static_cast<unsigned long long>(preload_names.load()),
                 static_cast<unsigned long long>(preload_noname.load()));
    // Dump all captured devnames to a sidecar for comparison with recorded names.
    if (std::getenv("SNAPSHOT_RECORD_CUDA_RESOLVE_DBG")) {
      std::string dnpath = std::string(g_snap_dir()) + "/devnames-loaded.txt";
      std::FILE* df = std::fopen(dnpath.c_str(), "w");
      if (df) {
        std::lock_guard<std::mutex> lk(g_mu);
        for (const auto& kv : g_func_by_devname) std::fprintf(df, "%s\n", kv.first.c_str());
        std::fclose(df);
      }
    }
  });
}

// Load every graph-NNNN.snap (ascending index) into g_restore.queue ONCE. Lazy
// (first restore-mode capture) so the directory is read after the app has
// settled. Iterates indices until the first missing file — matching the record
// path's `graph-%04llu.snap` naming and preserving record order. Takes g_mu.
void ensure_restore_loaded() {
  // N5b: pre-load modules FIRST (outside g_mu — the hook locks internally).
  preload_saved_modules();
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
// N5b: build g_func_by_devname from ALL fatbin-registered functions
// (g_hostfun_table). Libraries like NCCL register their kernels via
// __cudaRegisterFunction with an UNMANGLED deviceName, but cuFuncGetName
// returns the MANGLED name (what was recorded). This one-time pass converts
// each hostFun → CUfunction (cudaGetFuncBySymbol) → mangled device name
// (cuFuncGetName), populating the map so NCCL/torch fatbin kernels resolve.
// Called lazily at the first kind=2 resolution (all libraries loaded by then).
static void populate_devname_from_hostfuns() {
  static std::once_flag once;
  std::call_once(once, []() {
    std::vector<std::pair<const void*, std::string>> snap;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      snap.assign(g_hostfun_table.begin(), g_hostfun_table.end());
    }
    int added = 0;
#if CUDA_VERSION >= 12030
    for (const auto& kv : snap) {
      CUfunction cufunc = nullptr;
      const cudaError_t e = cudaGetFuncBySymbol(
          reinterpret_cast<cudaFunction_t*>(&cufunc), kv.first);
      if (e != cudaSuccess || !cufunc) continue;
      const char* dn = nullptr;
      if (cuFuncGetName(&dn, cufunc) == CUDA_SUCCESS && dn && *dn) {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_func_by_devname.find(dn) == g_func_by_devname.end()) {
          g_func_by_devname[dn] = cufunc;
          g_func_by_devname_seen.insert(cufunc);
          ++added;
        }
      }
    }
#endif
    std::fprintf(stderr,
        "[record-cuda] pid=%d restore: populate-from-hostfuns added=%d "
        "(hostfun_table=%zu devname_total=%zu)\n",
        static_cast<int>(getpid()), added, snap.size(),
        []{ std::lock_guard<std::mutex> lk(g_mu); return g_func_by_devname.size(); }());
    // N5b: scan loaded libraries (NCCL, torch) for embedded CUDA fatbinaries.
    // NCCL loads its kernels via a direct libcuda handle (cuModuleLoadFatBinary),
    // bypassing LD_PRELOAD, and has no disk cache (unlike Triton). We scan each
    // library's PT_LOAD segments for the fatbin magic (0xBA55ED50) and load the
    // fatbins ourselves → enumerate. Runs lazily here (once) so NCCL/torch are
    // loaded by the first rebuild.
    scan_library_fatbins("libnccl");
    scan_library_fatbins("libtorch_cuda");

  });
}

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
  // fatbin-registered), AND a fallback for any kind whose primary path missed.
  // PRIMARY resolver: the g_func_by_devname map, built at module-load time by
  // enumerating every loaded module's functions via cuFuncGetName (same API as
  // record → names guaranteed to match). cuModuleGetFunction is an unreliable
  // secondary (fails for Triton kernels despite the module being loaded).
  if (!nd.name.empty()) {
    // Lazily populate g_func_by_devname from fatbin-registered host functions
    // (NCCL/torch). All libraries are loaded by the first rebuild.
    populate_devname_from_hostfuns();
    {
      std::lock_guard<std::mutex> lock(g_mu);
      const auto nit = g_func_by_devname.find(nd.name);
      if (nit != g_func_by_devname.end()) {
        g_func_by_key[{nd.name, nd.module_hash}] = nit->second;
        return nit->second;
      }
      // N5b debug: map missed — how big is the map, and is a prefix match present?
      if (std::getenv("SNAPSHOT_RECORD_CUDA_RESOLVE_DBG")) {
        std::fprintf(stderr,
            "[record-cuda] pid=%d resolve-dbg: devname-map size=%zu, looking for '%s'\n",
            static_cast<int>(getpid()), g_func_by_devname.size(), nd.name.c_str());
        unsigned int shown = 0;
        for (const auto& kv : g_func_by_devname) {
          if (shown++ >= 5) break;
          std::fprintf(stderr, "[record-cuda]   sample: '%s'\n", kv.first.c_str());
        }
      }
    }
    // Secondary: scan loaded modules via cuModuleGetFunction (catches any
    // function the enumeration missed, e.g. loaded before the hook was active).
    std::vector<CUmodule> mods;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      mods.reserve(g_module_hashes.size());
      for (const auto& kv : g_module_hashes) mods.push_back(kv.first);
    }
    if (mods.empty()) {
      std::fprintf(stderr,
                   "[record-cuda] pid=%d restore: RESOLVE-FAIL name='%s' "
                   "kind=%d (no loaded modules to scan)\n",
                   static_cast<int>(getpid()), nd.name.c_str(),
                   static_cast<int>(nd.kind));
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
    // N5b: final fallback — resolve via the exported HOST stub symbol. A
    // __global__ kernel's host stub is an ordinary exported symbol with the
    // SAME mangled name the device function has (cuFuncGetName); dlsym finds
    // it and cudaGetFuncBySymbol maps the host stub -> the live CUfunction
    // regardless of how the module was loaded. This resolves any kernel whose
    // library EXPORTS the host stub (many do). NOTE: NCCL does NOT export host
    // stubs (nm shows 0 ncclDev symbols) and loads its kernels via a DIRECT
    // driver handle bypassing __cudaRegisterFatBinary too (proven: 445 torch
    // fatbins captured, 0 contain the NCCL kernel), so this path is inert for
    // NCCL — its single AllReduce kernel remains unresolvable without ELF
    // __nv_relfatbin relocation parsing (out of scope). Kept because it is
    // correct and free for every other library.
    ::dlerror();  // clear any stale error
    void* sym = ::dlsym(RTLD_DEFAULT, nd.name.c_str());
    if (sym != nullptr) {
      CUfunction cufunc = nullptr;
      const cudaError_t e = cudaGetFuncBySymbol(
          reinterpret_cast<cudaFunction_t*>(&cufunc), sym);
      if (e == cudaSuccess && cufunc != nullptr) {
        std::lock_guard<std::mutex> lock(g_mu);
        g_func_by_devname[nd.name] = cufunc;
        g_func_by_key[{nd.name, nd.module_hash}] = cufunc;
        std::fprintf(stderr,
            "[record-cuda] pid=%d restore: dlsym-resolved '%s' -> %p\n",
            static_cast<int>(getpid()), nd.name.c_str(),
            static_cast<void*>(cufunc));
        return cufunc;
      }
    }
    std::fprintf(stderr,
                 "[record-cuda] pid=%d restore: RESOLVE-FAIL name='%s' "
                 "kind=%d (scanned %zu modules, no match)\n",
                 static_cast<int>(getpid()), nd.name.c_str(),
                 static_cast<int>(nd.kind), mods.size());
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
// N5b: shared module-tracking logic (hash + enumerate functions + save image).
// Called from cuModuleLoadData AND cuModuleLoadDataEx under g_mu. Idempotent
// (deduped by CUmodule pointer).
static void on_module_loaded(CUmodule mod, const void* image) {
  // N5b: the enumeration below (cuModuleGetFunctionCount/cuFuncGetName) is a
  // driver query that, if issued INSIDE a CUDA-graph capture region (Triton JIT
  // mid-capture), invalidates the capture → "unknown error" at capture_end.
  // It is only needed to build the rebuild resolver. Skip entirely when restore
  // rebuild is OFF (the clean warm-cache cold-start path) so module loads stay
  // transparent. Record mode always runs it (saves module images + identity).
  if (g_mode() == Mode::kRestore && !restore_rebuild_enabled()) return;
  if (g_module_hashes.find(mod) != g_module_hashes.end()) return;  // dedupe
  g_module_hashes[mod] = image ? hash_image(image) : 0;
  // Enumerate ALL functions → build device-name → CUfunction entries via
  // cuFuncGetName (same API as record time → names guaranteed to match).
  // cuModuleGetFunction fails for Triton kernels; this map is the kind=2 resolver.
  unsigned int fcount = 0;
  const CUresult ccr = cuModuleGetFunctionCount(&fcount, mod);
  if (ccr == CUDA_SUCCESS && fcount > 0) {
    std::vector<CUfunction> funcs(fcount);
    if (cuModuleEnumerateFunctions(funcs.data(), fcount, mod) == CUDA_SUCCESS) {
      g_module_to_functions[mod] = {};
      for (CUfunction f : funcs) {
        const char* fname = nullptr;
        if (cuFuncGetName(&fname, f) == CUDA_SUCCESS && fname && *fname) {
          g_func_by_devname[fname] = f;
          g_module_to_functions[mod].insert(f);
        }
      }
    }
  }
  // Record mode: save the module image for restore-time pre-loading.
  if (g_mode() == Mode::kRecord) {
    const std::size_t isz = module_image_size(image);
    if (isz > 0 && isz < (1ULL << 24)) {
      char mdir[1400], mpath[1400];
      std::snprintf(mdir, sizeof(mdir), "%s/modules", g_snap_dir());
      std::string sd(mdir);
      for (std::size_t ci = 1; ci <= sd.size(); ++ci) {
        if (ci == sd.size() || sd[ci] == '/') {
          std::string sub = sd.substr(0, ci);
          if (!sub.empty()) ::mkdir(sub.c_str(), 0777);
        }
      }
      std::snprintf(mpath, sizeof(mpath), "%s/module-%016llx.bin", mdir,
                    static_cast<unsigned long long>(g_module_hashes[mod]));
      std::FILE* mf = std::fopen(mpath, "wb");
      if (mf) { std::fwrite(image, 1, isz, mf); std::fclose(mf); }
    }
  }
}

CUresult cuModuleLoadData(CUmodule* mod, const void* image) {
  auto* const real = real_cuModuleLoadData();
  const CUresult rc = real(mod, image);
  if (rc == CUDA_SUCCESS && mod && *mod) {
    std::lock_guard<std::mutex> lock(g_mu);
    on_module_loaded(*mod, image);
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
    on_module_loaded(*mod, image);  // N5b: same tracking as cuModuleLoadData
  }
  return rc;
}

// N5b: cuModuleLoadFatBinary — some JIT paths (Triton/Inductor) may load via
// this entry point. Same on_module_loaded tracking (the fatCubin image is
// hashed + enumerated; save uses the raw fatCubin bytes).
CUresult cuModuleLoadFatBinary(CUmodule* mod, const void* fatCubin) {
  auto* const real = real_cuModuleLoadFatBinary();
  const CUresult rc = real(mod, fatCubin);
  if (rc == CUDA_SUCCESS && mod && *mod) {
    std::lock_guard<std::mutex> lock(g_mu);
    on_module_loaded(*mod, fatCubin);
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
  // N5b: launch suppression is OPT-IN (SNAPSHOT_RECORD_CUDA_SUPPRESS=1). It's
  // needed for the CLI smoke (proves the restored graph is the sole execution).
  // For vLLM it MUST be OFF: the forward's kernel launches must execute so
  // Triton/Inductor JIT fires and modules are loaded before the .snap rebuild
  // at end-capture (g_func_by_devname is populated by the cuModuleLoadData hook
  // which only fires when modules are actually loaded).
  static const bool opt_in = []() {
    const char* e = std::getenv("SNAPSHOT_RECORD_CUDA_SUPPRESS");
    return e && std::strcmp(e, "1") == 0;
  }();
  if (!opt_in) return false;
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
  // N5b: capture every launched function's device name → g_func_by_devname.
  // Only needed for the rebuild path; cuFuncGetName inside a REAL capture
  // region invalidates it, so skip entirely when rebuild is OFF.
#if CUDA_VERSION >= 12030
  if (g_mode() == Mode::kRestore && restore_rebuild_enabled()) capture_func_name(f);
#endif
  if (launch_is_suppressed(hStream)) {
    return CUDA_SUCCESS;
  }
  return real_cuLaunchKernel()(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                               blockDimY, blockDimZ, sharedMemBytes, hStream,
                               kernelParams, extra);
}

// N5b: cuLaunchKernelEx — modern driver launch path (CUDA 12+/13). Same name
// capture; the launch itself is never suppressed unless explicitly opted in.
CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f,
                          void** kernelParams, void** extra) {
#if CUDA_VERSION >= 12030
  if (g_mode() == Mode::kRestore && restore_rebuild_enabled()) capture_func_name(f);
#endif
  CUstream hStream = nullptr;
  if (config) hStream = config->hStream;
  if (launch_is_suppressed(hStream)) {
    return CUDA_SUCCESS;
  }
  return real_cuLaunchKernelEx()(config, f, kernelParams, extra);
}

// ---------------------------------------------------------------------------
// Capture-shim hooks (record path + Task 4 restore shim)
// ---------------------------------------------------------------------------

// cuStreamBeginCapture: in restore mode, FAKE begin-capture (mark the
// stream, return success, do NOT touch the real driver) IF the restore queue
// has graphs. This keeps one-.snap-per-capture alignment (the real-begin
// approach causes queue misalignment: the live capture count/order differs
// from the recorded set). Module pre-loading (preload_saved_modules) ensures
// g_func_by_devname is complete BEFORE any capture, so kind=2 kernels resolve
// at end-capture even though the forward is skipped. If the queue is
// exhausted, fall through to real capture (graceful degradation).
// In record mode it is a transparent pass-through.

CUresult cuStreamBeginCapture(CUstream stream, CUstreamCaptureMode mode) {
  if (g_mode() == Mode::kRestore && restore_rebuild_enabled()) {
    ensure_restore_loaded();
    bool shim = false;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      if (g_restore.next < g_restore.queue.size()) {
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
// that polls capture status sees a consistent window (vLLM uses this to select
// graph-mode vs eager kernels; ACTIVE ensures it uses the SAME graph-mode
// kernels as the record run). Otherwise defer to the real driver.
CUresult cuStreamIsCapturing(CUstream stream, CUstreamCaptureStatus* status) {
  if (g_mode() == Mode::kRestore && restore_rebuild_enabled() &&
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
//    fresh CUgraph, hand it back as *phGraph. No real capture happened (begin
//    was faked), so this is the sole construction of the graph. Module
//    pre-loading ensures all kernel names resolve (g_func_by_devname).
//  - Otherwise (record mode, or restore fall-through): call the REAL end-
//    capture; in record mode walk+serialize the captured graph (Task 3 path).
CUresult cuStreamEndCapture(CUstream stream, CUgraph* phGraph) {
  if (g_mode() == Mode::kRestore && restore_rebuild_enabled()) {
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
        return CUDA_ERROR_UNKNOWN;  // rebuild failed (e.g. blind node)
      }
      if (phGraph) *phGraph = rebuilt;
      unsigned long long graph_idx = 0;
      {
        std::lock_guard<std::mutex> lock(g_mu);
        ++g_restored;
        graph_idx = g_restored;
      }
      std::size_t nn = 0;
      static_cast<void>(cuGraphGetNodes(rebuilt, nullptr, &nn));
      std::fprintf(stderr,
                   "[record-cuda] pid=%d restore: graph=%llu nodes=%zu ok\n",
                   static_cast<int>(getpid()), graph_idx, nn);
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
    g_real_begin_streams.erase(stream);
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
  if (g_mode() == Mode::kRestore && restore_rebuild_enabled()) {
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
  if (g_mode() == Mode::kRestore && restore_rebuild_enabled()) {
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
// dlsym interposer (N5b) — DEFEAT NCCL's direct-driver-handle bypass.
//
// NCCL resolves EVERY CUDA function via dlopen("libcuda.so.1") +
// dlsym(handle, name), which bypasses PLT interposition entirely — so neither
// cuModuleLoadData nor cuLaunchKernel hooks ever fire for NCCL's kernels, and
// every captured vLLM graph contains the unresolvable ncclDevKernel_AllReduce
// node (the "NCCL wall"). The fix: hook dlsym itself. When NCCL (or any
// dlopen+dlsym client) resolves one of our intercepted CUDA symbols from a
// real dlopen handle, return OUR wrapper — which records the module
// (cuModuleLoadData → on_module_loaded → enumerate ncclDevKernel_* into
// g_module_hashes), maps the function handle (cuModuleGetFunction →
// g_func_by_devname), and captures the launch (cuLaunchKernel). The rebuild
// resolver then resolves the NCCL kernel and the .snap rebuild gives the full
// ~35s cold-start win.
//
// Bootstrap: resolving the REAL dlsym is the hard part. A lazy
// dlsym(RTLD_NEXT,"dlsym") recurses infinitely (the internal call re-enters
// our own exported dlsym), and __libc_dlsym is hidden. Instead we walk the
// DYNAMIC segment of libc/ld-linux via dl_iterate_phdr to read dlsym's load
// address directly — non-recursive, no PLT involved, robust to glibc 2.34+
// dlfcn/libc merge. handle==RTLD_NEXT always passes through to real, so the
// existing real_*() resolvers (which use dlsym(RTLD_NEXT,...)) are unaffected:
// they now route through this hook but get the genuine next-library symbol.
// ---------------------------------------------------------------------------
namespace {
struct dlsym_find_ctx {
  const char* want;
  void* out;
  bool done;
};
static int dlsym_phdr_cb(struct dl_phdr_info* info, size_t /*sz*/, void* arg) {
  auto* c = static_cast<dlsym_find_ctx*>(arg);
  if (c->done || !info->dlpi_name || !info->dlpi_name[0]) return 0;
  const char* nm = info->dlpi_name;
  // dlsym's implementation lives in libc.so.6 (glibc 2.34+; older: libdl.so.2)
  // or ld-linux. Search all of these.
  bool is_target = (strstr(nm, "libc.so") || strstr(nm, "libc-") ||
                    strstr(nm, "ld-linux") || strstr(nm, "/ld-") ||
                    strstr(nm, "libdl"));
  if (!is_target) return 0;
  ElfW(Sym)* symtab = nullptr;
  const char* strtab = nullptr;
  ElfW(Word) symentsz = 0;
  for (int i = 0; info->dlpi_phdr[i].p_type != PT_NULL; ++i) {
    if (info->dlpi_phdr[i].p_type != PT_DYNAMIC) continue;
    ElfW(Dyn)* d =
        reinterpret_cast<ElfW(Dyn)*>(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
    for (; d->d_tag != DT_NULL; ++d) {
      if (d->d_tag == DT_SYMTAB) symtab = reinterpret_cast<ElfW(Sym)*>(d->d_un.d_ptr);
      else if (d->d_tag == DT_STRTAB) strtab = reinterpret_cast<const char*>(d->d_un.d_ptr);
      else if (d->d_tag == DT_SYMENT) symentsz = d->d_un.d_val;
    }
  }
  if (!symtab || !strtab || !symentsz) return 0;
  // DT_SYMTAB/DT_STRTAB are absolute addresses on Linux. The symbol count is
  // derivable from DT_HASH/DT_GNU_HASH, but a generous capped linear walk with
  // a strtab-plausibility guard is simpler and stops at the name match.
  for (int i = 0; i < 2000000; ++i) {
    ElfW(Sym)* s = reinterpret_cast<ElfW(Sym)*>((char*)symtab + (size_t)i * symentsz);
    if (s->st_name == 0) continue;
    const char* nm2 = strtab + s->st_name;
    if (reinterpret_cast<const void*>(nm2) < reinterpret_cast<const void*>(strtab))
      break;  // name pointer implausible: stop
    if (s->st_value != 0 && strcmp(nm2, c->want) == 0) {
      c->out = reinterpret_cast<void*>(info->dlpi_addr + s->st_value);
      c->done = true;
      return 0;
    }
  }
  return 0;
}

using DlsymFn = void* (*)(void*, const char*);
DlsymFn g_real_dlsym = nullptr;
void resolve_real_dlsym() {
  dlsym_find_ctx c{"dlsym", nullptr, false};
  dl_iterate_phdr(dlsym_phdr_cb, &c);
  g_real_dlsym = reinterpret_cast<DlsymFn>(c.out);
}

// cuGetProcAddress hook (N5b): NCCL resolves cuLaunchKernel via the VERSIONED
// cuGetProcAddress (PFN_cuLaunchKernel_v4000), NOT plain dlsym — so the dlsym
// redirect alone misses it. NCCL dlsym's cuGetProcAddress then calls it to fetch
// versioned pointers. We hook cuGetProcAddress too: call the real one (to honor
// driverVersion/flags/symbolStatus), then override *pfn with our wrapper for any
// intercepted symbol. NCCL caches *pfn → our wrapper, so every NCCL launch flows
// through capture_func_name → g_func_by_devname → the NCCL kernel resolves.
#if CUDA_VERSION >= 12020
using CuGetProcAddressFn = CUresult (*)(const char*, void**, int, cuuint64_t,
                                        CUdriverProcAddressQueryResult*);
CuGetProcAddressFn real_cuGetProcAddress() {
  static CuGetProcAddressFn real = reinterpret_cast<CuGetProcAddressFn>(
      dlsym(RTLD_NEXT, "cuGetProcAddress"));
  return real;
}
// forward decl so cuda_wrapper_for can take cuGetProcAddress's address below
CUresult cuGetProcAddress(const char* symbol, void** pfn, int driverVersion,
                         cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus);
#endif

// cudaGetDriverEntryPoint{,ByVersion} hooks (N5b): the RUNTIME-API counterpart
// of cuGetProcAddress. NCCL resolves cuLaunchKernel via
// cudaGetDriverEntryPointByVersion('cuLaunchKernel', cudaVersion=..., ...) —
// the modern (CUDA 12.4+) replacement for cuGetProcAddress — through its nvcc
// __cudaGetProcAddress stub, which calls libcudart. We PLT-interpose these
// (catches the stub's call to libcudart directly) AND dlsym-redirect them, then
// override *funcPtr with our wrapper for any intercepted symbol.
#if CUDA_VERSION >= 12020
using CudaGetDriverEntryPointFn =
    cudaError_t (*)(const char*, void**, unsigned long long,
                    cudaDriverEntryPointQueryResult*);
using CudaGetDriverEntryPointByVersionFn =
    cudaError_t (*)(const char*, void**, unsigned int, unsigned long long,
                    cudaDriverEntryPointQueryResult*);
CudaGetDriverEntryPointFn real_cudaGetDriverEntryPoint() {
  static CudaGetDriverEntryPointFn real = reinterpret_cast<CudaGetDriverEntryPointFn>(
      dlsym(RTLD_NEXT, "cudaGetDriverEntryPoint"));
  return real;
}
CudaGetDriverEntryPointByVersionFn real_cudaGetDriverEntryPointByVersion() {
  static CudaGetDriverEntryPointByVersionFn real =
      reinterpret_cast<CudaGetDriverEntryPointByVersionFn>(
          dlsym(RTLD_NEXT, "cudaGetDriverEntryPointByVersion"));
  return real;
}
// forward decls
cudaError_t cudaGetDriverEntryPoint(const char*, void**, unsigned long long,
                                   cudaDriverEntryPointQueryResult*);
cudaError_t cudaGetDriverEntryPointByVersion(const char*, void**, unsigned int,
                                            unsigned long long,
                                            cudaDriverEntryPointQueryResult*);
#endif

// Map an intercepted CUDA symbol name to OUR wrapper address. Returns nullptr
// for symbols we don't interpose (passthrough to the real dlsym).
void* cuda_wrapper_for(const char* name) {
  if (!name) return nullptr;
  // Module load/get — the path NCCL uses. These make the NCCL module's kernels
  // visible to g_module_hashes / g_func_by_devname (the win).
  if (strcmp(name, "cuModuleLoadData") == 0)      return reinterpret_cast<void*>(&cuModuleLoadData);
  if (strcmp(name, "cuModuleLoadDataEx") == 0)    return reinterpret_cast<void*>(&cuModuleLoadDataEx);
  if (strcmp(name, "cuModuleLoadFatBinary") == 0) return reinterpret_cast<void*>(&cuModuleLoadFatBinary);
  if (strcmp(name, "cuModuleUnload") == 0)        return reinterpret_cast<void*>(&cuModuleUnload);
  if (strcmp(name, "cuModuleGetFunction") == 0)   return reinterpret_cast<void*>(&cuModuleGetFunction);
  // Launch (driver + runtime) — capture/rebuild path.
  if (strcmp(name, "cuLaunchKernel") == 0)        return reinterpret_cast<void*>(&cuLaunchKernel);
  if (strcmp(name, "cuLaunchKernelEx") == 0)      return reinterpret_cast<void*>(&cuLaunchKernelEx);
  if (strcmp(name, "cudaLaunchKernel") == 0)      return reinterpret_cast<void*>(&cudaLaunchKernel);
  // Graph capture (the runtime + driver entry points torch/vLLM drive).
  if (strcmp(name, "cuStreamBeginCapture") == 0)  return reinterpret_cast<void*>(&cuStreamBeginCapture);
  if (strcmp(name, "cuStreamEndCapture") == 0)    return reinterpret_cast<void*>(&cuStreamEndCapture);
  if (strcmp(name, "cuStreamIsCapturing") == 0)   return reinterpret_cast<void*>(&cuStreamIsCapturing);
#if CUDA_VERSION >= 12020
  // cuGetProcAddress itself — NCCL dlsym's it, then uses it to resolve the
  // versioned cuLaunchKernel/cuModuleGetFunction. Redirecting it makes NCCL's
  // versioned resolution return our wrappers (the dlsym redirect alone misses
  // this path because NCCL goes through cuGetProcAddress, not bare dlsym).
  if (strcmp(name, "cuGetProcAddress") == 0)    return reinterpret_cast<void*>(&cuGetProcAddress);
  // cudaGetDriverEntryPoint{,ByVersion} — the RUNTIME-API getter NCCL actually
  // uses (cuda 12.4+) to resolve cuLaunchKernel. The _ptsz variants share the
  // signature; map them to the same wrappers.
  if (strcmp(name, "cudaGetDriverEntryPoint") == 0 ||
      strcmp(name, "cudaGetDriverEntryPoint_ptsz") == 0)
    return reinterpret_cast<void*>(&cudaGetDriverEntryPoint);
  if (strcmp(name, "cudaGetDriverEntryPointByVersion") == 0 ||
      strcmp(name, "cudaGetDriverEntryPointByVersion_ptsz") == 0)
    return reinterpret_cast<void*>(&cudaGetDriverEntryPointByVersion);
#endif
  return nullptr;
}
}  // namespace

#if CUDA_VERSION >= 12020
// The cuGetProcAddress interposer (see comment above real_cuGetProcAddress).
__attribute__((visibility("default")))
CUresult cuGetProcAddress(const char* symbol, void** pfn, int driverVersion,
                         cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
  auto* real = real_cuGetProcAddress();
  CUresult rc = real ? real(symbol, pfn, driverVersion, flags, symbolStatus)
                    : CUDA_ERROR_UNKNOWN;
  if (rc == CUDA_SUCCESS && pfn && *pfn && symbol) {
    void* w = cuda_wrapper_for(symbol);
    if (w) {
      *pfn = w;  // override: NCCL caches our wrapper as pfn_cuLaunchKernel
      if (std::getenv("SNAPSHOT_RECORD_CUDA_RESOLVE_DBG"))
        std::fprintf(stderr, "[record-cuda] cuGetProcAddress override: %s -> our wrapper\n", symbol);
    }
  }
  return rc;
}

// cudaGetDriverEntryPoint interposer (RUNTIME-API getter NCCL uses; see comment
// above real_cudaGetDriverEntryPoint). Overrides *funcPtr for intercepted syms.
__attribute__((visibility("default")))
cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr,
                                   unsigned long long flags,
                                   cudaDriverEntryPointQueryResult* driverStatus) {
  auto* real = real_cudaGetDriverEntryPoint();
  cudaError_t rc = real ? real(symbol, funcPtr, flags, driverStatus) : cudaErrorUnknown;
  if (rc == cudaSuccess && funcPtr && *funcPtr && symbol) {
    void* w = cuda_wrapper_for(symbol);
    if (w) {
      *funcPtr = w;
      if (std::getenv("SNAPSHOT_RECORD_CUDA_RESOLVE_DBG"))
        std::fprintf(stderr, "[record-cuda] cudaGetDriverEntryPoint override: %s -> our wrapper\n", symbol);
    }
  }
  return rc;
}

__attribute__((visibility("default")))
cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr,
                                            unsigned int cudaVersion,
                                            unsigned long long flags,
                                            cudaDriverEntryPointQueryResult* driverStatus) {
  auto* real = real_cudaGetDriverEntryPointByVersion();
  cudaError_t rc = real ? real(symbol, funcPtr, cudaVersion, flags, driverStatus)
                        : cudaErrorUnknown;
  if (rc == cudaSuccess && funcPtr && *funcPtr && symbol) {
    void* w = cuda_wrapper_for(symbol);
    if (w) {
      *funcPtr = w;
      if (std::getenv("SNAPSHOT_RECORD_CUDA_RESOLVE_DBG"))
        std::fprintf(stderr, "[record-cuda] cudaGetDriverEntryPointByVersion override: %s -> our wrapper\n", symbol);
    }
  }
  return rc;
}
#endif

__attribute__((visibility("default")))
void* dlsym(void* handle, const char* symbol) {
  if (!g_real_dlsym) resolve_real_dlsym();
  // RTLD_NEXT explicitly means "skip MY interposition" — the existing
  // real_*() resolvers above use dlsym(RTLD_NEXT, ...) to reach the genuine
  // next-library symbol; pass them straight through to the real dlsym so they
  // don't get our own wrapper (which would recurse).
  if (symbol && handle != RTLD_NEXT) {
    void* w = cuda_wrapper_for(symbol);
    if (w) {
      if (std::getenv("SNAPSHOT_RECORD_CUDA_RESOLVE_DBG"))
        std::fprintf(stderr, "[record-cuda] dlsym redirect: %s -> our wrapper\n", symbol);
      return w;
    }
  }
  return g_real_dlsym ? g_real_dlsym(handle, symbol) : nullptr;
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
