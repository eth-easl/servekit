// snapshot_record — an LD_PRELOAD shim that *records* a serializable graph IR
// by observing HIP module-load / get-function / launch / capture events inside
// an unmodified application, then writes each captured graph to a snapshot file
// via snapshot_core. This is the interceptor the design has always assumed: the
// recorder (not the workload) recovers kernel identity, because identity cannot
// be reverse-engineered from a captured node's opaque function pointer.
//
// Identity is *recorded*, not recovered, at the two clean interceptable sites:
//   * hipModuleLoadData / hipModuleLoad   -> module handle -> image bytes
//   * hipModuleGetFunction                -> function handle -> (module, entry)
// and correlated with hipModuleLaunchKernel inside a hipStreamBeginCapture /
// hipStreamEndCapture window. Pointer offsets within the param blob are NOT
// recorded (unknowable from the launch API); the proven M1 blind-scan
// relocation discovers them at restore time over the recorded region.
//
// Two modes, by what it observes (the shim itself is mode-agnostic):
//   * synthetic (M3a.2): preload alone over `snapshot capture`. The workload's
//     core allocator reserves/maps its VMM region via hipMemAddressReserve /
//     hipMemMap, which we also interpose to capture region_base + alloc events
//     so the snapshot is complete enough to restore bit-identically.
//   * real vLLM (M3a.3/4): LD_PRELOAD="libsnapshot_redirect.so libsnapshot_record.so".
//     Redirect serves hipMalloc (and owns the arena); record observes module /
//     capture / launch identity. Alloc events are immaterial here (no restore),
//     so the M3a.3 gate is nodes_without_identity == 0.
//
// Env knobs:
//   SNAPSHOT_RECORD_OUT_DIR=<dir>     write graph-<pid>-<idx>.snap here (required)
//   SNAPSHOT_RECORD_MAX_GRAPHS=<n>    stop recording after n graphs (default 4)
//   SNAPSHOT_RECORD_REGION_BASE=<hex> override region base (else from VMM reserve)
//   SNAPSHOT_RECORD_REGION_SIZE=<bytes> override region size
//   SNAPSHOT_RECORD_VERBOSE=1         per-graph log line

#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>  // M3a.6: ROCr layer — recover identity HIP launch hooks miss

#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "snapshot/allocator.hpp"
#include "snapshot/gpu_backend.hpp"
#include "snapshot/record.hpp"
#include "snapshot/snapshot_format.hpp"

namespace {

std::mutex g_mu;

// --- identity tables --------------------------------------------------------
struct ModRec {
  std::uint64_t hash = 0;
  std::vector<std::byte> image;  // empty if the ELF size could not be parsed
};
std::map<std::uintptr_t, ModRec> g_modules;  // hipModule_t (raw) -> record

// M3b: code objects loaded via the ROCr/HSA runtime directly (Triton JIT loads
// its HSACO here, NOT via hipModuleLoad* — confirmed on vLLM: no hipModuleLoad-
// Data FIRST probe, triton kernel names absent from every captured HIP module).
// hsa_code_object_reader_create_from_memory hands us the full HSACO bytes + size
// (unlike the HIP module API), so we copy them here and extract kernel names from
// the AMDGPU metadata note at FLUSH time. Keyed by reader handle for dedup.
struct HsaImage {
  std::uint64_t hash = 0;
  std::vector<std::byte> image;
};
std::map<std::uint64_t, HsaImage> g_hsa_images;  // reader.handle -> image
std::uint64_t g_hsa_capture_skipped = 0;  // ran past the cap/size budget

struct FuncRec {
  std::uint64_t id = 0;
  std::uint64_t module_hash = 0;
  std::string entry;
};
std::map<std::uintptr_t, FuncRec> g_functions;  // hipFunction_t (raw) -> record
std::uint64_t g_next_func_id = 1;
std::uint64_t g_get_function_calls = 0;  // total hipModuleGetFunction calls seen

// --- host-registered kernels (the <<<>>> / hipLaunchKernel path) ------------
// PyTorch/ATen kernels are NOT launched via hipModuleLaunchKernel; they are
// host-registered at static-init via __hipRegisterFatBinary / __hipRegisterFunction
// and launched by host function pointer (<<<>>>). A captured node's `func`
// (from hipGraphKernelNodeGetParams) IS that host pointer (confirmed on ROCm
// 6.3), so we correlate against a map keyed by hostFunc. The HSACO for these
// kernels lives in the fat binary the runtime registers — an AMDGPU ELF inside
// the __CLANG_OFFLOAD_BUNDLE__ at FatBinWrapper.binary (offset 8 of the data
// struct, magic "HPIF"). We scan for it the same way M1 relocates pointers:
// blindly, by ELF magic, then bound the copy with elf_code_object_size.
struct FatbinRec {
  std::vector<std::vector<std::byte>> elf_images;  // all AMDGPU HSACO ELFs found
  std::uint64_t primary_hash = 0;  // hash of the first ELF; functions reference it
};
std::map<std::uintptr_t, FatbinRec> g_fatbins;  // fatCubinHandle (void**) -> rec

struct HostFuncRec {
  std::uint64_t id = 0;
  std::uintptr_t fatbin_handle = 0;
  std::string device_name;
};
std::map<std::uintptr_t, HostFuncRec> g_host_functions;  // hostFunc ptr -> rec

// A captured kernel node's `func` (from hipGraphKernelNodeGetParams) for a
// host-registered (<<<>>>) kernel is an opaque device handle, NOT the host
// pointer __hipRegisterFunction was given, and NOT what hipGetFuncBySymbol
// returns (empirically: the canonical handle does not match the captured one).
// So host identity is recovered NOT from the handle but from the host function
// pointer recorded in issue order at hipLaunchKernel time (CaptureWindow::
// host_issue_ids), filled into the introspected host nodes by position.

// M3a.6 — ROCr/HSA identity map. The ~30-50% of captured nodes that launch via
// a path LD_PRELOAD can't hook (aiter/CK) still load their code objects through
// the HSA loader. At hsa_executable_freeze we iterate the executable's kernel
// symbols and record (KERNEL_OBJECT device address -> mangled name). HYPOTHESIS:
// a captured node's opaque func IS that HSA kernel_object (both are device
// addresses in the same range), so this map names the kernels HIP hooks miss.
std::map<std::uint64_t, std::string> g_hsa_kernels;  // kernel_object -> name
hsa_agent_t g_gpu_agent{};
bool g_have_gpu_agent = false;

// --- capture window state ---------------------------------------------------
// Per-hook launch counts snapshot (forward decl: PendingGraph embeds one).
struct CaptureHookCount {
  std::uint64_t mod = 0, host = 0, ext = 0, exc = 0, coop_host = 0,
      coop_mod = 0, addnode = 0;
};
struct CaptureWindow {
  std::vector<snapshot::RecordedLaunch> launches;
  // function_id of EVERY kernel launched on this stream during capture, in issue
  // order, resolved at launch time: module kernels (hipExtModuleLaunchKernel ->
  // g_functions) and host kernels (hipLaunchKernel -> g_host_functions), 0 if
  // unresolvable. Index-aligned 1:1 with the introspected graph nodes (both are
  // the stream's issue sequence), so each node's identity is issue[i] — needed
  // because a captured node's func is an opaque device handle.
  std::vector<std::uint64_t> issue_ids;
  // Raw kernarg bytes captured at LAUNCH time (tagged; see pack_kernel_args_*).
  // Index-aligned 1:1 with issue_ids and with the introspected kernel nodes —
  // filled into each node's param_blob at FLUSH so the rebuild can replay the
  // exact args hipGraphKernelNodeGetParams cannot recover (extra=NULL on ROCm).
  std::vector<std::vector<std::byte>> arg_blobs;
};
std::map<void*, CaptureWindow> g_windows;  // stream -> in-progress launches

// --- M3b deferred snapshots -------------------------------------------------
// A captured graph cannot be named inline (hipKernelNameRef deadlocks on the HIP
// capture lock), so EndCapture parks the graph's structure + per-node funcs here
// and the off-path namer writes it once every func is named. node_funcs is
// index-aligned with nodes.
struct PendingGraph {
  std::uint64_t index = 0;  // graph-<pid>-<index>.snap
  std::string arch;         // captured on the main thread (namer ctx is unsure)
  std::vector<snapshot::RecordedLaunch> nodes;
  std::vector<std::uintptr_t> node_funcs;
  // Launch-time kernarg blobs, index-aligned with nodes. Merged into
  // nodes[i].param_blob at FLUSH when counts match (see merge_arg_blobs).
  std::vector<std::vector<std::byte>> arg_blobs;
  // Per-hook launch counts snapshot (for the FLUSH diagnostic; see g_cap_hooks).
  CaptureHookCount hooks;
  std::uint64_t region_base = 0;
  std::uint64_t region_size = 0;
  std::uint64_t granularity = 4096;
  std::vector<snapshot::AllocEvent> alloc_events;
  bool written = false;
};
std::mutex g_pending_mu;
std::vector<PendingGraph> g_pending;

// Defined later (needs the module/name maps + serializer); the namer calls it.
void try_flush_pending();

// Fast-path gate: only touch the mutex when at least one capture is open and we
// have not yet hit the graph cap. hipModuleLaunchKernel is the HIP hot path; we
// must not serialize it when not recording.
std::atomic<int> g_active_captures{0};
std::atomic<std::uint64_t> g_graphs_written{0};

// record_issue outcome counters (cumulative; read deltas across graphs).
std::atomic<std::uint64_t> g_ri_matched{0};
std::atomic<std::uint64_t> g_ri_fallback{0};
std::atomic<std::uint64_t> g_ri_dropped{0};

// Per-hook launch counters, incremented ONLY while g_active_captures > 0 (i.e.
// during an open capture window). Read at FLUSH to see exactly which launch API
// the captured nodes flowed through — diagnoses why only some nodes get args.
// 'args' variants count only when arg capture was actually attempted.
struct CaptureHookStats {
  std::atomic<std::uint64_t> mod{0};          // hipModuleLaunchKernel
  std::atomic<std::uint64_t> host{0};         // hipLaunchKernel
  std::atomic<std::uint64_t> ext{0};          // hipExtModuleLaunchKernel
  std::atomic<std::uint64_t> exc{0};          // hipLaunchKernelExC
  std::atomic<std::uint64_t> coop_host{0};    // hipLaunchCooperativeKernel
  std::atomic<std::uint64_t> coop_mod{0};     // hipModuleLaunchCooperativeKernel
  std::atomic<std::uint64_t> addnode{0};     // hipGraphAddKernelNode (capture)
};
CaptureHookStats g_cap_hooks;

// --- deterministic-memory region (for the snapshot header + synthetic allocs) -
std::uint64_t g_region_base = 0;
std::uint64_t g_region_size = 0;
std::uint64_t g_granularity = 4096;
std::vector<snapshot::AllocEvent> g_alloc_events;

bool verbose() {
  static const bool v = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_VERBOSE");
    return e != nullptr && e[0] != '0';
  }();
  return v;
}

std::uint64_t max_graphs() {
  static const std::uint64_t n = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_MAX_GRAPHS");
    return e != nullptr && e[0] != '\0' ? std::strtoull(e, nullptr, 10) : 4ULL;
  }();
  return n == 0 ? 1 : n;
}

std::string out_dir() {
  static const std::string d = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_OUT_DIR");
    return e != nullptr ? std::string(e) : std::string();
  }();
  return d;
}

// M3b: when to drain the off-path namer. hipKernelNameRef SEGFAULTS if a stream
// capture starts while it runs (not merely blocks — confirmed job 509282), and
// vLLM fires captures back-to-back through init, so draining "whenever the lock
// looks free" has a check-then-call race that crashes the engine. Two modes:
//   * auto (default): drain as soon as g_active_captures == 0. Safe for our own
//     single-capture-then-exit synthetic workload (no concurrent capture).
//   * idle: drain ONLY once the recipe signals vLLM is at /health (fully past
//     the capture phase) by creating a sentinel file. Real vLLM uses this —
//     names resolve while the model serves, never mid-capture.
bool drain_idle_mode() {
  static const bool idle = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_DRAIN_MODE");
    return e != nullptr && e[0] != '\0' &&
           std::strstr(e, "idle") != nullptr;
  }();
  return idle;
}

const char* drain_sentinel_path() {
  static const char* const p = std::getenv("SNAPSHOT_RECORD_DRAIN_SENTINEL");
  return p;
}

bool drain_sentinel_exists() {
  const char* p = drain_sentinel_path();
  return p != nullptr && p[0] != '\0' && ::access(p, F_OK) == 0;
}

// M3b: hard cap on live hipKernelNameRef calls. The 2nd distinct host-kernel
// handle segfaults the call (unrecoverably — PyTorch's signal handler wins), so
// the safe budget on this ROCm stack is 1. Override with the env for experiments
// once a fix lands; 0 disables the live query entirely (module-map only).
std::atomic<int> g_nameref_calls{0};
int nameref_limit() {
  static const int n = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_NAMEREF_LIMIT");
    if (e == nullptr || e[0] == '\0') return 1;
    long v = std::strtol(e, nullptr, 10);
    return v < 0 ? 0 : static_cast<int>(v);
  }();
  return n;
}

std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* e = std::getenv(name);
  if (e == nullptr || e[0] == '\0') {
    return fallback;
  }
  if (std::strstr(name, "_BASE") != nullptr) {
    return std::strtoull(e, nullptr, 16);  // hex for addresses
  }
  return std::strtoull(e, nullptr, 10);
}

template <typename Fn>
Fn resolve(const char* name) {
  return reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
}

// Arch string cached on first use; must match HipBackend::arch() (gcnArchName).
std::string query_arch() {
  // Retry until non-empty: __hipRegisterFatBinary (which needs the gfx token
  // to pick the right per-arch bundle entry) can fire at static-init, before the
  // runtime has a live context; caching that empty result forever would disable
  // arch filtering. A benign race on `arch` is fine (same value either way).
  static std::string arch;
  if (!arch.empty()) {
    return arch;
  }
  int device = 0;
  hipDeviceProp_t props{};
  if (hipGetDevice(&device) == hipSuccess &&
      hipGetDeviceProperties(&props, device) == hipSuccess) {
    arch = props.gcnArchName;
  }
  return arch;
}

// The base ISA token (e.g. "gfx942") from gcnArchName ("gfx942:sramecc+:xnack-").
// Used to pick the right per-arch entry out of a multi-target clang offload
// bundle so a captured fatbin image matches the device it must reload on.
std::string device_gfx_token() {
  const std::string a = query_arch();
  const std::size_t c = a.find(':');
  return c == std::string::npos ? a : a.substr(0, c);
}

// ---- M3a.6 HSA identity helpers --------------------------------------------
// HSA functions we CALL are resolved via dlsym(RTLD_DEFAULT) (libhsa-runtime64
// is already loaded by HIP), so the recorder needs no extra link dependency.

hsa_status_t record_find_gpu_agent_cb(hsa_agent_t agent, void* /*data*/) {
  using FnInfo = hsa_status_t (*)(hsa_agent_t, hsa_agent_info_t, void*);
  static auto info =
      reinterpret_cast<FnInfo>(dlsym(RTLD_DEFAULT, "hsa_agent_get_info"));
  if (info == nullptr) {
    return HSA_STATUS_SUCCESS;
  }
  hsa_device_type_t type{};
  if (info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS &&
      type == HSA_DEVICE_TYPE_GPU) {
    g_gpu_agent = agent;
    g_have_gpu_agent = true;
    return HSA_STATUS_INFO_BREAK;  // first GPU agent is enough (TP=1)
  }
  return HSA_STATUS_SUCCESS;
}

void ensure_gpu_agent() {
  if (g_have_gpu_agent) {
    return;
  }
  using FnIter = hsa_status_t (*)(hsa_status_t (*)(hsa_agent_t, void*), void*);
  static auto iter =
      reinterpret_cast<FnIter>(dlsym(RTLD_DEFAULT, "hsa_iterate_agents"));
  if (iter != nullptr) {
    iter(record_find_gpu_agent_cb, nullptr);
  }
}

hsa_status_t record_hsa_symbol_cb(hsa_executable_t /*exe*/, hsa_agent_t /*agent*/,
                                  hsa_executable_symbol_t sym, void* /*data*/) {
  using FnSymInfo = hsa_status_t (*)(hsa_executable_symbol_t,
                                     hsa_executable_symbol_info_t, void*);
  static auto si = reinterpret_cast<FnSymInfo>(
      dlsym(RTLD_DEFAULT, "hsa_executable_symbol_get_info"));
  if (si == nullptr) {
    return HSA_STATUS_SUCCESS;
  }
  hsa_symbol_kind_t kind{};
  if (si(sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) != HSA_STATUS_SUCCESS ||
      kind != HSA_SYMBOL_KIND_KERNEL) {
    return HSA_STATUS_SUCCESS;
  }
  std::uint64_t ko = 0;
  si(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ko);
  std::uint32_t namelen = 0;
  si(sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &namelen);
  std::string name;
  if (namelen > 0 && namelen < (1u << 16)) {
    name.resize(namelen);
    si(sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
  }
  if (ko != 0) {
    const std::lock_guard<std::mutex> lock(g_mu);
    g_hsa_kernels.emplace(ko, std::move(name));
  }
  return HSA_STATUS_SUCCESS;
}

// Iterate one frozen executable's kernel symbols into the kernel_object map.
void record_executable_kernels(hsa_executable_t exe) {
  ensure_gpu_agent();
  if (!g_have_gpu_agent) {
    return;
  }
  using FnIterSyms = hsa_status_t (*)(
      hsa_executable_t, hsa_agent_t,
      hsa_status_t (*)(hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t,
                       void*),
      void*);
  static auto iters = reinterpret_cast<FnIterSyms>(
      dlsym(RTLD_DEFAULT, "hsa_executable_iterate_agent_symbols"));
  if (iters != nullptr) {
    iters(exe, g_gpu_agent, record_hsa_symbol_cb, nullptr);
  }
}

// M3a.6b — the clean bridge: ask HIP for a kernel node's name directly.
// hipKernelNameRef(const hipFunction_t) -> const char* maps a captured node's
// func (a hipFunction_t = ihipModuleSymbol_t*) straight to its kernel name, no
// matter which launch path created it — a live-handle runtime query, so it works
// even for aiter handles we never saw created. (hipKernelGetName takes a
// different type, hipKernel_t, and returns nothing for our func — wrong API.)
// Resolved via dlsym; libamdhip64 exports hipKernelNameRef (hip_4.2).
//
// M3b caveat (the hard-won finding): hipKernelNameRef SEGFAULTS on SOME captured
// node handles (the 2nd distinct one we touch — reproducibly, on/off the main
// thread, with no capture open). So we guard every call with a SIGSEGV+
// siglongjmp wrapper: a handle that faults is skipped (recorded empty) instead of
// killing the engine. That yields a partial-coverage snapshot (the nameable
// nodes named, the crashy ones empty) + a real rebuild verdict — and isolates
// exactly which handles the API can't handle.
namespace {
sigjmp_buf g_name_jmp;
volatile sig_atomic_t g_name_guard_active = 0;
// The fatal signals a bad hipKernelNameRef handle can raise. PyTorch's own
// c10::SignalHandler installs for these too, so we save/restore each one and
// only TAKE the fault during a guarded name call (otherwise chain to default).
constexpr int kFatalSigs[] = {SIGSEGV, SIGBUS, SIGILL, SIGABRT, SIGFPE};
struct sigaction g_name_old_sa[5];
bool g_name_old_sa_saved[5] = {false, false, false, false, false};

void name_fault_handler(int sig) {
  // async-signal-safe logging so we can see whether the guard actually fires.
  const char p1[] = "[record] NAME-GUARD sig=";
  char b[4];
  b[0] = '0' + (sig / 10 ? sig / 10 : 0);
  b[1] = '0' + (sig % 10);
  b[2] = '\n';
  if (sig >= 10) {
    ssize_t r = ::write(2, p1, sizeof(p1) - 1);
    (void)r; ::write(2, b, 3);
  } else {
    b[0] = '0' + sig;
    ssize_t r = ::write(2, p1, sizeof(p1) - 1);
    (void)r; ::write(2, b, 2);
  }
  if (g_name_guard_active) {
    g_name_guard_active = 0;
    siglongjmp(g_name_jmp, sig);  // jump back into kernel_name_via_hip
  }
  // Not ours: restore default disposition and re-raise so the normal crash
  // diagnostics (PyTorch/Python faulthandler) still fire.
  struct sigaction dfl;
  dfl.sa_handler = SIG_DFL;
  sigemptyset(&dfl.sa_mask);
  dfl.sa_flags = 0;
  sigaction(sig, &dfl, nullptr);
  raise(sig);
}

void name_guard_begin() {
  struct sigaction sa;
  sa.sa_handler = name_fault_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_NODEFER;  // don't block the signal in the handler
  for (std::size_t i = 0; i < 5; ++i) {
    sigaction(kFatalSigs[i], &sa, g_name_old_sa_saved[i] ? nullptr : &g_name_old_sa[i]);
    g_name_old_sa_saved[i] = true;
  }
}

void name_guard_end() {
  for (std::size_t i = 0; i < 5; ++i) {
    if (g_name_old_sa_saved[i]) {
      sigaction(kFatalSigs[i], &g_name_old_sa[i], nullptr);
    }
  }
}
}  // namespace

// Returns the name (possibly empty); sets crashed=true if the call faulted.
std::string kernel_name_via_hip(void* func, bool* crashed) {
  if (crashed != nullptr) *crashed = false;
  using Fn = const char* (*)(void* /*hipFunction_t*/);
  static auto fn =
      reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hipKernelNameRef"));
  if (fn == nullptr || func == nullptr) {
    return std::string();
  }
  name_guard_begin();
  g_name_guard_active = 1;
  int jumped = sigsetjmp(g_name_jmp, 1);
  const char* name = nullptr;
  if (jumped == 0) {
    name = fn(func);  // may fault -> handler longjmps back here with jumped!=0
  }
  g_name_guard_active = 0;
  name_guard_end();

  if (jumped != 0) {
    if (crashed != nullptr) *crashed = true;
    return std::string();
  }
  if (name != nullptr && name[0] != '\0') {
    return std::string(name);
  }
  return std::string();
}

// M3b: how to name a device-handle node that the module/host maps missed.
//   fork (default): isolate each hipKernelNameRef in a child so the 2nd-handle
//     crash cannot kill the recorder (and may name everything, since each call
//     starts from inherited state). SNAPSHOT_RECORD_NAMEREF_FORK_TIMEOUT_MS.
//   cap  : legacy — call inline in the parent, SIGSEGV-guarded, capped at
//     SNAPSHOT_RECORD_NAMEREF_LIMIT (default 1). Kept for A/B comparison.
//   off  : no live query; the node stays unnamed (coverage from maps only).
enum class NameRefMode { kFork, kCap, kOff };
NameRefMode nameref_mode() {
  static const NameRefMode m = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_NAMEREF_MODE");
    if (e != nullptr && std::strcmp(e, "cap") == 0) return NameRefMode::kCap;
    if (e != nullptr && std::strcmp(e, "off") == 0) return NameRefMode::kOff;
    return NameRefMode::kFork;
  }();
  return m;
}
int nameref_fork_timeout_ms() {
  static const int n = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_NAMEREF_FORK_TIMEOUT_MS");
    if (e == nullptr || e[0] == '\0') return 8000;
    long v = std::strtol(e, nullptr, 10);
    return v < 100 ? 8000 : static_cast<int>(v);
  }();
  return n;
}

// M3b crash-isolation: resolve a kernel name via hipKernelNameRef in a FORKED
// child. The 2nd distinct device handle reliably segfaults the call (jobs
// 509282–509295), and PyTorch's c10::SignalHandler wins the SIGSEGV so the
// in-process siglongjmp guard can't recover it. In a child the same crash is
// harmless: the parent reads the name over a pipe (or notices the child died)
// and SIGKILLs/reaps it. fork() duplicates the process address space, so the
// hipFunction_t (a pointer into the HIP runtime's heap data) stays valid in the
// child and a metadata read needs no live GPU work. Two watchdogs: an alarm()
// in the child (in case hipKernelNameRef deadlocks on a runtime lock held at
// fork time) and a poll() timeout in the parent. Each call is isolated, so this
// also answers whether the crash is cumulative-state (independent children each
// succeed) or handle-specific (each child still faults).
std::string kernel_name_via_fork(void* func) {
  if (func == nullptr) {
    return std::string();
  }
  using Fn = const char* (*)(void* /*hipFunction_t*/);
  static auto fn =
      reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hipKernelNameRef"));
  if (fn == nullptr) {
    return std::string();
  }
  int fds[2];
  if (::pipe(fds) != 0) {
    return std::string();
  }
  pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    return std::string();
  }
  if (pid == 0) {
    // ---- CHILD: async-signal-safe-adjacent; one runtime call then _exit. ----
    ::close(fds[0]);
    ::alarm(5);  // watchdog: don't hang forever if the runtime deadlocks
    const char* name = fn(func);  // may fault here -> child dies, parent notices
    char ok = (name != nullptr && name[0] != '\0') ? 1 : 0;
    ssize_t w = ::write(fds[1], &ok, 1);
    (void)w;
    if (ok) {
      std::uint32_t len =
          static_cast<std::uint32_t>(::strnlen(name, 8192));
      w = ::write(fds[1], &len, 4);
      (void)w;
      if (len > 0) {
        std::size_t put = 0;
        while (put < len) {
          ssize_t r = ::write(fds[1], name + put, len - put);
          if (r <= 0) break;
          put += static_cast<std::size_t>(r);
        }
      }
    }
    ::close(fds[1]);
    ::_exit(0);
  }
  // ---- PARENT ----
  ::close(fds[1]);
  std::string result;
  struct pollfd pfd;
  pfd.fd = fds[0];
  pfd.events = POLLIN;
  pfd.revents = 0;
  const int timeout_ms = nameref_fork_timeout_ms();
  if (::poll(&pfd, 1, timeout_ms) > 0 && (pfd.revents & POLLIN)) {
    char ok = 0;
    if (::read(fds[0], &ok, 1) == 1 && ok) {
      std::uint32_t len = 0;
      if (::read(fds[0], &len, 4) == 4 && len > 0 && len <= 8192) {
        std::vector<char> buf(len, '\0');
        std::size_t got = 0;
        while (got < len) {
          ssize_t r = ::read(fds[0], buf.data() + got, len - got);
          if (r <= 0) break;
          got += static_cast<std::size_t>(r);
        }
        result.assign(buf.data(), got);
      }
    }
  }
  ::close(fds[0]);
  // Reap; SIGKILL if the child is still alive (crash mid-write, or watchdog).
  int status = 0;
  if (::waitpid(pid, &status, WNOHANG) == 0) {
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &status, 0);
  }
  return result;
}

// M3a.6b off-path name resolution. hipKernelNameRef hangs if called while any
// stream capture holds the HIP global lock, but the captured node funcs stay
// valid handles, so a background thread drains a queue and resolves each when
// the lock frees (between captures / during eager work). This is how a real
// recorder would name the aiter/CK nodes the launch path can't reach. Here it
// also serves as the de-risk: do those 0x39.. handles resolve to real names?
std::mutex g_namer_mu;
std::deque<std::uintptr_t> g_namer_queue;
std::map<std::uintptr_t, std::string> g_resolved_names;
std::atomic<bool> g_namer_started{false};

// M3b fix: do NOT call hipKernelNameRef while any stream capture holds the HIP
// global lock. The call blocks on that lock, and when it briefly acquires it
// between two captures it serializes against the next BeginCapture — which
// STRETCHES vLLM's capture phase and defers the very idle window names need to
// resolve (self-defeating: the namer prevented the idle it was waiting for, so
// earlier runs drained only ~2 funcs and never FLUSHed). Instead we back off
// while g_active_captures > 0 and drain in a tight loop once capture is done.
// In production this means names resolve while the model serves requests.
void namer_thread() {
  std::size_t logged = 0;
  bool backing_off = false;
  for (;;) {
    std::uintptr_t f = 0;
    {
      const std::lock_guard<std::mutex> lock(g_namer_mu);
      if (!g_namer_queue.empty()) {
        f = g_namer_queue.front();
        g_namer_queue.pop_front();
      }
    }
    if (f == 0) {
      backing_off = false;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    // IDLE mode: hold the queued funcs until the recipe confirms vLLM is past
    // the capture phase (sentinel touched at /health). Draining mid-init races
    // the next BeginCapture and segfaults hipKernelNameRef (job 509282). In
    // AUTO mode (our synthetic workload) this is skipped: there is exactly one
    // capture then the process exits, so no concurrent capture can occur.
    if (drain_idle_mode() && !drain_sentinel_exists()) {
      {
        const std::lock_guard<std::mutex> lock(g_namer_mu);
        g_namer_queue.push_front(f);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }
    if (g_active_captures.load(std::memory_order_relaxed) > 0) {
      // Capture in progress: re-queue at the front and wait for an idle window.
      std::size_t pending = 0;
      {
        const std::lock_guard<std::mutex> lock(g_namer_mu);
        g_namer_queue.push_front(f);
        pending = g_namer_queue.size();
      }
      if (!backing_off) {
        std::fprintf(stderr,
                     "[record] NAMER backoff: %zu funcs pending, capture active "
                     "(will drain at idle)\n",
                     pending);
        std::fflush(stderr);
        backing_off = true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    if (backing_off) {
      std::size_t pending = 0;
      {
        const std::lock_guard<std::mutex> lock(g_namer_mu);
        pending = g_namer_queue.size() + 1;  // +1 for the func we just popped
      }
      std::fprintf(stderr, "[record] NAMER idle: draining %zu pending funcs\n",
                   pending);
      std::fflush(stderr);
      backing_off = false;
    }
    // Capture lock free: resolves quickly. (A capture starting just after this
    // check makes the call block briefly until it ends — harmless.)
    bool crashed = false;
    const std::string name =
        kernel_name_via_hip(reinterpret_cast<void*>(f), &crashed);
    {
      const std::lock_guard<std::mutex> lock(g_namer_mu);
      g_resolved_names[f] = name;
    }
    if (crashed) {
      std::fprintf(stderr,
                   "[record] NAMER func=0x%llx CRASHED hipKernelNameRef (skipped)\n",
                   static_cast<unsigned long long>(f));
      std::fflush(stderr);
    }
    if (logged < 80) {
      std::fprintf(stderr, "[record] NAMER func=0x%llx -> '%.80s'\n",
                   static_cast<unsigned long long>(f), name.c_str());
      std::fflush(stderr);
      ++logged;
    }
    // A name just landed; flush any pending graph now fully named.
    try_flush_pending();
  }
}

void ensure_namer() {
  // In IDLE mode naming happens INLINE on the main thread (drain_naming_inline),
  // driven by the inference launch path. We deliberately do NOT spawn the
  // background namer here: hipKernelNameRef is not thread-safe off the main HIP
  // thread and segfaults on its 2nd call (jobs 509282/509286). AUTO mode (the
  // single-capture synthetic workload) still uses the background thread.
  if (drain_idle_mode()) {
    return;
  }
  bool expected = false;
  if (g_namer_started.compare_exchange_strong(expected, true)) {
    std::thread(namer_thread).detach();
  }
}

void enqueue_for_naming(std::uintptr_t f) {
  if (f == 0) {
    return;
  }
  const std::lock_guard<std::mutex> lock(g_namer_mu);
  if (g_resolved_names.find(f) == g_resolved_names.end()) {
    g_namer_queue.push_back(f);
  }
}

// M3b: main-thread name resolution. hipKernelNameRef is NOT thread-safe from a
// background thread (segfaults on its 2nd call — jobs 509282/509286), but it IS
// safe on the main HIP thread (the NAMEREF-TEST resolves names inline during
// init, repeatedly). So in IDLE mode we drain the naming queue INLINE from the
// inference launch path (hipLaunchKernel / hipExtModuleLaunchKernel): that runs
// on the main thread, past the capture phase, after the recipe touches the
// sentinel (post-/health) and sends a request to drive inference. We resolve
// ONE func per launch (see drain_naming_inline).

void drain_naming_inline() {
  if (!drain_idle_mode() || !drain_sentinel_exists()) {
    return;
  }
  // Never name while a capture is open (the sentinel implies post-/health, but
  // guard anyway — hipKernelNameRef + an open capture is the original hang/crash).
  if (g_active_captures.load(std::memory_order_relaxed) > 0) {
    return;
  }
  // Resolve ONE func per launch (spread across inference launches). try_flush_
  // pending FLUSHes once every func in a pending graph is resolved.
  std::uintptr_t f = 0;
  {
    const std::lock_guard<std::mutex> lock(g_namer_mu);
    while (!g_namer_queue.empty()) {
      std::uintptr_t cand = g_namer_queue.front();
      g_namer_queue.pop_front();
      if (g_resolved_names.find(cand) == g_resolved_names.end()) {
        f = cand;
        break;
      }
    }
    if (f == 0) {
      return;
    }
  }
  std::string name;
  std::string source;
  // (1) Module kernels: the entry name was already captured at
  // hipModuleGetFunction time (g_functions) — use it directly, no hipKernelNameRef.
  {
    const std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_functions.find(f);
    if (it != g_functions.end() && !it->second.entry.empty()) {
      name = it->second.entry;
      source = "module-map";
    }
  }
  // (2) Host-launched kernels (PyTorch/ATen <<<>>>): a captured node's func IS
  // the host function pointer __hipRegisterFunction took (observed: handles in
  // the host VA range like 0x14c…/0x149c…). So look it up in g_host_functions
  // (device_name) — free and safe; hipKernelNameRef is the WRONG tool here (it
  // expects a device hipFunction_t and faults on a host pointer).
  if (name.empty()) {
    const std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_host_functions.find(f);
    if (it != g_host_functions.end() && !it->second.device_name.empty()) {
      name = it->second.device_name;
      source = "host-map";
    }
  }
  // (3) Device-handle nodes (Triton JIT, aiter/CK via a path we don't hook):
  // only hipKernelNameRef names these from the live handle. On this ROCm stack
  // it SEGFAULTS on the 2nd distinct handle (jobs 509282..509295), so the
  // default is to isolate each query in a forked child (crash is confined and
  // the parent never dies). SNAPSHOT_RECORD_NAMEREF_MODE = fork|cap|off.
  if (name.empty()) {
    const NameRefMode mode = nameref_mode();
    if (mode == NameRefMode::kFork) {
      name = kernel_name_via_fork(reinterpret_cast<void*>(f));
      source = name.empty() ? "nameref:fork(crash/timeout)" : "nameref:fork";
    } else if (mode == NameRefMode::kCap) {
      const int limit = nameref_limit();
      const int seen =
          g_nameref_calls.fetch_add(1, std::memory_order_relaxed) + 1;
      if (seen <= limit) {
        bool crashed = false;
        name = kernel_name_via_hip(reinterpret_cast<void*>(f), &crashed);
        source = crashed ? "nameref:CRASHED" : "nameref";
        if (crashed) name.clear();
      } else {
        source = "nameref:SKIPPED(cap)";
      }
    } else {
      source = "nameref:OFF";
    }
  }
  {
    const std::lock_guard<std::mutex> lock(g_namer_mu);
    g_resolved_names[f] = name;  // empty for unknown/crashed -> still "resolved"
  }
  std::fprintf(stderr, "[record] NAMER(main) func=0x%llx -> '%.80s' [%s]\n",
               static_cast<unsigned long long>(f), name.c_str(), source.c_str());
  std::fflush(stderr);
  try_flush_pending();
}

// Reconcile introspected node identity with the issue-order identities recorded
// at launch time. When the counts match (every launch on the stream became one
// node), the lists are index-aligned, so node[i]'s identity is issue_ids[i] —
// authoritative for host nodes whose node.func is an opaque handle, and a
// cross-check for module nodes introspection already named. On a count mismatch
// (a launch API still unrecorded) we fall back to sequentially consuming the
// recorded identities for the unnamed nodes, and log so the residual localizes.
void reconcile_identity_by_position(
    std::vector<snapshot::RecordedLaunch>& nodes,
    const std::vector<std::uint64_t>& issue_ids) {
  std::size_t unmatched_before = 0;
  for (const auto& n : nodes) {
    if (n.function_id == 0) ++unmatched_before;
  }
  std::size_t crosscheck_ok = 0, crosscheck_bad = 0;
  if (nodes.size() == issue_ids.size()) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].function_id == 0) {
        nodes[i].function_id = issue_ids[i];
      } else if (issue_ids[i] != 0) {
        (nodes[i].function_id == issue_ids[i] ? crosscheck_ok : crosscheck_bad)++;
      }
    }
  } else {
    std::size_t next = 0;
    for (auto& n : nodes) {
      if (n.function_id != 0) continue;
      while (next < issue_ids.size() && issue_ids[next] == 0) ++next;
      if (next < issue_ids.size()) n.function_id = issue_ids[next++];
    }
  }
  if (verbose()) {
    std::size_t remaining = 0;
    for (const auto& n : nodes) {
      if (n.function_id == 0) ++remaining;
    }
    std::size_t issue_zero = 0;
    for (std::uint64_t id : issue_ids) {
      if (id == 0) ++issue_zero;
    }
    std::fprintf(stderr,
                 "[record] reconcile: nodes=%zu issue=%zu(zero=%zu) "
                 "unmatched_before=%zu remaining_unknown=%zu xcheck_ok=%zu "
                 "xcheck_bad=%zu | record_issue matched=%llu fallback=%llu "
                 "dropped=%llu%s\n",
                 nodes.size(), issue_ids.size(), issue_zero, unmatched_before,
                 remaining, crosscheck_ok, crosscheck_bad,
                 static_cast<unsigned long long>(g_ri_matched.load()),
                 static_cast<unsigned long long>(g_ri_fallback.load()),
                 static_cast<unsigned long long>(g_ri_dropped.load()),
                 nodes.size() != issue_ids.size()
                     ? "  (COUNT MISMATCH: a launch API is unrecorded)"
                     : "");
    std::fflush(stderr);
  }
}

// Record one kernel launch's resolved identity into its stream's capture window,
// in issue order. fid is resolved by the caller (module via g_functions, host
// via g_host_functions; 0 if unknown). Gated to the recording window. `args` is
// the launch-time kernarg blob (tagged; see pack_kernel_args_*) — captured now
// because the caller's buffer is reused the instant real() returns.
void record_issue(hipStream_t stream, std::uint64_t fid,
                  std::vector<std::byte> args = {}) {
  if (g_active_captures.load(std::memory_order_relaxed) == 0 ||
      g_graphs_written.load(std::memory_order_relaxed) >= max_graphs()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(g_mu);
  auto push = [&](CaptureWindow& w) {
    w.issue_ids.push_back(fid);
    w.arg_blobs.push_back(std::move(args));
  };
  auto it = g_windows.find(static_cast<void*>(stream));
  if (it != g_windows.end()) {
    push(it->second);
    g_ri_matched.fetch_add(1, std::memory_order_relaxed);
  } else if (g_windows.size() == 1) {
    // Cross-stream capture: HIP graph capture forks helper streams whose handle
    // differs from the captured stream, but their launches still become nodes in
    // the one open capture graph. Attribute them to that single window so the
    // issue list stays 1:1 with the captured nodes.
    push(g_windows.begin()->second);
    g_ri_fallback.fetch_add(1, std::memory_order_relaxed);
  } else {
    // No window for this stream and not exactly one open: dropped. A high count
       // here means launches reach the graph on streams we can't attribute.
    g_ri_dropped.fetch_add(1, std::memory_order_relaxed);
  }
}

std::uint64_t lookup_module_fid(hipFunction_t f) {
  const std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_functions.find(reinterpret_cast<std::uintptr_t>(f));
  return it != g_functions.end() ? it->second.id : 0;
}

std::uint64_t lookup_host_fid(const void* host_func) {
  const std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_host_functions.find(reinterpret_cast<std::uintptr_t>(host_func));
  return it != g_host_functions.end() ? it->second.id : 0;
}

// Extract the contiguous kernarg buffer from a driver-style `extra` launch
// config: { BUFFER_POINTER, buf, BUFFER_SIZE, &size, END }. Returns empty if
// the launch uses kernelParams instead (sizes unknowable; identity is still
// recorded, relocation just has nothing to patch on that node).
std::vector<std::byte> extract_param_blob(void** extra) {
  // Log the first few calls BEFORE the early return: ROCm's
  // hipGraphKernelNodeGetParams returns extra=NULL, so the kernarg cannot be
  // recovered from the captured node at all (it must be captured at launch
  // time). This log makes that limitation visible per-run.
  static std::atomic<std::uint64_t> n_called{0};
  const std::uint64_t k = n_called.fetch_add(1, std::memory_order_relaxed);
  if (k < 8) {
    std::fprintf(stderr,
                 "[record] extract_param_blob #%llu extra=%p%s\n",
                 static_cast<unsigned long long>(k), extra,
                 extra == nullptr ? " (NULL: kernarg unavailable from graph node)"
                                  : "");
    std::fflush(stderr);
  }
  std::vector<std::byte> blob;
  if (extra == nullptr) {
    return blob;
  }
  void* buffer = nullptr;
  std::size_t size = 0;
  bool have_buffer = false;
  bool have_size = false;
  std::size_t i = 0;
  while (i < 32) {
    void* id = extra[i];
    if (id == HIP_LAUNCH_PARAM_END) {
      break;
    }
    if (id == HIP_LAUNCH_PARAM_BUFFER_POINTER && i + 1 < 32) {
      buffer = extra[i + 1];
      have_buffer = true;
      i += 2;
    } else if (id == HIP_LAUNCH_PARAM_BUFFER_SIZE && i + 1 < 32) {
      size = *reinterpret_cast<std::size_t*>(extra[i + 1]);
      have_size = true;
      i += 2;
    } else {
      i += 1;
    }
  }
  if (have_buffer && have_size && buffer != nullptr && size > 0 &&
      size <= (1ULL << 28)) {  // 256 MiB sanity cap on a single kernarg blob
    blob.assign(static_cast<const std::byte*>(buffer),
                static_cast<const std::byte*>(buffer) + size);
  }
  return blob;
}

// ---- kernel-argument capture at launch time -------------------------------
//
// ROCm's hipGraphKernelNodeGetParams returns extra=NULL, so the kernarg cannot
// be recovered from the captured node (confirmed: extract_param_blob #0..N all
// see extra=(nil)). We capture it instead at LAUNCH time, while the caller's
// arg buffer is still alive. Two HIP arg formats exist:
//
//   extra       (BUFFER format): {PTR, &buf, SIZE, &sz, END} — one contiguous
//                               buffer of known size. extract_param_blob() above.
//   kernelParams/args (ARRAY format): NULL-terminated void*[] — each entry
//                               points to ONE arg value (size unknown here;
//                               HIP reads the right count from the code object
//                               signature at launch/instantiate time).
//
// We store whichever format the caller used, tagged so the rebuild can pass it
// back verbatim (buffer->extra, array->kernelParams). No signature parsing is
// needed: for the array format we copy a generous per-arg slice (default 16 B,
// env SNAPSHOT_RECORD_ARG_BYTES) — HIP reads only the bytes the kernel actually
// declares, so trailing garbage in an over-sized slice is never consumed.
// Pointers captured here already point into the arena the rebuild recreates at
// region_base, so no relocation is needed for the in-arena case.
static void append_u32(std::vector<std::byte>& v, std::uint32_t x) {
  v.push_back(static_cast<std::byte>(x & 0xff));
  v.push_back(static_cast<std::byte>((x >> 8) & 0xff));
  v.push_back(static_cast<std::byte>((x >> 16) & 0xff));
  v.push_back(static_cast<std::byte>((x >> 24) & 0xff));
}

static std::uint32_t arg_copy_bytes() {
  static const std::uint32_t b = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_ARG_BYTES");
    return e && *e ? static_cast<std::uint32_t>(std::strtoul(e, nullptr, 10))
                   : 16u;
  }();
  return b;
}

// Signal-free readability probe + copy: process_vm_readv reads from our own
// process VA into `dst`, stopping cleanly (returning a short count or -1/
// EFAULT) at the first unmapped page — never SIGSEGV. This replaces a
// write(/dev/null) probe, which was useless because /dev/null's write handler
// discards input WITHOUT touching the source buffer (so it never fault-tested
// and the following memcpy segfaulted on args[12]=0x1000000040 in wvSplitK,
// jobs 509889..509902). Returns the number of bytes actually copied (0 if the
// source is fully unreadable, so the caller can zero-pad the rest).
#include <sys/uio.h>
std::size_t safe_copy_n(void* dst, const void* src, std::size_t n) {
  if (n == 0 || src == nullptr || dst == nullptr) return 0;
  const pid_t self = ::getpid();
  iovec liov{dst, n};
  iovec riov{const_cast<void*>(src), n};
  ssize_t r = ::process_vm_readv(self, &liov, 1, &riov, 1, 0);
  if (r > 0) return static_cast<std::size_t>(r);
  return 0;  // EFAULT / ENOSYS / unreadable: caller zero-pads
}
int devnull_fd() {
  static int fd = [] {
    int f = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
    return f;
  }();
  return fd;
}
std::size_t readable_bytes(const void* p, std::size_t n) {
  if (n == 0 || p == nullptr) return 0;
  const int fd = devnull_fd();
  if (fd < 0) return 0;
  ssize_t w = ::write(fd, p, n);
  return w > 0 ? static_cast<std::size_t>(w) : 0;
}

// Master gate for launch-time kernarg capture. Default OFF: the arg-packing
// dereferences caller-owned arg arrays, and a first-run fault (job 509884,
// segfault right after BeginCapture) must be isolated before this is safe to
// leave on. Flip with SNAPSHOT_RECORD_CAPTURE_ARGS=1.
bool capture_args_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("SNAPSHOT_RECORD_CAPTURE_ARGS");
    return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
  }();
  return on;
}

// First-call diagnostic: log params pointer + detected count exactly once per
// pack path, BEFORE any deref, so a fault on the first captured launch leaves a
// breadcrumb pointing at the offending API + arg layout.
void log_pack_once(const char* api, const char* fmt, void** params,
                   std::size_t count) {
  static std::atomic<int> n{0};
  if (n.fetch_add(1, std::memory_order_relaxed) == 0) {
    std::fprintf(stderr,
                 "[record] PACK first %s fmt=%s params=%p count=%zu per=%u\n",
                 api, fmt, static_cast<void*>(params), count, arg_copy_bytes());
    std::fflush(stderr);
  }
}

// Pack a NULL-terminated kernelParams/args array. Tag 1 = array format.
// The terminator is determined by BOTH the NULL sentinel AND readability: a
// real arg pointer is always readable host memory, so the first non-NULL but
// unreadable entry marks the true end (some launch sites — e.g. wvSplitK via
// hipLaunchKernel — leave a non-NULL garbage value where a terminator should
// be; reading past it pulled in device-range addresses and crashed the
// rebuild). Each arg is copied with a fault-tested process_vm_readv.
std::vector<std::byte> pack_kernel_args_array(void** params) {
  std::vector<std::byte> blob;
  if (params == nullptr) return blob;
  const std::uint32_t per = arg_copy_bytes();
  // First pass: find the count (first NULL or first non-NULL-unreadable entry).
  std::size_t count = 0;
  for (std::size_t i = 0; i < 48; ++i) {  // real kernels: <32 args
    if (params[i] == nullptr) break;
    // Probe readability with a 1-byte process_vm_readv: a valid host arg
    // pointer is always readable; a garbage/unmapped entry ends the array.
    char probe;
    if (safe_copy_n(&probe, params[i], 1) == 0) break;
    ++count;
  }
  if (count == 0) {
    log_pack_once("array", "empty(all-unreadable)", params, count);
    return blob;
  }
  log_pack_once("array", "ok", params, count);
  blob.push_back(std::byte{1});             // array-format tag
  append_u32(blob, static_cast<std::uint32_t>(count));
  static std::atomic<int> n_detail{0};
  const bool detail = n_detail.fetch_add(1, std::memory_order_relaxed) == 0;
  for (std::size_t i = 0; i < count; ++i) {
    append_u32(blob, per);  // record the logical (padded) arg size
    const std::size_t mark = blob.size();
    blob.resize(mark + per, std::byte{0});  // zero-pad to `per`
    const std::size_t got = safe_copy_n(blob.data() + mark, params[i], per);
    if (detail) {
      std::fprintf(stderr,
                   "[record] PACK arg[%zu] ptr=%p readable=%zu/%u%s\n",
                   i, params[i], got, per, got == 0 ? " (skipped-unreadable)" : "");
      std::fflush(stderr);
    }
  }
  return blob;
}

// Pack an `extra`-format launch config into a tagged blob. Tag 0 = buffer.
// Returns empty if extra is absent / unparseable (caller falls back to array).
std::vector<std::byte> pack_kernel_args_buffer(void** extra) {
  auto buf = extract_param_blob(extra);
  log_pack_once("buffer", buf.empty() ? "empty" : "ok", extra, buf.size());
  if (buf.empty()) return buf;
  std::vector<std::byte> tagged;
  tagged.reserve(1 + 4 + buf.size());
  tagged.push_back(std::byte{0});  // buffer-format tag
  append_u32(tagged, static_cast<std::uint32_t>(buf.size()));
  tagged.insert(tagged.end(), buf.begin(), buf.end());
  return tagged;
}

// Read a code object loaded from a file (hipModuleLoad path).
std::vector<std::byte> read_file_bytes(const char* fname) {
  std::ifstream in(fname, std::ios::binary);
  if (!in) {
    return {};
  }
  std::vector<char> chars((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  std::vector<std::byte> data(chars.size());
  if (!chars.empty()) {
    std::memcpy(data.data(), chars.data(), chars.size());
  }
  return data;
}

// Extract every AMDGPU HSACO ELF from a fat binary's Clang offload bundle.
// Bundle layout (ROCm 6.x, confirmed empirically):
//   char   magic[24]  "__CLANG_OFFLOAD_BUNDLE__"
//   uint64 numBundles
//   per bundle: { uint64 offset; uint64 size; uint64 tripleSize; char triple[] }
// We parse the header to find each bundle's (offset, size, triple) and take the
// entries whose triple names an amdgcn target — that is the device code object
// (an ELF whose e_machine == EM_AMDGPU). This is SAFE: every read is bounded by
// offsets parsed from the header, so we never scan past the bundle into
// unmapped memory (a blind scan segfaulted vLLM at startup). `cap` is a hard
// outer bound on trusted offsets (the bundle is never nearly this large).
std::vector<std::vector<std::byte>> extract_amdgpu_elfs(const void* binary,
                                                        std::size_t cap,
                                                        const std::string& target_gfx) {
  std::vector<std::vector<std::byte>> out;
  if (binary == nullptr || cap < 32) {
    return out;
  }
  const auto* p = static_cast<const unsigned char*>(binary);
  static const char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";  // 24 chars + NUL
  if (std::memcmp(p, kMagic, 24) != 0) {
    return out;  // not a recognized bundle; leave empty (identity still recorded)
  }
  std::size_t i = 24;
  if (i + 8 > cap) {
    return out;
  }
  std::uint64_t num = 0;
  std::memcpy(&num, p + i, 8);
  i += 8;
  if (num == 0 || num > 4096) {
    return out;  // sanity: a real bundle has a handful of targets
  }
  // First pass: collect amdgcn entries and every non-empty entry, reading each
  // header sequentially (triple is variable-length).
  struct Entry {
    std::uint64_t off;
    std::uint64_t size;
  };
  std::vector<Entry> amdgcn;
  std::vector<Entry> amdgcn_target;  // amdgcn AND matching the device gfx
  std::vector<Entry> any_nonempty;
  for (std::uint64_t b = 0; b < num; ++b) {
    if (i + 24 > cap) {
      break;
    }
    std::uint64_t off = 0, size = 0, tsize = 0;
    std::memcpy(&off, p + i, 8);
    std::memcpy(&size, p + i + 8, 8);
    std::memcpy(&tsize, p + i + 16, 8);
    i += 24;
    if (tsize > cap || i + tsize > cap) {
      break;
    }
    std::string triple(reinterpret_cast<const char*>(p + i),
                       static_cast<std::size_t>(tsize));
    i += static_cast<std::size_t>(tsize);
    if (off > cap || size > cap || off + size > cap) {
      break;  // entry points outside the trusted region; refuse
    }
    if (size >= 64) {
      any_nonempty.push_back({off, size});
    }
    // The device target triple contains "amdgcn" (e.g. hipv4-amdgcn-amd-amdhsa).
    if (triple.find("amdgcn") != std::string::npos) {
      amdgcn.push_back({off, size});
      // M3b: a PyTorch fat binary bundles an amdgcn entry PER target arch
      // (gfx942, gfx1100, ...). Keep only the one for THIS device — matching
      // a kernel name to a gfx1100 image made hipModuleLoadData reject it on
      // gfx942 ("no kernel image available"). Match the gfx token (e.g.
      // "gfx942") from gcnArchName; require a non-digit after so gfx942 !=
      // gfx9421.
      if (!target_gfx.empty()) {
        std::size_t pos = triple.find(target_gfx);
        if (pos != std::string::npos) {
          const char after =
              pos + target_gfx.size() < triple.size()
                  ? triple[pos + target_gfx.size()]
                  : '\0';
          if (!std::isdigit(static_cast<unsigned char>(after))) {
            amdgcn_target.push_back({off, size});
          }
        }
      }
    }
  }
  // Prefer amdgcn entries for the device arch; fall back to all amdgcn if none
  // matched (token parse miss); then to any non-empty ELF (triple spelling).
  std::vector<Entry> candidates = !amdgcn_target.empty() ? amdgcn_target
                                   : amdgcn.empty() ? any_nonempty : amdgcn;
  for (const Entry& e : candidates) {
    if (e.size < 64 || e.off + e.size > cap) {
      continue;
    }
    const auto* elf = p + e.off;
    if (!(elf[0] == 0x7f && elf[1] == 'E' && elf[2] == 'L' && elf[3] == 'F')) {
      continue;
    }
    // The offload-bundle header gives the AUTHORITATIVE per-entry size, so copy
    // the full entry. (Earlier this min'd with elf_code_object_size, which can
    // stop before a trailing .note/metadata segment and truncate the kernel
    // .text -> "no kernel image available" when the image is reloaded.)
    out.emplace_back(reinterpret_cast<const std::byte*>(elf),
                     reinterpret_cast<const std::byte*>(elf) + e.size);
  }
  return out;
}

// Log the first call to each interposed symbol per process. This is the
// decisive diagnostic for "which HIP entry points does the application actually
// reach" — e.g. hipModuleLoadDataEx (Triton/torch.compile) vs hipModuleLoadData,
// or hipStreamBeginCaptureToGraph (piecewise capture) vs hipStreamBeginCapture.
// Without it, a zero-output run leaves us guessing which symbol path was used.
void probe_once(const char* name) {
  static std::mutex probe_mu;
  static std::set<std::string> seen;
  const std::lock_guard<std::mutex> lock(probe_mu);
  if (seen.emplace(name).second) {
    std::fprintf(stderr, "[record] pid=%d FIRST %s\n",
                 static_cast<int>(getpid()), name);
    std::fflush(stderr);
  }
}

// Factor module-image capture (shared by hipModuleLoadData / LoadDataEx /
// hipModuleLoad): recover the HSACO/ELF length from the header (the load API
// passes no size), copy the image, hash it for later (module, entry) lookup.
void record_module_image(hipModule_t* module, const void* image) {
  const bool want = g_graphs_written.load(std::memory_order_relaxed) <
                        max_graphs() ||
                    g_active_captures.load(std::memory_order_relaxed) > 0;
  if (!want) {
    return;
  }
  const std::uint64_t size = snapshot::elf_code_object_size(
      static_cast<const std::byte*>(image), 1ULL << 30);
  ModRec rec;
  if (size > 0) {
    rec.image.assign(static_cast<const std::byte*>(image),
                     static_cast<const std::byte*>(image) + size);
    rec.hash = snapshot::hash_bytes(rec.image.data(), rec.image.size());
  } else if (verbose()) {
    std::fprintf(stderr,
                 "[record] module image not a parseable ELF; image empty\n");
  }
  const std::lock_guard<std::mutex> lock(g_mu);
  g_modules[reinterpret_cast<std::uintptr_t>(*module)] = std::move(rec);
}

// Factor capture-window opening (shared by hipStreamBeginCapture and the
// piecewise hipStreamBeginCaptureToGraph variant).
void open_capture_window(hipStream_t stream) {
  if (g_graphs_written.load(std::memory_order_relaxed) < max_graphs()) {
    const std::lock_guard<std::mutex> lock(g_mu);
    g_windows[static_cast<void*>(stream)] = CaptureWindow{};
    g_active_captures.fetch_add(1);
  }
}

// Introspect the just-captured graph into the launch list. The driver has
// already resolved every launch — whether issued via hipLaunchKernel
// (PyTorch/ATen's <<<>>> path) or hipModuleLaunchKernel (explicit module
// launch) — into kernel nodes. This is the AUTHORITATIVE node list,
// independent of which launch symbol the application used. A node's `func` is
// correlated against BOTH identity maps:
//   * g_functions      (module path: hipModuleGetFunction -> hipFunction_t)
//   * g_host_functions (host path:   __hipRegisterFunction -> hostFunc ptr;
//                      a captured <<<>>> node's func IS that hostFunc ptr)
// so every kernel recovers its (module, entry) regardless of launch API.
std::size_t introspect_graph_nodes(hipGraph_t graph,
                                   std::vector<snapshot::RecordedLaunch>& out,
                                   std::vector<std::uintptr_t>& out_funcs) {
  if (graph == nullptr) {
    return 0;
  }
  std::size_t count = 0;
  if (hipGraphGetNodes(graph, nullptr, &count) != hipSuccess || count == 0) {
    return 0;
  }
  std::vector<hipGraphNode_t> nodes(count);
  if (hipGraphGetNodes(graph, nodes.data(), &count) != hipSuccess) {
    return 0;
  }
  std::size_t kept = 0;
  std::size_t module_matched = 0;
  for (std::size_t i = 0; i < count; ++i) {
    hipGraphNodeType nt{};
    if (hipGraphNodeGetType(nodes[i], &nt) != hipSuccess) {
      continue;
    }
    if (nt != hipGraphNodeTypeKernel) {
      continue;  // memcpy/memset nodes: M3b buffer snapshotting, not yet.
    }
    hipKernelNodeParams p{};
    if (hipGraphKernelNodeGetParams(nodes[i], &p) != hipSuccess) {
      continue;
    }
    const std::uintptr_t fkey = reinterpret_cast<std::uintptr_t>(p.func);
    // Every node's name is resolved OFF-PATH by the background namer (calling
    // hipKernelNameRef inline here deadlocks on the HIP capture lock). The
    // snapshot for this graph is written later, once all its funcs are named.
    ensure_namer();
    enqueue_for_naming(fkey);

    snapshot::RecordedLaunch rec;
    {
      const std::lock_guard<std::mutex> lock(g_mu);
      auto fit = g_functions.find(fkey);
      if (fit != g_functions.end()) {
        rec.function_id = fit->second.id;  // module kernel, known at issue time
        ++module_matched;
      } else {
        rec.function_id = 0;  // named later via hipKernelNameRef
      }
    }
    rec.grid = snapshot::Dim3{p.gridDim.x, p.gridDim.y, p.gridDim.z};
    rec.block = snapshot::Dim3{p.blockDim.x, p.blockDim.y, p.blockDim.z};
    rec.shared_mem_bytes = p.sharedMemBytes;
    rec.param_blob = extract_param_blob(p.extra);
    out.push_back(std::move(rec));
    out_funcs.push_back(fkey);
    ++kept;
  }
  if (verbose()) {
    std::fprintf(stderr,
                 "[record] introspect: kernel_nodes=%zu module_matched=%zu "
                 "(rest named off-path)\n",
                 kept, module_matched);
    std::fflush(stderr);
  }
  return kept;
}

// Write any pending graph whose every node func has been named by the off-path
// namer. Called from the namer thread after each resolution. Builds node
// identity from the resolved name + (name -> module) recovered from the
// hipModuleGetFunction map, includes the referenced module images, and writes.
void try_flush_pending() {
  const std::string dir = out_dir();
  if (dir.empty()) {
    return;
  }
  std::vector<PendingGraph> todo;
  {
    const std::lock_guard<std::mutex> lock(g_pending_mu);
    for (const PendingGraph& pg : g_pending) {
      if (!pg.written) {
        todo.push_back(pg);
      }
    }
  }
  for (PendingGraph& g : todo) {
    std::vector<std::string> names(g.node_funcs.size());
    bool ready = true;
    {
      const std::lock_guard<std::mutex> lock(g_namer_mu);
      for (std::size_t i = 0; i < g.node_funcs.size(); ++i) {
        auto it = g_resolved_names.find(g.node_funcs[i]);
        if (it == g_resolved_names.end()) {
          ready = false;
          break;
        }
        names[i] = it->second;
      }
    }
    if (!ready) {
      continue;  // some funcs not named yet; a later resolution will retry
    }

    snapshot::RecordAssembly a;
    a.vendor = snapshot::Vendor::kHip;
    a.arch = g.arch;
    a.region_base = g.region_base;
    a.region_size = g.region_size;
    a.granularity = g.granularity;
    a.alloc_events = g.alloc_events;
    std::size_t named = 0, with_module = 0, with_module_from_syms = 0;
    std::size_t args_filled = 0;  // nodes whose param_blob came from launch time
    std::size_t n_modules_seen = 0, n_fatbins_seen = 0, n_funcs_seen = 0,
        n_hostfuncs_seen = 0, n_sym_names = 0, n_hsa_seen = 0;
    {
      const std::lock_guard<std::mutex> lock(g_mu);
      // name -> module image hash. Built from THREE sources so a kernel named by
      // hipKernelNameRef or __hipRegisterFunction still links to its code object
      // even when its hipFunction_t handle did not match the one
      // hipModuleGetFunction returned (the common case for Triton/aiter/CK):
      //   (a) g_functions (hipModuleGetFunction: entry -> module_hash), and
      //   (b) the ELF symbol tables of every captured module image, and
      //   (c) the ELF symbol tables of every captured fatbin image.
      std::map<std::string, std::uint64_t> name_to_module;
      for (const auto& [handle, fr] : g_functions) {
        (void)handle;
        if (!fr.entry.empty() && fr.module_hash != 0) {
          name_to_module.emplace(fr.entry, fr.module_hash);
        }
      }
      // hash -> image bytes, across both module-load and fatbin images.
      std::map<std::uint64_t, const std::vector<std::byte>*> hash_to_image;
      const std::string gfx = device_gfx_token();  // e.g. "gfx942" (empty pre-ctx)
      auto image_targets_device = [&](const std::vector<std::byte>& img) {
        if (gfx.empty() || img.size() < 7) return true;  // can't tell; keep
        const auto* needle = reinterpret_cast<const std::byte*>(gfx.data());
        const std::size_t nlen = gfx.size();
        // The amdhsa.target triple is embedded as an ASCII string even in
        // msgpack metadata, so a substring scan tells us the arch. Require a
        // non-digit after so "gfx942" != "gfx9421".
        for (std::size_t i = 0; i + nlen < img.size(); ++i) {
          if (std::memcmp(img.data() + i, needle, nlen) != 0) continue;
          const std::size_t j = i + nlen;
          const unsigned char after =
              j < img.size() ? static_cast<unsigned char>(img[j]) : '\0';
          if (!std::isdigit(after)) return true;
        }
        return false;
      };
      auto add_image = [&](const std::vector<std::byte>& img) {
        if (img.empty()) return;
        if (!image_targets_device(img)) {
          return;  // wrong-arch bundle entry (e.g. gfx1100 on a gfx942 device)
        }
        const std::uint64_t h =
            snapshot::hash_bytes(img.data(), img.size());
        hash_to_image.emplace(h, &img);
        for (const std::string& nm :
             snapshot::extract_elf_symbols(img.data(), img.size())) {
          name_to_module.emplace(nm, h);
          ++n_sym_names;
        }
      };
      for (const auto& [handle, mod] : g_modules) {
        (void)handle;
        ++n_modules_seen;
        if (mod.hash != 0 && !mod.image.empty()) {
          hash_to_image.emplace(mod.hash, &mod.image);
          for (const std::string& nm :
               snapshot::extract_elf_symbols(mod.image.data(),
                                             mod.image.size())) {
            name_to_module.emplace(nm, mod.hash);
            ++n_sym_names;
          }
        }
      }
      for (const auto& [handle, fb] : g_fatbins) {
        (void)handle;
        ++n_fatbins_seen;
        for (const std::vector<std::byte>& img : fb.elf_images) {
          add_image(img);
        }
      }
      // (c) HSACO loaded directly via ROCr (Triton JIT) — captured at
      // hsa_code_object_reader_create_from_memory. Kernel names come from the
      // AMDGPU metadata note (not .symtab), which extract_elf_symbols now reads.
      for (const auto& [handle, hi] : g_hsa_images) {
        (void)handle;
        ++n_hsa_seen;
        add_image(hi.image);
      }
      n_funcs_seen = g_functions.size();
      n_hostfuncs_seen = g_host_functions.size();
      std::set<std::uint64_t> needed;
      for (std::size_t i = 0; i < g.nodes.size(); ++i) {
        snapshot::RecordedLaunch rl = g.nodes[i];
        rl.function_id = static_cast<std::uint64_t>(i + 1);
        // Merge launch-time kernarg (the only source — hipGraphKernelNodeGetParams
        // returns extra=NULL on ROCm). Only fill when the issue-order window
        // aligns 1:1 with the introspected nodes (same invariant as the identity
        // reconcile). Count coverage for the FLUSH log.
        if (i < g.arg_blobs.size() && !g.arg_blobs[i].empty() &&
            rl.param_blob.empty()) {
          rl.param_blob = g.arg_blobs[i];
          ++args_filled;
        }
        a.launches.push_back(std::move(rl));
        std::uint64_t mh = 0;
        auto mit = name_to_module.find(names[i]);
        if (mit != name_to_module.end()) {
          mh = mit->second;
        }
        if (!names[i].empty()) {
          ++named;
        }
        if (mh != 0) {
          ++with_module;
          needed.insert(mh);
          // Was this link found only via the ELF symbol table (not g_functions)?
          bool in_funcs = false;
          for (const auto& [hh, fr] : g_functions) {
            (void)hh;
            if (fr.entry == names[i] && fr.module_hash == mh) {
              in_funcs = true;
              break;
            }
          }
          if (!in_funcs) ++with_module_from_syms;
        }
        snapshot::RecordedFunction rf;
        rf.function_id = static_cast<std::uint64_t>(i + 1);
        rf.module_hash = mh;
        rf.entry_name = names[i];
        a.functions.push_back(std::move(rf));
      }
      // M3b diagnostic: per-node identity +, for named-but-unmatched nodes, a
      // hint of whether the name is a substring of any indexed symbol (detects
      // the Triton-vs-ELF mangling mismatch). Gated on verbose(). One-time cost.
      if (verbose()) {
        for (std::size_t i = 0; i < g.nodes.size(); ++i) {
          const auto& rf = a.functions[i];
          if (!rf.entry_name.empty() && rf.module_hash == 0) {
            std::vector<std::string> hints;
            for (const auto& [kn, kh] : name_to_module) {
              if (kn.find(rf.entry_name) != std::string::npos ||
                  rf.entry_name.find(kn) != std::string::npos) {
                hints.push_back(kn);
                if (hints.size() >= 3) break;
              }
            }
            // Decisive: is the name a raw byte substring of any captured
            // module image? (Triton kernels are named only in the ROCm
            // code-object metadata note, not .symtab.) A hit means the module
            // IS captured and the metadata is uncompressed YAML (parse it); a
            // miss means the module isn't here or the metadata is compressed.
            bool raw_in_module = false;
            bool raw_in_fatbin = false;
            const void* needle = rf.entry_name.data();
            const std::size_t nlen = rf.entry_name.size();
            auto img_has = [&](const std::vector<std::byte>& img) {
              return img.size() >= nlen &&
                     std::search(img.data(), img.data() + img.size(),
                                 static_cast<const std::byte*>(needle),
                                 static_cast<const std::byte*>(needle) + nlen) !=
                         img.data() + img.size();
            };
            for (const auto& [hh, mod] : g_modules) {
              (void)hh;
              if (img_has(mod.image)) { raw_in_module = true; break; }
            }
            if (!raw_in_module) {
              for (const auto& [hh, fb] : g_fatbins) {
                (void)hh;
                bool found = false;
                for (const std::vector<std::byte>& img : fb.elf_images) {
                  if (img_has(img)) { found = true; break; }
                }
                if (found) { raw_in_fatbin = true; break; }
              }
            }
            std::string h;
            for (std::size_t k = 0; k < hints.size(); ++k) {
              h += "'";
              h += hints[k].substr(0, 60);
              h += "'";
              if (k + 1 < hints.size()) h += ", ";
            }
            std::fprintf(stderr,
                         "[record] FLUSH-node %zu name='%.90s' NO-MODULE "
                         "hints=[%s] raw_in_module=%d raw_in_fatbin=%d\n",
                         i, rf.entry_name.c_str(), h.c_str(),
                         raw_in_module ? 1 : 0, raw_in_fatbin ? 1 : 0);
          } else {
            std::fprintf(stderr,
                         "[record] FLUSH-node %zu name='%.90s' mh=0x%llx\n",
                         i, rf.entry_name.c_str(),
                         static_cast<unsigned long long>(rf.module_hash));
          }
        }
        std::fflush(stderr);
      }
      for (const auto& [hash, img] : hash_to_image) {
        if (needed.count(hash)) {
          snapshot::RecordedModule rm;
          rm.hash = hash;
          rm.image = *img;
          a.modules.push_back(std::move(rm));
        }
      }
    }

    snapshot::SnapshotData snap;
    const snapshot::Status st = assemble_recorded_snapshot(a, &snap);
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/graph-%d-%llu.snap", dir.c_str(),
                  static_cast<int>(getpid()),
                  static_cast<unsigned long long>(g.index));
    bool ok = st.ok() && snapshot::write_snapshot_file(path, snap).ok();
    // Debug dump (SNAPSHOT_RECORD_DUMP_HSA=1): write each image that goes into
    // the .snap (module/fatbin/HSA alike), named by hash, so modload_test can
    // load each in isolation and pinpoint which one ROCm rejects.
    if (std::getenv("SNAPSHOT_RECORD_DUMP_HSA") != nullptr) {
      for (const snapshot::RecordedModule& rm : a.modules) {
        char mpath[1024];
        std::snprintf(mpath, sizeof(mpath), "%s/mod-0x%llx.co", dir.c_str(),
                      static_cast<unsigned long long>(rm.hash));
        std::ofstream df(mpath, std::ios::binary);
        df.write(reinterpret_cast<const char*>(rm.image.data()),
                 static_cast<std::streamsize>(rm.image.size()));
      }
    }
    std::fprintf(stderr,
                 "[record] FLUSH idx=%llu nodes=%zu named=%zu with_module=%zu "
                 "(syms=%zu) args_filled=%zu modules=%zu wrote=%d "
                 "[tables mods=%zu fatbins=%zu hsa=%zu funcs=%zu hostfuncs=%zu "
                 "sym_names=%zu]%s\n",
                 static_cast<unsigned long long>(g.index), g.nodes.size(), named,
                 with_module, with_module_from_syms, args_filled, a.modules.size(),
                 ok ? 1 : 0, n_modules_seen, n_fatbins_seen, n_hsa_seen,
                 n_funcs_seen, n_hostfuncs_seen, n_sym_names,
                 g_hsa_capture_skipped ? " (HSA cap hit!)" : "");
    std::fflush(stderr);
    // Arg-coverage diagnostic: shows why args_filled may be < nodes. We log the
    // window's arg_blobs count, how many are non-empty, and a per-position fill
    // mask (1=node i got a blob, 0=no blob / empty / positional gap) so a
    // short window (unhooked launches) vs a misaligned window (reordering) are
    // distinguishable at a glance.
    {
      std::size_t win_size = g.arg_blobs.size();
      std::size_t win_nonempty = 0;
      std::string mask;
      mask.reserve(g.nodes.size());
      for (std::size_t i = 0; i < g.nodes.size(); ++i) {
        const bool have =
            i < win_size && !g.arg_blobs[i].empty();
        if (have) ++win_nonempty;
        mask.push_back(have ? '1' : '0');
      }
      std::fprintf(stderr,
                   "[record] FLUSH args: window=%zu nonempty=%zu nodes=%zu "
                   "mask=%s\n",
                   win_size, win_nonempty, g.nodes.size(), mask.c_str());
      std::fflush(stderr);
    }
    // Per-hook launch counts during capture (cumulative). Tells us which API
    // path the captured nodes flowed through — diagnoses why only some nodes
    // have args (e.g. mod+host=3 but nodes=6 means 3 nodes were added via a
    // non-launch path like hipGraphAddKernelNode, shown by addnode).
    std::fprintf(stderr,
                 "[record] FLUSH hooks: mod=%llu host=%llu ext=%llu exc=%llu "
                 "coop_host=%llu coop_mod=%llu addnode=%llu\n",
                 static_cast<unsigned long long>(g.hooks.mod),
                 static_cast<unsigned long long>(g.hooks.host),
                 static_cast<unsigned long long>(g.hooks.ext),
                 static_cast<unsigned long long>(g.hooks.exc),
                 static_cast<unsigned long long>(g.hooks.coop_host),
                 static_cast<unsigned long long>(g.hooks.coop_mod),
                 static_cast<unsigned long long>(g.hooks.addnode));
    std::fflush(stderr);
    {
      const std::lock_guard<std::mutex> lock(g_pending_mu);
      for (PendingGraph& pg : g_pending) {
        if (pg.index == g.index) {
          pg.written = true;
        }
      }
    }
  }
}

// Serialize one finished capture window to disk.
void write_window_snapshot(const CaptureWindow& window) {
  const std::string dir = out_dir();
  if (dir.empty() || window.launches.empty()) {
    return;
  }

  snapshot::RecordAssembly a;
  a.vendor = snapshot::Vendor::kHip;
  a.arch = query_arch();
  a.launches = window.launches;
  {
    // Copy the shared identity tables + region under the lock; the (heavier)
    // assemble + serialize below runs lock-free on this thread-local copy.
    //
    // PRUNE to only the functions/modules THIS graph references: PyTorch
    // registers thousands of host functions across hundreds of fat binaries at
    // static-init; serializing all of them per graph made snapshots gigabytes.
    // A captured vLLM graph references only a handful, so we resolve the
    // window's function_ids to their (module_hash, entry) and include just
    // those modules.
    const std::lock_guard<std::mutex> lock(g_mu);
    a.region_base = env_u64("SNAPSHOT_RECORD_REGION_BASE", g_region_base);
    a.region_size = env_u64("SNAPSHOT_RECORD_REGION_SIZE", g_region_size);
    a.granularity = g_granularity;
    a.alloc_events = g_alloc_events;

    std::set<std::uint64_t> needed_module_hashes;
    // Resolve every referenced function id to its (module_hash, entry) by
    // consulting both identity maps; collect the module hashes it needs.
    for (const snapshot::RecordedLaunch& rl : window.launches) {
      if (rl.function_id == 0) {
        continue;
      }
      auto fit = std::find_if(
          g_functions.begin(), g_functions.end(),
          [&](const auto& kv) { return kv.second.id == rl.function_id; });
      if (fit != g_functions.end()) {
        snapshot::RecordedFunction rf;
        rf.function_id = fit->second.id;
        rf.module_hash = fit->second.module_hash;
        rf.entry_name = fit->second.entry;
        a.functions.push_back(std::move(rf));
        if (fit->second.module_hash != 0) {
          needed_module_hashes.insert(fit->second.module_hash);
        }
        continue;
      }
      auto hfit = std::find_if(
          g_host_functions.begin(), g_host_functions.end(),
          [&](const auto& kv) { return kv.second.id == rl.function_id; });
      if (hfit != g_host_functions.end()) {
        snapshot::RecordedFunction rf;
        rf.function_id = hfit->second.id;
        rf.entry_name = hfit->second.device_name;
        auto fat = g_fatbins.find(hfit->second.fatbin_handle);
        if (fat != g_fatbins.end() && fat->second.primary_hash != 0) {
          rf.module_hash = fat->second.primary_hash;
          needed_module_hashes.insert(fat->second.primary_hash);
        }
        a.functions.push_back(std::move(rf));
      }
    }
    // Emit the modules whose hash is referenced, from either source.
    for (const auto& [handle, mod] : g_modules) {
      if (needed_module_hashes.count(mod.hash) == 0) {
        continue;
      }
      snapshot::RecordedModule rm;
      rm.hash = mod.hash;
      rm.image = mod.image;
      a.modules.push_back(rm);
    }
    for (const auto& [handle, fat] : g_fatbins) {
      for (const auto& elf : fat.elf_images) {
        const std::uint64_t h =
            snapshot::hash_bytes(elf.data(), elf.size());
        if (needed_module_hashes.count(h) == 0) {
          continue;
        }
        snapshot::RecordedModule rm;
        rm.hash = h;
        rm.image = elf;
        a.modules.push_back(rm);
      }
    }
  }

  snapshot::SnapshotData snap;
  snapshot::Status status = snapshot::assemble_recorded_snapshot(a, &snap);
  if (!status.ok()) {
    std::fprintf(stderr, "[record] assemble failed: %s\n",
                 status.message().c_str());
    return;
  }

  snapshot::SnapshotSummary sum;
  static_cast<void>(snapshot::summarize_snapshot(snap, &sum));

  char path[1024];
  std::snprintf(path, sizeof(path), "%s/graph-%d-%llu.snap", dir.c_str(),
                static_cast<int>(getpid()),
                static_cast<unsigned long long>(g_graphs_written.load()));
  status = snapshot::write_snapshot_file(path, snap);
  if (!status.ok()) {
    std::fprintf(stderr, "[record] write %s failed: %s\n", path,
                 status.message().c_str());
    return;
  }
  if (verbose()) {
    std::fprintf(stderr,
                 "[record] pid=%d wrote %s nodes=%zu identity=%zu "
                 "unknown=%zu modules=%zu image_bytes=%llu param_bytes=%llu\n",
                 static_cast<int>(getpid()), path, sum.node_count,
                 sum.nodes_with_identity, sum.nodes_without_identity,
                 sum.module_count,
                 static_cast<unsigned long long>(sum.total_module_bytes),
                 static_cast<unsigned long long>(sum.total_param_bytes));
  }
}

struct Summary {
  ~Summary() {
    std::fprintf(stderr,
                 "[record] pid=%d SUMMARY graphs_written=%llu modules=%zu "
                 "functions=%zu host_functions=%zu fatbins=%zu "
                 "get_function_calls=%llu region_base=0x%llx region_size=%llu\n",
                 static_cast<int>(getpid()),
                 static_cast<unsigned long long>(g_graphs_written.load()),
                 g_modules.size(), g_functions.size(),
                 g_host_functions.size(),
                 g_fatbins.size(),
                 static_cast<unsigned long long>(g_get_function_calls),
                 static_cast<unsigned long long>(g_region_base),
                 static_cast<unsigned long long>(g_region_size));
  }
};
Summary g_summary;

}  // namespace

extern "C" {

// ---- module load / unload --------------------------------------------------

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  probe_once("hipModuleLoadData");
  using Fn = hipError_t (*)(hipModule_t*, const void*);
  static Fn real = resolve<Fn>("hipModuleLoadData");
  const hipError_t status = real(module, image);
  if (status == hipSuccess && module != nullptr && image != nullptr) {
    record_module_image(module, image);
  }
  return status;
}

// Triton (and torch.compile/inductor on ROCm) load JIT'd HSACO via the _Ex
// variant with JIT options — NOT via hipModuleLoadData. On a real vLLM cold
// start almost all code objects arrive here, so interposing only LoadData
// yields modules=0 (the exact M3a.3 failure this variant fixes). Signature
// must match the ROCm header exactly (C rejects conflicting declarations).
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                              unsigned int numOptions, hipJitOption* options,
                              void** optionValues) {
  probe_once("hipModuleLoadDataEx");
  using Fn = hipError_t (*)(hipModule_t*, const void*, unsigned int,
                            hipJitOption*, void**);
  static Fn real = resolve<Fn>("hipModuleLoadDataEx");
  const hipError_t status =
      real(module, image, numOptions, options, optionValues);
  if (status == hipSuccess && module != nullptr && image != nullptr) {
    record_module_image(module, image);
  }
  return status;
}

hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  probe_once("hipModuleLoad");
  using Fn = hipError_t (*)(hipModule_t*, const char*);
  static Fn real = resolve<Fn>("hipModuleLoad");
  const hipError_t status = real(module, fname);
  if (status == hipSuccess && module != nullptr && fname != nullptr) {
    const bool want = g_graphs_written.load(std::memory_order_relaxed) <
                          max_graphs() ||
                      g_active_captures.load(std::memory_order_relaxed) > 0;
    if (want) {
      std::vector<std::byte> bytes = read_file_bytes(fname);
      ModRec rec;
      if (!bytes.empty()) {
        rec.image = std::move(bytes);
        rec.hash = snapshot::hash_bytes(rec.image.data(), rec.image.size());
      }
      const std::lock_guard<std::mutex> lock(g_mu);
      g_modules[reinterpret_cast<std::uintptr_t>(*module)] = std::move(rec);
    }
  }
  return status;
}

hipError_t hipModuleUnload(hipModule_t module) {
  probe_once("hipModuleUnload");
  using Fn = hipError_t (*)(hipModule_t);
  static Fn real = resolve<Fn>("hipModuleUnload");
  {
    const std::lock_guard<std::mutex> lock(g_mu);
    g_modules.erase(reinterpret_cast<std::uintptr_t>(module));
  }
  return real(module);
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                const char* kname) {
  probe_once("hipModuleGetFunction");
  using Fn = hipError_t (*)(hipFunction_t*, hipModule_t, const char*);
  static Fn real = resolve<Fn>("hipModuleGetFunction");
  const hipError_t status = real(function, module, kname);
  ++g_get_function_calls;
  if (verbose() && g_get_function_calls <= 12) {
    std::fprintf(stderr,
                 "[record] hipModuleGetFunction #%llu module=%p name='%s' -> *fn=%p rc=%d\n",
                 static_cast<unsigned long long>(g_get_function_calls),
                 reinterpret_cast<void*>(module), kname ? kname : "(null)",
                 function ? reinterpret_cast<void*>(*function) : nullptr,
                 static_cast<int>(status));
  }
  if (status == hipSuccess && function != nullptr) {
    // M3a.6b safe-context validation: does hipKernelNameRef name this fresh,
    // known-valid hipFunction_t correctly (vs the kname we already know)? This is
    // OUTSIDE any capture, so the global lock that hangs it inline is free here.
    if (verbose() && g_get_function_calls <= 6) {
      const std::string ref = kernel_name_via_hip(*function, nullptr);
      std::fprintf(stderr,
                   "[record] NAMEREF-TEST kname='%.50s' hipKernelNameRef='%.50s'\n",
                   kname ? kname : "(null)", ref.c_str());
      std::fflush(stderr);
    }
    // Record function -> (module hash, entry name) AT THE SITE IDENTITY IS
    // ESTABLISHED. This is the whole reason identity is recoverable later.
    const std::lock_guard<std::mutex> lock(g_mu);
    std::uint64_t module_hash = 0;
    auto mit = g_modules.find(reinterpret_cast<std::uintptr_t>(module));
    if (mit != g_modules.end()) {
      module_hash = mit->second.hash;
    }
    FuncRec rec;
    rec.id = g_next_func_id++;
    rec.module_hash = module_hash;
    rec.entry = kname != nullptr ? std::string(kname) : std::string();
    g_functions[reinterpret_cast<std::uintptr_t>(*function)] = std::move(rec);
  }
  return status;
}

// ---- capture window delimiters --------------------------------------------

hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  probe_once("hipStreamBeginCapture");
  using Fn = hipError_t (*)(hipStream_t, hipStreamCaptureMode);
  static Fn real = resolve<Fn>("hipStreamBeginCapture");
  const hipError_t status = real(stream, mode);
  if (status == hipSuccess) {
    open_capture_window(stream);
  }
  return status;
}

// Piecewise CUDA-graph capture: begins a capture that merges into an existing
// graph node. vLLM's "PIECEWISE" capture path (mixed prefill-decode) can route
// through here rather than the plain BeginCapture. Signature matches the ROCm
// header exactly (hipGraphNode_t is itself a pointer typedef, so
// const hipGraphNode_t* == hipGraphNode* const*).
hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph,
                                        const hipGraphNode_t* dependencies,
                                        const hipGraphEdgeData* dependencyData,
                                        size_t numDependencies,
                                        hipStreamCaptureMode mode) {
  probe_once("hipStreamBeginCaptureToGraph");
  using Fn = hipError_t (*)(hipStream_t, hipGraph_t, const hipGraphNode_t*,
                            const hipGraphEdgeData*, size_t,
                            hipStreamCaptureMode);
  static Fn real = resolve<Fn>("hipStreamBeginCaptureToGraph");
  const hipError_t status =
      real(stream, graph, dependencies, dependencyData, numDependencies, mode);
  if (status == hipSuccess) {
    open_capture_window(stream);
  }
  return status;
}

hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  probe_once("hipStreamEndCapture");
  using Fn = hipError_t (*)(hipStream_t, hipGraph_t*);
  static Fn real = resolve<Fn>("hipStreamEndCapture");
  const hipError_t status = real(stream, pGraph);

  // Grab the finished window before decrementing the active count so the launch
  // fast-path sees an accurate "still recording" signal.
  CaptureWindow window;
  bool have_window = false;
  {
    const std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_windows.find(static_cast<void*>(stream));
    if (it != g_windows.end()) {
      window = std::move(it->second);
      g_windows.erase(it);
      have_window = true;
    }
  }
  if (have_window) {
    g_active_captures.fetch_sub(1);
    if (g_graphs_written.load() < max_graphs() && pGraph != nullptr) {
      // Introspect the captured graph: authoritative node list + structure
      // (grid/block/params) + per-node func. Identity (names) cannot be resolved
      // inline (hipKernelNameRef deadlocks on the capture lock), so we PARK the
      // graph; the off-path namer writes it once every func is named.
      std::vector<snapshot::RecordedLaunch> intro;
      std::vector<std::uintptr_t> intro_funcs;
      introspect_graph_nodes(*pGraph, intro, intro_funcs);
      if (!intro.empty()) {
        PendingGraph pg;
        pg.index = g_graphs_written.fetch_add(1);
        pg.arch = query_arch();  // reliable on this (capture) thread
        pg.nodes = std::move(intro);
        pg.node_funcs = std::move(intro_funcs);
        pg.arg_blobs = std::move(window.arg_blobs);  // launch-time kernarg
        pg.hooks.mod = g_cap_hooks.mod.load(std::memory_order_relaxed);
        pg.hooks.host = g_cap_hooks.host.load(std::memory_order_relaxed);
        pg.hooks.ext = g_cap_hooks.ext.load(std::memory_order_relaxed);
        pg.hooks.exc = g_cap_hooks.exc.load(std::memory_order_relaxed);
        pg.hooks.coop_host =
            g_cap_hooks.coop_host.load(std::memory_order_relaxed);
        pg.hooks.coop_mod =
            g_cap_hooks.coop_mod.load(std::memory_order_relaxed);
        pg.hooks.addnode = g_cap_hooks.addnode.load(std::memory_order_relaxed);
        {
          const std::lock_guard<std::mutex> lock(g_mu);
          pg.region_base =
              env_u64("SNAPSHOT_RECORD_REGION_BASE", g_region_base);
          pg.region_size =
              env_u64("SNAPSHOT_RECORD_REGION_SIZE", g_region_size);
          pg.granularity = g_granularity;
          pg.alloc_events = g_alloc_events;
        }
        {
          const std::lock_guard<std::mutex> lock(g_pending_mu);
          g_pending.push_back(std::move(pg));
        }
      }
    }
  }
  return status;
}

// ---- kernel launch (the HIP hot path) --------------------------------------

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gx,
                                 unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by,
                                 unsigned int bz, unsigned int shared,
                                 hipStream_t stream, void** kernelParams,
                                 void** extra) {
  probe_once("hipModuleLaunchKernel");
  using Fn = hipError_t (*)(hipFunction_t, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, hipStream_t, void**,
                            void**);
  static Fn real = resolve<Fn>("hipModuleLaunchKernel");
  // Record module-kernel identity in issue order (this path does not fire on the
  // observed vLLM stack, but is recorded for completeness and other workloads).
  if (g_active_captures.load(std::memory_order_relaxed) > 0) {
    g_cap_hooks.mod.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::byte> args;
    if (capture_args_enabled()) {
      args = extra ? pack_kernel_args_buffer(extra)
                   : pack_kernel_args_array(kernelParams);
    }
    record_issue(stream, lookup_module_fid(f), std::move(args));
  }
  return real(f, gx, gy, gz, bx, by, bz, shared, stream, kernelParams, extra);
}

// ---- deterministic-memory region observation (for synthetic restore) -------
//
// The synthetic workload's core allocator reserves/maps its VMM region here.
// We learn region_base/region_size from the reserve and record each mapping as
// an AllocEvent (offset = va - region_base) so a later RestoreSession can
// recreate the buffers before rebuilding + launching the graph.

hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment,
                                void* addr, unsigned long long flags) {
  probe_once("hipMemAddressReserve");
  using Fn = hipError_t (*)(void**, size_t, size_t, void*, unsigned long long);
  static Fn real = resolve<Fn>("hipMemAddressReserve");
  const hipError_t status = real(ptr, size, alignment, addr, flags);
  if (status == hipSuccess && ptr != nullptr && *ptr != nullptr) {
    const std::lock_guard<std::mutex> lock(g_mu);
    if (g_region_base == 0) {
      g_region_base = reinterpret_cast<std::uint64_t>(*ptr);
      g_region_size = size;
    }
  }
  return status;
}

hipError_t hipMemMap(void* ptr, size_t size, size_t offset,
                     hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags) {
  probe_once("hipMemMap");
  using Fn = hipError_t (*)(void*, size_t, size_t, hipMemGenericAllocationHandle_t,
                            unsigned long long);
  static Fn real = resolve<Fn>("hipMemMap");
  const hipError_t status = real(ptr, size, offset, handle, flags);
  if (status == hipSuccess && ptr != nullptr && g_region_base != 0 &&
      g_region_size > 0) {
    const std::uint64_t va = reinterpret_cast<std::uint64_t>(ptr);
    // Only record mappings that fall inside the deterministic region. The HIP
    // runtime also maps its own internal VA ranges; those are not the workload's
    // buffers and must not become replay events.
    if (va >= g_region_base && va < g_region_base + g_region_size) {
      const std::lock_guard<std::mutex> lock(g_mu);
      snapshot::AllocEvent ev;
      ev.offset = va - g_region_base;
      ev.size = size;
      ev.tag = "map";
      g_alloc_events.push_back(ev);
    }
  }
  return status;
}

// ---- host-registered kernels (<<<>>> / hipLaunchKernel path) ----------------
//
// PyTorch/ATen kernels are registered at static-init: the compiler emits a call
// to __hipRegisterFatBinary (which returns a fatCubinHandle) and, per kernel,
// __hipRegisterFunction(handle, hostFunc, deviceFunc, deviceName, ...). They
// are later launched by host pointer via hipLaunchKernel/<<<>>>. The probe on
// real vLLM showed this is the path for the majority of captured nodes; without
// these interpositions, ~55-65%% of nodes have no recoverable identity.

void** __hipRegisterFatBinary(const void* data) {
  probe_once("__hipRegisterFatBinary");
  using Fn = void** (*)(const void*);
  static Fn real = resolve<Fn>("__hipRegisterFatBinary");
  void** handle = real(data);
  const bool want = g_graphs_written.load(std::memory_order_relaxed) <
                        max_graphs() ||
                    g_active_captures.load(std::memory_order_relaxed) > 0;
  if (handle != nullptr && data != nullptr && want) {
    // FatBinWrapper: { u32 magic "HPIF"; u32 version; const void* binary; ... }
    // The HSACO code objects live in the __CLANG_OFFLOAD_BUNDLE__ at `binary`.
    const auto* base = static_cast<const unsigned char*>(data);
    std::uint64_t binptr;
    std::memcpy(&binptr, base + 8, 8);
    FatbinRec rec;
    if (binptr > 0x1000 && binptr < 0x7fffffffffffULL) {
      // Scan a generous but bounded range for AMDGPU ELFs (each is bounded by
      // its own header via elf_code_object_size, so only the magic search reads
      // "blind"; 256 MiB covers any plausible PyTorch fat binary).
      rec.elf_images = extract_amdgpu_elfs(reinterpret_cast<void*>(binptr),
                                           1ULL << 28, device_gfx_token());
      if (!rec.elf_images.empty()) {
        rec.primary_hash = snapshot::hash_bytes(rec.elf_images[0].data(),
                                                rec.elf_images[0].size());
      } else if (verbose()) {
        std::fprintf(stderr,
                     "[record] __hipRegisterFatBinary: no AMDGPU ELF in bundle "
                     "(binary=%p); host kernels in this fatbin won't rebuild\n",
                     reinterpret_cast<void*>(binptr));
      }
    }
    const std::lock_guard<std::mutex> lock(g_mu);
    g_fatbins[reinterpret_cast<std::uintptr_t>(handle)] = std::move(rec);
  }
  return handle;
}

void __hipUnregisterFatBinary(void** handle) {
  probe_once("__hipUnregisterFatBinary");
  using Fn = void (*)(void**);
  static Fn real = resolve<Fn>("__hipUnregisterFatBinary");
  if (handle != nullptr) {
    const std::lock_guard<std::mutex> lock(g_mu);
    g_fatbins.erase(reinterpret_cast<std::uintptr_t>(handle));
  }
  real(handle);
}

void __hipRegisterFunction(void** modules, const void* hostFunction,
                           char* deviceFunction, const char* deviceName,
                           unsigned int threadLimit, void* tid, void* bid,
                           void* blockDim, void* gridDim, int* wSize) {
  probe_once("__hipRegisterFunction");
  using Fn = void (*)(void**, const void*, char*, const char*, unsigned int,
                      void*, void*, void*, void*, int*);
  static Fn real = resolve<Fn>("__hipRegisterFunction");
  const bool want = g_graphs_written.load(std::memory_order_relaxed) <
                        max_graphs() ||
                    g_active_captures.load(std::memory_order_relaxed) > 0;
  if (modules != nullptr && hostFunction != nullptr && want) {
    HostFuncRec rec;
    rec.fatbin_handle = reinterpret_cast<std::uintptr_t>(modules);
    rec.device_name = deviceName != nullptr ? std::string(deviceName)
                                            : std::string();
    {
      const std::lock_guard<std::mutex> lock(g_mu);
      rec.id = g_next_func_id++;
      g_host_functions[reinterpret_cast<std::uintptr_t>(hostFunction)] =
          std::move(rec);
    }
  }
  real(modules, hostFunction, deviceFunction, deviceName, threadLimit, tid, bid,
       blockDim, gridDim, wSize);
}

// ---- diagnostic probes: which launch / graph-build paths does vLLM use? -------
// These forward-only probes quantify which HIP entry points produce the
// captured kernel nodes, so the remaining identity gap localizes to a specific
// symbol instead of a guess. They do not record (introspection at EndCapture is
// authoritative); the probe_once line in the log is the signal.

hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks,
                           dim3 dimBlocks, void** args,
                           std::size_t sharedMemBytes, hipStream_t stream) {
  probe_once("hipLaunchKernel");
  using Fn = hipError_t (*)(const void*, dim3, dim3, void**, std::size_t,
                            hipStream_t);
  static Fn real = resolve<Fn>("hipLaunchKernel");
  // Record this host launch's identity, in issue order. The captured node's func
  // is an opaque device handle, but here we still hold the host function pointer
  // __hipRegisterFunction gave us. Fast-path gated inside record_issue.
  if (g_active_captures.load(std::memory_order_relaxed) > 0) {
    g_cap_hooks.host.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::byte> kernarg;
    if (capture_args_enabled()) {
      // hipLaunchKernel exposes only the array-format `args`; capture it now
      // (the buffer is caller-owned and reused on return).
      kernarg = pack_kernel_args_array(args);
    }
    record_issue(stream, lookup_host_fid(function_address), std::move(kernarg));
  }
  // Main-thread naming opportunity (IDLE mode, post-sentinel): the inference
  // path calls this thousands of times; the first call after /health drains the
  // whole naming queue safely on this thread.
  drain_naming_inline();
  return real(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
              stream);
}

// The aiter/Tensile launch path (and how vLLM's captured GEMM + many fused
// kernels reach the graph). Declared here matching hip_ext.h exactly so the C
// symbol is interposed; we record the function's identity in issue order like
// any other launch. f is a hipFunction_t (module kernels resolve via
// g_functions; others stay 0 and are characterized by the reconcile log).
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, std::uint32_t gWSx,
                                    std::uint32_t gWSy, std::uint32_t gWSz,
                                    std::uint32_t lWSx, std::uint32_t lWSy,
                                    std::uint32_t lWSz, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams,
                                    void** extra, hipEvent_t startEvent,
                                    hipEvent_t stopEvent, std::uint32_t flags) {
  probe_once("hipExtModuleLaunchKernel");
  using Fn = hipError_t (*)(hipFunction_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, size_t, hipStream_t, void**, void**,
                            hipEvent_t, hipEvent_t, std::uint32_t);
  static Fn real = resolve<Fn>("hipExtModuleLaunchKernel");
  if (g_active_captures.load(std::memory_order_relaxed) > 0) {
    g_cap_hooks.ext.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::byte> args;
    if (capture_args_enabled()) {
      args = extra ? pack_kernel_args_buffer(extra)
                   : pack_kernel_args_array(kernelParams);
    }
    record_issue(hStream, lookup_module_fid(f), std::move(args));
  }
  // Main-thread naming opportunity (IDLE mode, post-sentinel).
  drain_naming_inline();
  return real(f, gWSx, gWSy, gWSz, lWSx, lWSy, lWSz, sharedMemBytes, hStream,
              kernelParams, extra, startEvent, stopEvent, flags);
}

hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* dependencies,
                                std::size_t numDependencies,
                                const hipKernelNodeParams* nodeParams) {
  probe_once("hipGraphAddKernelNode");
  using Fn = hipError_t (*)(hipGraphNode_t*, hipGraph_t, const hipGraphNode_t*,
                            std::size_t, const hipKernelNodeParams*);
  static Fn real = resolve<Fn>("hipGraphAddKernelNode");
  if (nodeParams != nullptr && nodeParams->func != nullptr) {
    if (g_active_captures.load(std::memory_order_relaxed) > 0) {
      g_cap_hooks.addnode.fetch_add(1, std::memory_order_relaxed);
    }
    // Record the func this directly-added node uses, so it can be correlated
    // like any captured launch (covers torch.compile/inductor building graph
    // nodes explicitly rather than via stream capture of <<<>>> launches).
    const std::lock_guard<std::mutex> lock(g_mu);
    auto fit = g_functions.find(reinterpret_cast<std::uintptr_t>(nodeParams->func));
    auto hfit = g_host_functions.find(reinterpret_cast<std::uintptr_t>(nodeParams->func));
    if (verbose() && fit == g_functions.end() && hfit == g_host_functions.end()) {
      std::fprintf(stderr,
                   "[record] hipGraphAddKernelNode func=0x%llx NOT in any map\n",
                   static_cast<unsigned long long>(
                       reinterpret_cast<std::uintptr_t>(nodeParams->func)));
    }
  }
  return real(pGraphNode, graph, dependencies, numDependencies, nodeParams);
}

// Forward-only probes for the remaining module/cooperative launch variants
// (Composable-Kernel / cooperative-launch paths). The probe_once line tells us
// which one vLLM's CK kernels actually use, so the identity gap localizes.
hipError_t hipLaunchCooperativeKernel(const void* function_address, dim3 numBlocks,
                                      dim3 dimBlocks, void** args,
                                      unsigned int sharedMemBytes,
                                      hipStream_t stream) {
  probe_once("hipLaunchCooperativeKernel");
  using Fn = hipError_t (*)(const void*, dim3, dim3, void**, unsigned int,
                            hipStream_t);
  static Fn real = resolve<Fn>("hipLaunchCooperativeKernel");
  if (g_active_captures.load(std::memory_order_relaxed) > 0) {
    g_cap_hooks.coop_host.fetch_add(1, std::memory_order_relaxed);
    record_issue(stream, lookup_host_fid(function_address));
  }
  return real(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
              stream);
}

// Signature must match the ROCm header exactly (single void** kernelParams, no
// trailing `extra`) or the C function declaration conflicts and fails to build.
hipError_t hipModuleLaunchCooperativeKernel(
    hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz, unsigned int sharedMem,
    hipStream_t stream, void** kernelParams) {
  probe_once("hipModuleLaunchCooperativeKernel");
  using Fn = hipError_t (*)(hipFunction_t, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, hipStream_t, void**);
  static Fn real = resolve<Fn>("hipModuleLaunchCooperativeKernel");
  if (g_active_captures.load(std::memory_order_relaxed) > 0) {
    g_cap_hooks.coop_mod.fetch_add(1, std::memory_order_relaxed);
    record_issue(stream, lookup_module_fid(f));
  }
  return real(f, gx, gy, gz, bx, by, bz, sharedMem, stream, kernelParams);
}

// ---- M3a.6: ROCr/HSA layer interposition (de-risk only, now disabled) -------
// The HSA prototype proved identity is recoverable, but the bridge turned out to
// be hipKernelNameRef (HIP layer), NOT the HSA kernel_object map — and iterating
// 9683 symbols on EVERY freeze badly slowed vLLM init (it never reached idle, so
// the off-path namer couldn't drain). So the freeze interpose is now a pass-
// through; the HSA executable-symbol path stays as the documented M3b fallback
// for code objects of any kernel whose HSACO we never see via the HIP module API.
hsa_status_t hsa_executable_freeze(hsa_executable_t executable,
                                   const char* options) {
  using Fn = hsa_status_t (*)(hsa_executable_t, const char*);
  static Fn real = resolve<Fn>("hsa_executable_freeze");
  return real(executable, options);
}

// M3b: capture HSACO bytes loaded directly through the ROCr runtime. Triton JIT
// compiles each fused kernel to an HSACO and registers it here (NOT via
// hipModuleLoadData — confirmed: no such FIRST probe, and the triton kernel
// names are absent from every HIP-captured module/fatbin). The HIP module API
// interpositions never see these, so the graph's Triton nodes had no code object
// to rebuild from. This interpose copies the full HSACO (the API gives us a
// size, unlike hipModuleLoadData) once, at load — cheap, and not on the hot
// freeze path that previously stalled init. Kernel names are extracted from the
// AMDGPU metadata note at FLUSH and fed into the same name -> image map.
hsa_status_t hsa_code_object_reader_create_from_memory(
    const void* code_object_data, std::size_t code_object_data_size,
    hsa_code_object_reader_t* code_object_reader) {
  probe_once("hsa_code_object_reader_create_from_memory");
  using Fn = hsa_status_t (*)(const void*, std::size_t,
                              hsa_code_object_reader_t*);
  static Fn real = resolve<Fn>("hsa_code_object_reader_create_from_memory");
  const hsa_status_t rc =
      real(code_object_data, code_object_data_size, code_object_reader);
  const bool want = g_graphs_written.load(std::memory_order_relaxed) <
                        max_graphs() ||
                    g_active_captures.load(std::memory_order_relaxed) > 0;
  if (rc == 0 /*HSA_STATUS_SUCCESS*/ && code_object_reader != nullptr &&
      code_object_data != nullptr && code_object_data_size >= 64 && want) {
    const std::uint64_t handle = code_object_reader->handle;
    const std::byte* img = static_cast<const std::byte*>(code_object_data);
    // The HSA API hands us the AUTHORITATIVE size (unlike hipModuleLoadData,
    // which passes none), so copy exactly that many bytes. Bounding by
    // elf_code_object_size here would TRUNCATE trailing sections/notes (the
    // helper returns max(ehsize, sh-table end, ph-table end), which can stop
    // before a trailing .note / metadata segment), yielding an HSACO whose
    // kernel .text is incomplete -> "no kernel image available" at rebuild.
    const std::uint64_t elfsz = code_object_data_size;
    HsaImage rec;
    rec.image.assign(img, img + elfsz);
    rec.hash = snapshot::hash_bytes(rec.image.data(), rec.image.size());
    const bool has_triton =
        std::search(rec.image.data(), rec.image.data() + rec.image.size(),
                    reinterpret_cast<const std::byte*>("triton_"),
                    reinterpret_cast<const std::byte*>("triton_") + 7) !=
        rec.image.data() + rec.image.size();
    {
      const std::lock_guard<std::mutex> lock(g_mu);
      if (g_hsa_images.size() < 8192) {
        g_hsa_images.emplace(handle, std::move(rec));
      } else {
        ++g_hsa_capture_skipped;
      }
    }
    // Debug dump (SNAPSHOT_RECORD_DUMP_HSA=1): write each captured HSACO so we
    // can readelf/objdump it and confirm format + target arch when a rebuild
    // fails with "no kernel image available".
    if (std::getenv("SNAPSHOT_RECORD_DUMP_HSA") != nullptr) {
      const std::string dir = out_dir();
      if (!dir.empty()) {
        char dpath[1024];
        std::snprintf(dpath, sizeof(dpath), "%s/hsa-dump-0x%llx.co",
                      dir.c_str(),
                      static_cast<unsigned long long>(
                          snapshot::hash_bytes(img, elfsz)));
        std::ofstream df(dpath, std::ios::binary);
        df.write(reinterpret_cast<const char*>(img),
                 static_cast<std::streamsize>(elfsz));
      }
    }
    if (verbose() && has_triton) {
      std::fprintf(stderr,
                   "[record] HSA reader handle=0x%llx size=%llu hash=0x%llx "
                   "has_triton=1 (captured)\n",
                   static_cast<unsigned long long>(handle),
                   static_cast<unsigned long long>(elfsz),
                   static_cast<unsigned long long>(
                       g_hsa_images.count(handle)
                           ? g_hsa_images[handle].hash
                           : 0));
      std::fflush(stderr);
    }
  }
  return rc;
}

// ---- diagnostic: which launch API issues the unmatched (host) kernels? ------
// host_issue from hipLaunchKernel covered only a fraction of the captured host
// nodes (COUNT MISMATCH), so most launch through a different entry point. These
// forward-only probes count calls DURING active capture and report whether the
// launched function is a registered host kernel or a module function, so the
// culprit API and its identity source localize in one run. No recording yet.
void diag_launch(const char* api, const void* key, bool key_is_host_ptr) {
  if (!verbose() || g_active_captures.load(std::memory_order_relaxed) == 0) {
    return;
  }
  static std::atomic<std::uint64_t> n{0};
  const std::uint64_t i = n.fetch_add(1);
  if (i >= 40) {
    return;
  }
  bool in_host, in_mod;
  {
    const std::lock_guard<std::mutex> lock(g_mu);
    in_host = g_host_functions.count(reinterpret_cast<std::uintptr_t>(key)) > 0;
    in_mod = g_functions.count(reinterpret_cast<std::uintptr_t>(key)) > 0;
  }
  std::fprintf(stderr,
               "[record] CAP %s key=0x%llx (%s) in_host=%d in_mod=%d\n", api,
               static_cast<unsigned long long>(
                   reinterpret_cast<std::uintptr_t>(key)),
               key_is_host_ptr ? "hostptr" : "hipFunc", in_host ? 1 : 0,
               in_mod ? 1 : 0);
  std::fflush(stderr);
}

hipError_t hipLaunchByPtr(const void* func) {
  probe_once("hipLaunchByPtr");
  using Fn = hipError_t (*)(const void*);
  static Fn real = resolve<Fn>("hipLaunchByPtr");
  diag_launch("hipLaunchByPtr", func, /*key_is_host_ptr=*/true);
  return real(func);
}

hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, const void* fPtr,
                              void** args_param) {
  probe_once("hipLaunchKernelExC");
  using Fn = hipError_t (*)(const hipLaunchConfig_t*, const void*, void**);
  static Fn real = resolve<Fn>("hipLaunchKernelExC");
  diag_launch("hipLaunchKernelExC", fPtr, /*key_is_host_ptr=*/true);
  if (g_active_captures.load(std::memory_order_relaxed) > 0 && config != nullptr) {
    g_cap_hooks.exc.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::byte> kernarg;
    if (capture_args_enabled()) {
      kernarg = pack_kernel_args_array(args_param);
    }
    record_issue(config->stream, lookup_host_fid(fPtr), std::move(kernarg));
  }
  return real(config, fPtr, args_param);
}

hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, hipFunction_t f,
                                void** params, void** extra) {
  probe_once("hipDrvLaunchKernelEx");
  using Fn = hipError_t (*)(const HIP_LAUNCH_CONFIG*, hipFunction_t, void**,
                            void**);
  static Fn real = resolve<Fn>("hipDrvLaunchKernelEx");
  diag_launch("hipDrvLaunchKernelEx", reinterpret_cast<const void*>(f),
              /*key_is_host_ptr=*/false);
  return real(config, f, params, extra);
}

hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks,
                              dim3 dimBlocks, void** args, size_t sharedMemBytes,
                              hipStream_t stream, hipEvent_t startEvent,
                              hipEvent_t stopEvent, int flags) {
  probe_once("hipExtLaunchKernel");
  using Fn = hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t,
                            hipEvent_t, hipEvent_t, int);
  static Fn real = resolve<Fn>("hipExtLaunchKernel");
  diag_launch("hipExtLaunchKernel", function_address, /*key_is_host_ptr=*/true);
  return real(function_address, numBlocks, dimBlocks, args, sharedMemBytes,
              stream, startEvent, stopEvent, flags);
}

}  // extern "C"
