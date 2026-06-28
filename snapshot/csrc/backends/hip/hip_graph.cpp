#include "hip_backend.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <hip/hip_runtime_api.h>

namespace snapshot {
namespace {

Status hip_status(hipError_t error, const char* call) {
  if (error == hipSuccess) {
    return Status::Ok();
  }
  std::ostringstream message;
  message << call << " failed: " << hipGetErrorString(error);
  return Status::backend(message.str());
}

template <typename T>
T as_handle(OpaqueHandle handle) {
  return reinterpret_cast<T>(handle.value);
}

template <typename Handle, typename T>
Handle from_handle(T value) {
  Handle out;
  out.value = reinterpret_cast<std::uintptr_t>(value);
  return out;
}

// A rebuilt kernel node references its packed argument buffer through the
// driver-style `extra` launch config. That buffer (and the config array that
// points at it) must outlive the hipGraphAddKernelNode call and stay alive
// until the graph is instantiated, after which the executable graph owns a
// baked copy. We park it in a registry keyed by the graph handle and drop it
// inside instantiate(). Heap-stable storage (unique_ptr) keeps the pointers in
// `config` valid as more nodes are added.
//
// Two kernarg formats are recorded at launch time (see snapshot_record.cpp's
// pack_kernel_args_*):
//   tag 0 (buffer): one contiguous kernarg blob -> params.extra
//   tag 1 (array) : per-arg byte slices -> params.kernelParams (NULL-term)
// The recorder always emits a tag as the first byte; this decoder picks the
// right HIP node-param layout. Pointers in arg_ptrs point into arg_storage,
// which is populated to its final size before arg_ptrs is built and never
// resized afterwards, so the addresses stay valid through instantiate().
struct NodeParam {
  std::uint8_t tag = 0;       // 0=buffer, 1=array
  // tag 0 storage:
  std::vector<std::byte> blob;
  std::size_t size = 0;
  std::array<void*, 5> config{};  // {PTR, &blob, SIZE, &size, END}
  // tag 1 storage:
  std::vector<std::vector<std::byte>> arg_storage;
  std::vector<void*> arg_ptrs;    // N+1 entries; arg_ptrs[N] = nullptr (END)
  std::vector<std::byte> zero_pad;  // crash-safety backing for undercount pad
};

// Decode the recorder's tagged kernarg blob into a NodeParam. `exact_arg_count`
// (0 = unknown) overrides the blind 32-arg crash-safety pad: when the parsed
// AMDGPU metadata gives the kernel's true arg count, pad precisely to that
// count instead of blindly to 32. Returns false (and leaves an empty
// buffer-format param) on any parse error so the rebuild proceeds but the node
// fails at instantiate with a clear size-0 signal.
bool decode_recorded_args(const std::vector<std::byte>& src, NodeParam& out,
                          std::size_t exact_arg_count = 0) {
  if (src.empty()) {
    // No launch-time args were captured for this node (its launch did not
    // reach a hooked API during the capture window, or the positional merge
    // had no blob for it). Represent it as a zero-arg kernel node via the
    // array format: kernelParams -> {nullptr} (null-terminated empty array),
    // extra=null. This is the form hipGraphAddKernelNode accepts for a no-arg
    // node — unlike launch-time extra, the node-params extra does NOT honor
    // the {HIP_LAUNCH_PARAM_END} terminator on ROCm (END=0x03), so pointing
    // extra at {END,...} crashes AddKernelNode (regression in job 523212).
    //
    // However, if the parsed signature says the kernel HAS args (N>0), a true
    // zero-arg {nullptr} may be rejected by AddKernelNode (it reads N entries
    // from kernelParams and would deref the NULL terminator). Pad to N zero-
    // valued slots so the node matches the signature's count (wrong values, but
    // structurally valid — same philosophy as the crash-safety pad below).
    out.tag = 1;
    if (exact_arg_count == 0) {
      out.arg_ptrs.assign(1, nullptr);  // null-terminated empty array
    } else {
      out.zero_pad.assign(64, std::byte{0});
      out.arg_ptrs.reserve(exact_arg_count + 1);
      for (std::size_t i = 0; i < exact_arg_count; ++i) {
        out.arg_ptrs.push_back(out.zero_pad.data());
      }
      out.arg_ptrs.push_back(nullptr);  // terminator
    }
    return true;
  }
  auto rd_u32 = [&](std::size_t off) -> std::uint32_t {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(src[off])) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(src[off + 1]))
            << 8) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(src[off + 2]))
            << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(src[off + 3]))
            << 24);
  };
  out.tag = static_cast<std::uint8_t>(src[0]);
  if (out.tag == 0) {
    // {tag:0, u32 size, bytes[size]}
    if (src.size() < 5) return false;
    const std::uint32_t sz = rd_u32(1);
    if (static_cast<std::size_t>(sz) + 5 > src.size()) return false;
    out.blob.assign(src.begin() + 5, src.begin() + 5 + sz);
    out.size = sz;
    out.config = {HIP_LAUNCH_PARAM_BUFFER_POINTER, out.blob.data(),
                  HIP_LAUNCH_PARAM_BUFFER_SIZE, &out.size,
                  HIP_LAUNCH_PARAM_END};
    return true;
  }
  if (out.tag == 1) {
    // {tag:1, u32 count, per arg: {u32 per_bytes, bytes[per_bytes]}}
    if (src.size() < 5) return false;
    const std::uint32_t count = rd_u32(1);
    std::size_t off = 5;
    out.arg_storage.reserve(count);
    for (std::uint32_t a = 0; a < count; ++a) {
      if (off + 4 > src.size()) {
        if (std::getenv("SNAPSHOT_REBUILD_DEBUG")) {
          std::fprintf(stderr, "[rebuild] decode FAIL tag=1 count=%u a=%u "
                       "off=%zu blobsz=%zu (header overrun)\n",
                       count, a, off, src.size());
        }
        return false;
      }
      const std::uint32_t per = rd_u32(off);
      off += 4;
      if (off + per > src.size()) {
        if (std::getenv("SNAPSHOT_REBUILD_DEBUG")) {
          std::fprintf(stderr, "[rebuild] decode FAIL tag=1 count=%u a=%u "
                       "per=%u off=%zu blobsz=%zu (data overrun)\n",
                       count, a, per, off, src.size());
        }
        return false;
      }
      out.arg_storage.emplace_back(src.begin() + off,
                                   src.begin() + off + per);
      off += per;
    }
    // Build the NULL-terminated pointer array AFTER arg_storage has its final
    // size, so each arg_storage[i].data() address is stable for instantiate().
    out.arg_ptrs.reserve(count + 1);
    for (auto& v : out.arg_storage) out.arg_ptrs.push_back(v.data());
    out.arg_ptrs.push_back(nullptr);
    // Crash-safety pad: if the captured argcnt UNDERcounts the kernel's real
    // signature (seen for aiter fused_qk_rmsnorm: 12 captured vs 15 declared —
    // a real arg pointer probed unreadable mid-array), HIP reads the
    // signature's full count and would hit our NULL terminator early, then
    // dereference NULL. Pad with a zero-valued backing buffer so an undercount
    // yields zeros (wrong but safe) instead of a NULL-deref segfault. The
    // trailing real NULL terminator still sits at the end.
    //
    // When the parsed AMDGPU metadata supplies the exact arg count, pad to
    // EXACTLY that count (e.g. 15). If the captured count already meets or
    // exceeds it, no padding is needed. When the count is unknown (0), fall
    // back to the blind 32-arg crash-safety pad.
    const std::size_t pad_target =
        exact_arg_count > 0 ? exact_arg_count : 32;
    out.zero_pad.assign(64, std::byte{0});  // covers any single arg ≤ 64 B
    for (std::size_t i = count; i < pad_target; ++i) {
      out.arg_ptrs.back() = out.zero_pad.data();  // overwrite prior NULL
      out.arg_ptrs.push_back(nullptr);            // new terminator
    }
    return true;
  }
  return false;  // unknown tag
}

struct GraphSideData {
  std::vector<std::unique_ptr<NodeParam>> nodes;
};

std::map<std::uintptr_t, GraphSideData>& graph_side_registry() {
  static std::map<std::uintptr_t, GraphSideData> registry;
  return registry;
}

GraphNodeType from_hip_node_type(hipGraphNodeType type) {
  switch (type) {
    case hipGraphNodeTypeKernel:
      return GraphNodeType::kKernel;
    case hipGraphNodeTypeMemcpy:
      return GraphNodeType::kMemcpyD2D;
    case hipGraphNodeTypeMemset:
      return GraphNodeType::kMemset;
    default:
      return GraphNodeType::kKernel;
  }
}

}  // namespace

Status HipBackend::load_module(const std::byte* image, std::size_t n,
                               ModuleHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("module output is null");
  }
  if (image == nullptr || n == 0) {
    return Status::invalid_argument("module image is empty");
  }
  // Optionally patch the over-restrictive .max_flat_workgroup_size that
  // Triton's ROCm codegen emits for num_warps=8 pointwise kernels (it reports
  // 256 for a kernel compiled for/launched at 512). Eager launch ignores the
  // field; hipGraphAddKernelNode enforces block.x<=MAX_THREADS_PER_BLOCK and
  // rejects such nodes. Bumping to the device max (1024 on MI300A) makes the
  // code object self-consistent. Gated by SNAPSHOT_HSACO_PATCH_MAXWG=<threads>.
  // The patch is a same-width in-place msgpack int rewrite (no offset shifts).
  const std::byte* load_image = image;
  std::vector<std::byte> patched;
  if (const char* tg = std::getenv("SNAPSHOT_HSACO_PATCH_MAXWG")) {
    const long targetl = std::strtol(tg, nullptr, 10);
    if (targetl > 0) {
      const std::uint32_t target = static_cast<std::uint32_t>(targetl);
      patched.assign(image, image + n);
      std::size_t cnt = patch_amdgpu_max_flat_workgroup_size(
          patched.data(), patched.size(), target);
      if (cnt > 0) {
        if (const char* d = std::getenv("SNAPSHOT_DEBUG")) {
          if (d[0] == '1') {
            std::fprintf(stderr,
                         "[hsaco-patch] rewrote %zu max_flat_workgroup_size "
                         "fields -> %u\n",
                         cnt, target);
          }
        }
        load_image = patched.data();
      }
    }
  }
  hipModule_t module{};
  Status status = hip_status(hipModuleLoadData(&module, load_image),
                             "hipModuleLoadData");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<ModuleHandle>(module);
  // Parse kernel signatures from the ELF's AMDGPU metadata note so rebuild_graph
  // knows each kernel's exact arg count / pointer offsets. Best-effort: a parse
  // failure (e.g. code-object v3 YAML, or no metadata note) leaves the name
  // absent and rebuild falls back to the captured count / blind-scan reloc.
  // NOTE: parse from the ORIGINAL image (unchanged metadata either way; the
  // patch only touches max_flat_workgroup_size, which KernelSig ignores).
  for (const KernelSig& ks : extract_amdgpu_kernels(image, n)) {
    sig_by_name_[ks.name] = ks;
  }
  return Status::Ok();
}

Status HipBackend::unload_module(ModuleHandle module) {
  return hip_status(hipModuleUnload(as_handle<hipModule_t>(module)),
                    "hipModuleUnload");
}

Status HipBackend::get_function(ModuleHandle module, const std::string& name,
                                FunctionHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("function output is null");
  }
  hipFunction_t function{};
  Status status = hip_status(hipModuleGetFunction(
                                 &function, as_handle<hipModule_t>(module),
                                 name.c_str()),
                             "hipModuleGetFunction");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<FunctionHandle>(function);
  return Status::Ok();
}

Status HipBackend::stream_create(StreamHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("stream output is null");
  }
  hipStream_t stream{};
  Status status = hip_status(hipStreamCreate(&stream), "hipStreamCreate");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<StreamHandle>(stream);
  return Status::Ok();
}

Status HipBackend::stream_destroy(StreamHandle stream) {
  return hip_status(hipStreamDestroy(as_handle<hipStream_t>(stream)),
                    "hipStreamDestroy");
}

Status HipBackend::begin_capture(StreamHandle stream) {
  return hip_status(hipStreamBeginCapture(as_handle<hipStream_t>(stream),
                                          hipStreamCaptureModeGlobal),
                    "hipStreamBeginCapture");
}

Status HipBackend::end_capture(StreamHandle stream, GraphHandle* out_graph) {
  if (out_graph == nullptr) {
    return Status::invalid_argument("graph output is null");
  }
  hipGraph_t graph{};
  Status status = hip_status(hipStreamEndCapture(as_handle<hipStream_t>(stream),
                                                 &graph),
                             "hipStreamEndCapture");
  if (!status.ok()) {
    return status;
  }
  *out_graph = from_handle<GraphHandle>(graph);
  return Status::Ok();
}

Status HipBackend::introspect_graph(GraphHandle graph, GraphIR* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph IR output is null");
  }
  hipGraph_t hip_graph = as_handle<hipGraph_t>(graph);

  std::size_t count = 0;
  Status status = hip_status(hipGraphGetNodes(hip_graph, nullptr, &count),
                             "hipGraphGetNodes(count)");
  if (!status.ok()) {
    return status;
  }

  std::vector<hipGraphNode_t> nodes(count);
  if (count > 0) {
    status = hip_status(hipGraphGetNodes(hip_graph, nodes.data(), &count),
                        "hipGraphGetNodes");
    if (!status.ok()) {
      return status;
    }
  }

  // Introspection recovers the graph's structure (node count + node types) for
  // validation against the recorded IR. It cannot recover a kernel's identity
  // (module/entry) or arguments: a captured node holds only an opaque, process-
  // local function pointer, which is exactly why launch identity must be
  // recorded at issue time rather than reverse-engineered from the graph.
  out->nodes.clear();
  out->edges.clear();
  for (std::size_t i = 0; i < count; ++i) {
    hipGraphNodeType node_type{};
    status = hip_status(hipGraphNodeGetType(nodes[i], &node_type),
                        "hipGraphNodeGetType");
    if (!status.ok()) {
      return status;
    }
    GraphNodeIR node;
    node.id = static_cast<std::uint64_t>(i + 1);
    node.type = from_hip_node_type(node_type);
    out->nodes.push_back(std::move(node));
  }
  return Status::Ok();
}

Status HipBackend::rebuild_graph(const GraphIR& ir, GraphHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph output is null");
  }

  hipGraph_t graph{};
  Status status = hip_status(hipGraphCreate(&graph, 0), "hipGraphCreate");
  if (!status.ok()) {
    return status;
  }

  GraphSideData side;
  std::map<std::uint64_t, hipGraphNode_t> node_by_id;
  static const bool dbg = [] {
    const char* e = std::getenv("SNAPSHOT_REBUILD_DEBUG");
    return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
  }();

  for (const GraphNodeIR& node : ir.nodes) {
    if (node.type != GraphNodeType::kKernel) {
      static_cast<void>(hipGraphDestroy(graph));
      return Status::unsupported(
          "rebuild_graph currently supports kernel nodes only");
    }
    if (!node.kernel.function.valid()) {
      static_cast<void>(hipGraphDestroy(graph));
      return Status::invalid_argument(
          "kernel node is missing a resolved function handle");
    }

    auto param = std::make_unique<NodeParam>();
    // Look up the kernel's parsed signature for the EXACT declared arg count
    // (replaces the blind 32-arg crash-safety pad). Absent when the module's
    // metadata couldn't be parsed (v3 YAML, or no note) — then exact_count=0
    // and decode_recorded_args falls back to the blind pad.
    std::size_t exact_count = 0;
    auto sit = sig_by_name_.find(node.entry_name);
    if (sit != sig_by_name_.end()) {
      exact_count = sit->second.args.size();
    }
    const bool ok = decode_recorded_args(node.kernel.param_blob, *param,
                                         exact_count);
    static std::atomic<int> g_node_seq{0};
    const int seq = g_node_seq.fetch_add(1, std::memory_order_relaxed);
    if (dbg) {
      std::fprintf(stderr,
                   "[rebuild] node#%d name='%s' func_ok=%d argtag=%d "
                   "argcnt=%zu sigargc=%zu blob=%zu grid=%ux%ux%u block=%ux%ux%u\n",
                   seq, node.entry_name.c_str(),
                   node.kernel.function.valid() ? 1 : 0, param->tag,
                   param->arg_ptrs.empty() ? 0 : param->arg_ptrs.size() - 1,
                   exact_count, node.kernel.param_blob.size(), node.kernel.grid.x,
                   node.kernel.grid.y, node.kernel.grid.z, node.kernel.block.x,
                   node.kernel.block.y, node.kernel.block.z);
      std::fflush(stderr);
      // M3k: sharedMemBytes is validated by hipGraphAddKernelNode against the
      // device/kernel limit — an over-large value faults there WITHOUT
      // launching (reported as launch failure). Print it per node.
      std::fprintf(stderr, "[rebuild] node#%d shared=%u func=%p\n", seq,
                   node.kernel.shared_mem_bytes,
                   (void*)as_handle<hipFunction_t>(node.kernel.function));
      std::fflush(stderr);
      // M3k: dump per-arg values for tag-1 (array) nodes to see whether the
      // failing node's pointer args fall inside the mapped region or are
      // unmapped (trajectory-mismatch fault).
      if (param->tag == 1 && !param->arg_storage.empty()) {
        std::fprintf(stderr, "[rebuild] node#%d args:", seq);
        for (std::size_t a = 0; a < param->arg_storage.size(); ++a) {
          const auto& v = param->arg_storage[a];
          std::uint64_t lo = 0;
          for (std::size_t b = 0; b < v.size() && b < 8; ++b)
            lo |= static_cast<std::uint64_t>(
                       static_cast<unsigned char>(v[b]))
                  << (8 * b);
          std::fprintf(stderr, " [a%zu sz=%zu v0=0x%016llx]", a, v.size(),
                       static_cast<unsigned long long>(lo));
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
      }
    }

    hipKernelNodeParams params{};
    params.func = as_handle<hipFunction_t>(node.kernel.function);
    params.gridDim = dim3{node.kernel.grid.x, node.kernel.grid.y,
                          node.kernel.grid.z};
    params.blockDim = dim3{node.kernel.block.x, node.kernel.block.y,
                           node.kernel.block.z};
    params.sharedMemBytes = node.kernel.shared_mem_bytes;
    // tag 1 -> array format (kernelParams); tag 0 (or empty) -> buffer (extra).
    // When decode fails entirely (corrupted/truncated blob), fall back to a
    // safe zero-arg array format — NEVER pass an uninitialized config vector
    // as extra (that segfaults inside hipGraphAddKernelNode).
    if (ok && param->tag == 1) {
      params.kernelParams = param->arg_ptrs.data();
      params.extra = nullptr;
    } else if (ok && param->tag == 0 && !param->config.empty()) {
      params.kernelParams = nullptr;
      params.extra = param->config.data();
    } else {
      // Decode failed or unknown tag: use zero-arg array with sig-count pad.
      param->tag = 1;
      const std::size_t pad_target = exact_count > 0 ? exact_count : 32;
      param->zero_pad.assign(64, std::byte{0});
      param->arg_ptrs.assign(pad_target, param->zero_pad.data());
      param->arg_ptrs.push_back(nullptr);  // terminator
      params.kernelParams = param->arg_ptrs.data();
      params.extra = nullptr;
    }
    if (dbg) {
      std::fprintf(stderr, "[rebuild] node#%d calling hipGraphAddKernelNode kp=%p "
                   "extra=%p\n", seq, params.kernelParams, params.extra);
      std::fflush(stderr);
    }

    std::vector<hipGraphNode_t> deps;
    for (const GraphEdgeIR& edge : ir.edges) {
      if (edge.to == node.id) {
        auto it = node_by_id.find(edge.from);
        if (it != node_by_id.end()) {
          deps.push_back(it->second);
        }
      }
    }

    hipGraphNode_t graph_node{};
    // M3k sticky probe: peek (non-clearing) for a sticky HIP error left by a
    // PRIOR node's AddKernelNode (intra-graph stickiness), then optionally
    // clear it so THIS node gets a fair attempt. Distinguishes node#4's own
    // fault from a poisoned state inherited from nodes 0-3.
    static const bool sticky_dbg =
        std::getenv("SNAPSHOT_REBUILD_STICKY_DBG");
    static const bool sticky_clear =
        std::getenv("SNAPSHOT_REBUILD_CLEAR_STICKY");
    if (sticky_dbg || sticky_clear) {
      hipError_t peek = hipPeekAtLastError();
      if (sticky_dbg) {
        std::fprintf(stderr, "[rebuild] node#%d pre-add peek=%d(%s)\n", seq,
                     (int)peek, hipGetErrorString(peek));
        std::fflush(stderr);
      }
      if (sticky_clear) (void)hipGetLastError();  // clear
    }
    // M3k mechanism probe: query the kernel's resource footprint via
    // hipFuncGetAttribute and print it. Tests the hypothesis that the failing
    // pointwise@8-warp kernel trips a per-block resource cap (static LDS /
    // registers) that hipGraphAddKernelNode enforces at add-time but eager
    // launch does not. Comparing OK nodes (reduction@8, pointwise@4) vs the
    // FAIL node (pointwise@8) on these attributes localizes the cause.
    if (dbg) {
      hipFunction_t f = (hipFunction_t)params.func;
      auto q = [&](hipFunction_attribute a) -> int {
        int v = -999; hipFuncGetAttribute(&v, a, f);
        return v;
      };
      std::fprintf(stderr,
                   "[rebuild] node#%d attr regs=%d static_shared=%d "
                   "max_dyn_shared=%d max_threads=%d local=%d\n",
                   seq, q(HIP_FUNC_ATTRIBUTE_NUM_REGS),
                   q(HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
                   q(HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
                   q(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
                   q(HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES));
      std::fflush(stderr);
    }
    status = hip_status(
        hipGraphAddKernelNode(&graph_node, graph, deps.data(), deps.size(),
                              &params),
        "hipGraphAddKernelNode");
    if (dbg) {
      std::fprintf(stderr, "[rebuild] node#%d hipGraphAddKernelNode rc=%d\n", seq,
                   status.ok() ? 0 : -1);
      std::fflush(stderr);
    }
    if (!status.ok()) {
      static_cast<void>(hipGraphDestroy(graph));
      return status;
    }
    node_by_id[node.id] = graph_node;
    side.nodes.push_back(std::move(param));
  }

  graph_side_registry()[reinterpret_cast<std::uintptr_t>(graph)] =
      std::move(side);
  *out = from_handle<GraphHandle>(graph);
  return Status::Ok();
}

Status HipBackend::instantiate(GraphHandle graph, GraphExecHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph exec output is null");
  }
  hipGraphExec_t exec{};
  Status status = hip_status(hipGraphInstantiate(&exec,
                                                 as_handle<hipGraph_t>(graph),
                                                 nullptr, nullptr, 0),
                             "hipGraphInstantiate");
  if (!status.ok()) {
    return status;
  }
  // The executable graph now owns baked copies of every node's parameters, so
  // the rebuild-time param buffers can be released.
  graph_side_registry().erase(
      reinterpret_cast<std::uintptr_t>(as_handle<hipGraph_t>(graph)));
  *out = from_handle<GraphExecHandle>(exec);
  return Status::Ok();
}

Status HipBackend::exec_set_kernel_node_params(
    GraphExecHandle /*exec*/, std::uint64_t /*node_id*/,
    const KernelLaunchParams& /*params*/) {
  return Status::unsupported(
      "HIP graph exec kernel parameter update needs node-handle mapping");
}

Status HipBackend::launch(GraphExecHandle exec, StreamHandle stream) {
  return hip_status(hipGraphLaunch(as_handle<hipGraphExec_t>(exec),
                                   as_handle<hipStream_t>(stream)),
                    "hipGraphLaunch");
}

Status HipBackend::synchronize(StreamHandle stream) {
  return hip_status(hipStreamSynchronize(as_handle<hipStream_t>(stream)),
                    "hipStreamSynchronize");
}

Status HipBackend::launch_kernel(StreamHandle stream,
                                 const KernelLaunchParams& params) {
  // Decode the recorder's tagged kernarg blob. For the array format (tag 1) we
  // must build a NULL-terminated kernelParams[] in stable storage that outlives
  // the hipModuleLaunchKernel call; for the buffer format (tag 0) we pass extra.
  // arg_storage / arg_ptrs are kept on this stack frame so their addresses are
  // valid through the call below.
  NodeParam np{};
  const bool ok = decode_recorded_args(params.param_blob, np);
  void* extra[5] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, np.blob.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &np.size,
                    HIP_LAUNCH_PARAM_END};
  void** kp = (ok && np.tag == 1) ? np.arg_ptrs.data() : nullptr;
  void** ex = (ok && np.tag == 1) ? nullptr : extra;
  return hip_status(hipModuleLaunchKernel(
                        as_handle<hipFunction_t>(params.function),
                        params.grid.x, params.grid.y, params.grid.z,
                        params.block.x, params.block.y, params.block.z,
                        params.shared_mem_bytes, as_handle<hipStream_t>(stream),
                        kp, ex),
                    "hipModuleLaunchKernel");
}

Status HipBackend::memcpy_h2d(std::uint64_t dst_device, const void* src_host,
                              std::uint64_t bytes, StreamHandle stream) {
  return hip_status(hipMemcpyAsync(reinterpret_cast<void*>(dst_device),
                                   src_host, bytes, hipMemcpyHostToDevice,
                                   as_handle<hipStream_t>(stream)),
                    "hipMemcpyAsync H2D");
}

Status HipBackend::memcpy_d2h(void* dst_host, std::uint64_t src_device,
                              std::uint64_t bytes, StreamHandle stream) {
  return hip_status(hipMemcpyAsync(dst_host, reinterpret_cast<void*>(src_device),
                                   bytes, hipMemcpyDeviceToHost,
                                   as_handle<hipStream_t>(stream)),
                    "hipMemcpyAsync D2H");
}

}  // namespace snapshot
