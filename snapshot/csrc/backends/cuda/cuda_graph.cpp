#include "cuda_backend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <cuda.h>

namespace snapshot {
namespace {

Status cu_status(CUresult error, const char* call) {
  if (error == CUDA_SUCCESS) {
    return Status::Ok();
  }
  const char* msg = nullptr;
  cuGetErrorString(error, &msg);
  std::ostringstream message;
  message << call << " failed: " << (msg ? msg : "unknown CUDA error");
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

GraphNodeType from_cuda_node_type(CUgraphNodeType type) {
  switch (type) {
    case CU_GRAPH_NODE_TYPE_KERNEL:
      return GraphNodeType::kKernel;
    case CU_GRAPH_NODE_TYPE_MEMCPY:
      return GraphNodeType::kMemcpyD2D;
    case CU_GRAPH_NODE_TYPE_MEMSET:
      return GraphNodeType::kMemset;
    default:
      return GraphNodeType::kKernel;
  }
}

// CUDA's cuLaunchKernel (and cuGraphAddKernelNode) validate the kernarg buffer
// against the kernel's DECLARED argument size: an oversized buffer is rejected
// (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES on a plain stream, or
// CUDA_ERROR_INVALID_VALUE during graph capture). The synthetic workload pads
// some blobs past the true arg size (in_place: 12-byte signature padded to 16).
// HIP tolerates this because hipModuleLaunchKernel copies only the declared
// args into an internally aligned kernarg segment; CUDA does not. Walk
// cuFuncGetParamInfo (CUDA >= 12.4) to find the kernel's true total argument
// size and build an exactly-sized launch buffer. If the query yields nothing
// (older CUDA / no params), fall back to the source blob size verbatim.
std::size_t kernarg_size_for(CUfunction f) {
  std::size_t total = 0;
  for (std::size_t i = 0;; ++i) {
    std::size_t off = 0;
    std::size_t sz = 0;
    if (cuFuncGetParamInfo(f, i, &off, &sz) != CUDA_SUCCESS) {
      break;
    }
    const std::size_t end = off + sz;
    if (end > total) {
      total = end;
    }
  }
  return total;
}

std::vector<std::byte> exact_kernarg_buffer(CUfunction f,
                                            const std::vector<std::byte>& src) {
  std::size_t want = kernarg_size_for(f);
  if (want == 0) {
    want = src.size();  // fallback: trust the recorded blob size
  }
  std::vector<std::byte> out(want, std::byte{0});
  const std::size_t n = std::min(src.size(), want);
  if (n > 0) {
    std::memcpy(out.data(), src.data(), n);
  }
  return out;
}

// A rebuilt kernel node references its kernarg buffer through the driver `extra`
// launch config. That buffer, the `size` it points at, and the config array must
// outlive cuGraphAddKernelNode and stay valid until the graph is instantiated
// (after which the executable graph owns a baked copy). Park them in a registry
// keyed by the graph handle; drop it in instantiate(). unique_ptr gives each
// NodeParam a stable address as more nodes are added.
//
// The synthetic workload records a FLAT kernarg buffer (param_blob) padded to
// the kernel's exact kernarg-segment size, with ptr_offsets marking pointer
// args for relocation. This mirrors how the HIP backend's recorded tag-0 buffer
// format is rebuilt, but CUDA needs no tag decoding: the synthetic workload and
// the CLI's restore path both hand rebuild_graph a raw flat kernarg buffer
// (pointers already relocated by Δ by relocate_graph_ir).
struct NodeParam {
  std::vector<std::byte> blob;
  std::size_t size = 0;
  void* config[5] = {};  // {BUFFER_POINTER, blob.data(), BUFFER_SIZE, &size, END}
};

struct GraphSideData {
  std::vector<std::unique_ptr<NodeParam>> nodes;
};

std::map<std::uintptr_t, GraphSideData>& graph_side_registry() {
  static std::map<std::uintptr_t, GraphSideData> registry;
  return registry;
}

}  // namespace

Status CudaBackend::load_module(const std::byte* image, std::size_t n,
                                ModuleHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("module output is null");
  }
  if (image == nullptr || n == 0) {
    return Status::invalid_argument("module image is empty");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUmodule module{};
  status = cu_status(cuModuleLoadData(&module, image), "cuModuleLoadData");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<ModuleHandle>(module);
  return Status::Ok();
}

Status CudaBackend::unload_module(ModuleHandle module) {
  return cu_status(cuModuleUnload(as_handle<CUmodule>(module)),
                   "cuModuleUnload");
}

Status CudaBackend::get_function(ModuleHandle module, const std::string& name,
                                 FunctionHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("function output is null");
  }
  CUfunction function{};
  Status status = cu_status(
      cuModuleGetFunction(&function, as_handle<CUmodule>(module), name.c_str()),
      "cuModuleGetFunction");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<FunctionHandle>(function);
  return Status::Ok();
}

Status CudaBackend::stream_create(StreamHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("stream output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUstream stream{};
  status = cu_status(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                     "cuStreamCreate");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<StreamHandle>(stream);
  return Status::Ok();
}

Status CudaBackend::stream_destroy(StreamHandle stream) {
  return cu_status(cuStreamDestroy(as_handle<CUstream>(stream)),
                   "cuStreamDestroy");
}

Status CudaBackend::begin_capture(StreamHandle stream) {
  return cu_status(cuStreamBeginCapture(as_handle<CUstream>(stream),
                                        CU_STREAM_CAPTURE_MODE_GLOBAL),
                   "cuStreamBeginCapture");
}

Status CudaBackend::end_capture(StreamHandle stream, GraphHandle* out_graph) {
  if (out_graph == nullptr) {
    return Status::invalid_argument("graph output is null");
  }
  CUgraph graph{};
  Status status = cu_status(
      cuStreamEndCapture(as_handle<CUstream>(stream), &graph),
      "cuStreamEndCapture");
  if (!status.ok()) {
    return status;
  }
  *out_graph = from_handle<GraphHandle>(graph);
  return Status::Ok();
}

Status CudaBackend::introspect_graph(GraphHandle graph, GraphIR* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph IR output is null");
  }
  CUgraph cuda_graph = as_handle<CUgraph>(graph);

  std::size_t count = 0;
  Status status = cu_status(cuGraphGetNodes(cuda_graph, nullptr, &count),
                            "cuGraphGetNodes(count)");
  if (!status.ok()) {
    return status;
  }

  std::vector<CUgraphNode> nodes(count);
  if (count > 0) {
    status = cu_status(cuGraphGetNodes(cuda_graph, nodes.data(), &count),
                       "cuGraphGetNodes");
    if (!status.ok()) {
      return status;
    }
  }

  // Recovers structure (count + node types) for validation against the recorded
  // IR; a captured node's kernel identity/args are NOT recovered here (recorded
  // at issue time by the workload/interposer instead), matching the HIP backend.
  out->nodes.clear();
  out->edges.clear();
  for (std::size_t i = 0; i < count; ++i) {
    CUgraphNodeType node_type{};
    status = cu_status(cuGraphNodeGetType(nodes[i], &node_type),
                       "cuGraphNodeGetType");
    if (!status.ok()) {
      return status;
    }
    GraphNodeIR node;
    node.id = static_cast<std::uint64_t>(i + 1);
    node.type = from_cuda_node_type(node_type);
    out->nodes.push_back(std::move(node));
  }
  return Status::Ok();
}

Status CudaBackend::rebuild_graph(const GraphIR& ir, GraphHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }

  CUgraph graph{};
  status = cu_status(cuGraphCreate(&graph, 0), "cuGraphCreate");
  if (!status.ok()) {
    return status;
  }

  GraphSideData side;
  std::map<std::uint64_t, CUgraphNode> node_by_id;

  for (const GraphNodeIR& node : ir.nodes) {
    if (node.type != GraphNodeType::kKernel) {
      static_cast<void>(cuGraphDestroy(graph));
      return Status::unsupported(
          "rebuild_graph currently supports kernel nodes only");
    }
    if (!node.kernel.function.valid()) {
      static_cast<void>(cuGraphDestroy(graph));
      return Status::invalid_argument(
          "kernel node is missing a resolved function handle");
    }

    // Buffer-format kernargs: a stable copy of the (already-relocated) flat
    // kernarg blob, sized EXACTLY to the kernel's declared argument size (the
    // workload may over-pad; see exact_kernarg_buffer). Referenced via the
    // extra config array.
    auto param = std::make_unique<NodeParam>();
    param->blob = exact_kernarg_buffer(
        as_handle<CUfunction>(node.kernel.function), node.kernel.param_blob);
    param->size = param->blob.size();
    param->config[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
    param->config[1] = param->blob.data();
    param->config[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
    param->config[3] = &param->size;
    param->config[4] = CU_LAUNCH_PARAM_END;

    CUDA_KERNEL_NODE_PARAMS kparams{};
    kparams.func = as_handle<CUfunction>(node.kernel.function);
    kparams.gridDimX = node.kernel.grid.x;
    kparams.gridDimY = node.kernel.grid.y;
    kparams.gridDimZ = node.kernel.grid.z;
    kparams.blockDimX = node.kernel.block.x;
    kparams.blockDimY = node.kernel.block.y;
    kparams.blockDimZ = node.kernel.block.z;
    kparams.sharedMemBytes = node.kernel.shared_mem_bytes;
    kparams.kernelParams = nullptr;
    kparams.extra = param->config;

    std::vector<CUgraphNode> deps;
    for (const GraphEdgeIR& edge : ir.edges) {
      if (edge.to == node.id) {
        auto it = node_by_id.find(edge.from);
        if (it != node_by_id.end()) {
          deps.push_back(it->second);
        }
      }
    }

    CUgraphNode graph_node{};
    status = cu_status(
        cuGraphAddKernelNode(&graph_node, graph, deps.data(), deps.size(),
                             &kparams),
        "cuGraphAddKernelNode");
    if (!status.ok()) {
      static_cast<void>(cuGraphDestroy(graph));
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

Status CudaBackend::instantiate(GraphHandle graph, GraphExecHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph exec output is null");
  }
  CUgraphExec exec{};
  Status status = cu_status(
      cuGraphInstantiateWithFlags(&exec, as_handle<CUgraph>(graph), 0),
      "cuGraphInstantiateWithFlags");
  if (!status.ok()) {
    return status;
  }
  // The executable graph owns baked copies of every node's params now, so the
  // rebuild-time kernarg buffers can be released.
  graph_side_registry().erase(
      reinterpret_cast<std::uintptr_t>(as_handle<CUgraph>(graph)));
  *out = from_handle<GraphExecHandle>(exec);
  return Status::Ok();
}

Status CudaBackend::exec_set_kernel_node_params(
    GraphExecHandle /*exec*/, std::uint64_t /*node_id*/,
    const KernelLaunchParams& /*params*/) {
  return Status::unsupported(
      "CUDA graph exec kernel parameter update needs node-handle mapping");
}

Status CudaBackend::launch(GraphExecHandle exec, StreamHandle stream) {
  return cu_status(cuGraphLaunch(as_handle<CUgraphExec>(exec),
                                 as_handle<CUstream>(stream)),
                   "cuGraphLaunch");
}

Status CudaBackend::synchronize(StreamHandle stream) {
  return cu_status(cuStreamSynchronize(as_handle<CUstream>(stream)),
                   "cuStreamSynchronize");
}

Status CudaBackend::launch_kernel(StreamHandle stream,
                                  const KernelLaunchParams& params) {
  // The synthetic workload records a FLAT kernarg buffer (param_blob) padded to
  // a round size, with ptr_offsets marking pointer args for relocation. CUDA's
  // cuLaunchKernel rejects an oversized buffer, so size it exactly to the
  // kernel's declared argument size (exact_kernarg_buffer). The buffer must be
  // valid only for the duration of this call (the driver copies kernargs at
  // enqueue time), so a local copy is safe.
  std::vector<std::byte> blob = exact_kernarg_buffer(
      as_handle<CUfunction>(params.function), params.param_blob);
  std::size_t size = blob.size();
  void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, blob.data(),
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  return cu_status(
      cuLaunchKernel(as_handle<CUfunction>(params.function), params.grid.x,
                     params.grid.y, params.grid.z, params.block.x,
                     params.block.y, params.block.z, params.shared_mem_bytes,
                     as_handle<CUstream>(stream), nullptr, config),
      "cuLaunchKernel");
}

Status CudaBackend::memcpy_h2d(std::uint64_t dst_device, const void* src_host,
                               std::uint64_t bytes, StreamHandle stream) {
  return cu_status(
      cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(dst_device), src_host, bytes,
                        as_handle<CUstream>(stream)),
      "cuMemcpyHtoDAsync");
}

Status CudaBackend::memcpy_d2h(void* dst_host, std::uint64_t src_device,
                               std::uint64_t bytes, StreamHandle stream) {
  return cu_status(
      cuMemcpyDtoHAsync(dst_host, static_cast<CUdeviceptr>(src_device), bytes,
                        as_handle<CUstream>(stream)),
      "cuMemcpyDtoHAsync");
}

}  // namespace snapshot
