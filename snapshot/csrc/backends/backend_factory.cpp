#include "snapshot/gpu_backend.hpp"

#include <map>
#include <memory>
#include <set>

#include "snapshot/workload_kernels.hpp"

namespace snapshot {
namespace {

class StubBackend final : public GpuBackend {
 public:
  Vendor vendor() const override { return Vendor::kStub; }

  Status arch(ArchInfo* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("arch output is null");
    }
    out->name = "stub-host";
    return Status::Ok();
  }

  Status get_allocation_granularity(std::uint64_t* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("granularity output is null");
    }
    *out = 4096;
    return Status::Ok();
  }

  Status reserve_address(std::uint64_t size, std::uint64_t alignment,
                         std::uint64_t requested_base,
                         std::uint64_t* out_base) override {
    if (out_base == nullptr) {
      return Status::invalid_argument("reserve output is null");
    }
    if (size == 0) {
      return Status::invalid_argument("reserve size is zero");
    }
    if (alignment == 0) {
      alignment = 4096;
    }
    std::uint64_t base = requested_base == 0 ? 0x600000000000ULL : requested_base;
    base += reserve_count_ * 0x100000000ULL;
    base = ((base + alignment - 1) / alignment) * alignment;
    ++reserve_count_;
    reservations_[base] = size;
    *out_base = base;
    return Status::Ok();
  }

  Status release_address(std::uint64_t base, std::uint64_t /*size*/) override {
    reservations_.erase(base);
    return Status::Ok();
  }

  Status create_physical(std::uint64_t size, MemHandle* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("physical memory output is null");
    }
    if (size == 0) {
      return Status::invalid_argument("physical memory size is zero");
    }
    MemHandle handle;
    handle.value = next_handle_++;
    physical_.insert(handle.value);
    *out = handle;
    return Status::Ok();
  }

  Status release_physical(MemHandle handle) override {
    physical_.erase(handle.value);
    return Status::Ok();
  }

  Status map(std::uint64_t va, std::uint64_t size, std::uint64_t /*offset*/,
             MemHandle handle) override {
    if (!handle.valid() || physical_.count(handle.value) == 0) {
      return Status::invalid_argument("unknown physical memory handle");
    }
    mappings_[va] = size;
    return Status::Ok();
  }

  Status unmap(std::uint64_t va, std::uint64_t /*size*/) override {
    mappings_.erase(va);
    return Status::Ok();
  }

  Status set_access(std::uint64_t /*va*/, std::uint64_t /*size*/,
                    const MemoryAccess& /*access*/) override {
    return Status::Ok();
  }

  Status load_module(const std::byte* image, std::size_t n,
                     ModuleHandle* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("module output is null");
    }
    if (image == nullptr && n != 0) {
      return Status::invalid_argument("module image pointer is null");
    }
    ModuleHandle handle;
    handle.value = next_handle_++;
    modules_.insert(handle.value);
    *out = handle;
    return Status::Ok();
  }

  Status unload_module(ModuleHandle module) override {
    modules_.erase(module.value);
    return Status::Ok();
  }

  Status get_function(ModuleHandle module, const std::string& /*name*/,
                      FunctionHandle* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("function output is null");
    }
    if (modules_.count(module.value) == 0) {
      return Status::invalid_argument("unknown module handle");
    }
    FunctionHandle handle;
    handle.value = next_handle_++;
    *out = handle;
    return Status::Ok();
  }

  Status stream_create(StreamHandle* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("stream output is null");
    }
    StreamHandle handle;
    handle.value = next_handle_++;
    streams_.insert(handle.value);
    *out = handle;
    return Status::Ok();
  }

  Status stream_destroy(StreamHandle stream) override {
    streams_.erase(stream.value);
    return Status::Ok();
  }

  Status begin_capture(StreamHandle stream) override {
    if (streams_.count(stream.value) == 0) {
      return Status::invalid_argument("unknown stream handle");
    }
    capturing_stream_ = stream.value;
    return Status::Ok();
  }

  Status end_capture(StreamHandle stream, GraphHandle* out_graph) override {
    if (out_graph == nullptr) {
      return Status::invalid_argument("graph output is null");
    }
    if (capturing_stream_ != stream.value) {
      return Status::invalid_argument("stream is not being captured");
    }
    GraphHandle handle;
    handle.value = next_handle_++;
    graphs_[handle.value] = captured_graph_;
    captured_graph_ = {};
    capturing_stream_ = 0;
    *out_graph = handle;
    return Status::Ok();
  }

  Status introspect_graph(GraphHandle graph, GraphIR* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("graph IR output is null");
    }
    auto it = graphs_.find(graph.value);
    if (it == graphs_.end()) {
      return Status::invalid_argument("unknown graph handle");
    }
    *out = it->second;
    return Status::Ok();
  }

  Status rebuild_graph(const GraphIR& ir, GraphHandle* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("graph output is null");
    }
    GraphHandle handle;
    handle.value = next_handle_++;
    graphs_[handle.value] = ir;
    *out = handle;
    return Status::Ok();
  }

  Status instantiate(GraphHandle graph, GraphExecHandle* out) override {
    if (out == nullptr) {
      return Status::invalid_argument("graph exec output is null");
    }
    if (graphs_.count(graph.value) == 0) {
      return Status::invalid_argument("unknown graph handle");
    }
    GraphExecHandle handle;
    handle.value = next_handle_++;
    execs_[handle.value] = graph.value;
    *out = handle;
    return Status::Ok();
  }

  Status exec_set_kernel_node_params(GraphExecHandle exec,
                                     std::uint64_t /*node_id*/,
                                     const KernelLaunchParams& /*params*/) override {
    if (execs_.count(exec.value) == 0) {
      return Status::invalid_argument("unknown graph exec handle");
    }
    return Status::Ok();
  }

  Status launch(GraphExecHandle exec, StreamHandle stream) override {
    if (execs_.count(exec.value) == 0 || streams_.count(stream.value) == 0) {
      return Status::invalid_argument("unknown exec or stream handle");
    }
    return Status::Ok();
  }

  Status synchronize(StreamHandle stream) override {
    if (streams_.count(stream.value) == 0) {
      return Status::invalid_argument("unknown stream handle");
    }
    return Status::Ok();
  }

  Status launch_kernel(StreamHandle stream,
                       const KernelLaunchParams& params) override {
    if (streams_.count(stream.value) == 0) {
      return Status::invalid_argument("unknown stream handle");
    }
    if (capturing_stream_ == stream.value) {
      GraphNodeIR node;
      node.id = static_cast<std::uint64_t>(captured_graph_.nodes.size() + 1);
      node.type = GraphNodeType::kKernel;
      node.kernel = params;
      captured_graph_.nodes.push_back(std::move(node));
    }
    return Status::Ok();
  }

  Status memcpy_h2d(std::uint64_t /*dst_device*/, const void* src_host,
                    std::uint64_t bytes, StreamHandle stream) override {
    if (bytes != 0 && src_host == nullptr) {
      return Status::invalid_argument("host source pointer is null");
    }
    return synchronize(stream);
  }

  Status memcpy_d2h(void* dst_host, std::uint64_t /*src_device*/,
                    std::uint64_t bytes, StreamHandle stream) override {
    if (bytes != 0 && dst_host == nullptr) {
      return Status::invalid_argument("host destination pointer is null");
    }
    return synchronize(stream);
  }

 private:
  std::uint64_t reserve_count_ = 0;
  std::uintptr_t next_handle_ = 1;
  std::uintptr_t capturing_stream_ = 0;
  GraphIR captured_graph_;
  std::map<std::uint64_t, std::uint64_t> reservations_;
  std::map<std::uint64_t, std::uint64_t> mappings_;
  std::map<std::uintptr_t, GraphIR> graphs_;
  std::map<std::uintptr_t, std::uintptr_t> execs_;
  std::set<std::uintptr_t> physical_;
  std::set<std::uintptr_t> modules_;
  std::set<std::uintptr_t> streams_;
};

}  // namespace

// These are defined in the backend translation units with external linkage, so
// their declarations must live outside the anonymous namespace above.
#if SNAPSHOT_BACKEND_HIP
std::unique_ptr<GpuBackend> make_hip_backend();
#endif

#if SNAPSHOT_BACKEND_CUDA
std::unique_ptr<GpuBackend> make_cuda_backend();
#endif

std::unique_ptr<GpuBackend> make_backend() {
#if SNAPSHOT_BACKEND_HIP
  return make_hip_backend();
#elif SNAPSHOT_BACKEND_CUDA
  return make_cuda_backend();
#else
  return std::make_unique<StubBackend>();
#endif
}

std::unique_ptr<GpuBackend> make_stub_backend() {
  return std::make_unique<StubBackend>();
}

#if !SNAPSHOT_BACKEND_HIP && !SNAPSHOT_BACKEND_CUDA
// The stub build has no real device, so there is no code object to compile.
Status compile_synthetic_module(std::vector<std::byte>* /*image*/,
                                std::vector<std::string>* /*entry_names*/) {
  return Status::unsupported(
      "synthetic kernel compilation requires the HIP or CUDA backend");
}
#endif

}  // namespace snapshot
