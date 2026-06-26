#pragma once

#include "snapshot/gpu_backend.hpp"

namespace snapshot {

// Establishes the device-0 primary CUDA context once (driver API requires
// explicit cuInit + a current context). Defined in cuda_vmm.cpp; called by every
// CudaBackend entry point that issues driver calls.
Status ensure_cuda_context();

class CudaBackend final : public GpuBackend {
 public:
  Vendor vendor() const override { return Vendor::kCuda; }
  Status arch(ArchInfo* out) override;
  Status get_allocation_granularity(std::uint64_t* out) override;
  Status reserve_address(std::uint64_t size, std::uint64_t alignment,
                         std::uint64_t requested_base,
                         std::uint64_t* out_base) override;
  Status release_address(std::uint64_t base, std::uint64_t size) override;
  Status create_physical(std::uint64_t size, MemHandle* out) override;
  Status release_physical(MemHandle handle) override;
  Status map(std::uint64_t va, std::uint64_t size, std::uint64_t offset,
             MemHandle handle) override;
  Status unmap(std::uint64_t va, std::uint64_t size) override;
  Status set_access(std::uint64_t va, std::uint64_t size,
                    const MemoryAccess& access) override;
  Status load_module(const std::byte* image, std::size_t n,
                     ModuleHandle* out) override;
  Status unload_module(ModuleHandle module) override;
  Status get_function(ModuleHandle module, const std::string& name,
                      FunctionHandle* out) override;
  Status stream_create(StreamHandle* out) override;
  Status stream_destroy(StreamHandle stream) override;
  Status begin_capture(StreamHandle stream) override;
  Status end_capture(StreamHandle stream, GraphHandle* out_graph) override;
  Status introspect_graph(GraphHandle graph, GraphIR* out) override;
  Status rebuild_graph(const GraphIR& ir, GraphHandle* out) override;
  Status instantiate(GraphHandle graph, GraphExecHandle* out) override;
  Status exec_set_kernel_node_params(GraphExecHandle exec, std::uint64_t node_id,
                                     const KernelLaunchParams& params) override;
  Status launch(GraphExecHandle exec, StreamHandle stream) override;
  Status synchronize(StreamHandle stream) override;
  Status launch_kernel(StreamHandle stream,
                       const KernelLaunchParams& params) override;
  Status memcpy_h2d(std::uint64_t dst_device, const void* src_host,
                    std::uint64_t bytes, StreamHandle stream) override;
  Status memcpy_d2h(void* dst_host, std::uint64_t src_device,
                    std::uint64_t bytes, StreamHandle stream) override;
};

}  // namespace snapshot
