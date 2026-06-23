#include "cuda_backend.hpp"

namespace snapshot {

Status CudaBackend::load_module(const std::byte* /*image*/, std::size_t /*n*/,
                                ModuleHandle* /*out*/) {
  return Status::unsupported("CUDA module scaffold is not implemented yet");
}

Status CudaBackend::unload_module(ModuleHandle /*module*/) {
  return Status::unsupported("CUDA module scaffold is not implemented yet");
}

Status CudaBackend::get_function(ModuleHandle /*module*/,
                                 const std::string& /*name*/,
                                 FunctionHandle* /*out*/) {
  return Status::unsupported("CUDA module scaffold is not implemented yet");
}

Status CudaBackend::stream_create(StreamHandle* /*out*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::stream_destroy(StreamHandle /*stream*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::begin_capture(StreamHandle /*stream*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::end_capture(StreamHandle /*stream*/, GraphHandle* /*out*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::introspect_graph(GraphHandle /*graph*/, GraphIR* /*out*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::rebuild_graph(const GraphIR& /*ir*/, GraphHandle* /*out*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::instantiate(GraphHandle /*graph*/, GraphExecHandle* /*out*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::exec_set_kernel_node_params(
    GraphExecHandle /*exec*/, std::uint64_t /*node_id*/,
    const KernelLaunchParams& /*params*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::launch(GraphExecHandle /*exec*/, StreamHandle /*stream*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::synchronize(StreamHandle /*stream*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::launch_kernel(StreamHandle /*stream*/,
                                  const KernelLaunchParams& /*params*/) {
  return Status::unsupported("CUDA graph scaffold is not implemented yet");
}

Status CudaBackend::memcpy_h2d(std::uint64_t /*dst_device*/,
                               const void* /*src_host*/, std::uint64_t /*bytes*/,
                               StreamHandle /*stream*/) {
  return Status::unsupported("CUDA memcpy scaffold is not implemented yet");
}

Status CudaBackend::memcpy_d2h(void* /*dst_host*/, std::uint64_t /*src_device*/,
                               std::uint64_t /*bytes*/, StreamHandle /*stream*/) {
  return Status::unsupported("CUDA memcpy scaffold is not implemented yet");
}

}  // namespace snapshot
