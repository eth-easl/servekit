#include "cuda_backend.hpp"

namespace snapshot {

Status CudaBackend::arch(ArchInfo* out) {
  if (out == nullptr) {
    return Status::invalid_argument("arch output is null");
  }
  return Status::unsupported("CUDA backend scaffold is not implemented yet");
}

Status CudaBackend::get_allocation_granularity(std::uint64_t* /*out*/) {
  return Status::unsupported("CUDA VMM scaffold is not implemented yet");
}

Status CudaBackend::reserve_address(std::uint64_t /*size*/,
                                    std::uint64_t /*alignment*/,
                                    std::uint64_t /*requested_base*/,
                                    std::uint64_t* /*out_base*/) {
  return Status::unsupported("CUDA VMM scaffold is not implemented yet");
}

Status CudaBackend::release_address(std::uint64_t /*base*/,
                                    std::uint64_t /*size*/) {
  return Status::unsupported("CUDA VMM scaffold is not implemented yet");
}

Status CudaBackend::create_physical(std::uint64_t /*size*/, MemHandle* /*out*/) {
  return Status::unsupported("CUDA VMM scaffold is not implemented yet");
}

Status CudaBackend::release_physical(MemHandle /*handle*/) {
  return Status::unsupported("CUDA VMM scaffold is not implemented yet");
}

Status CudaBackend::map(std::uint64_t /*va*/, std::uint64_t /*size*/,
                        std::uint64_t /*offset*/, MemHandle /*handle*/) {
  return Status::unsupported("CUDA VMM scaffold is not implemented yet");
}

Status CudaBackend::unmap(std::uint64_t /*va*/, std::uint64_t /*size*/) {
  return Status::unsupported("CUDA VMM scaffold is not implemented yet");
}

Status CudaBackend::set_access(std::uint64_t /*va*/, std::uint64_t /*size*/,
                               const MemoryAccess& /*access*/) {
  return Status::unsupported("CUDA VMM scaffold is not implemented yet");
}

}  // namespace snapshot
