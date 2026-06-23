#include "cuda_backend.hpp"

#include "snapshot/workload_kernels.hpp"

namespace snapshot {

std::unique_ptr<GpuBackend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();
}

// TODO: implement nvrtc compilation of the synthetic module for the CUDA
// backend. Stubbed so the CUDA build links; CUDA validation is a later
// milestone (Clariden GH200 / Bristen A100).
Status compile_synthetic_module(std::vector<std::byte>* /*image*/,
                                std::vector<std::string>* /*entry_names*/) {
  return Status::unsupported(
      "synthetic kernel compilation is not yet implemented for CUDA");
}

}  // namespace snapshot
