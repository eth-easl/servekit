#include "hip_backend.hpp"

namespace snapshot {

std::unique_ptr<GpuBackend> make_hip_backend() {
  return std::make_unique<HipBackend>();
}

}  // namespace snapshot
