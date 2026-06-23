#pragma once

#include <string>

#include "snapshot/allocator.hpp"
#include "snapshot/gpu_backend.hpp"
#include "snapshot/snapshot_format.hpp"

namespace snapshot {

class CaptureSession {
 public:
  CaptureSession(GpuBackend& backend, std::uint64_t region_size);

  Status begin();
  Status finish(const std::vector<ModuleImage>& modules, GraphHandle graph,
                SnapshotData* out);

  DeterministicAllocator& allocator() { return allocator_; }
  const DeterministicAllocator& allocator() const { return allocator_; }

 private:
  GpuBackend& backend_;
  std::uint64_t region_size_ = 0;
  DeterministicAllocator allocator_;
};

}  // namespace snapshot
