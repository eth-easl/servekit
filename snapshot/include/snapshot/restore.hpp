#pragma once

#include "snapshot/allocator.hpp"
#include "snapshot/gpu_backend.hpp"
#include "snapshot/relocation.hpp"
#include "snapshot/snapshot_format.hpp"

namespace snapshot {

struct RestoreResult {
  std::uint64_t restored_base = 0;
  RelocationStats relocation_stats;
  GraphHandle graph;
  GraphExecHandle exec;
};

class RestoreSession {
 public:
  explicit RestoreSession(GpuBackend& backend);

  Status restore(const SnapshotData& snapshot, RestoreResult* out);

  DeterministicAllocator& allocator() { return allocator_; }

 private:
  GpuBackend& backend_;
  DeterministicAllocator allocator_;
};

}  // namespace snapshot
