#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "snapshot/gpu_backend.hpp"
#include "snapshot/status.hpp"

namespace snapshot {

constexpr std::uint64_t kDefaultRequestedBase = 0x600000000000ULL;

struct AllocEvent {
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  std::string tag;
};

struct AllocatorState {
  std::uint64_t requested_base = kDefaultRequestedBase;
  std::uint64_t region_base = 0;
  std::uint64_t region_size = 0;
  std::uint64_t cursor = 0;
  std::uint64_t granularity = 0;
  bool fixed_base_honored = false;
  std::vector<AllocEvent> events;
};

class DeterministicAllocator {
 public:
  DeterministicAllocator() = default;

  Status init(GpuBackend& backend, std::uint64_t region_size,
              std::uint64_t requested_base = kDefaultRequestedBase);
  Status alloc(GpuBackend& backend, std::uint64_t bytes,
               const std::string& tag, std::uint64_t* out_device_va);
  Status replay(GpuBackend& backend, const AllocatorState& captured,
                std::uint64_t requested_base, std::uint64_t* out_base);
  Status release(GpuBackend& backend);

  const AllocatorState& state() const { return state_; }
  AllocatorState& mutable_state() { return state_; }

  static std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment);

 private:
  AllocatorState state_;
  std::vector<MemHandle> physical_;
};

}  // namespace snapshot
