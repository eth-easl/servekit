#include "snapshot/allocator.hpp"

namespace snapshot {

std::uint64_t DeterministicAllocator::round_up(std::uint64_t value,
                                               std::uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return value + (alignment - remainder);
}

Status DeterministicAllocator::init(GpuBackend& backend,
                                    std::uint64_t region_size,
                                    std::uint64_t requested_base) {
  if (region_size == 0) {
    return Status::invalid_argument("allocator region size must be non-zero");
  }

  std::uint64_t granularity = 0;
  Status status = backend.get_allocation_granularity(&granularity);
  if (!status.ok()) {
    return status;
  }
  if (granularity == 0) {
    return Status::backend("backend returned zero allocation granularity");
  }

  const std::uint64_t rounded_region = round_up(region_size, granularity);
  std::uint64_t region_base = 0;
  status = backend.reserve_address(rounded_region, granularity, requested_base,
                                   &region_base);
  if (!status.ok()) {
    return status;
  }

  state_ = {};
  physical_.clear();
  state_.requested_base = requested_base;
  state_.region_base = region_base;
  state_.region_size = rounded_region;
  state_.granularity = granularity;
  state_.fixed_base_honored = (region_base == requested_base);
  return Status::Ok();
}

Status DeterministicAllocator::alloc(GpuBackend& backend, std::uint64_t bytes,
                                     const std::string& tag,
                                     std::uint64_t* out_device_va) {
  if (out_device_va == nullptr) {
    return Status::invalid_argument("alloc output pointer is null");
  }
  if (bytes == 0) {
    return Status::invalid_argument("allocation size must be non-zero");
  }
  if (state_.region_base == 0 || state_.granularity == 0) {
    return Status::invalid_argument("allocator is not initialized");
  }

  const std::uint64_t aligned = round_up(bytes, state_.granularity);
  if (aligned > state_.region_size ||
      state_.cursor > state_.region_size - aligned) {
    return {StatusCode::kOutOfMemory,
            "deterministic allocator region exhausted"};
  }

  const std::uint64_t offset = state_.cursor;
  const std::uint64_t va = state_.region_base + offset;
  if (va < state_.region_base) {
    return Status::overflow("device virtual address overflow");
  }

  MemHandle physical;
  Status status = backend.create_physical(aligned, &physical);
  if (!status.ok()) {
    return status;
  }
  status = backend.map(va, aligned, 0, physical);
  if (!status.ok()) {
    backend.release_physical(physical);
    return status;
  }
  status = backend.set_access(va, aligned, MemoryAccess{});
  if (!status.ok()) {
    backend.unmap(va, aligned);
    backend.release_physical(physical);
    return status;
  }

  physical_.push_back(physical);
  state_.events.push_back(AllocEvent{offset, aligned, tag});
  state_.cursor += aligned;
  *out_device_va = va;
  return Status::Ok();
}

Status DeterministicAllocator::replay(GpuBackend& backend,
                                      const AllocatorState& captured,
                                      std::uint64_t requested_base,
                                      std::uint64_t* out_base) {
  if (captured.region_size == 0 || captured.granularity == 0) {
    return Status::invalid_argument("captured allocator state is incomplete");
  }
  if (out_base == nullptr) {
    return Status::invalid_argument("replay output pointer is null");
  }

  std::uint64_t region_base = 0;
  Status status = backend.reserve_address(captured.region_size,
                                          captured.granularity, requested_base,
                                          &region_base);
  if (!status.ok()) {
    return status;
  }

  state_ = captured;
  state_.requested_base = requested_base;
  state_.region_base = region_base;
  state_.fixed_base_honored = (region_base == requested_base);
  physical_.clear();

  for (const AllocEvent& event : captured.events) {
    if (event.size == 0 || event.offset > captured.region_size ||
        event.size > captured.region_size - event.offset) {
      release(backend);
      return Status::format("allocation event exceeds captured region");
    }

    MemHandle physical;
    status = backend.create_physical(event.size, &physical);
    if (!status.ok()) {
      release(backend);
      return status;
    }

    const std::uint64_t va = region_base + event.offset;
    status = backend.map(va, event.size, 0, physical);
    if (!status.ok()) {
      backend.release_physical(physical);
      release(backend);
      return status;
    }
    status = backend.set_access(va, event.size, MemoryAccess{});
    if (!status.ok()) {
      backend.unmap(va, event.size);
      backend.release_physical(physical);
      release(backend);
      return status;
    }
    physical_.push_back(physical);
  }

  *out_base = region_base;
  return Status::Ok();
}

Status DeterministicAllocator::release(GpuBackend& backend) {
  Status first_error = Status::Ok();

  for (const AllocEvent& event : state_.events) {
    if (state_.region_base != 0 && event.size != 0) {
      Status status = backend.unmap(state_.region_base + event.offset,
                                    event.size);
      if (!status.ok() && first_error.ok()) {
        first_error = status;
      }
    }
  }

  for (MemHandle handle : physical_) {
    Status status = backend.release_physical(handle);
    if (!status.ok() && first_error.ok()) {
      first_error = status;
    }
  }
  physical_.clear();

  if (state_.region_base != 0 && state_.region_size != 0) {
    Status status = backend.release_address(state_.region_base,
                                            state_.region_size);
    if (!status.ok() && first_error.ok()) {
      first_error = status;
    }
  }

  state_ = {};
  return first_error;
}

}  // namespace snapshot
