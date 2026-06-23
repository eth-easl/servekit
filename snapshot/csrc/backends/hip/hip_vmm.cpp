#include "hip_backend.hpp"

#include <sstream>

#include <hip/hip_runtime_api.h>

namespace snapshot {
namespace {

Status hip_status(hipError_t error, const char* call) {
  if (error == hipSuccess) {
    return Status::Ok();
  }
  std::ostringstream message;
  message << call << " failed: " << hipGetErrorString(error);
  return Status::backend(message.str());
}

hipMemGenericAllocationHandle_t to_physical(MemHandle handle) {
  return reinterpret_cast<hipMemGenericAllocationHandle_t>(handle.value);
}

MemHandle from_physical(hipMemGenericAllocationHandle_t handle) {
  MemHandle out;
  out.value = reinterpret_cast<std::uintptr_t>(handle);
  return out;
}

// The physical allocation and its access grant must target the device the
// caller is actually using. Hardcoding device 0 breaks when more than one GPU
// is visible (e.g. a vLLM worker bound to a non-zero ordinal), where
// hipMemSetAccess then fails with invalid argument.
int current_device() {
  int device = 0;
  hipGetDevice(&device);
  return device;
}

}  // namespace

Status HipBackend::arch(ArchInfo* out) {
  if (out == nullptr) {
    return Status::invalid_argument("arch output is null");
  }
  int device = 0;
  hipDeviceProp_t props{};
  Status status = hip_status(hipGetDevice(&device), "hipGetDevice");
  if (!status.ok()) {
    return status;
  }
  status = hip_status(hipGetDeviceProperties(&props, device),
                      "hipGetDeviceProperties");
  if (!status.ok()) {
    return status;
  }
  out->name = props.gcnArchName;
  return Status::Ok();
}

Status HipBackend::get_allocation_granularity(std::uint64_t* out) {
  if (out == nullptr) {
    return Status::invalid_argument("granularity output is null");
  }
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = current_device();
  std::size_t granularity = 0;
  Status status = hip_status(hipMemGetAllocationGranularity(
                                 &granularity, &prop,
                                 hipMemAllocationGranularityRecommended),
                             "hipMemGetAllocationGranularity");
  if (!status.ok()) {
    return status;
  }
  *out = granularity;
  return Status::Ok();
}

Status HipBackend::reserve_address(std::uint64_t size, std::uint64_t alignment,
                                   std::uint64_t requested_base,
                                   std::uint64_t* out_base) {
  if (out_base == nullptr) {
    return Status::invalid_argument("reserve output is null");
  }
  void* ptr = reinterpret_cast<void*>(requested_base);
  Status status = hip_status(hipMemAddressReserve(&ptr, size, alignment, ptr, 0),
                             "hipMemAddressReserve");
  if (!status.ok()) {
    return status;
  }
  *out_base = reinterpret_cast<std::uint64_t>(ptr);
  return Status::Ok();
}

Status HipBackend::release_address(std::uint64_t base, std::uint64_t size) {
  return hip_status(hipMemAddressFree(reinterpret_cast<void*>(base), size),
                    "hipMemAddressFree");
}

Status HipBackend::create_physical(std::uint64_t size, MemHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("physical output is null");
  }
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = current_device();
  prop.requestedHandleTypes = hipMemHandleTypeNone;
  hipMemGenericAllocationHandle_t handle{};
  Status status = hip_status(hipMemCreate(&handle, size, &prop, 0),
                             "hipMemCreate");
  if (!status.ok()) {
    return status;
  }
  *out = from_physical(handle);
  return Status::Ok();
}

Status HipBackend::release_physical(MemHandle handle) {
  return hip_status(hipMemRelease(to_physical(handle)), "hipMemRelease");
}

Status HipBackend::map(std::uint64_t va, std::uint64_t size,
                       std::uint64_t offset, MemHandle handle) {
  return hip_status(hipMemMap(reinterpret_cast<void*>(va), size, offset,
                              to_physical(handle), 0),
                    "hipMemMap");
}

Status HipBackend::unmap(std::uint64_t va, std::uint64_t size) {
  return hip_status(hipMemUnmap(reinterpret_cast<void*>(va), size),
                    "hipMemUnmap");
}

Status HipBackend::set_access(std::uint64_t va, std::uint64_t size,
                              const MemoryAccess& access) {
  hipMemAccessDesc desc{};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = access.device_ordinal != 0 ? access.device_ordinal
                                                 : current_device();
  desc.flags = access.read && access.write
                   ? hipMemAccessFlagsProtReadWrite
                   : (access.read ? hipMemAccessFlagsProtRead
                                  : hipMemAccessFlagsProtNone);
  return hip_status(hipMemSetAccess(reinterpret_cast<void*>(va), size, &desc, 1),
                    "hipMemSetAccess");
}

}  // namespace snapshot
