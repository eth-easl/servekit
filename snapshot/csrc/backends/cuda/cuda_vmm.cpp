#include "cuda_backend.hpp"

#include <sstream>
#include <string>

#include <cuda.h>

namespace snapshot {
namespace {

Status cu_status(CUresult error, const char* call) {
  if (error == CUDA_SUCCESS) {
    return Status::Ok();
  }
  const char* msg = nullptr;
  cuGetErrorString(error, &msg);
  std::ostringstream message;
  message << call << " failed: " << (msg ? msg : "unknown CUDA error");
  return Status::backend(message.str());
}

CUmemGenericAllocationHandle to_physical(MemHandle handle) {
  return static_cast<CUmemGenericAllocationHandle>(handle.value);
}

MemHandle from_physical(CUmemGenericAllocationHandle handle) {
  MemHandle out;
  out.value = static_cast<std::uintptr_t>(handle);
  return out;
}

// The physical allocation and its access grant must target the device the
// caller is actually using (matches hip_vmm.cpp's current_device rationale:
// hardcoding device 0 breaks a multi-GPU worker bound to a non-zero ordinal).
int current_device() {
  CUdevice dev = 0;
  cuCtxGetDevice(&dev);
  return static_cast<int>(dev);
}

}  // namespace

Status ensure_cuda_context() {
  // Run-once: cuInit + retain device-0 primary context + make it current. The
  // primary context is shared with any CUDA runtime-API user in the process.
  static Status init = [] {
    CUresult rc = cuInit(0);
    if (rc != CUDA_SUCCESS) return cu_status(rc, "cuInit");
    CUdevice dev = 0;
    rc = cuDeviceGet(&dev, 0);
    if (rc != CUDA_SUCCESS) return cu_status(rc, "cuDeviceGet");
    CUcontext ctx{};
    rc = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (rc != CUDA_SUCCESS) return cu_status(rc, "cuDevicePrimaryCtxRetain");
    rc = cuCtxSetCurrent(ctx);
    if (rc != CUDA_SUCCESS) return cu_status(rc, "cuCtxSetCurrent");
    return Status::Ok();
  }();
  return init;
}

Status CudaBackend::arch(ArchInfo* out) {
  if (out == nullptr) {
    return Status::invalid_argument("arch output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUdevice dev = 0;
  status = cu_status(cuCtxGetDevice(&dev), "cuCtxGetDevice");
  if (!status.ok()) {
    return status;
  }
  int major = 0, minor = 0;
  status = cu_status(
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           dev),
      "cuDeviceGetAttribute(major)");
  if (!status.ok()) {
    return status;
  }
  status = cu_status(
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           dev),
      "cuDeviceGetAttribute(minor)");
  if (!status.ok()) {
    return status;
  }
  out->name = "sm_" + std::to_string(major) + std::to_string(minor);
  return Status::Ok();
}

Status CudaBackend::get_allocation_granularity(std::uint64_t* out) {
  if (out == nullptr) {
    return Status::invalid_argument("granularity output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = current_device();
  std::size_t granularity = 0;
  status = cu_status(
      cuMemGetAllocationGranularity(&granularity, &prop,
                                    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
      "cuMemGetAllocationGranularity");
  if (!status.ok()) {
    return status;
  }
  *out = granularity;
  return Status::Ok();
}

Status CudaBackend::reserve_address(std::uint64_t size, std::uint64_t alignment,
                                    std::uint64_t requested_base,
                                    std::uint64_t* out_base) {
  if (out_base == nullptr) {
    return Status::invalid_argument("reserve output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUdeviceptr ptr = 0;
  status = cu_status(
      cuMemAddressReserve(&ptr, size, alignment,
                          static_cast<CUdeviceptr>(requested_base), 0),
      "cuMemAddressReserve");
  if (!status.ok()) {
    return status;
  }
  *out_base = static_cast<std::uint64_t>(ptr);
  return Status::Ok();
}

Status CudaBackend::release_address(std::uint64_t base, std::uint64_t size) {
  return cu_status(cuMemAddressFree(static_cast<CUdeviceptr>(base), size),
                   "cuMemAddressFree");
}

Status CudaBackend::create_physical(std::uint64_t size, MemHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("physical output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = current_device();
  CUmemGenericAllocationHandle handle{};
  status = cu_status(cuMemCreate(&handle, size, &prop, 0), "cuMemCreate");
  if (!status.ok()) {
    return status;
  }
  *out = from_physical(handle);
  return Status::Ok();
}

Status CudaBackend::release_physical(MemHandle handle) {
  return cu_status(cuMemRelease(to_physical(handle)), "cuMemRelease");
}

Status CudaBackend::map(std::uint64_t va, std::uint64_t size,
                        std::uint64_t offset, MemHandle handle) {
  return cu_status(cuMemMap(static_cast<CUdeviceptr>(va), size, offset,
                            to_physical(handle), 0),
                   "cuMemMap");
}

Status CudaBackend::unmap(std::uint64_t va, std::uint64_t size) {
  return cu_status(cuMemUnmap(static_cast<CUdeviceptr>(va), size), "cuMemUnmap");
}

Status CudaBackend::set_access(std::uint64_t va, std::uint64_t size,
                               const MemoryAccess& access) {
  CUmemAccessDesc desc{};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = access.device_ordinal != 0 ? access.device_ordinal
                                                 : current_device();
  desc.flags = access.read && access.write
                   ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                   : (access.read ? CU_MEM_ACCESS_FLAGS_PROT_READ
                                  : CU_MEM_ACCESS_FLAGS_PROT_NONE);
  return cu_status(
      cuMemSetAccess(static_cast<CUdeviceptr>(va), size, &desc, 1),
      "cuMemSetAccess");
}

}  // namespace snapshot
