#include "cuda_backend.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#include "snapshot/workload_kernels.hpp"

namespace snapshot {

std::unique_ptr<GpuBackend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();
}

namespace {

// Identical exact-unsigned-integer arithmetic to hip_kernels.cpp, so a
// captured-then-restored pipeline reproduces byte-identical device memory
// regardless of base address. extern "C" => unmangled names, so
// cuModuleGetFunction resolves them by name.
constexpr const char* kSource = R"cuda(
extern "C" __global__ void mul_bias(unsigned int* a, unsigned int* b,
                                    unsigned int* c, int bias, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i] + (unsigned int)bias;
}

extern "C" __global__ void relu_offset(unsigned int* c, unsigned int* out,
                                       int offset, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = c[i] + (unsigned int)offset;
}

extern "C" __global__ void in_place(unsigned int* out, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = out[i] ^ 0x9e3779b9u;
}
)cuda";

Status nvrtc_fail(nvrtcResult result, const char* call) {
  std::ostringstream message;
  message << call << " failed: " << nvrtcGetErrorString(result);
  return Status::backend(message.str());
}

// "--gpu-architecture=sm_XY" from the current device's compute capability.
// Defaults to sm_80 (A100) when no context is current yet (nvrtc itself needs
// no context; this only picks the codegen target).
std::string arch_option() {
  int major = 8, minor = 0;
  CUdevice dev = 0;
  if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) {
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                         dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                         dev);
  }
  return "--gpu-architecture=sm_" + std::to_string(major) +
         std::to_string(minor);
}

}  // namespace

Status compile_synthetic_module(std::vector<std::byte>* image,
                                std::vector<std::string>* entry_names) {
  if (image == nullptr || entry_names == nullptr) {
    return Status::invalid_argument("compile_synthetic_module output is null");
  }

  // Establish a context first so arch_option targets the real device.
  (void)ensure_cuda_context();

  nvrtcProgram program{};
  nvrtcResult result = nvrtcCreateProgram(&program, kSource,
                                          "snapshot_synthetic.cu", 0, nullptr,
                                          nullptr);
  if (result != NVRTC_SUCCESS) {
    return nvrtc_fail(result, "nvrtcCreateProgram");
  }

  const std::string arch = arch_option();
  const char* options[] = {arch.c_str()};
  result = nvrtcCompileProgram(program, 1, options);
  if (result != NVRTC_SUCCESS) {
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 0) {
      nvrtcGetProgramLog(program, log.data());
    }
    nvrtcDestroyProgram(&program);
    std::ostringstream message;
    message << "nvrtcCompileProgram failed: " << log;
    return Status::backend(message.str());
  }

  std::size_t cubin_size = 0;
  result = nvrtcGetCUBINSize(program, &cubin_size);
  if (result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    return nvrtc_fail(result, "nvrtcGetCUBINSize");
  }
  std::vector<char> cubin(cubin_size);
  result = nvrtcGetCUBIN(program, cubin.data());
  if (result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    return nvrtc_fail(result, "nvrtcGetCUBIN");
  }
  nvrtcDestroyProgram(&program);

  image->assign(reinterpret_cast<const std::byte*>(cubin.data()),
                reinterpret_cast<const std::byte*>(cubin.data()) + cubin.size());
  *entry_names = {"mul_bias", "relu_offset", "in_place"};
  return Status::Ok();
}

}  // namespace snapshot
