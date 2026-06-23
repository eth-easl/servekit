#include <sstream>
#include <string>
#include <vector>

#include <hip/hiprtc.h>

#include "snapshot/workload_kernels.hpp"

namespace snapshot {
namespace {

// All arithmetic is exact unsigned-integer, so the captured-then-restored
// pipeline reproduces byte-identical device memory regardless of base address.
constexpr const char* kSource = R"hip(
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
)hip";

Status hiprtc_fail(hiprtcResult result, const char* call) {
  std::ostringstream message;
  message << call << " failed: " << hiprtcGetErrorString(result);
  return Status::backend(message.str());
}

}  // namespace

Status compile_synthetic_module(std::vector<std::byte>* image,
                                std::vector<std::string>* entry_names) {
  if (image == nullptr || entry_names == nullptr) {
    return Status::invalid_argument("compile_synthetic_module output is null");
  }

  hiprtcProgram program{};
  hiprtcResult result = hiprtcCreateProgram(&program, kSource,
                                            "snapshot_synthetic.hip", 0, nullptr,
                                            nullptr);
  if (result != HIPRTC_SUCCESS) {
    return hiprtc_fail(result, "hiprtcCreateProgram");
  }

  const char* options[] = {"-O3"};
  result = hiprtcCompileProgram(program, 1, options);
  if (result != HIPRTC_SUCCESS) {
    std::size_t log_size = 0;
    hiprtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 0) {
      hiprtcGetProgramLog(program, log.data());
    }
    hiprtcDestroyProgram(&program);
    std::ostringstream message;
    message << "hiprtcCompileProgram failed: " << log;
    return Status::backend(message.str());
  }

  std::size_t code_size = 0;
  result = hiprtcGetCodeSize(program, &code_size);
  if (result != HIPRTC_SUCCESS) {
    hiprtcDestroyProgram(&program);
    return hiprtc_fail(result, "hiprtcGetCodeSize");
  }

  std::vector<char> code(code_size);
  result = hiprtcGetCode(program, code.data());
  if (result != HIPRTC_SUCCESS) {
    hiprtcDestroyProgram(&program);
    return hiprtc_fail(result, "hiprtcGetCode");
  }
  hiprtcDestroyProgram(&program);

  image->assign(reinterpret_cast<const std::byte*>(code.data()),
                reinterpret_cast<const std::byte*>(code.data()) + code.size());
  *entry_names = {"mul_bias", "relu_offset", "in_place"};
  return Status::Ok();
}

}  // namespace snapshot
