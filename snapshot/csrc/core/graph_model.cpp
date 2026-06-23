#include "snapshot/gpu_backend.hpp"

namespace snapshot {

const char* vendor_name(Vendor vendor) {
  switch (vendor) {
    case Vendor::kStub:
      return "STUB";
    case Vendor::kHip:
      return "HIP";
    case Vendor::kCuda:
      return "CUDA";
  }
  return "UNKNOWN";
}

Vendor vendor_from_name(const std::string& name) {
  if (name == "HIP") {
    return Vendor::kHip;
  }
  if (name == "CUDA") {
    return Vendor::kCuda;
  }
  return Vendor::kStub;
}

}  // namespace snapshot
