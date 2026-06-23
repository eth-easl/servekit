#include <iostream>

#include "snapshot/gpu_backend.hpp"

int main() {
  auto backend = snapshot::make_backend();
  if (backend->vendor() == snapshot::Vendor::kStub) {
    std::cout << "SKIP: GPU round-trip test requires HIP or CUDA backend\n";
    return 0;
  }
  std::cout << "GPU end-to-end round-trip is gated on graph rebuild support\n";
  return 0;
}
