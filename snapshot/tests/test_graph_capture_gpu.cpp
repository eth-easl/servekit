#include <iostream>

#include "snapshot/gpu_backend.hpp"

int main() {
  auto backend = snapshot::make_backend();
  if (backend->vendor() == snapshot::Vendor::kStub) {
    std::cout << "SKIP: GPU graph capture test requires HIP or CUDA backend\n";
    return 0;
  }
  std::uint64_t granularity = 0;
  snapshot::Status status = backend->get_allocation_granularity(&granularity);
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  std::cout << "GPU backend granularity=" << granularity << "\n";
  return 0;
}
