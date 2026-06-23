#include <cstdlib>
#include <iostream>
#include <memory>

#include "snapshot/allocator.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

void require_ok(const snapshot::Status& status, const char* message) {
  if (!status.ok()) {
    std::cerr << "FAIL: " << message << ": " << status.message() << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  // Host-only logic test: always use the stub backend so the determinism math
  // is validated without a device and regardless of the compiled-in backend.
  auto backend = snapshot::make_stub_backend();
  snapshot::DeterministicAllocator allocator;

  require_ok(allocator.init(*backend, 10 * 4096, snapshot::kDefaultRequestedBase),
             "allocator init");
  require(allocator.state().region_base == snapshot::kDefaultRequestedBase,
          "stub backend should honor first requested base");
  require(allocator.state().granularity == 4096, "stub granularity");

  std::uint64_t a = 0;
  std::uint64_t b = 0;
  require_ok(allocator.alloc(*backend, 1, "a", &a), "alloc a");
  require_ok(allocator.alloc(*backend, 4097, "b", &b), "alloc b");
  require(a == allocator.state().region_base, "first allocation at base");
  require(b == allocator.state().region_base + 4096,
          "second allocation rounded to granularity");
  require(allocator.state().cursor == 3 * 4096, "cursor rounded");
  require(allocator.state().events.size() == 2, "event count");
  require(allocator.state().events[0].tag == "a", "first event tag");
  require(allocator.state().events[1].size == 8192, "second event size");

  snapshot::AllocatorState captured = allocator.state();
  require_ok(allocator.release(*backend), "release capture allocator");

  snapshot::DeterministicAllocator restored;
  std::uint64_t restored_base = 0;
  require_ok(restored.replay(*backend, captured, captured.region_base,
                             &restored_base),
             "replay allocator");
  require(restored_base != captured.region_base,
          "stub backend should shift second reservation");
  require(restored.state().events.size() == captured.events.size(),
          "replayed event count");
  require_ok(restored.release(*backend), "release restored allocator");
  return 0;
}
