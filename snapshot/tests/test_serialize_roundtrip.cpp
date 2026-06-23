#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "snapshot/snapshot_format.hpp"

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

snapshot::SnapshotData sample_snapshot() {
  snapshot::SnapshotData data;
  data.vendor = snapshot::Vendor::kStub;
  data.arch = "stub-host";
  data.allocator.requested_base = snapshot::kDefaultRequestedBase;
  data.allocator.region_base = snapshot::kDefaultRequestedBase;
  data.allocator.region_size = 4096 * 8;
  data.allocator.cursor = 4096 * 2;
  data.allocator.granularity = 4096;
  data.allocator.fixed_base_honored = true;
  data.allocator.events = {
      snapshot::AllocEvent{0, 4096, "A"},
      snapshot::AllocEvent{4096, 4096, "B"},
  };

  const char image[] = "module-image";
  snapshot::ModuleImage module;
  module.image.assign(reinterpret_cast<const std::byte*>(image),
                      reinterpret_cast<const std::byte*>(image) +
                          sizeof(image) - 1);
  module.hash = snapshot::hash_bytes(module.image.data(), module.image.size());
  module.entry_names = {"kernel"};
  data.modules.push_back(module);

  snapshot::GraphNodeIR node;
  node.id = 1;
  node.type = snapshot::GraphNodeType::kKernel;
  node.module_hash = module.hash;
  node.entry_name = "kernel";
  node.kernel.grid = snapshot::Dim3{1, 2, 3};
  node.kernel.block = snapshot::Dim3{4, 5, 6};
  node.kernel.param_blob.resize(8);
  node.kernel.ptr_offsets = {0};
  data.graph.nodes.push_back(node);
  data.graph.edges.push_back(snapshot::GraphEdgeIR{1, 2});
  return data;
}

}  // namespace

int main() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "snapshot-roundtrip.snap";
  snapshot::SnapshotData data = sample_snapshot();
  require_ok(snapshot::write_snapshot_file(path.string(), data),
             "write snapshot");

  snapshot::SnapshotData loaded;
  require_ok(snapshot::read_snapshot_file(path.string(), &loaded),
             "read snapshot");
  require(loaded.vendor == data.vendor, "vendor round-trip");
  require(loaded.arch == data.arch, "arch round-trip");
  require(loaded.allocator.events.size() == 2, "alloc events round-trip");
  require(loaded.modules.size() == 1, "modules round-trip");
  require(loaded.graph.nodes.size() == 1, "nodes round-trip");
  require(loaded.graph.edges.size() == 1, "edges round-trip");
  require(loaded.graph.nodes[0].kernel.grid.y == 2, "dim3 round-trip");

  std::filesystem::remove(path);
  return 0;
}
