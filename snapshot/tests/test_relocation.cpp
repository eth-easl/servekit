#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "snapshot/relocation.hpp"

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

std::uint64_t load_u64(const std::vector<std::byte>& blob, std::size_t offset) {
  std::uint64_t value = 0;
  std::memcpy(&value, blob.data() + offset, sizeof(value));
  return value;
}

void store_u64(std::vector<std::byte>* blob, std::size_t offset,
               std::uint64_t value) {
  std::memcpy(blob->data() + offset, &value, sizeof(value));
}

}  // namespace

int main() {
  const snapshot::Relocation relocation{0x600000000000ULL, 0x610000000000ULL,
                                        0x1000000ULL};

  std::uint64_t value = 0x600000000123ULL;
  bool patched = false;
  require_ok(snapshot::relocate_value(&value, relocation, &patched),
             "relocate value");
  require(patched, "value patched");
  require(value == 0x610000000123ULL, "value relocated by delta");

  std::vector<std::byte> blob(24);
  store_u64(&blob, 8, 0x600000001000ULL);
  snapshot::RelocationStats stats;
  require_ok(snapshot::relocate_param_blob(&blob, {8}, relocation, true, &stats),
             "known-offset relocation");
  require(load_u64(blob, 8) == 0x610000001000ULL, "known pointer relocated");
  require(stats.known_patches == 1, "known patch count");
  require(stats.blind_patches == 0, "blind skipped after known patch");

  std::vector<std::byte> blind(24);
  store_u64(&blind, 3, 0x600000002000ULL);
  snapshot::RelocationStats blind_stats;
  require_ok(snapshot::relocate_param_blob(&blind, {}, relocation, true,
                                           &blind_stats),
             "blind relocation");
  require(load_u64(blind, 3) == 0x610000002000ULL, "blind pointer relocated");
  require(blind_stats.blind_patches == 1, "blind patch count");

  snapshot::GraphIR graph;
  snapshot::GraphNodeIR node;
  node.type = snapshot::GraphNodeType::kMemcpyD2D;
  node.dst = 0x600000003000ULL;
  node.src = 0x600000004000ULL;
  graph.nodes.push_back(node);
  snapshot::RelocationStats graph_stats;
  require_ok(snapshot::relocate_graph_ir(&graph, relocation, false,
                                         &graph_stats),
             "graph relocation");
  require(graph.nodes[0].dst == 0x610000003000ULL, "graph dst relocated");
  require(graph.nodes[0].src == 0x610000004000ULL, "graph src relocated");
  return 0;
}
