// Standalone CPU-only diagnostic v2: reads a .snap and scans each kernel
// node's param_blob for 8-byte values that look like device pointers, against
// BOTH (a) the snapshot's stored region_base and (b) an externally-supplied
// REAL record base (since the stored region_base may be garbage). Also dumps a
// histogram of pointer-ish values to see where args actually point.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "snapshot/record.hpp"
#include "snapshot/snapshot_format.hpp"

static bool in_range(std::uint64_t v, std::uint64_t lo, std::uint64_t size) {
  return v >= lo && v < lo + size;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <foo.snap> [real_record_base_hex]\n", argv[0]);
    return 2;
  }
  std::uint64_t real_base = 0;
  if (argc >= 3) real_base = std::strtoull(argv[2], nullptr, 16);

  snapshot::SnapshotData snap;
  if (!snapshot::read_snapshot_file(argv[1], &snap).ok()) {
    std::fprintf(stderr, "read failed: %s\n", argv[1]);
    return 1;
  }
  const std::uint64_t sb = snap.allocator.region_base;
  const std::uint64_t size = snap.allocator.region_size;
  std::printf("STORED region_base = 0x%llx (decimal %llu)\n",
              static_cast<unsigned long long>(sb),
              static_cast<unsigned long long>(sb));
  std::printf("REAL  record_base  = 0x%llx (decimal %llu)  region_size=%lluGiB\n",
              static_cast<unsigned long long>(real_base),
              static_cast<unsigned long long>(real_base),
              static_cast<unsigned long long>(size >> 30));
  std::printf("nodes=%zu\n\n", snap.graph.nodes.size());

  std::size_t in_stored = 0, in_real = 0, in_real_any_total = 0;
  std::size_t total_ptr_ish = 0;  // values >= 0x100000000 (32-bit) that look like ptrs
  // histogram: top 4 bits of pointer-ish values -> count
  std::uint64_t hist[16] = {0};

  for (std::size_t ni = 0; ni < snap.graph.nodes.size(); ++ni) {
    const auto& node = snap.graph.nodes[ni];
    if (node.type != snapshot::GraphNodeType::kKernel) continue;
    const auto& blob = node.kernel.param_blob;
    std::size_t node_stored = 0, node_real = 0;
    std::size_t node_real_any = 0;  // blind-scan (every byte offset) into REAL region
    std::vector<std::uint64_t> ptrs;
    // 8-aligned scan (kernarg slots)
    for (std::size_t off = 0; off + 8 <= blob.size(); off += 8) {
      std::uint64_t v = 0;
      std::memcpy(&v, blob.data() + off, 8);
      if (v >= 0x100000000ULL) {  // plausible 64-bit device/host ptr
        ++total_ptr_ish;
        ptrs.push_back(v);
        hist[(v >> 60) & 0xF]++;
        if (in_range(v, sb, size)) ++node_stored;
        if (real_base && in_range(v, real_base, size)) ++node_real;
      }
    }
    // blind scan (every byte offset) into REAL region -- mirrors relocate_param_blob
    for (std::size_t off = 0; off + 8 <= blob.size(); ++off) {
      std::uint64_t v = 0;
      std::memcpy(&v, blob.data() + off, 8);
      if (real_base && in_range(v, real_base, size)) ++node_real_any;
    }
    in_stored += node_stored;
    in_real += node_real;
    in_real_any_total += node_real_any;
    if (ni < 3 || node_real > 0 || node_real_any > 0) {
      std::printf("  node[%zu] '%s' blob=%zuB ptr-ish=%zu stored=%zu real(8al)=%zu real(blind)=%zu",
                  ni, node.entry_name.substr(0, 40).c_str(), blob.size(),
                  ptrs.size(), node_stored, node_real, node_real_any);
      if (!ptrs.empty() && ni < 3) {
        std::printf(" [0x%llx", (unsigned long long)ptrs[0]);
        for (std::size_t k = 1; k < ptrs.size() && k < 4; ++k)
          std::printf(",0x%llx", (unsigned long long)ptrs[k]);
        std::printf("]");
      }
      std::printf("\n");
    }
  }
  std::printf("\nSUMMARY ptr-ish=%zu  in_stored_region=%zu  in_real_region(8al)=%zu  in_real_region(BLIND)=%zu\n",
              total_ptr_ish, in_stored, in_real, in_real_any_total);
  std::printf("histogram of (value>>60):\n");
  for (int i = 0; i < 16; ++i)
    if (hist[i]) std::printf("  top4=0x%X: %llu values\n", i,
                             (unsigned long long)hist[i]);
  // Also: where do MOST pointer-ish values cluster (top byte)?
  std::uint64_t tb[256] = {0};
  for (std::size_t ni = 0; ni < snap.graph.nodes.size(); ++ni) {
    const auto& node = snap.graph.nodes[ni];
    if (node.type != snapshot::GraphNodeType::kKernel) continue;
    const auto& blob = node.kernel.param_blob;
    for (std::size_t off = 0; off + 8 <= blob.size(); off += 8) {
      std::uint64_t v = 0;
      std::memcpy(&v, blob.data() + off, 8);
      if (v >= 0x100000000ULL) tb[(v >> 56) & 0xFF]++;
    }
  }
  std::printf("top-byte clusters:\n");
  for (int i = 0; i < 256; ++i)
    if (tb[i]) std::printf("  0x%02X_xxxxxxxx_xxxxxxxx: %llu values\n", i,
                           (unsigned long long)tb[i]);
  return 0;
}
