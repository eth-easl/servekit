// Standalone HIP-free snapshot analyzer. Compiles with g++ on the login node
// (no ROCm needed): reads a recorded snapshot, parses AMDGPU kernel signatures
// from module ELF images, and reports per-node arg counts / pointer offsets /
// signature matches. Validates the msgpack parser + exact-arg-count logic
// against real snapshots without spending a GPU allocation.
//
// Build (login node):
//   g++ -std=c++17 -O2 -Wall -Iinclude \
//     csrc/core/record.cpp csrc/core/serialize.cpp csrc/core/hashing.cpp \
//     csrc/cli/analyze_tool.cpp -o /tmp/analyze
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "snapshot/record.hpp"
#include "snapshot/snapshot_format.hpp"

using namespace snapshot;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <snapshot.snap>\n";
    return 2;
  }
  SnapshotData snap;
  Status st = read_snapshot_file(argv[1], &snap);
  if (!st.ok()) {
    std::cerr << "read failed: " << st.message() << "\n";
    return 1;
  }

  // Extract all kernel signatures from module ELF images (AMDGPU msgpack).
  std::unordered_map<std::string, KernelSig> sig_by_name;
  std::uint64_t modules_with_sigs = 0;
  for (std::size_t mi = 0; mi < snap.modules.size(); ++mi) {
    const ModuleImage& module = snap.modules[mi];
    if (module.image.empty()) {
      std::cout << "  [debug] module#" << mi << ": empty image\n";
      continue;
    }
    // Debug: check ELF structure
    const auto* img = module.image.data();
    const auto sz = module.image.size();
    bool elf_ok = sz >= 64 && img[0] == std::byte{0x7f} &&
                  img[1] == std::byte{'E'} && img[2] == std::byte{'L'} &&
                  img[3] == std::byte{'F'};
    std::cout << "  [debug] module#" << mi << ": size=" << sz
              << " elf_ok=" << elf_ok;
    if (elf_ok) {
      // Walk section headers for SHT_NOTE
      std::uint16_t e_shnum = static_cast<std::uint16_t>(
          static_cast<unsigned char>(img[0x3c])) |
          (static_cast<std::uint16_t>(static_cast<unsigned char>(img[0x3d])) << 8);
      std::uint64_t e_shoff = 0;
      for (int b = 0; b < 8; ++b)
        e_shoff |= static_cast<std::uint64_t>(
            static_cast<unsigned char>(img[0x28 + b])) << (b * 8);
      int note_sections = 0;
      std::cout << "    sections:";
      for (int si = 0; si < e_shnum && si < 30; ++si) {
        std::size_t sh = e_shoff + static_cast<std::size_t>(si) * 64;
        if (sh + 40 > sz) break;
        std::uint32_t sh_type = static_cast<std::uint32_t>(
            static_cast<unsigned char>(img[sh + 4])) |
            (static_cast<std::uint32_t>(static_cast<unsigned char>(img[sh + 5])) << 8);
        if (sh_type == 7) {
          ++note_sections;
          std::uint64_t so = 0, ss = 0;
          for (int b = 0; b < 8; ++b) so |= static_cast<std::uint64_t>(
              static_cast<unsigned char>(img[sh + 0x18 + b])) << (b * 8);
          for (int b = 0; b < 8; ++b) ss |= static_cast<std::uint64_t>(
              static_cast<unsigned char>(img[sh + 0x20 + b])) << (b * 8);
          std::cout << " [" << si << "]=NOTE(off=" << so << ",sz=" << ss << ")";
        }
      }
      std::cout << "\n";
      std::cout << " shnum=" << e_shnum << " shoff=" << e_shoff
                << " note_secs=" << note_sections;
    }
    std::cout << "\n";
    auto sigs = extract_amdgpu_kernels(module.image.data(), module.image.size());
    std::cout << "    -> extract returned " << sigs.size() << " kernels\n";
    if (!sigs.empty()) ++modules_with_sigs;
    for (const KernelSig& ks : sigs) {
      sig_by_name[ks.name] = ks;
    }
  }
  std::cout << "analyze: modules=" << snap.modules.size()
            << " modules_with_sigs=" << modules_with_sigs
            << " unique_kernels=" << sig_by_name.size() << "\n";

  // List all parsed kernel names (for debugging name mismatches).
  if (!sig_by_name.empty()) {
    std::cout << "  parsed kernel names:\n";
    for (auto& [name, sig] : sig_by_name) {
      std::uint32_t nptr = 0;
      for (auto& a : sig.args) if (a.is_pointer) ++nptr;
      std::cout << "    '" << name << "' args=" << sig.args.size()
                << " ptrs=" << nptr
                << " kernarg_sz=" << sig.kernarg_segment_size << "\n";
    }
  }

  // Per-node analysis.
  std::uint64_t matched = 0, unmatched_named = 0, unnamed = 0;
  for (std::size_t i = 0; i < snap.graph.nodes.size(); ++i) {
    const GraphNodeIR& node = snap.graph.nodes[i];
    if (node.type != GraphNodeType::kKernel) continue;
    std::cout << "  node#" << i << " entry='" << node.entry_name << "'";
    if (node.entry_name.empty()) {
      std::cout << " [UNNAMED]\n";
      ++unnamed;
      continue;
    }
    auto sit = sig_by_name.find(node.entry_name);
    if (sit == sig_by_name.end()) {
      std::cout << " [NO SIG MATCH]\n";
      ++unmatched_named;
      continue;
    }
    const KernelSig& sig = sit->second;
    ++matched;
    const std::vector<std::byte>& blob = node.kernel.param_blob;
    std::uint32_t captured_count = 0;
    int tag = -1;
    if (!blob.empty()) {
      tag = static_cast<int>(static_cast<unsigned char>(blob[0]));
      if (tag == 1 && blob.size() >= 5) {
        captured_count = static_cast<std::uint32_t>(
            static_cast<unsigned char>(blob[1])) |
            (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[2]))
             << 8) |
            (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[3]))
             << 16) |
            (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[4]))
             << 24);
      }
    }
    std::uint32_t sig_ptrs = 0;
    for (const KernelArgSig& a : sig.args) if (a.is_pointer) ++sig_ptrs;
    auto ptr_offs = tag1_blob_ptr_offsets(blob, sig);
    std::cout << " sig_args=" << sig.args.size()
              << " sig_ptrs=" << sig_ptrs
              << " kernarg_sz=" << sig.kernarg_segment_size
              << " captured_tag=" << tag
              << " captured_args=" << captured_count
              << " blob_bytes=" << blob.size()
              << " ptr_offsets=" << ptr_offs.size();
    if (!blob.empty() && captured_count > 0 &&
        captured_count < sig.args.size()) {
      std::cout << " [UNDERCOUNT pad " << (sig.args.size() - captured_count)
                << "]";
    } else if (blob.empty() && sig.args.size() > 0) {
      std::cout << " [EMPTY BLOB pad " << sig.args.size() << " zeros]";
    }
    std::cout << "\n";
  }
  std::cout << "analyze: matched=" << matched
            << " unmatched_named=" << unmatched_named
            << " unnamed=" << unnamed << "\n";
  std::cout << "ANALYZE_GATE="
            << ((unmatched_named + unnamed) == 0 ? "PASS" : "PARTIAL") << "\n";
  return 0;
}
