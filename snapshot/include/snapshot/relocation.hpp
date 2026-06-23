#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "snapshot/gpu_backend.hpp"
#include "snapshot/status.hpp"

namespace snapshot {

struct Relocation {
  std::uint64_t captured_base = 0;
  std::uint64_t restored_base = 0;
  std::uint64_t region_size = 0;

  std::int64_t delta() const {
    return static_cast<std::int64_t>(restored_base - captured_base);
  }
};

struct RelocationStats {
  std::uint64_t known_patches = 0;
  std::uint64_t blind_patches = 0;
};

Status relocate_value(std::uint64_t* value, const Relocation& relocation,
                      bool* patched);
Status relocate_param_blob(std::vector<std::byte>* blob,
                           const std::vector<std::uint32_t>& ptr_offsets,
                           const Relocation& relocation,
                           bool blind_scan_fallback,
                           RelocationStats* stats = nullptr);
Status relocate_graph_ir(GraphIR* graph, const Relocation& relocation,
                         bool blind_scan_fallback,
                         RelocationStats* stats = nullptr);

}  // namespace snapshot
