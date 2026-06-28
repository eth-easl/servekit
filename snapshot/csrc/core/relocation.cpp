#include "snapshot/relocation.hpp"

#include <cstring>
#include <cstdlib>
#include <limits>

namespace snapshot {
namespace {

bool in_region(std::uint64_t value, const Relocation& relocation) {
  if (relocation.region_size == 0 || value < relocation.captured_base) {
    return false;
  }
  return value - relocation.captured_base < relocation.region_size;
}

Status checked_add_delta(std::uint64_t value, std::int64_t delta,
                         std::uint64_t* out) {
  if (delta >= 0) {
    const auto unsigned_delta = static_cast<std::uint64_t>(delta);
    if (value > std::numeric_limits<std::uint64_t>::max() - unsigned_delta) {
      return Status::overflow("relocated pointer overflow");
    }
    *out = value + unsigned_delta;
    return Status::Ok();
  }

  const auto unsigned_delta = static_cast<std::uint64_t>(-delta);
  if (value < unsigned_delta) {
    return Status::overflow("relocated pointer underflow");
  }
  *out = value - unsigned_delta;
  return Status::Ok();
}

std::uint64_t load_u64(const std::byte* ptr) {
  std::uint64_t value = 0;
  std::memcpy(&value, ptr, sizeof(value));
  return value;
}

void store_u64(std::byte* ptr, std::uint64_t value) {
  std::memcpy(ptr, &value, sizeof(value));
}

}  // namespace

Status relocate_value(std::uint64_t* value, const Relocation& relocation,
                      bool* patched) {
  if (value == nullptr || patched == nullptr) {
    return Status::invalid_argument("relocate_value received null pointer");
  }
  *patched = false;
  if (relocation.captured_base == relocation.restored_base ||
      !in_region(*value, relocation)) {
    return Status::Ok();
  }

  std::uint64_t relocated = 0;
  Status status = checked_add_delta(*value, relocation.delta(), &relocated);
  if (!status.ok()) {
    return status;
  }
  *value = relocated;
  *patched = true;
  return Status::Ok();
}

Status relocate_param_blob(std::vector<std::byte>* blob,
                           const std::vector<std::uint32_t>& ptr_offsets,
                           const Relocation& relocation,
                           bool blind_scan_fallback,
                           RelocationStats* stats) {
  if (blob == nullptr) {
    return Status::invalid_argument("relocate_param_blob received null blob");
  }
  if (relocation.captured_base == relocation.restored_base) {
    return Status::Ok();
  }

  bool patched_known = false;
  for (std::uint32_t offset : ptr_offsets) {
    if (offset > blob->size() ||
        blob->size() - offset < sizeof(std::uint64_t)) {
      return Status::format("pointer offset exceeds kernel parameter blob");
    }

    std::uint64_t value = load_u64(blob->data() + offset);
    bool patched = false;
    Status status = relocate_value(&value, relocation, &patched);
    if (!status.ok()) {
      return status;
    }
    if (patched) {
      store_u64(blob->data() + offset, value);
      patched_known = true;
      if (stats != nullptr) {
        stats->known_patches++;
      }
    }
  }

  if (!ptr_offsets.empty() || patched_known || !blind_scan_fallback ||
      blob->size() < sizeof(std::uint64_t)) {
    // If ptr_offsets is non-empty we have a parsed signature: trust it and
    // NEVER blind-scan (blind-scan corrupts scalar args / dimension values
    // that happen to fall in the arena range — the root cause of the (nil)
    // fault in pre-signature restores).
    return Status::Ok();
  }

  // Alignment filter for blind-scan: real GPU pointers from the arena are
  // at least 256-byte aligned (arena granularity is 4K; buffer starts are
  // 4K-aligned, element pointers are dtype-aligned >= 16). Cross-boundary
  // phantom values (from reading across arg/field boundaries) have random
  // alignment and are the source of false-positive corruption -> (nil) fault.
  // Configurable via SNAPSHOT_RESTORE_BLIND_ALIGN (default 256; 0 = off).
  static const std::uint64_t blind_align = [] {
    const char* e = std::getenv("SNAPSHOT_RESTORE_BLIND_ALIGN");
    if (!e || !*e) return 256ULL;
    const long v = std::atol(e);
    return v > 0 ? static_cast<std::uint64_t>(v) : 0ULL;
  }();

  for (std::size_t offset = 0; offset <= blob->size() - sizeof(std::uint64_t);
       ++offset) {
    std::uint64_t value = load_u64(blob->data() + offset);
    if (blind_align != 0 && (value & (blind_align - 1)) != 0) {
      continue;  // not aligned enough to be a real pointer
    }
    bool patched = false;
    Status status = relocate_value(&value, relocation, &patched);
    if (!status.ok()) {
      return status;
    }
    if (patched) {
      store_u64(blob->data() + offset, value);
      if (stats != nullptr) {
        stats->blind_patches++;
      }
      offset += sizeof(std::uint64_t) - 1;
    }
  }
  return Status::Ok();
}

Status relocate_graph_ir(GraphIR* graph, const Relocation& relocation,
                         bool blind_scan_fallback,
                         RelocationStats* stats) {
  if (graph == nullptr) {
    return Status::invalid_argument("relocate_graph_ir received null graph");
  }
  if (relocation.captured_base == relocation.restored_base) {
    return Status::Ok();
  }

  for (GraphNodeIR& node : graph->nodes) {
    if (node.type == GraphNodeType::kKernel) {
      Status status = relocate_param_blob(&node.kernel.param_blob,
                                          node.kernel.ptr_offsets, relocation,
                                          blind_scan_fallback, stats);
      if (!status.ok()) {
        return status;
      }
    }

    bool patched = false;
    Status status = relocate_value(&node.dst, relocation, &patched);
    if (!status.ok()) {
      return status;
    }
    if (patched && stats != nullptr) {
      stats->known_patches++;
    }

    status = relocate_value(&node.src, relocation, &patched);
    if (!status.ok()) {
      return status;
    }
    if (patched && stats != nullptr) {
      stats->known_patches++;
    }
  }
  return Status::Ok();
}

}  // namespace snapshot
