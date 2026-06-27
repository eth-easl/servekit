// record_cuda_format.hpp — `.snap` binary format for recorded CUDA graphs.
//
// N5a Task 3 introduced v1 (kernel nodes only, N5a CLI smoke). N5b Task 2 bumps
// to **v2**: type-tagged nodes so FULL/PIECEWISE graphs containing MEMCPY and
// MEMSET nodes (plus the kernel nodes) are recorded structurally and rebuilt —
// the N5a kernel-only gap that dropped non-kernel dependency edges. A node the
// interposer cannot yet record/replay is tagged BLIND with an explicit reason
// (no silent edge-drop). `extra`-buffer (CU_LAUNCH_PARAM_BUFFER_POINTER)
// kernarg launches are recorded verbatim (the N5a kernelParams-only gap).
//
// Standalone, header-only, plain little-endian binary (length-prefixed blobs).
// NO msgpack, NO dependency on snapshot/csrc/core/** — the record/restore .so
// links only libcuda + libcudart + libdl and must stay self-contained for the
// vLLM-CUDA image. Produced by snapshot_record_cuda.cpp, consumed VERBATIM by
// the restore path.
//
// Verbatim Δ=0 contract: because snapshot_redirect_cuda pins device pointers at
// a fixed base, the recorded kernarg blob AND the device pointers inside a
// CUDA_MEMCPY3D / CUDA_MEMSET_NODE_PARAMS struct are byte-identical across
// runs. We store them VERBATIM — no cubin/PTX parser, no pointer relocation.
//
// One `.snap` file == one captured CUDA graph. Layout (all integers little-
// endian; record + restore run on the same x86_64 A100 nodes, so byte order is
// fixed and the format is endian-explicit regardless):
//
//   magic        u8[8]            "SNAPCUD1"  (kSnapMagic — format family)
//   version      u32              kSnapVersion (=2)
//   node_count   u32              number of node records that follow
//   repeat node_count times (one record per captured node, in cuGraphGetNodes
//   order; a record's position == its node index, which dep_indices reference):
//     tag          u8             0=KERNEL, 1=MEMCPY, 2=MEMSET, 255=BLIND
//     if tag == KERNEL:
//       kind         i32          0 = fatbin/static, 1 = module/nvrtc
//       module_hash  u64          FNV-1a64 of the PTX/cubin image (0 for fatbin)
//       name_len     u32
//       name         u8[name_len] mangled device kernel name (no NUL stored)
//       grid         u32[3]       gridDimX/Y/Z
//       block        u32[3]       blockDimX/Y/Z
//       shared_mem   u32          sharedMemBytes
//       kernarg_form u8           0 = packed from kernelParams, 1 = raw `extra`
//       kernarg_size u32
//       kernarg      u8[kernarg_size]   VERBATIM kernarg blob (either form)
//     elif tag == MEMCPY:
//       blob_size    u32
//       blob         u8[blob_size]      VERBATIM CUDA_MEMCPY3D struct
//     elif tag == MEMSET:
//       blob_size    u32
//       blob         u8[blob_size]      VERBATIM CUDA_MEMSET_NODE_PARAMS struct
//     elif tag == BLIND:
//       reason_len   u32
//       reason       u8[reason_len]     why this node is not restorable
//     dep_count    u32
//     dep_indices  u32[dep_count]       record indices (0-based) of predecessors
//
// For KERNEL nodes, (kind, name, module_hash) is the IdentityMap key resolving
// the CUfunction; (grid, block, shared_mem, kernarg) is the verbatim launch;
// kernarg_form records how the blob was obtained (both forms replay identically
// via the driver `extra` buffer-pointer config). For MEMCPY/MEMSET, the verbatim
// params struct (device pointers included) is correct unmodified under Δ=0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace snapshot_cuda {

inline constexpr char          kSnapMagic[8] = {'S', 'N', 'A', 'P',
                                                'C', 'U', 'D', '1'};
inline constexpr std::uint32_t kSnapVersion  = 2u;

// Node-type tag. BLIND (255) marks a node the interposer could not record or
// rebuild verbatim; a non-empty reason explains why. The restore gate requires
// blind=0 for the chosen strategy's graphs.
enum class NodeTag : std::uint8_t {
  Kernel = 0u,
  Memcpy = 1u,
  Memset = 2u,
  Sync   = 3u,   // N5b: wait-event / event-record / empty / host / child — rebuilt as empty node
  Blind  = 255u,
};

// One recorded graph node. The tag selects which fields are meaningful; all
// fields share one struct so serialization stays a flat switch on the tag.
struct RecordedNode {
  NodeTag tag = NodeTag::Blind;     // selects the meaningful fields below
  std::vector<std::uint32_t> deps;  // record indices of predecessors (all tags)

  // KERNEL fields (tag == Kernel).
  std::int32_t  kind = 0;           // 0=fatbin/static, 1=module/nvrtc
  std::uint64_t module_hash = 0;
  std::string   name;               // mangled device kernel name
  std::uint32_t grid[3]  = {1u, 1u, 1u};
  std::uint32_t block[3] = {1u, 1u, 1u};
  std::uint32_t shared_mem_bytes = 0;
  std::uint8_t  kernarg_form = 0;   // 0=packed-kernelParams, 1=raw `extra`
  std::vector<std::uint8_t> kernarg;

  // MEMCPY / MEMSET fields (tag == Memcpy|Memset): verbatim params struct.
  std::vector<std::uint8_t> blob;

  // BLIND fields (tag == Blind): human-readable reason (no NUL stored).
  std::string reason;
};

// One recorded CUDA graph (all node types, in cuGraphGetNodes order).
struct RecordedGraph {
  std::vector<RecordedNode> nodes;
};

namespace detail {

inline void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
  b.push_back(static_cast<std::uint8_t>(v & 0xffu));
  b.push_back(static_cast<std::uint8_t>((v >> 8) & 0xffu));
  b.push_back(static_cast<std::uint8_t>((v >> 16) & 0xffu));
  b.push_back(static_cast<std::uint8_t>((v >> 24) & 0xffu));
}

inline void put_i32(std::vector<std::uint8_t>& b, std::int32_t v) {
  put_u32(b, static_cast<std::uint32_t>(v));
}

inline void put_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
  put_u32(b, static_cast<std::uint32_t>(v & 0xffffffffULL));
  put_u32(b, static_cast<std::uint32_t>((v >> 32) & 0xffffffffULL));
}

inline void put_u8(std::vector<std::uint8_t>& b, std::uint8_t v) {
  b.push_back(v);
}

inline void put_bytes(std::vector<std::uint8_t>& b, const void* p,
                      std::size_t n) {
  if (n == 0) return;
  const auto* q = static_cast<const std::uint8_t*>(p);
  b.insert(b.end(), q, q + n);
}

// Bounds-checked little-endian reader over a buffer + cursor. Any out-of-bounds
// read latches ok=false and yields zeroes thereafter, so callers can parse
// optimistically and check ok at the end.
struct Reader {
  const std::uint8_t* p = nullptr;
  std::size_t         n = 0;
  std::size_t         pos = 0;
  bool                ok = true;

  bool need(std::size_t k) {
    if (!ok || pos + k > n) {
      ok = false;
      return false;
    }
    return true;
  }
  std::uint8_t u8() {
    if (!need(1)) return 0;
    return p[pos++];
  }
  std::uint32_t u32() {
    if (!need(4)) return 0;
    const std::uint32_t v =
        static_cast<std::uint32_t>(p[pos]) |
        (static_cast<std::uint32_t>(p[pos + 1]) << 8) |
        (static_cast<std::uint32_t>(p[pos + 2]) << 16) |
        (static_cast<std::uint32_t>(p[pos + 3]) << 24);
    pos += 4;
    return v;
  }
  std::int32_t  i32() { return static_cast<std::int32_t>(u32()); }
  std::uint64_t u64() {
    const std::uint64_t lo = u32();
    const std::uint64_t hi = u32();
    return lo | (hi << 32);
  }
  bool bytes(void* dst, std::size_t k) {
    if (!need(k)) return false;
    std::memcpy(dst, p + pos, k);
    pos += k;
    return true;
  }
};

}  // namespace detail

// Serialize a recorded graph to `path`. Deterministic: identical RecordedGraph
// inputs produce byte-identical files (the Δ=0 cmp gate depends on this).
// Returns true on success.
inline bool serialize_graph(const RecordedGraph& g, const char* path) {
  std::vector<std::uint8_t> buf;
  detail::put_bytes(buf, kSnapMagic, sizeof(kSnapMagic));
  detail::put_u32(buf, kSnapVersion);
  detail::put_u32(buf, static_cast<std::uint32_t>(g.nodes.size()));
  for (const RecordedNode& nd : g.nodes) {
    detail::put_u8(buf, static_cast<std::uint8_t>(nd.tag));
    switch (nd.tag) {
      case NodeTag::Kernel: {
        detail::put_i32(buf, nd.kind);
        detail::put_u64(buf, nd.module_hash);
        detail::put_u32(buf, static_cast<std::uint32_t>(nd.name.size()));
        detail::put_bytes(buf, nd.name.data(), nd.name.size());
        for (int k = 0; k < 3; ++k) detail::put_u32(buf, nd.grid[k]);
        for (int k = 0; k < 3; ++k) detail::put_u32(buf, nd.block[k]);
        detail::put_u32(buf, nd.shared_mem_bytes);
        detail::put_u8(buf, nd.kernarg_form);
        detail::put_u32(buf, static_cast<std::uint32_t>(nd.kernarg.size()));
        detail::put_bytes(buf, nd.kernarg.data(), nd.kernarg.size());
        break;
      }
      case NodeTag::Memcpy:
      case NodeTag::Memset: {
        detail::put_u32(buf, static_cast<std::uint32_t>(nd.blob.size()));
        detail::put_bytes(buf, nd.blob.data(), nd.blob.size());
        break;
      }
      case NodeTag::Sync: {
        // No payload — rebuilt as an empty (no-op) node preserving deps.
        break;
      }
      case NodeTag::Blind:
      default: {
        detail::put_u32(buf, static_cast<std::uint32_t>(nd.reason.size()));
        detail::put_bytes(buf, nd.reason.data(), nd.reason.size());
        break;
      }
    }
    // Dependency edges (common to all tags).
    detail::put_u32(buf, static_cast<std::uint32_t>(nd.deps.size()));
    for (std::uint32_t d : nd.deps) detail::put_u32(buf, d);
  }

  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) return false;
  const std::size_t wrote =
      buf.empty() ? 0 : std::fwrite(buf.data(), 1, buf.size(), f);
  const bool wrote_ok = (wrote == buf.size());
  const bool close_ok = (std::fclose(f) == 0);
  return wrote_ok && close_ok;
}

// Deserialize a `.snap` file into *out. Returns true on success; on any
// malformed/truncated input returns false (with *out cleared/partial). Rejects
// a wrong magic or a version mismatch cleanly (v1 readers reject v2 and vice
// versa) so a stale snapshot dir cannot be mis-parsed.
inline bool deserialize_graph(const char* path, RecordedGraph* out) {
  if (out == nullptr) return false;
  out->nodes.clear();

  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return false;
  if (std::fseek(f, 0, SEEK_END) != 0) {
    std::fclose(f);
    return false;
  }
  const long sz = std::ftell(f);
  if (sz < 0 || std::fseek(f, 0, SEEK_SET) != 0) {
    std::fclose(f);
    return false;
  }
  std::vector<std::uint8_t> buf(static_cast<std::size_t>(sz));
  const std::size_t rd =
      buf.empty() ? 0 : std::fread(buf.data(), 1, buf.size(), f);
  std::fclose(f);
  if (rd != buf.size()) return false;

  detail::Reader r{buf.data(), buf.size(), 0, true};
  char magic[sizeof(kSnapMagic)];
  if (!r.bytes(magic, sizeof(magic))) return false;
  if (std::memcmp(magic, kSnapMagic, sizeof(magic)) != 0) return false;
  if (r.u32() != kSnapVersion) return false;
  const std::uint32_t node_count = r.u32();
  // Cap the reserve so a corrupt/hostile file cannot request a huge up-front
  // allocation. A captured graph with >1M nodes is implausible for the
  // workloads this targets; reject rather than reserve gigabytes.
  if (node_count > (1u << 20)) return false;
  out->nodes.reserve(node_count);
  for (std::uint32_t i = 0; i < node_count && r.ok; ++i) {
    RecordedNode nd;
    nd.tag = static_cast<NodeTag>(r.u8());
    switch (nd.tag) {
      case NodeTag::Kernel: {
        nd.kind        = r.i32();
        nd.module_hash = r.u64();
        const std::uint32_t name_len = r.u32();
        if (!r.need(name_len)) return false;
        nd.name.assign(reinterpret_cast<const char*>(buf.data() + r.pos),
                       name_len);
        r.pos += name_len;
        for (int k = 0; k < 3; ++k) nd.grid[k] = r.u32();
        for (int k = 0; k < 3; ++k) nd.block[k] = r.u32();
        nd.shared_mem_bytes = r.u32();
        nd.kernarg_form     = r.u8();
        const std::uint32_t ksize = r.u32();
        if (!r.need(ksize)) return false;
        nd.kernarg.resize(ksize);
        if (ksize) r.bytes(nd.kernarg.data(), ksize);
        break;
      }
      case NodeTag::Memcpy:
      case NodeTag::Memset: {
        const std::uint32_t bsize = r.u32();
        if (!r.need(bsize)) return false;
        nd.blob.resize(bsize);
        if (bsize) r.bytes(nd.blob.data(), bsize);
        break;
      }
      case NodeTag::Sync: {
        break;  // no payload
      }
      case NodeTag::Blind:
      default: {
        nd.tag = NodeTag::Blind;  // coerce any unknown tag to BLIND on read
        const std::uint32_t rlen = r.u32();
        if (!r.need(rlen)) return false;
        nd.reason.assign(reinterpret_cast<const char*>(buf.data() + r.pos),
                         rlen);
        r.pos += rlen;
        break;
      }
    }
    // Dependency edges (common to all tags).
    const std::uint32_t dep_count = r.u32();
    if (!r.need(static_cast<std::size_t>(dep_count) * 4u)) return false;
    nd.deps.reserve(dep_count);
    for (std::uint32_t d = 0; d < dep_count; ++d) nd.deps.push_back(r.u32());
    if (!r.ok) return false;
    out->nodes.push_back(std::move(nd));
  }
  return r.ok;
}

}  // namespace snapshot_cuda
