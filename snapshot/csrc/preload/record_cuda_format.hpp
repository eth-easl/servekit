// record_cuda_format.hpp — N5a Task 3: the `.snap` binary format for recorded
// CUDA graphs.
//
// Standalone, header-only, plain little-endian binary (length-prefixed blobs).
// NO msgpack, NO dependency on snapshot/csrc/core/** — the record/restore .so
// links only libcuda + libcudart + libdl and must stay self-contained for the
// vLLM-CUDA image (N5b). Produced by snapshot_record_cuda.cpp (Task 3),
// consumed VERBATIM by the restore path (Task 4).
//
// Verbatim Δ=0 contract: because snapshot_redirect_cuda pins device pointers at
// a fixed base, the recorded kernarg blob is byte-identical across runs. We
// store it VERBATIM — no cubin/PTX parser, no pointer relocation.
//
// One `.snap` file == one captured CUDA graph. Layout (all integers are stored
// little-endian; record + restore run on the same x86_64 A100 nodes, so byte
// order is fixed and the format is endian-explicit regardless):
//
//   magic        u8[8]            "SNAPCUD1"  (kSnapMagic)
//   version      u32              kSnapVersion
//   node_count   u32              number of kernel-node records that follow
//   repeat node_count times (one record per captured KERNEL node, in
//   cuGraphGetNodes order; a record's position == its node index, which is what
//   dep_indices reference):
//     kind         i32            0 = fatbin/static (__cudaRegisterFunction),
//                                 1 = module/nvrtc (cuModuleGetFunction)
//     module_hash  u64            FNV-1a64 of the PTX/cubin image (0 for fatbin)
//     name_len     u32
//     name         u8[name_len]   mangled device kernel name (no NUL stored)
//     grid         u32[3]         gridDimX / gridDimY / gridDimZ
//     block        u32[3]         blockDimX / blockDimY / blockDimZ
//     shared_mem   u32            sharedMemBytes
//     kernarg_size u32
//     kernarg      u8[kernarg_size]   VERBATIM contiguous kernarg blob
//     dep_count    u32
//     dep_indices  u32[dep_count]     record indices (0-based) of predecessors
//
// (kind, name, module_hash) is the IdentityMap key resolving the CUfunction;
// (grid, block, shared_mem, kernarg) is the verbatim launch; dep_indices
// reproduce the DAG edges.

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
inline constexpr std::uint32_t kSnapVersion  = 1u;

// One recorded kernel-graph node.
struct RecordedNode {
  std::int32_t               kind = 0;      // 0=fatbin/static, 1=module/nvrtc
  std::uint64_t              module_hash = 0;
  std::string                name;          // mangled device kernel name
  std::uint32_t              grid[3]  = {1u, 1u, 1u};
  std::uint32_t              block[3] = {1u, 1u, 1u};
  std::uint32_t              shared_mem_bytes = 0;
  std::vector<std::uint8_t>  kernarg;       // verbatim contiguous kernarg blob
  std::vector<std::uint32_t> deps;          // record indices of predecessors
};

// One recorded CUDA graph (kernel nodes only, in cuGraphGetNodes order).
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
    detail::put_i32(buf, nd.kind);
    detail::put_u64(buf, nd.module_hash);
    detail::put_u32(buf, static_cast<std::uint32_t>(nd.name.size()));
    detail::put_bytes(buf, nd.name.data(), nd.name.size());
    for (int k = 0; k < 3; ++k) detail::put_u32(buf, nd.grid[k]);
    for (int k = 0; k < 3; ++k) detail::put_u32(buf, nd.block[k]);
    detail::put_u32(buf, nd.shared_mem_bytes);
    detail::put_u32(buf, static_cast<std::uint32_t>(nd.kernarg.size()));
    detail::put_bytes(buf, nd.kernarg.data(), nd.kernarg.size());
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
// malformed/truncated input returns false (with *out cleared/partial).
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
  out->nodes.reserve(node_count);
  for (std::uint32_t i = 0; i < node_count && r.ok; ++i) {
    RecordedNode nd;
    nd.kind        = r.i32();
    nd.module_hash = r.u64();
    const std::uint32_t name_len = r.u32();
    if (!r.need(name_len)) return false;
    nd.name.assign(reinterpret_cast<const char*>(buf.data() + r.pos), name_len);
    r.pos += name_len;
    for (int k = 0; k < 3; ++k) nd.grid[k] = r.u32();
    for (int k = 0; k < 3; ++k) nd.block[k] = r.u32();
    nd.shared_mem_bytes = r.u32();
    const std::uint32_t ksize = r.u32();
    if (!r.need(ksize)) return false;
    nd.kernarg.resize(ksize);
    if (ksize) r.bytes(nd.kernarg.data(), ksize);
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
