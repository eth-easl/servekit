#include "snapshot/snapshot_format.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>

namespace snapshot {
namespace {

constexpr std::array<char, 8> kMagic = {'S', 'N', 'A', 'P', 'S', 'H', 'O', 'T'};

struct Section {
  SectionTag tag = SectionTag::kAllocLog;
  std::vector<std::byte> payload;
};

void append_u8(std::vector<std::byte>* out, std::uint8_t value) {
  out->push_back(static_cast<std::byte>(value));
}

void append_u32(std::vector<std::byte>* out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<std::byte>((value >> (i * 8)) & 0xFFU));
  }
}

void append_u64(std::vector<std::byte>* out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out->push_back(static_cast<std::byte>((value >> (i * 8)) & 0xFFU));
  }
}

Status append_size_prefixed(std::vector<std::byte>* out,
                            const std::byte* data, std::size_t size) {
  if (size > std::numeric_limits<std::uint32_t>::max()) {
    return Status::overflow("serialized field exceeds uint32 length");
  }
  append_u32(out, static_cast<std::uint32_t>(size));
  out->insert(out->end(), data, data + size);
  return Status::Ok();
}

Status append_string(std::vector<std::byte>* out, const std::string& value) {
  return append_size_prefixed(out,
                              reinterpret_cast<const std::byte*>(value.data()),
                              value.size());
}

class Reader {
 public:
  explicit Reader(const std::vector<std::byte>& bytes) : bytes_(bytes) {}

  Status u8(std::uint8_t* out) {
    if (!need(1)) {
      return Status::format("unexpected EOF reading uint8");
    }
    *out = static_cast<std::uint8_t>(bytes_[offset_++]);
    return Status::Ok();
  }

  Status u32(std::uint32_t* out) {
    if (!need(4)) {
      return Status::format("unexpected EOF reading uint32");
    }
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      value |= static_cast<std::uint32_t>(
                   static_cast<std::uint8_t>(bytes_[offset_++]))
               << (i * 8);
    }
    *out = value;
    return Status::Ok();
  }

  Status u64(std::uint64_t* out) {
    if (!need(8)) {
      return Status::format("unexpected EOF reading uint64");
    }
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value |= static_cast<std::uint64_t>(
                   static_cast<std::uint8_t>(bytes_[offset_++]))
               << (i * 8);
    }
    *out = value;
    return Status::Ok();
  }

  Status string(std::string* out) {
    std::uint32_t size = 0;
    Status status = u32(&size);
    if (!status.ok()) {
      return status;
    }
    if (!need(size)) {
      return Status::format("unexpected EOF reading string");
    }
    out->assign(reinterpret_cast<const char*>(bytes_.data() + offset_), size);
    offset_ += size;
    return Status::Ok();
  }

  Status bytes(std::vector<std::byte>* out) {
    std::uint32_t size = 0;
    Status status = u32(&size);
    if (!status.ok()) {
      return status;
    }
    if (!need(size)) {
      return Status::format("unexpected EOF reading bytes");
    }
    out->assign(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + size));
    offset_ += size;
    return Status::Ok();
  }

  bool done() const { return offset_ == bytes_.size(); }

 private:
  bool need(std::size_t n) const { return n <= bytes_.size() - offset_; }

  const std::vector<std::byte>& bytes_;
  std::size_t offset_ = 0;
};

void append_dim3(std::vector<std::byte>* out, const Dim3& dim) {
  append_u32(out, dim.x);
  append_u32(out, dim.y);
  append_u32(out, dim.z);
}

Status read_dim3(Reader* reader, Dim3* out) {
  Status status = reader->u32(&out->x);
  if (!status.ok()) {
    return status;
  }
  status = reader->u32(&out->y);
  if (!status.ok()) {
    return status;
  }
  return reader->u32(&out->z);
}

Status build_alloc_section(const AllocatorState& allocator,
                           std::vector<std::byte>* out) {
  append_u64(out, allocator.requested_base);
  append_u64(out, allocator.region_base);
  append_u64(out, allocator.region_size);
  append_u64(out, allocator.cursor);
  append_u64(out, allocator.granularity);
  append_u8(out, allocator.fixed_base_honored ? 1 : 0);
  if (allocator.events.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::overflow("too many allocation events");
  }
  append_u32(out, static_cast<std::uint32_t>(allocator.events.size()));
  for (const AllocEvent& event : allocator.events) {
    append_u64(out, event.offset);
    append_u64(out, event.size);
    Status status = append_string(out, event.tag);
    if (!status.ok()) {
      return status;
    }
  }
  return Status::Ok();
}

Status parse_alloc_section(const std::vector<std::byte>& payload,
                           AllocatorState* out) {
  Reader reader(payload);
  std::uint8_t fixed = 0;
  Status status = reader.u64(&out->requested_base);
  if (!status.ok()) {
    return status;
  }
  status = reader.u64(&out->region_base);
  if (!status.ok()) {
    return status;
  }
  status = reader.u64(&out->region_size);
  if (!status.ok()) {
    return status;
  }
  status = reader.u64(&out->cursor);
  if (!status.ok()) {
    return status;
  }
  status = reader.u64(&out->granularity);
  if (!status.ok()) {
    return status;
  }
  status = reader.u8(&fixed);
  if (!status.ok()) {
    return status;
  }
  out->fixed_base_honored = fixed != 0;

  std::uint32_t count = 0;
  status = reader.u32(&count);
  if (!status.ok()) {
    return status;
  }
  out->events.clear();
  out->events.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    AllocEvent event;
    status = reader.u64(&event.offset);
    if (!status.ok()) {
      return status;
    }
    status = reader.u64(&event.size);
    if (!status.ok()) {
      return status;
    }
    status = reader.string(&event.tag);
    if (!status.ok()) {
      return status;
    }
    out->events.push_back(std::move(event));
  }
  return reader.done() ? Status::Ok()
                       : Status::format("trailing bytes in allocation section");
}

Status build_modules_section(const std::vector<ModuleImage>& modules,
                             std::vector<std::byte>* out) {
  if (modules.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::overflow("too many modules");
  }
  append_u32(out, static_cast<std::uint32_t>(modules.size()));
  for (const ModuleImage& module : modules) {
    append_u64(out, module.hash);
    Status status = append_size_prefixed(out, module.image.data(),
                                         module.image.size());
    if (!status.ok()) {
      return status;
    }
    if (module.entry_names.size() > std::numeric_limits<std::uint32_t>::max()) {
      return Status::overflow("too many module entry names");
    }
    append_u32(out, static_cast<std::uint32_t>(module.entry_names.size()));
    for (const std::string& entry : module.entry_names) {
      status = append_string(out, entry);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

Status parse_modules_section(const std::vector<std::byte>& payload,
                             std::vector<ModuleImage>* out) {
  Reader reader(payload);
  std::uint32_t count = 0;
  Status status = reader.u32(&count);
  if (!status.ok()) {
    return status;
  }
  out->clear();
  out->reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    ModuleImage module;
    status = reader.u64(&module.hash);
    if (!status.ok()) {
      return status;
    }
    status = reader.bytes(&module.image);
    if (!status.ok()) {
      return status;
    }
    std::uint32_t entry_count = 0;
    status = reader.u32(&entry_count);
    if (!status.ok()) {
      return status;
    }
    module.entry_names.reserve(entry_count);
    for (std::uint32_t j = 0; j < entry_count; ++j) {
      std::string entry;
      status = reader.string(&entry);
      if (!status.ok()) {
        return status;
      }
      module.entry_names.push_back(std::move(entry));
    }
    out->push_back(std::move(module));
  }
  return reader.done() ? Status::Ok()
                       : Status::format("trailing bytes in modules section");
}

Status build_nodes_section(const GraphIR& graph, std::vector<std::byte>* out) {
  if (graph.nodes.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::overflow("too many graph nodes");
  }
  append_u32(out, static_cast<std::uint32_t>(graph.nodes.size()));
  for (const GraphNodeIR& node : graph.nodes) {
    append_u64(out, node.id);
    append_u32(out, static_cast<std::uint32_t>(node.type));
    append_u64(out, node.module_hash);
    Status status = append_string(out, node.entry_name);
    if (!status.ok()) {
      return status;
    }
    append_dim3(out, node.kernel.grid);
    append_dim3(out, node.kernel.block);
    append_u32(out, node.kernel.shared_mem_bytes);
    status = append_size_prefixed(out, node.kernel.param_blob.data(),
                                  node.kernel.param_blob.size());
    if (!status.ok()) {
      return status;
    }
    if (node.kernel.ptr_offsets.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      return Status::overflow("too many pointer offsets");
    }
    append_u32(out, static_cast<std::uint32_t>(node.kernel.ptr_offsets.size()));
    for (std::uint32_t offset : node.kernel.ptr_offsets) {
      append_u32(out, offset);
    }
    append_u64(out, node.dst);
    append_u64(out, node.src);
    append_u64(out, node.bytes);
    append_u32(out, node.memset_value);
  }
  return Status::Ok();
}

Status parse_nodes_section(const std::vector<std::byte>& payload,
                           GraphIR* out) {
  Reader reader(payload);
  std::uint32_t count = 0;
  Status status = reader.u32(&count);
  if (!status.ok()) {
    return status;
  }
  out->nodes.clear();
  out->nodes.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    GraphNodeIR node;
    std::uint32_t type = 0;
    status = reader.u64(&node.id);
    if (!status.ok()) {
      return status;
    }
    status = reader.u32(&type);
    if (!status.ok()) {
      return status;
    }
    node.type = static_cast<GraphNodeType>(type);
    status = reader.u64(&node.module_hash);
    if (!status.ok()) {
      return status;
    }
    status = reader.string(&node.entry_name);
    if (!status.ok()) {
      return status;
    }
    status = read_dim3(&reader, &node.kernel.grid);
    if (!status.ok()) {
      return status;
    }
    status = read_dim3(&reader, &node.kernel.block);
    if (!status.ok()) {
      return status;
    }
    status = reader.u32(&node.kernel.shared_mem_bytes);
    if (!status.ok()) {
      return status;
    }
    status = reader.bytes(&node.kernel.param_blob);
    if (!status.ok()) {
      return status;
    }
    std::uint32_t ptr_count = 0;
    status = reader.u32(&ptr_count);
    if (!status.ok()) {
      return status;
    }
    node.kernel.ptr_offsets.reserve(ptr_count);
    for (std::uint32_t j = 0; j < ptr_count; ++j) {
      std::uint32_t offset = 0;
      status = reader.u32(&offset);
      if (!status.ok()) {
        return status;
      }
      node.kernel.ptr_offsets.push_back(offset);
    }
    status = reader.u64(&node.dst);
    if (!status.ok()) {
      return status;
    }
    status = reader.u64(&node.src);
    if (!status.ok()) {
      return status;
    }
    status = reader.u64(&node.bytes);
    if (!status.ok()) {
      return status;
    }
    status = reader.u32(&node.memset_value);
    if (!status.ok()) {
      return status;
    }
    out->nodes.push_back(std::move(node));
  }
  return reader.done() ? Status::Ok()
                       : Status::format("trailing bytes in graph nodes section");
}

Status build_edges_section(const GraphIR& graph, std::vector<std::byte>* out) {
  if (graph.edges.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Status::overflow("too many graph edges");
  }
  append_u32(out, static_cast<std::uint32_t>(graph.edges.size()));
  for (const GraphEdgeIR& edge : graph.edges) {
    append_u64(out, edge.from);
    append_u64(out, edge.to);
  }
  return Status::Ok();
}

Status parse_edges_section(const std::vector<std::byte>& payload,
                           GraphIR* out) {
  Reader reader(payload);
  std::uint32_t count = 0;
  Status status = reader.u32(&count);
  if (!status.ok()) {
    return status;
  }
  out->edges.clear();
  out->edges.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    GraphEdgeIR edge;
    status = reader.u64(&edge.from);
    if (!status.ok()) {
      return status;
    }
    status = reader.u64(&edge.to);
    if (!status.ok()) {
      return status;
    }
    out->edges.push_back(edge);
  }
  return reader.done() ? Status::Ok()
                       : Status::format("trailing bytes in graph edges section");
}

Status checked_read(std::ifstream* in, char* data, std::size_t n,
                    const char* what) {
  in->read(data, static_cast<std::streamsize>(n));
  if (!*in) {
    return Status::format(std::string("failed to read ") + what);
  }
  return Status::Ok();
}

}  // namespace

Status write_snapshot_file(const std::string& path, const SnapshotData& data) {
  std::vector<Section> sections;
  sections.reserve(4);

  Section alloc{SectionTag::kAllocLog, {}};
  Status status = build_alloc_section(data.allocator, &alloc.payload);
  if (!status.ok()) {
    return status;
  }
  sections.push_back(std::move(alloc));

  Section modules{SectionTag::kModules, {}};
  status = build_modules_section(data.modules, &modules.payload);
  if (!status.ok()) {
    return status;
  }
  sections.push_back(std::move(modules));

  Section nodes{SectionTag::kGraphNodes, {}};
  status = build_nodes_section(data.graph, &nodes.payload);
  if (!status.ok()) {
    return status;
  }
  sections.push_back(std::move(nodes));

  Section edges{SectionTag::kGraphEdges, {}};
  status = build_edges_section(data.graph, &edges.payload);
  if (!status.ok()) {
    return status;
  }
  sections.push_back(std::move(edges));

  std::vector<std::byte> header;
  header.insert(header.end(), reinterpret_cast<const std::byte*>(kMagic.data()),
                reinterpret_cast<const std::byte*>(kMagic.data() + kMagic.size()));
  append_u32(&header, kSnapshotVersion);
  append_u32(&header, static_cast<std::uint32_t>(data.vendor));
  append_u64(&header, data.allocator.region_base);
  append_u64(&header, data.allocator.region_size);
  append_u8(&header, data.allocator.fixed_base_honored ? 1 : 0);
  append_u64(&header, data.allocator.granularity);
  append_u32(&header, static_cast<std::uint32_t>(sections.size()));
  status = append_string(&header, data.arch);
  if (!status.ok()) {
    return status;
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return Status::io("failed to open snapshot for write: " + path);
  }
  out.write(reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(header.size()));
  for (const Section& section : sections) {
    std::vector<std::byte> section_header;
    append_u32(&section_header, static_cast<std::uint32_t>(section.tag));
    append_u64(&section_header, section.payload.size());
    append_u32(&section_header, crc32(section.payload.data(),
                                      section.payload.size()));
    out.write(reinterpret_cast<const char*>(section_header.data()),
              static_cast<std::streamsize>(section_header.size()));
    out.write(reinterpret_cast<const char*>(section.payload.data()),
              static_cast<std::streamsize>(section.payload.size()));
  }
  if (!out) {
    return Status::io("failed while writing snapshot: " + path);
  }
  return Status::Ok();
}

Status read_snapshot_file(const std::string& path, SnapshotData* out) {
  if (out == nullptr) {
    return Status::invalid_argument("read_snapshot_file output is null");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::io("failed to open snapshot for read: " + path);
  }

  std::array<char, 8> magic{};
  Status status = checked_read(&in, magic.data(), magic.size(), "magic");
  if (!status.ok()) {
    return status;
  }
  if (magic != kMagic) {
    return Status::format("snapshot magic mismatch");
  }

  std::vector<std::byte> fixed_header(4 + 4 + 8 + 8 + 1 + 8 + 4);
  status = checked_read(&in, reinterpret_cast<char*>(fixed_header.data()),
                        fixed_header.size(), "fixed header");
  if (!status.ok()) {
    return status;
  }
  Reader header_reader(fixed_header);
  std::uint32_t version = 0;
  std::uint32_t vendor = 0;
  std::uint8_t fixed = 0;
  std::uint32_t section_count = 0;
  status = header_reader.u32(&version);
  if (!status.ok()) {
    return status;
  }
  if (version != kSnapshotVersion) {
    return Status::format("unsupported snapshot version");
  }
  status = header_reader.u32(&vendor);
  if (!status.ok()) {
    return status;
  }
  out->vendor = static_cast<Vendor>(vendor);
  status = header_reader.u64(&out->allocator.region_base);
  if (!status.ok()) {
    return status;
  }
  status = header_reader.u64(&out->allocator.region_size);
  if (!status.ok()) {
    return status;
  }
  status = header_reader.u8(&fixed);
  if (!status.ok()) {
    return status;
  }
  out->allocator.fixed_base_honored = fixed != 0;
  status = header_reader.u64(&out->allocator.granularity);
  if (!status.ok()) {
    return status;
  }
  status = header_reader.u32(&section_count);
  if (!status.ok()) {
    return status;
  }

  std::vector<std::byte> arch_payload(4);
  status = checked_read(&in, reinterpret_cast<char*>(arch_payload.data()),
                        arch_payload.size(), "arch length");
  if (!status.ok()) {
    return status;
  }
  Reader arch_len_reader(arch_payload);
  std::uint32_t arch_len = 0;
  status = arch_len_reader.u32(&arch_len);
  if (!status.ok()) {
    return status;
  }
  std::vector<char> arch(arch_len);
  status = checked_read(&in, arch.data(), arch.size(), "arch string");
  if (!status.ok()) {
    return status;
  }
  out->arch.assign(arch.begin(), arch.end());

  std::map<SectionTag, std::vector<std::byte>> sections;
  for (std::uint32_t i = 0; i < section_count; ++i) {
    std::vector<std::byte> section_header(4 + 8 + 4);
    status = checked_read(&in, reinterpret_cast<char*>(section_header.data()),
                          section_header.size(), "section header");
    if (!status.ok()) {
      return status;
    }
    Reader section_reader(section_header);
    std::uint32_t tag_raw = 0;
    std::uint64_t payload_size = 0;
    std::uint32_t expected_crc = 0;
    status = section_reader.u32(&tag_raw);
    if (!status.ok()) {
      return status;
    }
    status = section_reader.u64(&payload_size);
    if (!status.ok()) {
      return status;
    }
    status = section_reader.u32(&expected_crc);
    if (!status.ok()) {
      return status;
    }
    if (payload_size > std::numeric_limits<std::size_t>::max()) {
      return Status::overflow("section too large for host");
    }
    std::vector<std::byte> payload(static_cast<std::size_t>(payload_size));
    status = checked_read(&in, reinterpret_cast<char*>(payload.data()),
                          payload.size(), "section payload");
    if (!status.ok()) {
      return status;
    }
    const std::uint32_t actual_crc = crc32(payload.data(), payload.size());
    if (actual_crc != expected_crc) {
      return {StatusCode::kCrcMismatch, "section CRC mismatch"};
    }
    SectionTag tag = static_cast<SectionTag>(tag_raw);
    if (sections.count(tag) != 0) {
      return Status::format("duplicate snapshot section");
    }
    sections.emplace(tag, std::move(payload));
  }

  auto alloc_it = sections.find(SectionTag::kAllocLog);
  auto modules_it = sections.find(SectionTag::kModules);
  auto nodes_it = sections.find(SectionTag::kGraphNodes);
  auto edges_it = sections.find(SectionTag::kGraphEdges);
  if (alloc_it == sections.end() || modules_it == sections.end() ||
      nodes_it == sections.end() || edges_it == sections.end()) {
    return Status::format("snapshot missing required section");
  }

  status = parse_alloc_section(alloc_it->second, &out->allocator);
  if (!status.ok()) {
    return status;
  }
  status = parse_modules_section(modules_it->second, &out->modules);
  if (!status.ok()) {
    return status;
  }
  status = parse_nodes_section(nodes_it->second, &out->graph);
  if (!status.ok()) {
    return status;
  }
  status = parse_edges_section(edges_it->second, &out->graph);
  if (!status.ok()) {
    return status;
  }
  return Status::Ok();
}

Status validate_snapshot_for_backend(const SnapshotData& data,
                                     GpuBackend& backend) {
  if (data.vendor != backend.vendor()) {
    return Status::mismatch(std::string("snapshot vendor ") +
                            vendor_name(data.vendor) + " does not match backend " +
                            vendor_name(backend.vendor()));
  }

  ArchInfo arch;
  Status status = backend.arch(&arch);
  if (!status.ok()) {
    return status;
  }
  if (data.arch != arch.name) {
    return Status::mismatch("snapshot arch " + data.arch +
                            " does not match backend arch " + arch.name);
  }
  return Status::Ok();
}

}  // namespace snapshot
