#include <cstring>
#include <iostream>
#include <vector>

#include "snapshot/record.hpp"

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

// Minimal well-formed ELF64 header (64 bytes) with the section-header table at
// the end, so the parser's recovered size == header_size + one section entry.
// We only need the fields elf_code_object_size reads; the rest is zero.
std::vector<std::byte> elf64_sized(std::uint16_t e_shnum,
                                   std::uint16_t e_shentsize) {
  std::vector<std::byte> img(
      64 + static_cast<std::size_t>(e_shnum) * e_shentsize, std::byte{0});
  auto put16 = [&](std::size_t off, std::uint16_t v) {
    std::memcpy(img.data() + off, &v, sizeof(v));
  };
  auto put64 = [&](std::size_t off, std::uint64_t v) {
    std::memcpy(img.data() + off, &v, sizeof(v));
  };
  img[0] = static_cast<std::byte>(0x7f);
  img[1] = static_cast<std::byte>('E');
  img[2] = static_cast<std::byte>('L');
  img[3] = static_cast<std::byte>('F');
  img[4] = static_cast<std::byte>(2);   // ELFCLASS64
  put16(0x34, 64);                     // e_ehsize
  put64(0x28, 64);                     // e_shoff (section headers right after header)
  put16(0x3a, e_shentsize);            // e_shentsize
  put16(0x3c, e_shnum);                // e_shnum
  return img;
}

// ---- ELF size parser -------------------------------------------------------

void test_elf_size() {
  // Section-header table trailing: total = 64 + shnum*shentsize.
  auto img = elf64_sized(7, 64);
  require(snapshot::elf_code_object_size(img.data(), img.size()) == img.size(),
          "ELF size: trailing section headers");

  // Program headers can be later than section headers.
  auto img2 = elf64_sized(0, 0);
  std::uint16_t phentsize = 56;
  std::uint16_t phnum = 3;
  std::uint64_t phoff = 200;
  img2.resize(static_cast<std::size_t>(phoff) + phnum * phentsize, std::byte{0});
  std::memcpy(img2.data() + 0x20, &phoff, sizeof(phoff));
  std::memcpy(img2.data() + 0x36, &phentsize, sizeof(phentsize));
  std::memcpy(img2.data() + 0x38, &phnum, sizeof(phnum));
  require(snapshot::elf_code_object_size(img2.data(), img2.size()) == img2.size(),
          "ELF size: trailing program headers");

  // Bad magic -> 0.
  std::vector<std::byte> bad(128, std::byte{'A'});
  require(snapshot::elf_code_object_size(bad.data(), bad.size()) == 0,
          "ELF size: bad magic -> 0");

  // ELF32 -> 0 (HSACO on amd64 is ELF64).
  std::vector<std::byte> elf32(64, std::byte{0});
  elf32[0] = static_cast<std::byte>(0x7f);
  elf32[1] = static_cast<std::byte>('E');
  elf32[2] = static_cast<std::byte>('L');
  elf32[3] = static_cast<std::byte>('F');
  elf32[4] = static_cast<std::byte>(1);  // ELFCLASS32
  require(snapshot::elf_code_object_size(elf32.data(), elf32.size()) == 0,
          "ELF size: ELF32 -> 0");

  // Header claiming a size larger than the cap -> 0 (refuse to trust).
  auto img3 = elf64_sized(1, 64);
  std::uint16_t huge = 60000;
  std::memcpy(img3.data() + 0x3c, &huge, sizeof(huge));  // shnum * shentsize overflow
  require(snapshot::elf_code_object_size(img3.data(), 128) == 0,
          "ELF size: implausible size -> 0");

  // Null pointer -> 0.
  require(snapshot::elf_code_object_size(nullptr, 128) == 0,
          "ELF size: null -> 0");
}

// ---- assemble_recorded_snapshot -------------------------------------------

snapshot::RecordAssembly sample_assembly() {
  snapshot::RecordAssembly a;
  a.vendor = snapshot::Vendor::kHip;
  a.arch = "gfx942:sramecc+:xnack-";
  a.region_base = 0x600000000000ULL;
  a.region_size = 64ULL * 1024 * 1024;
  a.granularity = 4096;
  a.alloc_events = {snapshot::AllocEvent{0, 1 << 20, "A"},
                    snapshot::AllocEvent{1 << 20, 1 << 20, "B"}};

  const auto img = elf64_sized(2, 64);
  snapshot::RecordedModule m;
  m.hash = snapshot::hash_bytes(img.data(), img.size());
  m.image = img;
  a.modules.push_back(m);
  // A duplicate load of the same hash (different image bytes) must be deduped.
  snapshot::RecordedModule m2;
  m2.hash = m.hash;
  m2.image = std::vector<std::byte>(m.image.size(), std::byte{0xCC});
  a.modules.push_back(m2);

  // Two functions in the module, one of them never launched.
  a.functions.push_back({101, m.hash, std::string{"mul_bias"}});
  a.functions.push_back({102, m.hash, std::string{"relu_offset"}});
  // A function whose module hash is unknown (module never recorded).
  a.functions.push_back({103, 0xDEAD, std::string{"orphan"}});

  auto blob = [](std::uint64_t a, std::uint64_t b) {
    std::vector<std::byte> p(16, std::byte{0});
    std::memcpy(p.data(), &a, 8);
    std::memcpy(p.data() + 8, &b, 8);
    return p;
  };
  snapshot::RecordedLaunch l1;
  l1.function_id = 101;
  l1.grid = snapshot::Dim3{4096, 1, 1};
  l1.block = snapshot::Dim3{256, 1, 1};
  l1.shared_mem_bytes = 0;
  l1.param_blob = blob(0x600000000000ULL, 0x6000100000ULL);
  a.launches.push_back(l1);

  snapshot::RecordedLaunch l2;
  l2.function_id = 999;  // never seen by hipModuleGetFunction -> truly "unknown" node
  l2.grid = snapshot::Dim3{1, 1, 1};
  l2.param_blob = blob(0, 0);
  a.launches.push_back(l2);
  return a;
}

void test_assemble() {
  snapshot::RecordAssembly a = sample_assembly();
  snapshot::SnapshotData snap;
  require_ok(snapshot::assemble_recorded_snapshot(a, &snap),
             "assemble_recorded_snapshot");

  // Module dedupe by hash.
  require(snap.modules.size() == 1, "one module after dedupe");
  require(snap.modules[0].hash == a.modules[0].hash, "module hash preserved");
  require(snap.modules[0].image == a.modules[0].image,
          "first-seen module image kept");
  // Both entries the module is queried for are recorded (even unlaunched ones).
  require(snap.modules[0].entry_names.size() == 2, "both entry names recorded");

  // Serial chain: 2 nodes, 1 edge.
  require(snap.graph.nodes.size() == 2, "two nodes");
  require(snap.graph.edges.size() == 1, "one serial edge");
  require(snap.graph.edges[0].from == 1 && snap.graph.edges[0].to == 2,
          "serial chain 1->2");

  // Node 0: identity resolved.
  const auto& n0 = snap.graph.nodes[0];
  require(n0.module_hash == a.modules[0].hash, "node0 module resolved");
  require(n0.entry_name == "mul_bias", "node0 entry resolved");
  require(n0.kernel.param_blob.size() == 16, "node0 param blob copied");
  require(n0.kernel.ptr_offsets.empty(),
          "node0 ptr_offsets empty (blind-scan at restore)");
  require(n0.kernel.grid.x == 4096, "node0 grid copied");

  // Node 1: unknown identity (module 0xDEAD never recorded).
  const auto& n1 = snap.graph.nodes[1];
  require(n1.module_hash == 0 && n1.entry_name.empty(),
          "node1 identity unknown");

  // Allocator section.
  require(snap.allocator.region_base == a.region_base, "region base");
  require(snap.allocator.region_size == a.region_size, "region size");
  require(snap.allocator.events.size() == 2, "alloc events copied");
  require(snap.allocator.cursor == (1ULL << 20) * 2,
          "cursor is high-water mark");

  // The assembled snapshot must serialize + reload intact.
  snapshot::SnapshotData reloaded;
  require_ok(snapshot::write_snapshot_file("/tmp/snap-record-roundtrip.snap",
                                           snap),
             "write assembled snapshot");
  require_ok(snapshot::read_snapshot_file("/tmp/snap-record-roundtrip.snap",
                                          &reloaded),
             "read assembled snapshot");
  require(reloaded.graph.nodes.size() == 2, "reload: two nodes");
  require(reloaded.modules.size() == 1, "reload: one module");
  require(reloaded.graph.nodes[0].entry_name == "mul_bias",
          "reload: node0 entry");

  // Summary statistics: the M3a.3 gate is nodes_without_identity == 0.
  snapshot::SnapshotSummary sum;
  require_ok(snapshot::summarize_snapshot(reloaded, &sum), "summarize");
  require(sum.module_count == 1, "summary: module count");
  require(sum.kernel_nodes == 2, "summary: kernel nodes");
  require(sum.nodes_with_identity == 1, "summary: one node has identity");
  require(sum.nodes_without_identity == 1, "summary: one node is unknown");
  require(sum.modules_with_empty_image == 0, "summary: image present");
  require(sum.total_param_bytes == 32, "summary: total param bytes");
}

// An assembly whose only module had an unparseable (empty) image: the node has
// identity but the image is missing, which summarize must flag distinctly from
// "no identity" (different blocker, different fix).
void test_empty_image_flagged() {
  snapshot::RecordAssembly a;
  a.vendor = snapshot::Vendor::kHip;
  a.arch = "gfx942";
  a.region_base = 0;
  a.region_size = 0;
  snapshot::RecordedModule m;
  m.hash = 42;
  m.image = {};  // ELF parse failed at record time
  a.modules.push_back(m);
  a.functions.push_back({7, 42, std::string{"k"}});
  snapshot::RecordedLaunch l;
  l.function_id = 7;
  a.launches.push_back(l);

  snapshot::SnapshotData snap;
  require_ok(snapshot::assemble_recorded_snapshot(a, &snap), "assemble empty-img");
  snapshot::SnapshotSummary sum;
  require_ok(snapshot::summarize_snapshot(snap, &sum), "summarize empty-img");
  require(sum.nodes_with_identity == 1, "empty-img: identity present");
  require(sum.nodes_without_identity == 0, "empty-img: none unknown");
  require(sum.modules_with_empty_image == 1, "empty-img: image flagged missing");
  require(sum.nodes_with_empty_module_image == 1,
          "empty-img: node flagged as image-missing");
}

}  // namespace

int main() {
  test_elf_size();
  test_assemble();
  test_empty_image_flagged();
  std::cout << "record IR assembly + ELF parse: OK\n";
  return 0;
}
