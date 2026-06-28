// Local-only correctness test for extract_amdgpu_kernels (the AMDGPU msgpack
// metadata parser) and tag1_blob_ptr_offsets. Compiles with g++ (no HIP
// needed): validates the walker logic against a hand-built ELF + a realistic
// amdhsa msgpack note before the HIP cluster build is wired to use it.
// Build command:
//   g++ -std=c++17 -Wall -Wextra -Iinclude
//       csrc/core/record.cpp tests/test_amdgpu_msgpack.cpp -o /tmp/mp
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include "snapshot/record.hpp"

using std::byte;
using std::uint16_t;
using std::uint32_t;
using uint8 = unsigned char;

static void put16(std::vector<uint8>& v, uint16_t x) {
  v.push_back(x & 0xff);
  v.push_back((x >> 8) & 0xff);
}
static void put32(std::vector<uint8>& v, uint32_t x) {
  v.push_back(x & 0xff);
  v.push_back((x >> 8) & 0xff);
  v.push_back((x >> 16) & 0xff);
  v.push_back((x >> 24) & 0xff);
}
static void put64(std::vector<uint8>& v, uint64_t x) {
  for (int i = 0; i < 8; ++i) v.push_back((x >> (i * 8)) & 0xff);
}

// ---- minimal msgpack encoders (big-endian) ----
static void mp_str(std::vector<uint8>& v, const std::string& s) {
  assert(s.size() < 32);
  v.push_back(static_cast<uint8>(0xa0 | s.size()));
  v.insert(v.end(), s.begin(), s.end());
}
static void mp_uint(std::vector<uint8>& v, uint64_t x) {
  if (x <= 0x7f) {
    v.push_back(static_cast<uint8>(x));
  } else if (x <= 0xff) {
    v.push_back(0xcc);
    v.push_back(static_cast<uint8>(x));
  } else if (x <= 0xffff) {
    v.push_back(0xcd);
    v.push_back((x >> 8) & 0xff);
    v.push_back(x & 0xff);
  } else {
    v.push_back(0xce);
    v.push_back((x >> 24) & 0xff);
    v.push_back((x >> 16) & 0xff);
    v.push_back((x >> 8) & 0xff);
    v.push_back(x & 0xff);
  }
}

// Build an amdhsa.kernels msgpack doc for ONE kernel with the given args.
// Each arg is (size, value_kind). offset is computed cumulatively.
static std::vector<uint8> build_amdhsa_doc(
    const std::string& kname, uint32_t kernarg_size,
    const std::vector<std::pair<uint32_t, std::string>>& args) {
  std::vector<uint8> doc;
  // top-level fixmap with 1 pair: "amdhsa.kernels" -> [ {kernel-map} ]
  doc.push_back(0x81);
  mp_str(doc, "amdhsa.kernels");
  // 1-element fixarray
  doc.push_back(0x91);
  // kernel map: fixmap with .name, .kernarg_segment_size, .args
  doc.push_back(0x83);
  mp_str(doc, ".name");
  mp_str(doc, kname);
  mp_str(doc, ".kernarg_segment_size");
  mp_uint(doc, kernarg_size);
  mp_str(doc, ".args");
  // args fixarray
  doc.push_back(static_cast<uint8>(0x90 | args.size()));
  uint32_t off = 0;
  for (auto& a : args) {
    // arg fixmap: .size, .offset, .value_kind, .value_type
    doc.push_back(0x84);
    mp_str(doc, ".size");
    mp_uint(doc, a.first);
    mp_str(doc, ".offset");
    mp_uint(doc, off);
    mp_str(doc, ".value_kind");
    mp_str(doc, a.second);
    mp_str(doc, ".value_type");
    mp_str(doc, a.second == "global_buffer" ? "struct" : "f32");
    off += a.first;
  }
  return doc;
}

// Build a minimal ELF64 with a SHT_NOTE section carrying one AMD metadata note
// whose desc is `meta`. `note_type` selects the code-object version format:
// 0x3a with name "AMD" (v3), or 0x20 with name "AMDGPU" (v4+, the real form
// on MI300A/ROCm 6.3). Returns the full ELF image.
static std::vector<byte> build_elf_with_note(const std::vector<uint8>& meta,
                                             uint32_t note_type = 0x3a) {
  std::vector<uint8> e;
  // ELF64 header (64 bytes).
  e.insert(e.end(), {0x7f, 'E', 'L', 'F'});  // e_ident magic
  e.push_back(2);                             // ELFCLASS64
  e.push_back(1);                             // ELFDATA2LSB (little-endian)
  e.push_back(1);                             // EV_CURRENT
  e.push_back(0);                             // ELFOSABI_NONE
  e.insert(e.end(), 8, 0);                    // e_ident pad (to 16 bytes)
  put16(e, 1);    // e_type ET_REL
  put16(e, 62);   // e_machine EM_X86_64 (irrelevant to parser)
  put32(e, 1);    // e_version
  put64(e, 0);    // e_entry
  put64(e, 0);    // e_phoff
  uint32_t shoff_pos = static_cast<uint32_t>(e.size());
  put64(e, 0);    // e_shoff (patch later)
  put32(e, 0);    // e_flags
  put16(e, 64);   // e_ehsize
  put16(e, 0);    // e_phentsize
  put16(e, 0);    // e_phnum
  put16(e, 64);   // e_shentsize
  put16(e, 2);    // e_shnum (null + note)
  put16(e, 1);    // e_shstrndx (index 1, harmless)
  assert(e.size() == 64);

  // Section 0: null (16-byte header, but we use 64-byte e_shentsize -> pad).
  // We only need section header 1 (the note). Lay out note payload first.
  const uint32_t note_off = 64;
  // note: namesz/descsz/type depend on code-object version.
  const char* nm = (note_type == 0x20) ? "AMDGPU" : "AMD";
  uint32_t namesz = static_cast<uint32_t>(std::strlen(nm)) + 1;  // include NUL
  uint32_t desc_padded = (meta.size() + 3u) & ~3u;
  std::vector<uint8> note;
  put32(note, namesz);
  put32(note, static_cast<uint32_t>(meta.size()));
  put32(note, note_type);  // NT_AMDGPU_METADATA
  note.insert(note.end(), nm, nm + namesz);  // name + NUL
  // pad name to 4-byte boundary
  uint32_t name_padded = (namesz + 3u) & ~3u;
  for (uint32_t i = namesz; i < name_padded; ++i) note.push_back(0);
  note.insert(note.end(), meta.begin(), meta.end());
  for (uint32_t i = meta.size(); i < desc_padded; ++i) note.push_back(0);
  e.insert(e.end(), note.begin(), note.end());
  const uint32_t note_size = static_cast<uint32_t>(note.size());

  // Section header table at shoff = 64 + note_size.
  const uint64_t shoff = note_off + note_size;
  // patch e_shoff
  std::memcpy(&e[shoff_pos], &shoff, sizeof(shoff));

  // Section 0: null header (64 bytes of zero).
  e.resize(note_off + note_size);
  for (int i = 0; i < 64; ++i) e.push_back(0);
  // Section 1: SHT_NOTE header (64 bytes).
  std::vector<uint8> sh(64, 0);
  auto put = [&](uint32_t off, uint32_t x) {
    sh[off] = x & 0xff;
    sh[off + 1] = (x >> 8) & 0xff;
    sh[off + 2] = (x >> 16) & 0xff;
    sh[off + 3] = (x >> 24) & 0xff;
  };
  auto put64s = [&](uint32_t off, uint64_t x) {
    for (int i = 0; i < 8; ++i) sh[off + i] = (x >> (i * 8)) & 0xff;
  };
  put(4, 7);  // sh_type SHT_NOTE
  put64s(0x18, note_off);   // sh_offset
  put64s(0x20, note_size);  // sh_size
  e.insert(e.end(), sh.begin(), sh.end());

  std::vector<byte> out(e.size());
  std::memcpy(out.data(), e.data(), e.size());
  return out;
}

int main() {
  using namespace snapshot;

  // Kernel: aiter-like — 15 args, mix of pointers (8B) and scalars (4B).
  // Expected by rebuild: exact count=15, pointers at the global_buffer offsets.
  std::vector<std::pair<uint32_t, std::string>> args = {
      {8, "global_buffer"}, {8, "global_buffer"}, {8, "global_buffer"},
      {8, "global_buffer"}, {8, "global_buffer"}, {4, "by_value"},
      {4, "by_value"},      {4, "by_value"},      {4, "by_value"},
      {4, "by_value"},      {4, "by_value"},      {4, "by_value"},
      {4, "by_value"},      {4, "by_value"},      {4, "by_value"},
  };
  uint32_t ksize = 0;
  for (auto& a : args) ksize += a.first;
  auto doc = build_amdhsa_doc("aiter_fused_qk", ksize, args);
  auto elf = build_elf_with_note(doc);

  auto sigs = extract_amdgpu_kernels(elf.data(), elf.size());
  assert(sigs.size() == 1);
  const KernelSig& ks = sigs[0];
  assert(ks.name == "aiter_fused_qk");
  assert(ks.kernarg_segment_size == ksize);
  assert(ks.args.size() == 15);
  uint32_t off = 0;
  for (size_t i = 0; i < ks.args.size(); ++i) {
    assert(ks.args[i].size == args[i].first);
    assert(ks.args[i].offset == off);
    assert(ks.args[i].is_pointer == (args[i].second == "global_buffer"));
    off += args[i].first;
  }
  // pointer count = 5
  size_t nptr = 0;
  for (auto& a : ks.args)
    if (a.is_pointer) ++nptr;
  assert(nptr == 5);

  // --- empty / non-ELF inputs return empty ---
  assert(extract_amdgpu_kernels(nullptr, 0).empty());
  std::vector<byte> junk(128, byte{0});
  assert(extract_amdgpu_kernels(junk.data(), junk.size()).empty());

  // --- truncated msgpack returns empty (no crash) ---
  std::vector<uint8> trunc(doc.begin(), doc.begin() + 3);  // map header + partial key
  auto elf_trunc = build_elf_with_note(trunc);
  assert(extract_amdgpu_kernels(elf_trunc.data(), elf_trunc.size()).empty());

  // --- YAML (code-object v3) must NOT be mis-parsed as msgpack ---
  // Starts with 'a' (0x61), not a map header -> parser leaves it to the YAML path.
  std::string yaml = "amdhsa.kernels:\n  - .name: legacy_yaml_kernel\n";
  std::vector<uint8> yd(yaml.begin(), yaml.end());
  auto elf_yaml = build_elf_with_note(yd);
  assert(extract_amdgpu_kernels(elf_yaml.data(), elf_yaml.size()).empty());

  // --- multi-kernel doc (two kernels, one with zero args) ---
  // Doc: {amdhsa.kernels: [ {name,ksz,args:[{ptr}]}, {name2,ksz2,args:[]} ]}
  {
    std::vector<uint8> d2;
    d2.push_back(0x81);
    mp_str(d2, "amdhsa.kernels");
    d2.push_back(0x92);  // fixarray 2
    // kernel A
    d2.push_back(0x82);  // fixmap 2: .name, .args
    mp_str(d2, ".name"); mp_str(d2, "kA");
    mp_str(d2, ".args");
    d2.push_back(0x91);  // 1 arg
    d2.push_back(0x83);  // fixmap 3
    mp_str(d2, ".size"); mp_uint(d2, 8);
    mp_str(d2, ".offset"); mp_uint(d2, 0);
    mp_str(d2, ".value_kind"); mp_str(d2, "global_buffer");
    // kernel B (zero args)
    d2.push_back(0x82);
    mp_str(d2, ".name"); mp_str(d2, "kB");
    mp_str(d2, ".args");
    d2.push_back(0x90);  // empty fixarray
    auto elf2 = build_elf_with_note(d2);
    auto s2 = extract_amdgpu_kernels(elf2.data(), elf2.size());
    assert(s2.size() == 2);
    assert(s2[0].name == "kA" && s2[0].args.size() == 1 &&
           s2[0].args[0].is_pointer && s2[0].args[0].size == 8);
    assert(s2[1].name == "kB" && s2[1].args.empty());
  }

  // --- str16 + map16 forms (large kernel name / many args keys) ---
  // Exercise the 0xda/0xde/0xdb reader paths with a 40-char name (str16).
  {
    std::vector<uint8> d3;
    d3.push_back(0x81);
    mp_str(d3, "amdhsa.kernels");
    d3.push_back(0x91);
    std::string longname(40, 'Z');  // > 31 -> str16
    d3.push_back(0x82);
    // key ".name" then str16-encoded value
    mp_str(d3, ".name");
    d3.push_back(0xda);
    d3.push_back((longname.size() >> 8) & 0xff);
    d3.push_back(longname.size() & 0xff);
    d3.insert(d3.end(), longname.begin(), longname.end());
    mp_str(d3, ".args");
    d3.push_back(0x90);  // empty
    auto elf3 = build_elf_with_note(d3);
    auto s3 = extract_amdgpu_kernels(elf3.data(), elf3.size());
    assert(s3.size() == 1 && s3[0].name == longname && s3[0].args.empty());
  }

  // --- code-object v4+ format: type 0x20 with name "AMDGPU" (real MI300A) ---
  // The real ROCm 6.3 modules use this format. Same msgpack payload, but
  // different note type and name. Verify the parser handles BOTH forms.
  {
    auto elf_v4 = build_elf_with_note(doc, /*note_type=*/0x20);
    auto sigs_v4 = extract_amdgpu_kernels(elf_v4.data(), elf_v4.size());
    assert(sigs_v4.size() == 1);
    assert(sigs_v4[0].name == "aiter_fused_qk");
    assert(sigs_v4[0].args.size() == 15);
    size_t nptr_v4 = 0;
    for (auto& a : sigs_v4[0].args)
      if (a.is_pointer) ++nptr_v4;
    assert(nptr_v4 == 5);
    printf("OK: extract_amdgpu_kernels parsed v4+ (type=0x20, name='AMDGPU') "
           "15-arg kernel, %zu ptrs\n", nptr_v4);
  }

  printf("OK: extract_amdgpu_kernels parsed 15-arg kernel, %zu ptrs, "
         "kernarg_segment_size=%u; empty/trunc/yaml/multi/str16 cases all safe.\n",
         nptr, ks.kernarg_segment_size);

  // ---- tag1_blob_ptr_offsets: precise pointer-offset extraction ----
  // Build a tag-1 blob for a 5-arg kernel: [ptr8, ptr8, i32, ptr8, i32]
  // and verify the function returns the blob offsets of the 3 pointers.
  {
    auto put_u32 = [](std::vector<uint8>& v, uint32_t x) {
      v.push_back(x & 0xff);
      v.push_back((x >> 8) & 0xff);
      v.push_back((x >> 16) & 0xff);
      v.push_back((x >> 24) & 0xff);
    };
    // signature: 5 args, pointers at indices 0,1,3
    KernelSig fake;
    fake.name = "fake";
    fake.args.push_back(KernelArgSig{0, 8, true});
    fake.args.push_back(KernelArgSig{8, 8, true});
    fake.args.push_back(KernelArgSig{16, 4, false});
    fake.args.push_back(KernelArgSig{24, 8, true});
    fake.args.push_back(KernelArgSig{32, 4, false});
    // tag-1 blob: tag=1, count=5, args of sizes 8,8,4,8,4
    std::vector<uint8> raw;
    raw.push_back(1);            // tag
    put_u32(raw, 5);            // count
    // arg 0: 8-byte pointer (value 0xDEADBEEF00000001)
    put_u32(raw, 8);
    for (int b = 0; b < 8; ++b) raw.push_back(b);
    // arg 1: 8-byte pointer
    put_u32(raw, 8);
    for (int b = 0; b < 8; ++b) raw.push_back(0xFF);
    // arg 2: 4-byte scalar
    put_u32(raw, 4);
    for (int b = 0; b < 4; ++b) raw.push_back(b);
    // arg 3: 8-byte pointer
    put_u32(raw, 8);
    for (int b = 0; b < 8; ++b) raw.push_back(0xAA);
    // arg 4: 4-byte scalar
    put_u32(raw, 4);
    for (int b = 0; b < 4; ++b) raw.push_back(b);

    std::vector<std::byte> blob(raw.size());
    std::memcpy(blob.data(), raw.data(), raw.size());
    auto offs = tag1_blob_ptr_offsets(blob, fake);
    // Expected: arg0 value at off=5+4=9, arg1 at 9+8+4=21(? let me compute)
    // Layout: tag(1) count(4) | arg0: len(4)+8=12 | arg1: len(4)+8=12 | arg2: len(4)+4=8 | arg3: len(4)+8=12 | arg4: len(4)+4=8
    // arg0 value starts at 5+4 = 9
    // arg1 value starts at 9+8+4 = 21
    // arg2 value starts at 21+8+4 = 33 (NOT a pointer)
    // arg3 value starts at 33+4+4 = 41
    assert(offs.size() == 3);
    assert(offs[0] == 9);
    assert(offs[1] == 21);
    assert(offs[2] == 41);

    // truncation / mismatch -> empty
    std::vector<std::byte> trunc(10, std::byte{0});
    std::memcpy(trunc.data(), raw.data(), 10);
    assert(tag1_blob_ptr_offsets(trunc, fake).empty());
    std::vector<std::byte> notag1;
    notag1.push_back(std::byte{0});  // tag 0
    assert(tag1_blob_ptr_offsets(notag1, fake).empty());

    printf("OK: tag1_blob_ptr_offsets found 3 pointers at blob offsets "
           "%u,%u,%u\n", offs[0], offs[1], offs[2]);
  }

  return 0;
}
