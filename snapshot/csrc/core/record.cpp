#include "snapshot/record.hpp"

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace snapshot {
namespace {

// ---- ELF64 header field readers (little-endian code objects on amd64) ------
//
// A HIP/HSACO code object (ROCm code object v3+) is an ELF file. hipModuleLoadData
// gives a pointer with no length, so we recover the total image size from the
// header: the file ends at the last byte of the section header table (the
// common case for code objects) or the program header table, whichever is later.
// We also accept the ELF header's own size as a floor.

bool elf_magic_ok(const std::byte* image) {
  const unsigned char* p = reinterpret_cast<const unsigned char*>(image);
  return p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F';
}

std::uint16_t le16(const std::byte* p) {
  const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
  return static_cast<std::uint16_t>(b[0]) |
         (static_cast<std::uint16_t>(b[1]) << 8);
}

std::uint64_t le64(const std::byte* p) {
  const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(b[i]) << (i * 8);
  }
  return v;
}

std::uint32_t le32(const std::byte* p) {
  const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
  return static_cast<std::uint32_t>(b[0]) |
         (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) |
         (static_cast<std::uint32_t>(b[3]) << 24);
}

}  // namespace

std::uint64_t elf_code_object_size(const std::byte* image, std::size_t max_cap) {
  if (image == nullptr) {
    return 0;
  }
  // ELF64 header is 64 bytes; require at least that much to parse.
  if (max_cap < 64 || !elf_magic_ok(image)) {
    return 0;
  }
  const std::uint8_t ei_class = static_cast<std::uint8_t>(image[4]);
  if (ei_class != 2) {  // 2 == ELFCLASS64 (HSACO on amd64)
    return 0;
  }

  const std::uint64_t e_ehsize = le16(image + 0x34);
  const std::uint64_t e_phoff = le64(image + 0x20);
  const std::uint16_t e_phentsize = le16(image + 0x36);
  const std::uint16_t e_phnum = le16(image + 0x38);
  const std::uint64_t e_shoff = le64(image + 0x28);
  const std::uint16_t e_shentsize = le16(image + 0x3a);
  const std::uint16_t e_shnum = le16(image + 0x3c);

  std::uint64_t end = e_ehsize;
  if (e_shnum > 0 && e_shoff > 0) {
    const std::uint64_t sh_end =
        e_shoff + static_cast<std::uint64_t>(e_shnum) * e_shentsize;
    if (sh_end > end) {
      end = sh_end;
    }
  }
  if (e_phnum > 0 && e_phoff > 0) {
    const std::uint64_t ph_end =
        e_phoff + static_cast<std::uint64_t>(e_phnum) * e_phentsize;
    if (ph_end > end) {
      end = ph_end;
    }
  }

  if (end < 64 || end > max_cap) {
    return 0;  // header inconsistent or implausibly large; refuse to trust it
  }
  return end;
}

// Walk the symbol tables of an AMDGPU ELF64 code object and collect every symbol
// name (both SHT_SYMTAB and SHT_DYNSYM), each bounded by its linked string
// table's size so a missing terminator can't overrun. Every read is bounded by
// parsed section-header offsets, so a malformed image returns partial/empty
// rather than faulting. Used to build name -> image associations for kernels
// whose name we learned from hipKernelNameRef or __hipRegisterFunction but
// whose hipFunction_t handle did not match the one hipModuleGetFunction saw.
std::vector<std::string> extract_elf_symbols(const std::byte* image,
                                             std::size_t size) {
  std::vector<std::string> out;
  if (image == nullptr || size < 64 || !elf_magic_ok(image)) {
    return out;
  }
  if (static_cast<std::uint8_t>(image[4]) != 2) {  // ELFCLASS64
    return out;
  }
  const std::uint64_t e_shoff = le64(image + 0x28);
  const std::uint16_t e_shentsize = le16(image + 0x3a);
  const std::uint16_t e_shnum = le16(image + 0x3c);
  if (e_shnum == 0 || e_shoff == 0 || e_shentsize < 40 ||
      e_shoff + static_cast<std::uint64_t>(e_shnum) * e_shentsize > size) {
    return out;
  }
  const auto section =
      [&](std::uint16_t i) -> const std::byte* {
    return image + e_shoff + static_cast<std::uint64_t>(i) * e_shentsize;
  };
  // Helper: pull kernel names out of an AMDGPU code-object metadata note desc.
  // The desc is an ASCII YAML doc whose amdhsa.kernels entries look like
  // '  - .name: triton_red_fused__...'. We scan for ".name:" and take the
  // following token (up to newline/whitespace). Works for uncompressed metadata
  // (ROCm code-object v3; Triton). Compressed v4+ metadata won't contain the
  // literal and yields nothing (fall back to symtab / a zlib path later).
  auto scan_note_names = [&](const unsigned char* desc, std::size_t dlen,
                             std::vector<std::string>& dst) {
    static const char kTag[] = ".name:";
    for (std::size_t i = 0; i + 5 < dlen; ++i) {
      if (desc[i] != '.') continue;
      if (std::memcmp(desc + i, kTag, 6) != 0) continue;
      std::size_t j = i + 6;
      while (j < dlen && (desc[j] == ' ' || desc[j] == '\t')) ++j;
      std::size_t start = j;
      while (j < dlen) {
        unsigned char c = desc[j];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t' || c == '\0') break;
        ++j;
      }
      if (j > start) dst.emplace_back(reinterpret_cast<const char*>(desc + start),
                                      j - start);
      i = j;
    }
  };
  for (std::uint16_t s = 0; s < e_shnum; ++s) {
    const std::byte* shdr = section(s);
    const std::uint32_t sh_type = le32(shdr + 4);
    if (sh_type != 2 /*SHT_SYMTAB*/ && sh_type != 11 /*SHT_DYNSYM*/ &&
        sh_type != 8 /*SHT_NOTE*/) {
      continue;
    }
    const std::uint64_t sh_offset = le64(shdr + 0x18);
    const std::uint64_t sh_size = le64(shdr + 0x20);
    if (sh_offset >= size || sh_offset + sh_size > size) {
      continue;
    }
    if (sh_type == 8 /*SHT_NOTE*/) {
      // Walk ELF notes: {u32 namesz, u32 descsz, u32 type, name[pad4], desc[pad4]}.
      const unsigned char* base = reinterpret_cast<const unsigned char*>(image) +
                                  sh_offset;
      std::size_t off = 0;
      while (off + 12 <= sh_size) {
        const std::uint32_t namesz = le32(image + sh_offset + off);
        const std::uint32_t descsz = le32(image + sh_offset + off + 4);
        const std::uint32_t type = le32(image + sh_offset + off + 8);
        const std::size_t name_padded = (namesz + 3u) & ~3u;
        const std::size_t desc_padded = (descsz + 3u) & ~3u;
        if (off + 12 + name_padded + desc_padded > sh_size) break;
        const unsigned char* nm = base + 12;
        const unsigned char* desc = nm + name_padded;
        const bool is_amd = (namesz >= 3 && nm[0] == 'A' && nm[1] == 'M' &&
                             nm[2] == 'D');
        // NT_AMDGPU_METADATA (0x3a, code-object v3+) and the legacy
        // NT_AMD_HSA_METADATA (0x20) both carry the amdhsa YAML.
        if (is_amd && (type == 0x3a || type == 0x20)) {
          scan_note_names(desc, descsz, out);
        }
        off += 12 + name_padded + desc_padded;
      }
      continue;
    }
    const std::uint32_t sh_link = le32(shdr + 0x28);  // -> strtab section idx
    const std::uint64_t sh_entsize = le64(shdr + 0x38);
    if (sh_entsize == 0 || sh_link >= e_shnum) {
      continue;
    }
    const std::byte* strhdr = section(static_cast<std::uint16_t>(sh_link));
    const std::uint64_t str_off = le64(strhdr + 0x18);
    const std::uint64_t str_size = le64(strhdr + 0x20);
    if (str_off >= size || str_off + str_size > size) {
      continue;
    }
    const char* strtab = reinterpret_cast<const char*>(image) + str_off;
    const std::uint64_t nsym = sh_size / sh_entsize;
    for (std::uint64_t n = 0; n < nsym; ++n) {
      const std::byte* sym = image + sh_offset + n * sh_entsize;
      const std::uint32_t st_name = le32(sym);
      if (st_name == 0 || st_name >= str_size) {
        continue;
      }
      const char* s = strtab + st_name;
      const std::size_t maxlen = static_cast<std::size_t>(str_size - st_name);
      const std::size_t len = strnlen(s, maxlen);
      if (len > 0) {
        out.emplace_back(s, len);
      }
    }
  }
  return out;
}

// ---- AMDGPU code-object metadata (MessagePack, code-object v4+) -------------
//
// The NT_AMDGPU_METADATA note carries the amdhsa YAML. For code-object v3 it is
// plain ASCII YAML (handled by scan_note_names above). For v4+ it is encoded as
// MessagePack. The msgpack map looks like:
//   { "amdhsa.version": [int,int], "amdhsa.target": str,
//     "amdhsa.kernels": [ { ".name": str, ".kernarg_segment_size": int,
//       ".args": [ { ".size": int, ".offset": int,
//                    ".value_kind": "global_buffer"|"by_value"|... } ] } ] }
// We walk this with a minimal reader to extract, per kernel, the name, the
// total kernarg size, and each arg's (offset, size, pointer-ness). The rebuild
// uses these for EXACT arg counts (the launch-time readability heuristic
// undercounts — aiter fused_qk_rmsnorm: 12 captured vs 15 declared) and for
// precise pointer offsets (replaces blind-scan relocation). Pure parsing.
namespace {

// Minimal big-endian MessagePack cursor. Supports the full value set except
// that it never materializes floats/exts (only skips them). err is sticky.
struct Mp {
  const unsigned char* p;
  std::size_t len;
  std::size_t off = 0;
  bool err = false;

  bool need(std::size_t n) {
    if (err || off + n > len) {
      err = true;
      return false;
    }
    return true;
  }
  std::uint16_t be16() {
    return (static_cast<std::uint16_t>(p[off]) << 8) |
           static_cast<std::uint16_t>(p[off + 1]);
  }
  std::uint32_t be32() {
    return (static_cast<std::uint32_t>(p[off]) << 24) |
           (static_cast<std::uint32_t>(p[off + 1]) << 16) |
           (static_cast<std::uint32_t>(p[off + 2]) << 8) |
           static_cast<std::uint32_t>(p[off + 3]);
  }
  std::uint64_t be64() {
    return (static_cast<std::uint64_t>(p[off]) << 56) |
           (static_cast<std::uint64_t>(p[off + 1]) << 48) |
           (static_cast<std::uint64_t>(p[off + 2]) << 40) |
           (static_cast<std::uint64_t>(p[off + 3]) << 32) |
           (static_cast<std::uint64_t>(p[off + 4]) << 24) |
           (static_cast<std::uint64_t>(p[off + 5]) << 16) |
           (static_cast<std::uint64_t>(p[off + 6]) << 8) |
           static_cast<std::uint64_t>(p[off + 7]);
  }

  bool skip_body(std::size_t n) { return need(n) ? (off += n, true) : false; }
  bool skip_array(std::size_t n) {
    for (; n; --n)
      if (!skip()) return false;
    return true;
  }
  bool skip_map(std::size_t n) {
    for (; n; --n) {
      if (!skip() || !skip()) return false;
    }
    return true;
  }

  // Skip exactly one msgpack value.
  bool skip() {
    if (!need(1)) return false;
    unsigned char c = p[off];
    if (c <= 0x7f || c >= 0xe0) {
      off += 1;
      return true;  // fixint
    }
    if ((c & 0xf0) == 0x80) {
      off += 1;
      return skip_map(c & 0x0f);  // fixmap
    }
    if ((c & 0xf0) == 0x90) {
      off += 1;
      return skip_array(c & 0x0f);  // fixarray
    }
    if ((c & 0xe0) == 0xa0) {
      off += 1;
      return skip_body(c & 0x1f);  // fixstr (advance past tag, then body)
    }
    switch (c) {
      case 0xc0: case 0xc2: case 0xc3:
        off += 1;
        return true;  // nil/false/true
      case 0xc4: {  // bin8
        off += 1;
        if (!need(1)) return false;
        std::size_t n = p[off++];
        return skip_body(n);
      }
      case 0xc5: {  // bin16
        off += 1;
        if (!need(2)) return false;
        std::size_t n = be16();
        off += 2;
        return skip_body(n);
      }
      case 0xc6: {  // bin32
        off += 1;
        if (!need(4)) return false;
        std::size_t n = be32();
        off += 4;
        return skip_body(n);
      }
      case 0xca: off += 1; return need(4) && (off += 4, true);   // float32
      case 0xcb: off += 1; return need(8) && (off += 8, true);   // float64
      case 0xcc: off += 1; return need(1) && (off += 1, true);   // uint8
      case 0xcd: off += 1; return need(2) && (off += 2, true);   // uint16
      case 0xce: off += 1; return need(4) && (off += 4, true);   // uint32
      case 0xcf: off += 1; return need(8) && (off += 8, true);   // uint64
      case 0xd0: off += 1; return need(1) && (off += 1, true);   // int8
      case 0xd1: off += 1; return need(2) && (off += 2, true);   // int16
      case 0xd2: off += 1; return need(4) && (off += 4, true);   // int32
      case 0xd3: off += 1; return need(8) && (off += 8, true);   // int64
      case 0xd4: off += 1; return need(2) && (off += 2, true);   // fixext1
      case 0xd5: off += 1; return need(3) && (off += 3, true);   // fixext2
      case 0xd6: off += 1; return need(5) && (off += 5, true);   // fixext4
      case 0xd7: off += 1; return need(9) && (off += 9, true);   // fixext8
      case 0xd8: off += 1; return need(17) && (off += 17, true);  // fixext16
      case 0xd9: {  // str8
        off += 1;
        if (!need(1)) return false;
        std::size_t n = p[off++];
        return skip_body(n);
      }
      case 0xda: {  // str16
        off += 1;
        if (!need(2)) return false;
        std::size_t n = be16();
        off += 2;
        return skip_body(n);
      }
      case 0xdb: {  // str32
        off += 1;
        if (!need(4)) return false;
        std::size_t n = be32();
        off += 4;
        return skip_body(n);
      }
      case 0xdc: {  // array16
        off += 1;
        if (!need(2)) return false;
        std::size_t n = be16();
        off += 2;
        return skip_array(n);
      }
      case 0xdd: {  // array32
        off += 1;
        if (!need(4)) return false;
        std::size_t n = be32();
        off += 4;
        return skip_array(n);
      }
      case 0xde: {  // map16
        off += 1;
        if (!need(2)) return false;
        std::size_t n = be16();
        off += 2;
        return skip_map(n);
      }
      case 0xdf: {  // map32
        off += 1;
        if (!need(4)) return false;
        std::size_t n = be32();
        off += 4;
        return skip_map(n);
      }
      case 0xc7: case 0xc8: case 0xc9: {  // ext8/16/32
        off += 1;
        std::size_t n;
        if (c == 0xc7) {
          if (!need(1)) return false;
          n = p[off++];
        } else if (c == 0xc8) {
          if (!need(2)) return false;
          n = be16();
          off += 2;
        } else {
          if (!need(4)) return false;
          n = be32();
          off += 4;
        }
        off += 1;  // type byte
        return skip_body(n);
      }
      default:
        err = true;
        return false;
    }
  }

  // Read a non-negative integer into out. Returns false on non-int/negative.
  bool read_uint(std::uint64_t& out) {
    if (!need(1)) return false;
    unsigned char c = p[off];
    if (c <= 0x7f) {
      out = c;
      off += 1;
      return true;
    }
    switch (c) {
      case 0xcc: off += 1; return need(1) && (out = p[off], off += 1, true);
      case 0xcd: off += 1; return need(2) && (out = be16(), off += 2, true);
      case 0xce: off += 1; return need(4) && (out = be32(), off += 4, true);
      case 0xcf: off += 1; return need(8) && (out = be64(), off += 8, true);
      default:
        return false;  // negative/sint not expected for sizes/offsets
    }
  }

  // Read a str into out. Returns false on non-str.
  bool read_str(std::string& out) {
    if (!need(1)) return false;
    unsigned char c = p[off];
    std::size_t n;
    if ((c & 0xe0) == 0xa0) {
      n = c & 0x1f;
      off += 1;
    } else if (c == 0xd9) {
      off += 1;
      if (!need(1)) return false;
      n = p[off++];
    } else if (c == 0xda) {
      off += 1;
      if (!need(2)) return false;
      n = be16();
      off += 2;
    } else if (c == 0xdb) {
      off += 1;
      if (!need(4)) return false;
      n = be32();
      off += 4;
    } else {
      return false;
    }
    if (!need(n)) return false;
    out.assign(reinterpret_cast<const char*>(p + off), n);
    off += n;
    return true;
  }
};

bool is_pointer_value_kind(const std::string& k) {
  // AMDGPU .value_kind device-address kinds (per AMDHSA code-object spec).
  return k == "global_buffer" || k == "dynamic_shared_pointer" ||
         k == "image" || k == "sampler" || k == "queue" || k == "pipe";
}

// Parse one amdhsa.kernels entry (a msgpack map) into a KernelSig, advancing
// m to just past the map. Returns false on any structural error.
bool parse_kernel_map(Mp& m, KernelSig& ks) {
  if (!m.need(1)) return false;
  unsigned char c = m.p[m.off];
  std::size_t npair;
  if ((c & 0xf0) == 0x80) {
    npair = c & 0x0f;
    m.off += 1;
  } else if (c == 0xde) {
    m.off += 1;
    if (!m.need(2)) return false;
    npair = m.be16();
    m.off += 2;
  } else if (c == 0xdf) {
    m.off += 1;
    if (!m.need(4)) return false;
    npair = m.be32();
    m.off += 4;
  } else {
    return false;
  }
  for (std::size_t i = 0; i < npair; ++i) {
    std::string key;
    if (!m.read_str(key)) return false;
    if (key == ".name") {
      if (!m.read_str(ks.name)) return false;
    } else if (key == ".kernarg_segment_size") {
      std::uint64_t v = 0;
      if (!m.read_uint(v)) return false;
      ks.kernarg_segment_size = static_cast<std::uint32_t>(v);
    } else if (key == ".kernarg_segment_align") {
      std::uint64_t v = 0;
      if (!m.read_uint(v)) return false;
      ks.kernarg_alignment = static_cast<std::uint32_t>(v ? v : 1);
    } else if (key == ".args") {
      if (!m.need(1)) return false;
      unsigned char ac = m.p[m.off];
      std::size_t nargs;
      if ((ac & 0xf0) == 0x90) {
        nargs = ac & 0x0f;
        m.off += 1;
      } else if (ac == 0xdc) {
        m.off += 1;
        if (!m.need(2)) return false;
        nargs = m.be16();
        m.off += 2;
      } else if (ac == 0xdd) {
        m.off += 1;
        if (!m.need(4)) return false;
        nargs = m.be32();
        m.off += 4;
      } else {
        return false;
      }
      ks.args.reserve(nargs);
      for (std::size_t a = 0; a < nargs; ++a) {
        if (!m.need(1)) return false;
        unsigned char amc = m.p[m.off];
        std::size_t argpairs;
        if ((amc & 0xf0) == 0x80) {
          argpairs = amc & 0x0f;
          m.off += 1;
        } else if (amc == 0xde) {
          m.off += 1;
          if (!m.need(2)) return false;
          argpairs = m.be16();
          m.off += 2;
        } else {
          return false;
        }
        KernelArgSig as;
        for (std::size_t ap = 0; ap < argpairs; ++ap) {
          std::string ak;
          if (!m.read_str(ak)) return false;
          if (ak == ".size") {
            std::uint64_t v = 0;
            if (!m.read_uint(v)) return false;
            as.size = static_cast<std::uint32_t>(v);
          } else if (ak == ".offset") {
            std::uint64_t v = 0;
            if (!m.read_uint(v)) return false;
            as.offset = static_cast<std::uint32_t>(v);
          } else if (ak == ".value_kind") {
            std::string vk;
            if (!m.read_str(vk)) return false;
            as.is_pointer = is_pointer_value_kind(vk);
          } else {
            if (!m.skip()) return false;
          }
        }
        ks.args.push_back(std::move(as));
      }
    } else {
      if (!m.skip()) return false;
    }
  }
  return true;
}

// Parse the full amdhsa metadata msgpack document (the note desc bytes).
std::vector<KernelSig> parse_amdhsa_msgpack(const unsigned char* desc,
                                            std::size_t dlen) {
  std::vector<KernelSig> out;
  Mp m{desc, dlen, 0, false};
  if (!m.need(1)) return out;
  unsigned char c = m.p[m.off];
  std::size_t npair;
  if ((c & 0xf0) == 0x80) {
    npair = c & 0x0f;
    m.off += 1;
  } else if (c == 0xde) {
    m.off += 1;
    if (!m.need(2)) return out;
    npair = m.be16();
    m.off += 2;
  } else if (c == 0xdf) {
    m.off += 1;
    if (!m.need(4)) return out;
    npair = m.be32();
    m.off += 4;
  } else {
    return out;  // not a map -> not amdhsa msgpack
  }
  for (std::size_t i = 0; i < npair; ++i) {
    std::string key;
    if (!m.read_str(key) || m.err) return out;
    if (key == "amdhsa.kernels") {
      if (!m.need(1)) return out;
      unsigned char kc = m.p[m.off];
      std::size_t nkernels;
      if ((kc & 0xf0) == 0x90) {
        nkernels = kc & 0x0f;
        m.off += 1;
      } else if (kc == 0xdc) {
        m.off += 1;
        if (!m.need(2)) return out;
        nkernels = m.be16();
        m.off += 2;
      } else if (kc == 0xdd) {
        m.off += 1;
        if (!m.need(4)) return out;
        nkernels = m.be32();
        m.off += 4;
      } else {
        if (!m.skip()) return out;
        continue;
      }
      out.reserve(nkernels);
      for (std::size_t k = 0; k < nkernels; ++k) {
        KernelSig ks;
        if (!parse_kernel_map(m, ks) || m.err) return out;
        out.push_back(std::move(ks));
      }
    } else {
      if (!m.skip()) return out;
    }
  }
  return out;
}

}  // namespace

std::vector<KernelSig> extract_amdgpu_kernels(const std::byte* image,
                                              std::size_t size) {
  std::vector<KernelSig> out;
  if (image == nullptr || size < 64 || !elf_magic_ok(image)) {
    return out;
  }
  if (static_cast<std::uint8_t>(image[4]) != 2) {  // ELFCLASS64
    return out;
  }
  const std::uint64_t e_shoff = le64(image + 0x28);
  const std::uint16_t e_shentsize = le16(image + 0x3a);
  const std::uint16_t e_shnum = le16(image + 0x3c);
  if (e_shnum == 0 || e_shoff == 0 || e_shentsize < 40 ||
      e_shoff + static_cast<std::uint64_t>(e_shnum) * e_shentsize > size) {
    return out;
  }
  for (std::uint16_t s = 0; s < e_shnum; ++s) {
    const std::byte* shdr = image + e_shoff +
                             static_cast<std::uint64_t>(s) * e_shentsize;
    if (le32(shdr + 4) != 8 /*SHT_NOTE*/) continue;
    const std::uint64_t sh_offset = le64(shdr + 0x18);
    const std::uint64_t sh_size = le64(shdr + 0x20);
    if (sh_offset >= size || sh_offset + sh_size > size) continue;
    std::size_t off = 0;
    while (off + 12 <= sh_size) {
      const std::uint32_t namesz = le32(image + sh_offset + off);
      const std::uint32_t descsz = le32(image + sh_offset + off + 4);
      const std::uint32_t type = le32(image + sh_offset + off + 8);
      const std::size_t name_padded = (namesz + 3u) & ~3u;
      const std::size_t desc_padded = (descsz + 3u) & ~3u;
      if (off + 12 + name_padded + desc_padded > sh_size) break;
      const unsigned char* nm =
          reinterpret_cast<const unsigned char*>(image) + sh_offset + 12;
      const unsigned char* desc = nm + name_padded;
      const bool is_amd = (namesz >= 3 && nm[0] == 'A' && nm[1] == 'M' &&
                           nm[2] == 'D');
      // NT_AMDGPU_METADATA carries the amdhsa kernels as msgpack (code-object
      // v4+). The note TYPE varies by code-object version: older v3 uses 0x3a
      // with name "AMD"; v4+ uses 0x20 with name "AMDGPU". Both carry the
      // same logical payload. We require a msgpack map header (0x80-0x8f/
      // 0xde/0xdf) to distinguish msgpack from the v3 YAML form (handled by
      // extract_elf_symbols for names only). parse_amdhsa_msgpack safely
      // returns empty if the payload lacks an amdhsa.kernels key.
      const bool map_hdr = (desc[0] & 0xf0) == 0x80 || desc[0] == 0xde ||
                           desc[0] == 0xdf;
      if (is_amd && (type == 0x3a || type == 0x20) && descsz > 0 && map_hdr) {
        auto ks = parse_amdhsa_msgpack(desc, descsz);
        for (auto& k : ks) out.push_back(std::move(k));
      }
      off += 12 + name_padded + desc_padded;
    }
  }
  return out;
}

namespace {

const ModuleImage* find_module_by_hash(const std::vector<ModuleImage>& modules,
                                       std::uint64_t hash) {
  for (const ModuleImage& m : modules) {
    if (m.hash == hash) {
      return &m;
    }
  }
  return nullptr;
}

}  // namespace

std::vector<std::uint32_t> tag1_blob_ptr_offsets(
    const std::vector<std::byte>& blob, const KernelSig& sig) {
  std::vector<std::uint32_t> out;
  // tag-1 layout: {u8 tag=1, u32 count, per arg: {u32 len, bytes[len]}}
  if (blob.size() < 5 || static_cast<unsigned char>(blob[0]) != 1) return out;
  const std::uint32_t count = static_cast<std::uint32_t>(
      static_cast<unsigned char>(blob[1])) |
      (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[2])) << 8) |
      (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[3])) << 16) |
      (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[4])) << 24);
  std::size_t off = 5;  // cursor past tag + count
  const std::size_t nargs =
      std::min<std::size_t>(count, sig.args.size());
  for (std::size_t i = 0; i < nargs; ++i) {
    if (off + 4 > blob.size()) return {};  // truncated -> bail (empty)
    const std::uint32_t len = static_cast<std::uint32_t>(
        static_cast<unsigned char>(blob[off])) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[off + 1]))
         << 8) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[off + 2]))
         << 16) |
        (static_cast<std::uint32_t>(static_cast<unsigned char>(blob[off + 3]))
         << 24);
    off += 4;  // past the length prefix
    if (off + len > blob.size()) return {};
    // The arg value occupies [off, off+len). If the signature says this arg is
    // a pointer, record its start offset in the blob.
    if (sig.args[i].is_pointer) {
      out.push_back(static_cast<std::uint32_t>(off));
    }
    off += len;
  }
  return out;
}

Status assemble_recorded_snapshot(const RecordAssembly& in, SnapshotData* out) {
  if (out == nullptr) {
    return Status::invalid_argument("assemble_recorded_snapshot output is null");
  }

  // Dedupe modules by hash, preserving first-seen order. Each module's entry
  // names are the union of recorded functions that reference it.
  std::map<std::uint64_t, std::size_t> module_index_by_hash;
  out->modules.clear();
  for (const RecordedModule& rm : in.modules) {
    if (rm.hash == 0) {
      continue;
    }
    auto it = module_index_by_hash.find(rm.hash);
    if (it == module_index_by_hash.end()) {
      ModuleImage m;
      m.hash = rm.hash;
      m.image = rm.image;  // possibly empty if the ELF size could not be parsed
      module_index_by_hash.emplace(rm.hash, out->modules.size());
      out->modules.push_back(std::move(m));
    } else if (out->modules[it->second].image.empty() && !rm.image.empty()) {
      // A later load of the same hash finally gave us the image bytes.
      out->modules[it->second].image = rm.image;
    }
  }

  // Attach the entry names each module is launched under. (A module may be
  // queried for functions it never launches; we still record those names so a
  // restore can resolve any of them.)
  for (const RecordedFunction& rf : in.functions) {
    if (rf.module_hash == 0 || rf.entry_name.empty()) {
      continue;
    }
    auto it = module_index_by_hash.find(rf.module_hash);
    if (it == module_index_by_hash.end()) {
      continue;  // function whose module was never recorded; launch is "unknown"
    }
    std::vector<std::string>& names = out->modules[it->second].entry_names;
    bool seen = false;
    for (const std::string& n : names) {
      if (n == rf.entry_name) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      names.push_back(rf.entry_name);
    }
  }

  // Build the kernel-node chain. node.id is 1-based to match the rest of the
  // codebase; edges form a serial chain (launch i depends on launch i-1), which
  // is the structure hipStreamBeginCapture/EndCapture produces.
  out->graph.nodes.clear();
  out->graph.edges.clear();
  out->graph.nodes.reserve(in.launches.size());
  for (std::size_t i = 0; i < in.launches.size(); ++i) {
    const RecordedLaunch& rl = in.launches[i];
    GraphNodeIR node;
    node.id = static_cast<std::uint64_t>(i + 1);
    node.type = GraphNodeType::kKernel;

    // Resolve identity. A launch whose function was never recorded (or whose
    // module we never saw loaded) becomes an "unknown" node: module_hash=0,
    // entry_name empty. restore refuses such nodes; summarize_snapshot flags
    // them so the M3a.3 gate can fail loudly on a non-zero unknown count.
    std::uint64_t module_hash = 0;
    std::string entry_name;
    for (const RecordedFunction& rf : in.functions) {
      if (rf.function_id == rl.function_id) {
        module_hash = rf.module_hash;
        entry_name = rf.entry_name;
        break;
      }
    }
    node.module_hash = module_hash;
    node.entry_name = std::move(entry_name);

    node.kernel.grid = rl.grid;
    node.kernel.block = rl.block;
    node.kernel.shared_mem_bytes = rl.shared_mem_bytes;
    node.kernel.param_blob = rl.param_blob;
    // ptr_offsets intentionally empty: the interposer cannot tell which param
    // bytes are pointers. Blind-scan relocation at restore time discovers them
    // over the recorded region [region_base, region_base + region_size).
    out->graph.nodes.push_back(std::move(node));
    if (i > 0) {
      out->graph.edges.push_back(GraphEdgeIR{static_cast<std::uint64_t>(i),
                                            static_cast<std::uint64_t>(i + 1)});
    }
  }

  out->vendor = in.vendor;
  out->arch = in.arch;
  out->allocator.requested_base = kDefaultRequestedBase;
  out->allocator.region_base = in.region_base;
  out->allocator.region_size = in.region_size;
  out->allocator.granularity = in.granularity == 0 ? 4096 : in.granularity;
  out->allocator.fixed_base_honored =
      in.region_base == kDefaultRequestedBase && in.region_base != 0;
  out->allocator.events = in.alloc_events;
  // cursor is informational (high-water mark of the bump); reconstruct it so a
  // reader does not see a stale zero.
  std::uint64_t cursor = 0;
  for (const AllocEvent& e : in.alloc_events) {
    const std::uint64_t end = e.offset + e.size;
    if (end > cursor) {
      cursor = end;
    }
  }
  out->allocator.cursor = cursor;
  return Status::Ok();
}

Status summarize_snapshot(const SnapshotData& in, SnapshotSummary* out) {
  if (out == nullptr) {
    return Status::invalid_argument("summarize_snapshot output is null");
  }
  *out = {};
  out->captured_base = in.allocator.region_base;
  out->region_size = in.allocator.region_size;
  out->alloc_events = in.allocator.events.size();

  out->module_count = in.modules.size();
  for (const ModuleImage& m : in.modules) {
    out->total_module_bytes += m.image.size();
    if (m.image.empty()) {
      ++out->modules_with_empty_image;
    }
  }

  out->node_count = in.graph.nodes.size();
  for (const GraphNodeIR& n : in.graph.nodes) {
    out->total_param_bytes += n.kernel.param_blob.size();
    if (n.type != GraphNodeType::kKernel) {
      continue;
    }
    ++out->kernel_nodes;
    const bool has_identity = n.module_hash != 0 && !n.entry_name.empty();
    if (has_identity) {
      ++out->nodes_with_identity;
      // Is the referenced module's image actually present (reloadable)?
      const ModuleImage* m = find_module_by_hash(in.modules, n.module_hash);
      if (m == nullptr || m->image.empty()) {
        ++out->nodes_with_empty_module_image;
      }
    } else {
      ++out->nodes_without_identity;
    }
  }
  return Status::Ok();
}

}  // namespace snapshot
