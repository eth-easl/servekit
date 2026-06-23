#include "snapshot/snapshot_format.hpp"

namespace snapshot {

std::uint32_t crc32(const std::byte* data, std::size_t n) {
  static constexpr std::uint32_t kPolynomial = 0xEDB88320U;
  std::uint32_t crc = 0xFFFFFFFFU;

  for (std::size_t i = 0; i < n; ++i) {
    crc ^= static_cast<std::uint32_t>(data[i]);
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (kPolynomial & mask);
    }
  }

  return ~crc;
}

std::uint64_t hash_bytes(const std::byte* data, std::size_t n) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) {
    hash ^= static_cast<std::uint8_t>(data[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace snapshot
