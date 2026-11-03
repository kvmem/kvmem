#pragma once
#include <cstdint>
namespace kvmem::radiance {
enum class KvPacking : uint32_t { Plain = 0, Fp8V4 = 1, K6V4 = 2 };
constexpr uint32_t kv_row_elements(uint32_t head_dim, KvPacking packing) {
    return packing == KvPacking::Fp8V4 ? head_dim * 25 / 16 :
           packing == KvPacking::K6V4 ? head_dim * 22 / 16 : 2 * head_dim;
}
// Layout version 1: upstream 203064b. 16 channels share four high-bit bytes;
// each byte holds two channels from each half of that 16-channel chunk.
#if defined(__HIPCC__)
__host__ __device__
#endif
inline int packed_k6_code(const uint8_t* row, uint32_t hd, uint32_t d) {
    const unsigned low=(row[d/2] >> (4*(d&1))) & 15;
    const unsigned high=(row[hd/2+(d/16)*4+(d&7)/2] >> (((d&8)?4:0)+2*(d&1))) & 3;
    const int code=int(low|(high<<4));
    return code>=32 ? code-64 : code;
}
} // namespace kvmem::radiance
