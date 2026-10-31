#pragma once
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace kvmem::radiance {
struct PoolGeometry { uint32_t block_tokens; uint32_t resident_tokens; };
// One live view plus bounded append/output slack. An immutable prefix stays
// shared on GPU while it fits; under pressure it moves to host before its refs
// are released. Exclusive evicted pages recycle AFTER spill with rollback.
inline PoolGeometry pool_geometry(uint32_t requested_block, uint32_t budget,
                                  uint32_t batch, const std::vector<uint32_t>& native_blocks) {
    constexpr uint64_t limit = std::numeric_limits<int32_t>::max();
    if (!requested_block || !batch || native_blocks.empty())
        throw std::invalid_argument("missing KV pool geometry");
    uint64_t block = requested_block;
    for (uint32_t native : native_blocks) {
        if (!native) throw std::invalid_argument("zero native page size");
        block = block / std::gcd(block, uint64_t(native)) * native;
        if (block > limit) throw std::invalid_argument("KV block alignment overflow");
    }
    if (!budget) return {uint32_t(block), 0};
    if (budget % block || budget < 2 * block)
        throw std::invalid_argument("selection budget needs whole blocks and sink + tail");
    const uint64_t append = ((uint64_t(batch) + block - 1) / block + 1) * block;
    const uint64_t resident = uint64_t(budget) + 2 * append;
    if (resident > limit) throw std::invalid_argument("KV residency exceeds index range");
    return {uint32_t(block), uint32_t(resident)};
}
inline uint32_t leased_pool_tokens(PoolGeometry geometry, uint32_t budget, uint32_t output_reserve) {
    if (!geometry.block_tokens || !output_reserve)
        throw std::invalid_argument("missing generation reserve");
    const uint64_t reserve = (uint64_t(output_reserve) + geometry.block_tokens - 1) /
                             geometry.block_tokens * geometry.block_tokens;
    if (!budget || geometry.resident_tokens < budget)
        throw std::invalid_argument("missing live/prefix pool allowance");
    const uint64_t total = uint64_t(geometry.resident_tokens) + reserve;
    if (total > uint64_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("leased KV pool overflow");
    return uint32_t(total);
}
inline bool output_fits_reserve(int64_t output, uint32_t batch, uint32_t reserve) {
    return output > 0 && uint64_t(output) <= reserve && batch <= reserve - uint64_t(output);
}
} // namespace kvmem::radiance
