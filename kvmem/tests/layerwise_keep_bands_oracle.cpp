// Exact primary-source fixtures: kvmem-vllm 72a81cc, third_party/qw3/src/kvmem_store.cpp.
#include "kvmem/kvmem_store.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
namespace upstream_keep_oracle {
using kvmem::KvMemKeepAllocation;using kvmem::KvMemKeepSource;
uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    if (value == 0) return 0;
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(value) + divisor - 1) / divisor);
}

uint32_t clamp_u32(uint32_t value, uint32_t lo, uint32_t hi) {
    return std::min(std::max(value, lo), hi);
}

uint32_t ceil_percent(uint32_t value, uint32_t percent) {
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(value) * percent + 99u) / 100u);
}

KvMemKeepAllocation resolve_kvmem_keep_allocation(
        uint32_t block_tokens,
        uint32_t select_budget,
        int64_t sink_blocks,
        int64_t recent_blocks,
        int64_t sink_tokens,
        int64_t recent_tokens) {
    if (block_tokens == 0) {
        throw std::runtime_error(
            "KVMem keep allocation requires block_tokens > 0");
    }
    if (select_budget < block_tokens) {
        throw std::runtime_error(
            "KVMem selection budget must contain at least one block");
    }
    if (sink_blocks >= 0 && sink_tokens >= 0) {
        throw std::runtime_error(
            "KVMem sink allocation cannot specify both blocks and tokens");
    }
    if (recent_blocks >= 0 && recent_tokens >= 0) {
        throw std::runtime_error(
            "KVMem recent allocation cannot specify both blocks and tokens");
    }

    KvMemKeepAllocation out;
    const uint32_t budget_blocks = select_budget / block_tokens;

    auto resolve_band = [&](int64_t explicit_blocks,
                            int64_t explicit_tokens,
                            uint32_t auto_tokens,
                            const char *name,
                            uint32_t &target_tokens,
                            uint32_t &blocks,
                            uint32_t &effective_tokens,
                            KvMemKeepSource &source) {
        if (explicit_blocks >= 0) {
            if (static_cast<uint64_t>(explicit_blocks) >
                std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    std::string("KVMem ") + name +
                    " block allocation is too large");
            }
            blocks = static_cast<uint32_t>(explicit_blocks);
            const uint64_t tokens =
                static_cast<uint64_t>(blocks) * block_tokens;
            if (tokens > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    std::string("KVMem ") + name +
                    " block allocation overflows token accounting");
            }
            target_tokens = static_cast<uint32_t>(tokens);
            effective_tokens = target_tokens;
            source = KvMemKeepSource::Blocks;
            return;
        }

        const uint64_t requested = explicit_tokens >= 0
            ? static_cast<uint64_t>(explicit_tokens)
            : static_cast<uint64_t>(auto_tokens);
        if (requested > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                std::string("KVMem ") + name +
                " token allocation is too large");
        }
        target_tokens = static_cast<uint32_t>(requested);
        blocks = ceil_div_u32(target_tokens, block_tokens);
        const uint64_t rounded_tokens =
            static_cast<uint64_t>(blocks) * block_tokens;
        if (rounded_tokens > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                std::string("KVMem ") + name +
                " rounded token allocation overflows accounting");
        }
        effective_tokens = static_cast<uint32_t>(rounded_tokens);
        source = explicit_tokens >= 0
            ? KvMemKeepSource::Tokens
            : KvMemKeepSource::Auto;
    };

    const uint32_t auto_sink_tokens =
        clamp_u32(ceil_percent(select_budget, 1), 1024, 2048);
    const uint32_t auto_recent_tokens =
        clamp_u32(ceil_percent(select_budget, 8), 4096, 16384);
    resolve_band(
        sink_blocks, sink_tokens, auto_sink_tokens, "sink",
        out.sink_target_tokens, out.sink_blocks,
        out.sink_effective_tokens, out.sink_source);
    resolve_band(
        recent_blocks, recent_tokens, auto_recent_tokens, "recent",
        out.recent_target_tokens, out.recent_blocks,
        out.recent_effective_tokens, out.recent_source);

    if (out.sink_blocks > budget_blocks ||
        out.recent_blocks > budget_blocks ||
        static_cast<uint64_t>(out.sink_blocks) + out.recent_blocks >
            budget_blocks) {
        throw std::runtime_error(
            "KVMem sink + recent allocation exceeds the selection budget");
    }
    return out;
}

// Only the fixture directory and method qualification differ. The method body
// below is the pinned implementation, with a dense, live canonical directory.
class FixedFixture {
public:
    uint32_t n;
    struct Config { uint32_t sink_blocks, recent_blocks; } cfg_;
    uint32_t block_count() const { return n; }
    bool is_live(uint32_t id) const { return id<n; }
    bool sink_block(uint32_t id) const { return id<cfg_.sink_blocks; }
    std::vector<uint32_t> fixed_selection_from(const std::vector<uint32_t>&,
        const std::vector<uint32_t>&,bool,uint32_t) const;
};
std::vector<uint32_t> FixedFixture::fixed_selection_from(const std::vector<uint32_t> &order,
        const std::vector<uint32_t> &mandatory, bool pressure, uint32_t budget) const {
    if (!budget || order.size() <= budget) return order;
    std::vector<uint8_t> kept(block_count(), 0);
    uint32_t count = 0;
    auto keep = [&](uint32_t id) {
        if (id < kept.size() && is_live(id) && !kept[id]) { kept[id] = 1; ++count; }
    };
    for (auto id : order) if (sink_block(id) && count < budget) keep(id);
    for (auto id : mandatory) keep(id);
    if (count > budget)
        throw std::runtime_error("KVMem mandatory selection plus sink blocks exceeds the configured selection budget");
    if (!pressure) {
        const auto recent = std::min<size_t>(cfg_.recent_blocks, order.size());
        for (size_t i = 0; i < recent && count < budget; ++i) keep(order[order.size()-1-i]);
    }
    std::vector<uint32_t> result;
    for (auto id : order) if (kept[id]) result.push_back(id);
    return result;
}
std::vector<uint32_t> fixed_selection(uint32_t n,uint32_t sink,uint32_t recent,
        const std::vector<uint32_t>& mandatory,uint32_t budget,bool pressure) {
    std::vector<uint32_t> order(n); std::iota(order.begin(),order.end(),0u);
    return FixedFixture{n,{sink,recent}}.fixed_selection_from(order,mandatory,pressure,budget);
}
}
