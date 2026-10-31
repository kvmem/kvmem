#pragma once

#include "kvmem/memory_contract.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace rad { struct KVGroupInfo; struct KVGroupPlan; }

namespace kvmem::radiance {

// Physical IDs for one logical selection block, in chronological page order.
// backed_blocks must come from KVManager::live_blocks(), not the reservation's
// n_blocks: elastic pools may lend the unbacked tail to model weights.
struct GroupPages {
    int32_t group = -1;
    uint64_t backed_blocks = 0;
    std::vector<int32_t> pages;
};
struct CopySpan {
    int32_t group = -1;
    uint64_t pool_offset = 0; // relative to that group's base, NOT the entire KV pool
    uint64_t record_offset = 0;
    uint64_t bytes = 0;
};

// Address translation only. No device allocation, policy, or capability claims.
// Stateful groups have a separate checkpoint contract and are never KV pages.
class NativePages {
public:
    NativePages(const std::vector<rad::KVGroupPlan>&,
                const std::vector<rad::KVGroupInfo>&,
                uint32_t selection_tokens, const std::string& encoding_identity);
    const PayloadLayout& layout() const noexcept { return layout_; }
    // Generate all spans before issuing any asynchronous copies. The caller
    // holds page ownership and pool backing until the transfer has completed.
    std::vector<CopySpan> spans(const std::vector<rad::KVGroupPlan>& current,
                                const std::vector<GroupPages>&) const;
private:
    struct Group {
        int32_t id;
        int kind;
        uint32_t pages_per_record;
        uint64_t page_tokens, layers, fragment, capacity, layer_stride, record_offset;
    };
    PayloadLayout layout_;
    std::vector<Group> groups_;
};
} // namespace kvmem::radiance
