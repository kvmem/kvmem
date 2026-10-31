#pragma once
#include "kvmem/memory_contract.hpp"
#include <cstdint>
#include <optional>
#include <vector>

namespace kvmem::radiance {

struct BlockPages {
    BlockDescriptor block;
    std::vector<int32_t> pages;
};

// One FULL-attention group's published execution table. The shared policy has
// already selected blocks; this class only binds them to backend-owned pages.
// Original positions (including RoPE positions) are never renumbered.
class ExecutionPages {
public:
    ExecutionPages(const ExecutionView&, uint32_t selection_tokens,
                   uint32_t native_page_tokens, uint64_t backed_pages,
                   const std::vector<BlockPages>& selected,
                   const std::vector<int32_t>& append_pages = {});

    const std::vector<int32_t>& table() const noexcept { return table_; }
    const MemoryStamp& stamp() const noexcept { return stamp_; }
    uint64_t visible_tokens() const noexcept { return visible_; }
    uint64_t append_capacity() const noexcept { return capacity_ - visible_; }
    // For NEW tokens only: compact address = original position - append_offset.
    // This is not the original position of the first retained historical page.
    uint64_t append_offset() const noexcept { return stamp_.frontier - visible_; }
    uint64_t used_after(uint64_t original_frontier) const;
    std::optional<int32_t> slot(uint64_t original_position) const noexcept;
private:
    struct Span { uint64_t original, count, compact; };
    MemoryStamp stamp_;
    uint64_t visible_ = 0, capacity_ = 0;
    uint32_t page_tokens_ = 0;
    std::vector<int32_t> table_;
    std::vector<Span> spans_;
};
} // namespace kvmem::radiance
