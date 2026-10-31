#include "execution_pages.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace kvmem::radiance {
namespace {
[[noreturn]] void invalid(const char* message) {
    throw MemoryError(MemoryErrorCode::InvalidPlan, message);
}
}

ExecutionPages::ExecutionPages(const ExecutionView& view, uint32_t selection_tokens,
                             uint32_t page_tokens, uint64_t backed_pages,
                             const std::vector<BlockPages>& selected,
                             const std::vector<int32_t>& append_pages)
    : stamp_(view.stamp), visible_(view.visible_tokens), page_tokens_(page_tokens) {
    // Radiance's batch ABI stores positions, counts and flat slots as int32.
    constexpr uint64_t limit = std::numeric_limits<int32_t>::max();
    if (!stamp_.valid || !selection_tokens || !page_tokens ||
        selection_tokens % page_tokens || !backed_pages ||
        backed_pages > (limit + 1) / page_tokens || stamp_.frontier > limit ||
        visible_ > stamp_.frontier || selected.size() != view.blocks.size())
        invalid("invalid Radiance execution geometry or stamp");

    std::set<uint64_t> ids;
    std::set<int32_t> physical;
    auto add_page = [&](int32_t p) {
        if (p < 0 || uint64_t(p) >= backed_pages || !physical.insert(p).second)
            invalid("Radiance execution page is duplicate or unbacked");
        table_.push_back(p);
    };
    uint64_t compact = 0, previous_end = 0;
    for (size_t i = 0; i < view.blocks.size(); ++i) {
        const auto& v = view.blocks[i];
        const auto& b = v.block;
        const auto& binding = selected[i];
        if (!(binding.block == b) || !(b.session == stamp_.session) ||
            !ids.insert(b.id).second || !b.content_version || !b.valid_tokens ||
            b.valid_tokens > selection_tokens || b.original_start % selection_tokens ||
            b.original_start > stamp_.frontier ||
            b.valid_tokens > stamp_.frontier - b.original_start ||
            b.original_start < previous_end || v.compact_start != compact ||
            compact % page_tokens)
            invalid("stale, overlapping or misaligned Radiance selection");
        const uint64_t page_count = (uint64_t(b.valid_tokens) + page_tokens - 1) / page_tokens;
        if (binding.pages.size() != page_count)
            invalid("Radiance binding does not cover exactly the selected block");
        for (auto p : binding.pages) add_page(p);
        spans_.push_back({b.original_start, b.valid_tokens, compact});
        compact += b.valid_tokens;
        previous_end = b.original_start + b.valid_tokens;
    }
    if (compact != visible_ || (stamp_.frontier - visible_) % page_tokens)
        invalid("Radiance compact frontier is not page aligned with original frontier");
    // A partially filled final native page is safe to append ONLY if it is the
    // true history tail. Otherwise a new token would overwrite retained old KV.
    if (visible_ % page_tokens && previous_end != stamp_.frontier)
        invalid("partial retained page is not the accepted history tail");
    for (auto p : append_pages) add_page(p);
    if (table_.size() > limit / page_tokens)
        invalid("Radiance compact table exceeds int32 token range");
    capacity_ = uint64_t(table_.size()) * page_tokens;
    if (capacity_ < visible_) invalid("Radiance table is shorter than its valid tokens");
}

uint64_t ExecutionPages::used_after(uint64_t frontier) const {
    if (frontier < stamp_.frontier || frontier > uint64_t(std::numeric_limits<int32_t>::max()) ||
        frontier - stamp_.frontier > append_capacity())
        invalid("Radiance append is outside the reserved execution table");
    return visible_ + frontier - stamp_.frontier;
}

std::optional<int32_t> ExecutionPages::slot(uint64_t position) const noexcept {
    uint64_t compact;
    if (position >= stamp_.frontier) {
        const auto delta = position - stamp_.frontier;
        if (delta >= append_capacity()) return std::nullopt;
        compact = visible_ + delta;
    } else {
        auto it = std::upper_bound(spans_.begin(), spans_.end(), position,
            [](uint64_t pos, const Span& span) { return pos < span.original; });
        if (it == spans_.begin()) return std::nullopt;
        --it;
        if (position - it->original >= it->count) return std::nullopt;
        compact = it->compact + position - it->original;
    }
    return int32_t(uint64_t(table_[compact / page_tokens_]) * page_tokens_ +
                   compact % page_tokens_);
}
} // namespace kvmem::radiance
