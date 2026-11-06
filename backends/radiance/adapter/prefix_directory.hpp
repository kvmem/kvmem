#pragma once
#include "kvmem/prefix_checkpoint.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace kvmem::radiance {
// Prefix leases include accepted draft-window bytes. A restore stays behind the
// final token (no saved logits) and before a new Query.
inline uint64_t restorable_prefix_limit(uint64_t prompt, int64_t query_begin) {
    if (!prompt) return 0;
    uint64_t limit = prompt - 1; // A cached state carries no logits for the final token.
    if (query_begin >= 0) limit = std::min(limit, uint64_t(query_begin));
    return limit;
}
// The scheduler already stops at this declared input anchor. It is outside
// the Query rows and any suffix the probe may later rewind. Pressure before
// this point is accepted history, captured with its actual sparse native view.
// Without a checkpoint yet, monotone prefill can only save one at/after here.
inline bool retain_drafter_input_anchor(uint64_t frontier, int64_t requested_anchor,
        int64_t query_begin, uint64_t draft_window, int64_t replay_checkpoint) {
    return draft_window && requested_anchor > 0 && frontier == uint64_t(requested_anchor) &&
        query_begin >= 0 && frontier <= uint64_t(query_begin) && replay_checkpoint >= -1 &&
        (replay_checkpoint < 0 || frontier <= uint64_t(replay_checkpoint));
}
// Payloads remain native and immutable. A directory only selects a proven
// complete frontier; an LCP alone is never a restorable recurrent state.
struct PrefixCandidate {
    const std::vector<int32_t>* tokens = nullptr;
    const void* storage = nullptr;
    CheckpointPurpose purposes = CheckpointPurpose::None;
    bool eligible = false;
    CheckpointInfo checkpoint_info() const {
        return {tokens ? tokens->size() : 0, purposes, storage, 0};
    }
};
inline const void* select_prefix_checkpoint(const std::vector<int32_t>& prompt,
        const std::vector<PrefixCandidate>& candidates, uint64_t query_limit = UINT64_MAX) {
    if (prompt.empty()) return nullptr;
    CheckpointCatalog<PrefixCandidate> directory({candidates.size(), UINT64_MAX});
    for (const auto& candidate : candidates) {
        if (!candidate.storage || !candidate.tokens || candidate.tokens->empty()) continue;
        directory.slot(candidate.purposes) = candidate;
    }
    const auto* selected = directory.select(std::min<uint64_t>(prompt.size()-1, query_limit),
        [&](const PrefixCandidate& candidate) {
            if (!candidate.eligible) return false;
            const auto& saved = *candidate.tokens;
            const auto n = std::min(prompt.size(), saved.size());
            const auto lcp = size_t(std::mismatch(saved.begin(), saved.begin()+n, prompt.begin()).first-saved.begin());
            return saved.size() <= prefix_restore_limit(lcp, saved.size(), prompt.size(), true);
        });
    return selected ? selected->storage : nullptr;
}
} // namespace kvmem::radiance
