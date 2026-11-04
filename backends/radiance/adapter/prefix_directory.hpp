#pragma once
#include "kvmem/prefix_checkpoint.hpp"
#include "kvmem/content_cache.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace kvmem::radiance {
// Payloads remain native and immutable. A directory only selects a proven
// complete frontier; an LCP alone is never a restorable recurrent state.
struct PrefixCandidate {
    const std::vector<int32_t>* tokens = nullptr;
    const void* storage = nullptr;
    CheckpointPurpose purposes = CheckpointPurpose::None;
    bool eligible = false;
    bool continuation = false;
    CheckpointInfo checkpoint_info() const {
        return {tokens ? tokens->size() : 0, purposes, storage, 0};
    }
};
inline const void* select_prefix_checkpoint(const std::vector<int32_t>& prompt,
        const std::vector<PrefixCandidate>& candidates, uint64_t query_limit = UINT64_MAX,
        bool new_user_query = true) {
    if (prompt.empty()) return nullptr;
    std::vector<ContentCacheCandidate> proven;
    for (size_t slot=0;slot<candidates.size();++slot) {
        const auto& candidate=candidates[slot];
        if (!candidate.storage || !candidate.tokens || candidate.tokens->empty()) continue;
        if (candidate.eligible) {
            const auto& saved = *candidate.tokens;
            const auto n = std::min(prompt.size(), saved.size());
            const auto lcp = size_t(std::mismatch(saved.begin(), saved.begin()+n, prompt.begin()).first-saved.begin());
            if(saved.size()<=query_limit && saved.size()<=prefix_restore_limit(lcp,saved.size(),prompt.size(),true))
                proven.push_back({slot,saved.size(),0,0,candidate.continuation});
        }
    }
    const auto ranked=rank_content_candidates(std::move(proven),new_user_query);
    return ranked.empty() ? nullptr : candidates[ranked.front().slot].storage;
}
} // namespace kvmem::radiance
