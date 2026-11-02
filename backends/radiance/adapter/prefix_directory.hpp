#pragma once
#include "kvmem/prefix_checkpoint.hpp"
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
