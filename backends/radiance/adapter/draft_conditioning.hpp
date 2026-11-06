#pragma once
#include "kvmem/kvmem_store.hpp"

namespace kvmem::radiance {
inline KvMemKeepAllocation draft_keep_allocation(uint32_t block, uint32_t budget,
                                                 int64_t sink, int64_t recent,
                                                 uint32_t draft_window) {
    const auto requested = resolve_kvmem_keep_allocation(block, budget, -1, -1, sink, recent);
    if (!draft_window) return requested;
    // The newest selected page can contain just one accepted token. Reserve
    // its unused tail too, so protected pages still cover W actual positions.
    const auto minimum = uint64_t(draft_window) + block - 1;
    if (requested.recent_effective_tokens >= minimum) return requested;
    // The common allocator rounds the window upward to whole selection pages
    // and rejects sink + recent exceeding the existing selected-token budget.
    return resolve_kvmem_keep_allocation(block, budget, -1, -1, sink, int64_t(minimum));
}
} // namespace kvmem::radiance
