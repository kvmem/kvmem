#pragma once
#include "kvmem/kvmem_store.hpp"

namespace kvmem::radiance {
inline KvMemKeepAllocation draft_keep_allocation(uint32_t block, uint32_t budget,
                                                 int64_t sink, int64_t recent,
                                                 uint32_t draft_window) {
    const auto requested = resolve_kvmem_keep_allocation(block, budget, -1, -1, sink, recent);
    if (requested.recent_effective_tokens >= draft_window) return requested;
    // The common allocator rounds the window upward to whole selection pages
    // and rejects sink + recent exceeding the existing selected-token budget.
    return resolve_kvmem_keep_allocation(block, budget, -1, -1, sink, draft_window);
}
} // namespace kvmem::radiance
