#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace kvmem {

constexpr uint32_t default_query_tokens = 512;
constexpr uint32_t default_query_replay_tokens = 2048;

struct QueryRange {
    uint64_t begin = 0, end = 0;
    bool empty() const noexcept { return end <= begin; }
};

// Frontends supply a proven user range. Tool continuations have no new Query.
inline QueryRange bounded_query(uint64_t begin, uint64_t count, uint64_t total,
                                uint32_t limit) noexcept {
    begin = std::min(begin, total);
    const auto end = begin + std::min(count, total - begin);
    return {end - std::min<uint64_t>(end - begin, limit), end};
}

struct QueryReplayPlan {
    uint64_t begin = 0, end = 0;
    bool admitted = false;
    const char* reason = "no_query";
};

// Includes partial-page overlap exactly once, plus the native draft workspace.
// Independent of generation length: later generation uses DecodePressure.
inline QueryReplayPlan query_replay_plan(QueryRange query, uint64_t restore_begin,
    uint64_t end, uint32_t limit, uint32_t block, uint64_t history_pages,
    uint64_t device_pages, uint32_t workspace = 0) noexcept {
    QueryReplayPlan plan{restore_begin, end, false, "no_query"};
    if (query.empty()) return plan;
    if (!block || restore_begin > query.begin || query.end > end || end < restore_begin ||
        end > UINT64_MAX - workspace) {
        plan.reason = "invalid_replay_boundary"; return plan;
    }
    if (end - restore_begin > limit) {
        plan.reason = "replay_token_limit"; return plan;
    }
    const auto ceil_page = [block](uint64_t n) { return n / block + (n % block != 0); };
    const auto extra = ceil_page(end + uint64_t(workspace)) - ceil_page(restore_begin);
    if (history_pages > device_pages || extra > device_pages - history_pages) {
        plan.reason = "replay_device_capacity"; return plan;
    }
    plan.admitted = true; plan.reason = "bounded_replay";
    return plan;
}

// NInfer can rebuild a causal suffix over several B+R windows. The frozen
// first-pass Query scores only the prefix evaluated so far in the second pass.
// Zero removes the token cap; the context, checkpoint and physical budgets still
// apply. Other backends retain query_replay_plan's bounded admission contract.
inline QueryReplayPlan streaming_query_replay_plan(QueryRange query, uint64_t restore_begin,
    uint64_t end, uint32_t limit, uint32_t block, uint64_t history_pages,
    uint64_t device_pages, uint32_t workspace = 0) noexcept {
    auto plan = query_replay_plan(query, restore_begin, end, limit ? limit : UINT32_MAX,
                                 block, history_pages, device_pages, workspace);
    if (plan.admitted || std::strcmp(plan.reason, "replay_device_capacity") != 0) return plan;
    // A refresh must leave room for an append and the native draft workspace.
    // Do not turn invalid boundaries, explicit token limits or an oversized
    // historical view into a streaming admission.
    const uint64_t minimum = (uint64_t(workspace) + 1 + block - 1) / block;
    if (history_pages > device_pages || minimum > device_pages - history_pages) return plan;
    plan.admitted = true;
    plan.reason = "streaming_query_replay";
    return plan;
}

// Both executors select the same recoverable prefix, then append the mutable
// suffix. Sink/recent here refer to that prefix, not the end of the probe.
inline std::vector<uint32_t> query_prefix_required(uint64_t prefix, uint32_t block,
    uint32_t sink, uint32_t recent, const std::vector<uint32_t>& media) {
    const auto pages = uint32_t(prefix / block + (prefix % block != 0));
    std::vector<uint32_t> required;
    for (uint32_t id = 0; id < std::min(pages, sink); ++id) required.push_back(id);
    for (uint32_t id = pages - std::min(pages, recent); id < pages; ++id) required.push_back(id);
    for (auto id : media) if (id < pages) required.push_back(id);
    // Even recent=0 must preserve the partial page into which replay appends.
    if (prefix % block) required.push_back(pages - 1);
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    return required;
}

} // namespace kvmem
