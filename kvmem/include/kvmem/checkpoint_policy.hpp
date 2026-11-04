#pragma once
#include "prefix_checkpoint.hpp"
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

namespace kvmem {

// Experimental startup policy. Existing native execution slots remain separate
// from optional coverage records; disabling this policy keeps their old behavior.
struct CheckpointPolicy {
    bool enabled = false;
    // NInfer's native admission/retention policy also runs in legacy mode.
    // This bounds additional tokens, not checkpoint reconstruction or I/O time.
    uint64_t extra_recompute_tokens = 0;
    uint64_t interval = 16384;
    uint64_t min_gap = 0;
    size_t count = 24;
    uint64_t bytes = uint64_t(4) << 30;
    size_t hot_count = 4;
    uint64_t host_bytes = uint64_t(1) << 30;
    static CheckpointPolicy environment();
};

enum class CoveragePurpose : uint8_t { Periodic, Tail, Root, Anchor, Input, Endpoint };
struct CoverageInfo {
    uint64_t frontier = 0, bytes = 0, used = 0;
    const void * identity = nullptr;
    CoveragePurpose purpose = CoveragePurpose::Periodic;
    uint8_t protections = 0; // Root, anchor and input roles may share a payload.
};
// Stable slot indices; retained indices are a subset, in original slot order.
std::vector<size_t> retain_checkpoint_coverage(const std::vector<CoverageInfo> &,
    size_t count, uint64_t bytes, uint64_t interval, uint64_t min_gap = 0);

// A consumptive restore must not mistake a shared root for an anonymous
// continuation. Allow dropped generated output and a short re-templated tail;
// deeper edits require an explicit owner. Cancellation can leave a shorter
// accepted frontier than the input that was planned.
inline bool checkpoint_anonymous_continuation(uint64_t lcp, uint64_t input_end,
                                             uint64_t accepted_end) {
    const auto boundary = std::min(input_end, accepted_end);
    return lcp >= boundary || boundary - lcp <= 64;
}

// Exact semantic cuts may split a text chunk. Optional periodic cuts do not.
struct CheckpointCapturePlan {
    uint64_t next = 0, interval = 0, root = 0, tail = 0;
    bool root_done = false, tail_done = false;
    CheckpointCapturePlan() = default;
    CheckpointCapturePlan(uint64_t base, uint64_t stride, uint64_t root_at = 0, uint64_t tail_at = 0);
    uint64_t cut(uint64_t begin, uint64_t end) const;
    std::optional<CoveragePurpose> due(uint64_t frontier) const;
    void saved(uint64_t frontier);
};

// Record must be copyable and supply checkpoint_info(). The payload's owner,
// prefix proof and readiness remain native. Cold records still have an identity.
template<class Record> class CheckpointArchive {
public:
    struct Entry { Record record; CoverageInfo info; };
    CheckpointArchive() : CheckpointArchive(CheckpointPolicy::environment()) {}
    explicit CheckpointArchive(CheckpointPolicy policy) : policy_(policy) {}
    const CheckpointPolicy & policy() const { return policy_; }
    auto begin() { return entries_.begin(); }
    auto end() { return entries_.end(); }
    auto begin() const { return entries_.begin(); }
    auto end() const { return entries_.end(); }
    size_t slots() const { return entries_.size(); }
    size_t capacity() const { return entries_.capacity(); }
    Entry * at(size_t i) { return i < entries_.size() && entries_[i].info.identity ? &entries_[i] : nullptr; }
    const Entry * at(size_t i) const { return i < entries_.size() && entries_[i].info.identity ? &entries_[i] : nullptr; }
    void clear() { entries_.clear(); }
    void swap(CheckpointArchive & other) { entries_.swap(other.entries_); std::swap(clock_, other.clock_); }
    void truncate(uint64_t frontier) {
        for (auto & entry : entries_) if (entry.info.frontier > frontier) entry = {};
    }
    template<class Valid> void erase_invalid(Valid valid) {
        for (auto & entry : entries_) if (entry.info.identity && !valid(entry.record)) entry = {};
    }
    void touch(const void * identity) {
        for (auto & entry : entries_) if (entry.info.identity == identity) entry.info.used = ++clock_;
    }
    void erase(const void * identity) {
        for (auto & entry : entries_) if (entry.info.identity == identity) entry = {};
    }
    bool remember(Record record, CoveragePurpose purpose) {
        if (!policy_.enabled) return false;
        const auto info = record.checkpoint_info();
        if (!info.storage || !info.frontier || info.bytes > policy_.bytes) return false;
        // Pointer identity denotes one immutable payload, never frontier equality.
        for (auto & entry : entries_) if (entry.info.identity == info.storage) {
            if (entry.info.frontier != info.frontier) throw std::logic_error("checkpoint identity changed frontier");
            entry.info.used = ++clock_;
            entry.info.bytes = info.bytes;
            entry.info.protections |= purpose == CoveragePurpose::Root ? 1 : purpose == CoveragePurpose::Anchor ? 2 :
                                      purpose == CoveragePurpose::Input ? 4 : 0;
            if (purpose > entry.info.purpose) entry.info.purpose = purpose;
            return true;
        }
        // Publication at an existing position explicitly replaces its old version.
        // The caller is responsible for truncating any invalidated later history.
        size_t slot = entries_.size();
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].info.identity && entries_[i].info.frontier == info.frontier) { slot = i; break; }
            if (!entries_[i].info.identity && slot == entries_.size()) slot = i;
        }
        Entry incoming{std::move(record), {info.frontier, info.bytes, ++clock_, info.storage, purpose}};
        incoming.info.protections = purpose == CoveragePurpose::Root ? 1 : purpose == CoveragePurpose::Anchor ? 2 :
                                    purpose == CoveragePurpose::Input ? 4 : 0;
        if (slot == entries_.size()) entries_.push_back(std::move(incoming));
        else entries_[slot] = std::move(incoming);
        prune();
        return entries_[slot].info.identity == info.storage;
    }
    void prune(const std::vector<CheckpointInfo> & fixed = {}) {
        // Fixed execution leases cannot be evicted here. Charge distinct payloads
        // once across both directories, leaving the optional archive the remainder.
        size_t count = policy_.count;
        uint64_t bytes = policy_.bytes;
        std::vector<const void *> seen;
        for (const auto & info : fixed) if (info.storage &&
                std::find(seen.begin(), seen.end(), info.storage) == seen.end()) {
            seen.push_back(info.storage);
            count -= count != 0;
            bytes -= std::min(bytes, info.bytes);
        }
        std::vector<CoverageInfo> infos;
        for (auto & entry : entries_) {
            if (entry.info.identity) entry.info.bytes = entry.record.checkpoint_info().bytes;
            auto info = entry.info;
            if (std::find(seen.begin(), seen.end(), info.identity) != seen.end()) info.identity = nullptr;
            infos.push_back(info);
        }
        const auto keep = retain_checkpoint_coverage(infos, count, bytes, policy_.interval, policy_.min_gap);
        for (size_t i = 0; i < entries_.size(); ++i)
            if (std::find(keep.begin(), keep.end(), i) == keep.end() &&
                std::find(seen.begin(), seen.end(), entries_[i].info.identity) == seen.end()) entries_[i] = {};
    }
    template<class Valid> const Record * select(uint64_t limit, Valid valid) const {
        const Record * result = nullptr;
        uint64_t best = 0;
        for (const auto & entry : entries_) if (entry.info.identity && entry.info.frontier <= limit &&
                entry.info.frontier > best && valid(entry.record)) {
            result = &entry.record; best = entry.info.frontier;
        }
        return result;
    }
    // A nearby earlier state substitutes replay work, never an exact-state alias.
    // The backend supplies its existing native validity proof; no SSD read occurs.
    template<class Valid> const Record * nearby(uint64_t frontier, Valid valid) const {
        if (!policy_.enabled || !policy_.min_gap || !frontier) return nullptr;
        return select(frontier - 1, [&](const Record & record) {
            return frontier - record.checkpoint_info().frontier < policy_.min_gap && valid(record);
        });
    }
    uint64_t bytes() const { uint64_t n = 0; for (const auto & e : entries_) if (e.info.identity) n += e.record.checkpoint_info().bytes; return n; }
    size_t size() const { size_t n = 0; for (const auto & e : entries_) n += bool(e.info.identity); return n; }
private:
    CheckpointPolicy policy_;
    uint64_t clock_ = 0;
    std::vector<Entry> entries_;
};
} // namespace kvmem
