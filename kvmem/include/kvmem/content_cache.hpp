#pragma once

// Content-cache policy and an optional immutable directory. Neither knows a
// client conversation/key, device allocator, native state format or scheduler.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvmem {

// Test the complete, safely restorable logical history at a save boundary.
// This does not govern prefix lookup, working KV or individual checkpoints.
// Call before reserving a directory slot, replacing an ancestor or copying an
// optional retained snapshot. Zero preserves the existing retention policy.
inline bool admit_content_history(uint64_t accepted_tokens, uint64_t minimum_tokens) noexcept {
    return accepted_tokens >= minimum_tokens;
}

// Adapters add only native-eligible candidates. In particular, continuation is
// an adapter proof about this request's input/accepted boundary, never ID equality.
struct ContentCacheCandidate {
    size_t slot = 0;
    uint64_t frontier = 0, cold_bytes = 0, last_used = 0;
    bool continuation = false;
};

inline std::vector<ContentCacheCandidate> rank_content_candidates(
        std::vector<ContentCacheCandidate> candidates, bool new_user_query,
        size_t max_candidates = 16) {
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](const auto& value) {
        return !value.frontier || (!new_user_query && !value.continuation);
    }), candidates.end());
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.frontier != b.frontier) return a.frontier > b.frontier;
        if (a.cold_bytes != b.cold_bytes) return a.cold_bytes < b.cold_bytes;
        if (a.last_used != b.last_used) return a.last_used > b.last_used;
        return a.slot < b.slot;
    });
    if (candidates.size() > max_candidates) candidates.resize(max_candidates);
    return candidates;
}

struct ContentCacheSlot {
    bool occupied = false, pinned = false;
    uint64_t last_used = 0;
    const void* immutable_identity = nullptr;
    uint64_t group = 0; // internal capacity grouping; never reuse/replace proof
};

// Origins denote complete immutable restore points, including their native
// state, KV/position proof and policy baseline. Adapters preserve an origin only
// while preserving ALL those inputs. Tokens/frontier equality cannot mint one.
// Unknown (zero) or empty coverage is deliberately not a dominance proof.
inline bool content_snapshot_covers(const std::vector<uint64_t>& incoming,
        const std::vector<uint64_t>& previous) {
    if (incoming.empty() || previous.empty() ||
            std::find(incoming.begin(), incoming.end(), uint64_t(0)) != incoming.end()) return false;
    for (auto origin : previous)
        if (!origin || std::find(incoming.begin(), incoming.end(), origin) == incoming.end()) return false;
    return true;
}

enum class ContentPublicationReason {
    Inserted, Replaced, Deduplicated, Disabled, AllPinned, MetadataLimit, Incomplete
};
inline const char* content_publication_reason(ContentPublicationReason reason) noexcept {
    switch (reason) {
        case ContentPublicationReason::Inserted: return "inserted";
        case ContentPublicationReason::Replaced: return "replaced";
        case ContentPublicationReason::Deduplicated: return "deduplicated";
        case ContentPublicationReason::Disabled: return "disabled";
        case ContentPublicationReason::AllPinned: return "all_pinned";
        case ContentPublicationReason::MetadataLimit: return "metadata_limit";
        case ContentPublicationReason::Incomplete: return "incomplete";
    }
    return "unknown";
}
struct ContentPublicationPlan {
    std::optional<size_t> slot;
    ContentPublicationReason reason = ContentPublicationReason::Disabled;
    explicit operator bool() const noexcept { return slot.has_value(); }
    size_t operator*() const { return slot.value(); }
};

// A same-frontier or same-token record is NOT an identity match. Pass an identity
// only for one already sealed native object. A returned slots.size() means append.
// Pinned is for an adapter's non-evictable mutable/native bindings; ordinary
// immutable reader leases need not pin the directory entry itself.
inline ContentPublicationPlan select_content_publication(
        const std::vector<ContentCacheSlot>& slots, size_t max_entries,
        const void* incoming_identity = nullptr) {
    if (!max_entries) return {{}, ContentPublicationReason::Disabled};
    size_t count = 0;
    std::optional<size_t> empty, victim;
    for (size_t index = 0; index < slots.size(); ++index) {
        const auto& value = slots[index];
        if (!value.occupied) { if (!empty) empty = index; continue; }
        ++count;
        if (incoming_identity && value.immutable_identity == incoming_identity)
            return {index, ContentPublicationReason::Deduplicated};
        if (!value.pinned && (!victim || value.last_used < slots[*victim].last_used)) victim = index;
    }
    if (count < max_entries)
        return {empty.value_or(slots.size()), ContentPublicationReason::Inserted};
    if (victim) return {victim, ContentPublicationReason::Replaced};
    return {{}, ContentPublicationReason::AllPinned};
}

struct ContentGroupPublicationPlan : ContentPublicationPlan {
    std::vector<size_t> retired_slots;
};

// Shared victim policy for real Host/Disk/metadata pressure as well as full
// directory slots. Remove a redundant version before the only retained version
// of another group. A pinned sibling still preserves the group's source; zero
// means an unknown group and every such record remains independent.
inline std::optional<size_t> select_content_eviction(
        const std::vector<ContentCacheSlot>& slots) {
    std::optional<size_t> victim;
    bool redundant = false;
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto& slot = slots[i];
        if (!slot.occupied || slot.pinned) continue;
        const bool multiple = slot.group && std::any_of(slots.begin(), slots.end(), [&](const auto& other) {
            return &other != &slot && other.occupied && other.group == slot.group;
        });
        if (!victim || (multiple && !redundant) ||
                (multiple == redundant && slot.last_used < slots[*victim].last_used)) {
            victim = i; redundant = multiple;
        }
    }
    return victim;
}

// Capacity groups are internal fairness hints, not mutable conversation owners.
// A new version NEVER replaces an earlier version merely because groups match.
// max_entries bounds physical directory slots; max_groups bounds retained groups.
inline ContentGroupPublicationPlan plan_content_group_publication(
        const std::vector<ContentCacheSlot>& slots, size_t max_entries,
        size_t max_groups, uint64_t incoming_group) {
    ContentGroupPublicationPlan plan;
    if (!max_entries || !max_groups) return plan;
    if (!incoming_group) { plan.reason = ContentPublicationReason::Incomplete; return plan; }
    struct Group {
        uint64_t id = 0, used = 0;
        std::vector<size_t> slots;
        bool pinned = false, retired = false;
    };
    std::vector<Group> groups;
    size_t occupied = 0;
    bool known = false;
    for (size_t i = 0; i < slots.size(); ++i) if (slots[i].occupied) {
        ++occupied;
        const auto& slot = slots[i];
        auto group = groups.end();
        if (slot.group) group = std::find_if(groups.begin(), groups.end(),
            [&](const auto& value) { return value.id == slot.group; });
        // Every unknown origin remains a separate resource group.
        if (group == groups.end()) {
            groups.push_back({slot.group, slot.last_used, {i}, slot.pinned, false});
        } else {
            group->slots.push_back(i); group->used = std::max(group->used, slot.last_used);
            group->pinned = group->pinned || slot.pinned;
        }
        known = known || slot.group == incoming_group;
    }
    size_t group_count = groups.size() + size_t(!known);
    std::vector<bool> retired(slots.size(), false);
    while (group_count > max_groups) {
        std::optional<size_t> victim;
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto& group = groups[i];
            if (group.retired || group.pinned || group.id == incoming_group) continue;
            if (!victim || group.used < groups[*victim].used) victim = i;
        }
        if (!victim) {
            plan.retired_slots.clear(); plan.reason = ContentPublicationReason::AllPinned; return plan;
        }
        auto& group = groups[*victim];
        group.retired = true; --group_count;
        for (auto i : group.slots) { retired[i] = true; plan.retired_slots.push_back(i); --occupied; }
    }
    auto remaining = slots;
    for (size_t i = 0; i < slots.size(); ++i) if (retired[i]) remaining[i].occupied = false;
    while (occupied >= max_entries) {
        const auto victim = select_content_eviction(remaining);
        if (!victim) {
            plan.retired_slots.clear(); plan.reason = ContentPublicationReason::AllPinned; return plan;
        }
        retired[*victim] = true; remaining[*victim].occupied = false;
        plan.retired_slots.push_back(*victim); --occupied;
    }
    for (size_t i = 0; i < slots.size(); ++i) if (!slots[i].occupied || retired[i]) {
        plan.slot = i; break;
    }
    if (!plan.slot) plan.slot = slots.size();
    plan.reason = plan.retired_slots.empty() ? ContentPublicationReason::Inserted : ContentPublicationReason::Replaced;
    return plan;
}

using ContentPrefixDigest = std::array<uint64_t, 2>;

// A token/opaque-input digest shortlist, never a restore proof. Adapters still
// compare exact tokens, media/positions and accepted/native execution state.
// The optional per-token input identity can describe media/position metadata.
class ContentPrefixDigests {
public:
    ContentPrefixDigests() { digests_.push_back({0x6a09e667f3bcc909ULL, 0xbb67ae8584caa73bULL}); }
    explicit ContentPrefixDigests(const std::vector<int32_t>& tokens) : ContentPrefixDigests() {
        digests_.reserve(tokens.size() + 1);
        for (auto token : tokens) append(token);
    }
    void append(int32_t token, ContentPrefixDigest input_identity = {}) {
        auto value = digests_.back();
        // Explicit integer mixing avoids padding, platform byte order and text
        // encoding dependencies. Collisions are deliberately handled by proof.
        const auto bits = uint64_t(uint32_t(token));
        value[0] = mix(value[0] ^ bits ^ mix(input_identity[0]) ^ uint64_t(digests_.size()));
        value[1] = mix(value[1] ^ (bits << 32) ^ mix(input_identity[1]) ^ value[0]);
        digests_.push_back(value);
    }
    size_t size() const noexcept { return digests_.size() - 1; }
    const ContentPrefixDigest& at(size_t frontier) const { return digests_.at(frontier); }
private:
    static uint64_t mix(uint64_t x) noexcept {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    std::vector<ContentPrefixDigest> digests_;
};

struct ContentPrefixKey {
    // An internal compatible loaded-model/backend domain, never a client key.
    uint64_t domain = 0, frontier = 0;
    ContentPrefixDigest digest{};
    bool operator==(const ContentPrefixKey& other) const noexcept {
        return domain == other.domain && frontier == other.frontier && digest == other.digest;
    }
};
struct ContentPrefixKeyHash {
    size_t operator()(const ContentPrefixKey& key) const noexcept {
        const uint64_t value = key.digest[0] ^ (key.digest[1] << 1) ^
            (key.domain * 0x9e3779b97f4a7c15ULL) ^ key.frontier;
        return size_t(value ^ (value >> 32));
    }
};
struct ContentSnapshotDescriptor {
    ContentPrefixKey key;
    uint64_t cold_bytes = 0, metadata_bytes = 0;
    const void* immutable_identity = nullptr;
    bool complete = false;
};
struct ContentSnapshotLimits {
    size_t max_entries = 64;
    uint64_t max_metadata_bytes = uint64_t(64) << 20;
};
struct ContentSnapshotProof {
    bool compatible = false, continuation = false;
};

// Optional typed directory for adapters that can publish immutable native
// objects. Existing native catalogs can use the policy helpers above instead.
//
// Record owns sealed HostKvSource/fork references, not long-lived RAM read
// leases. GPU work, payload accounting and state proof remain native. A Record
// requiring device-safe destruction must have a deferred deleter OR callers
// must release all returned leases/retired entries at the native safe boundary.
// No device work or last native-object destruction is done under mutex_.
template<class Record> class ContentSnapshotDirectory {
public:
    struct Snapshot {
        uint64_t id;
        ContentSnapshotDescriptor descriptor;
        std::shared_ptr<const Record> native;
    };
    using Lease = std::shared_ptr<const Snapshot>;
    struct Publication {
        ContentPublicationReason reason = ContentPublicationReason::Incomplete;
        Lease snapshot;
        // Caller schedules these native releases at the correct engine boundary.
        std::vector<Lease> retired;
        explicit operator bool() const noexcept { return bool(snapshot); }
    };
    struct Stats { size_t entries = 0; uint64_t metadata_bytes = 0; };
    struct Lookup {
        std::vector<Lease> candidates;
        size_t content_matches = 0, native_checks = 0;
    };
    explicit ContentSnapshotDirectory(ContentSnapshotLimits limits = {}) : limits_(limits) {}
    ContentSnapshotDirectory(const ContentSnapshotDirectory&) = delete;
    ContentSnapshotDirectory& operator=(const ContentSnapshotDirectory&) = delete;

    Publication publish(ContentSnapshotDescriptor descriptor, std::shared_ptr<const Record> native) {
        Publication result;
        if (!native || !descriptor.complete || !descriptor.key.frontier) return result;
        if (!limits_.max_entries) { result.reason = ContentPublicationReason::Disabled; return result; }
        const auto charge = metadata_charge(descriptor);
        if (charge > limits_.max_metadata_bytes) {
            result.reason = ContentPublicationReason::MetadataLimit; return result;
        }
        // Prepare native ownership before taking the metadata lock. The extra
        // owner also prevents failed allocation paths from destroying it there.
        auto incoming = std::make_shared<Snapshot>(Snapshot{0, descriptor, std::move(native)});
        State next; // after swap this retains old references until outside lock
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& slot : state_.slots) if (slot.snapshot && descriptor.immutable_identity &&
                    slot.snapshot->descriptor.immutable_identity == descriptor.immutable_identity &&
                    slot.snapshot->descriptor.key == descriptor.key) {
                slot.used = ++clock_;
                result.reason = ContentPublicationReason::Deduplicated;
                result.snapshot = slot.snapshot;
                return result;
            }
            next = state_;
            // All potentially throwing allocations operate on a private metadata
            // copy. Failure cannot erase the old directory or consume its source.
            while (next.count >= limits_.max_entries || next.bytes > limits_.max_metadata_bytes - charge) {
                std::optional<size_t> victim;
                for (size_t i = 0; i < next.slots.size(); ++i) if (next.slots[i].snapshot &&
                        (!victim || next.slots[i].used < next.slots[*victim].used)) victim = i;
                if (!victim) throw std::logic_error("content directory accounting is inconsistent");
                result.retired.push_back(next.slots[*victim].snapshot);
                next.slots[*victim] = {};
                rebuild(next);
            }
            size_t target = 0;
            while (target < next.slots.size() && next.slots[target].snapshot) ++target;
            incoming->id = ++next_id_;
            if (target == next.slots.size()) next.slots.push_back({incoming, clock_ + 1});
            else next.slots[target] = {incoming, clock_ + 1};
            rebuild(next);
            state_.swap(next);
            ++clock_;
            result.reason = result.retired.empty() ? ContentPublicationReason::Inserted : ContentPublicationReason::Replaced;
            result.snapshot = std::move(incoming);
        }
        return result;
    }

    // Native proof executes outside the lock with every matching source held.
    // The entry count bounds checks; max_candidates bounds the returned restore
    // plans AFTER proof/gating, so ineligible records cannot hide valid ones.
    template<class Validate> Lookup lookup(const ContentPrefixDigests& request, uint64_t domain,
            uint64_t restore_limit, bool new_user_query, Validate validate, size_t max_candidates = 16) const {
        Lookup result;
        std::vector<std::pair<Lease, uint64_t>> matches;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto limit = std::min<uint64_t>(restore_limit, request.size());
            for (auto it = state_.frontiers.begin(); it != state_.frontiers.end() && it->first <= limit; ++it) {
                const ContentPrefixKey key{domain, it->first, request.at(size_t(it->first))};
                const auto found = state_.index.find(key);
                if (found == state_.index.end()) continue;
                for (auto index : found->second)
                    matches.emplace_back(state_.slots[index].snapshot, state_.slots[index].used);
            }
        }
        result.content_matches = matches.size();
        std::vector<ContentCacheCandidate> eligible;
        for (size_t i = 0; i < matches.size(); ++i) {
            ++result.native_checks;
            const ContentSnapshotProof proof = validate(*matches[i].first);
            if (proof.compatible) eligible.push_back({i, matches[i].first->descriptor.key.frontier,
                matches[i].first->descriptor.cold_bytes, matches[i].second, proof.continuation});
        }
        for (const auto& candidate : rank_content_candidates(std::move(eligible), new_user_query, max_candidates))
            result.candidates.push_back(std::move(matches[candidate.slot].first));
        return result;
    }

    bool touch(const Lease& lease) {
        if (!lease) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& slot : state_.slots) if (slot.snapshot == lease) { slot.used = ++clock_; return true; }
        return false;
    }
    std::vector<Lease> clear() {
        State old;
        std::vector<Lease> retired;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            retired.reserve(state_.count);
            for (const auto& slot : state_.slots) if (slot.snapshot) retired.push_back(slot.snapshot);
            state_.swap(old);
        }
        return retired;
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {state_.count, state_.bytes};
    }
private:
    struct Slot { Lease snapshot; uint64_t used = 0; };
    struct State {
        std::vector<Slot> slots;
        std::unordered_map<ContentPrefixKey, std::vector<size_t>, ContentPrefixKeyHash> index;
        std::map<uint64_t, size_t> frontiers;
        uint64_t bytes = 0;
        size_t count = 0;
        void swap(State& other) noexcept {
            slots.swap(other.slots); index.swap(other.index); frontiers.swap(other.frontiers);
            std::swap(bytes, other.bytes); std::swap(count, other.count);
        }
    };
    static uint64_t metadata_charge(const ContentSnapshotDescriptor& descriptor) {
        // Conservative fixed metadata charge plus adapter-declared proof bytes.
        // This is a retained-directory budget, not payload/RSS or leased bytes.
        const uint64_t overhead = sizeof(Snapshot) + sizeof(Slot) + sizeof(ContentPrefixKey) + 8 * sizeof(void*);
        if (descriptor.metadata_bytes > std::numeric_limits<uint64_t>::max() - overhead)
            throw std::overflow_error("content snapshot metadata bytes overflow");
        return descriptor.metadata_bytes + overhead;
    }
    static void rebuild(State& state) {
        state.index.clear(); state.frontiers.clear(); state.bytes = 0; state.count = 0;
        for (size_t i = 0; i < state.slots.size(); ++i) if (const auto& snapshot = state.slots[i].snapshot) {
            state.index[snapshot->descriptor.key].push_back(i);
            ++state.frontiers[snapshot->descriptor.key.frontier];
            state.bytes += metadata_charge(snapshot->descriptor);
            ++state.count;
        }
    }
    ContentSnapshotLimits limits_;
    mutable std::mutex mutex_;
    State state_;
    uint64_t clock_ = 0, next_id_ = 0;
};

} // namespace kvmem
