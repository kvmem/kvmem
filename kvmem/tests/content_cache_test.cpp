#include "kvmem/content_cache.hpp"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <thread>

using namespace kvmem;
static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Native {
    std::vector<int32_t> tokens;
    int state;
    std::function<void()> release;
    Native(std::vector<int32_t> input, int value, std::function<void()> done = {})
        : tokens(std::move(input)), state(value), release(std::move(done)) {}
    ~Native() { if (release) release(); }
};
using Directory = ContentSnapshotDirectory<Native>;

static ContentSnapshotDescriptor descriptor(const std::vector<int32_t>& tokens,
        uint64_t domain = 7, uint64_t cold = 0) {
    ContentPrefixDigests digests(tokens);
    return {{domain, tokens.size(), digests.at(tokens.size())}, cold, 0, nullptr, true};
}
static auto proof(const std::vector<int32_t>& input, bool continuation = true) {
    return [&input, continuation](const Directory::Snapshot& snapshot) {
        const auto& prefix = snapshot.native->tokens;
        return ContentSnapshotProof{prefix.size() <= input.size() &&
            std::equal(prefix.begin(), prefix.end(), input.begin()), continuation};
    };
}

static void policy_tests() {
    check(admit_content_history(0, 0) && admit_content_history(4095, 0), "zero disables history filtering");
    check(!admit_content_history(4095, 4096) && admit_content_history(4096, 4096) &&
        admit_content_history(4097, 4096), "minimum is inclusive without page alignment");
    check(admit_content_history(100000 + 30, 4096), "long history with short append remains eligible");
    const std::vector<ContentCacheCandidate> candidates{
        {0, 100, 10, 3, true}, {1, 200, 40, 9, false},
        {2, 200, 20, 1, true}, {3, 200, 20, 5, true}, {4, 0, 0, 99, true}};
    auto ranked = rank_content_candidates(candidates, true, 3);
    check(ranked.size() == 3 && ranked[0].slot == 3 && ranked[1].slot == 2 && ranked[2].slot == 1,
        "ranking must prefer frontier then cold bytes then recency");
    ranked = rank_content_candidates(candidates, false, 8);
    check(ranked.size() == 3 && ranked[2].slot == 0, "tool continuation must exclude prefix discovery");
    check(rank_content_candidates(candidates, true, 0).empty(), "zero candidate budget");

    int first = 1, second = 2;
    std::vector<ContentCacheSlot> slots{{true, false, 3, &first}, {true, false, 9, &second}};
    auto plan = select_content_publication(slots, 2);
    check(plan && *plan == 0 && plan.reason == ContentPublicationReason::Replaced, "full publication LRU");
    slots[0].pinned = true;
    check(*select_content_publication(slots, 2) == 1, "pinned native binding cannot be evicted");
    slots[1].pinned = true;
    plan = select_content_publication(slots, 2);
    check(!plan && plan.reason == ContentPublicationReason::AllPinned, "all pinned refusal");
    plan = select_content_publication(slots, 2, &first);
    check(plan && *plan == 0 && plan.reason == ContentPublicationReason::Deduplicated,
        "the same sealed object can be deduplicated while leased");
    slots[0] = {};
    check(*select_content_publication(slots, 2) == 0, "empty slot reused");
    check(*select_content_publication(slots, 3) == 0, "empty slot precedes append");
    check(!select_content_publication(slots, 0), "disabled publication cannot deduplicate");
    check(content_snapshot_covers({11, 12, 13}, {12, 11}), "all immutable restore origins retained");
    check(!content_snapshot_covers({11, 13}, {11, 12}), "longer content cannot replace a missing restore origin");
    check(!content_snapshot_covers({0, 11}, {11}) && !content_snapshot_covers({11}, {0}),
        "unknown native origin cannot prove dominance");
    check(!content_snapshot_covers({11}, {}) && !content_snapshot_covers({}, {11}), "empty coverage is no proof");
}

static void branching_and_proof_tests() {
    Directory directory({8, 1 << 20});
    auto source = directory.publish(descriptor({1, 2}), std::make_shared<Native>(std::vector<int32_t>{1, 2}, 10));
    check(bool(source), "source published");
    std::mutex gate;
    std::condition_variable changed;
    bool a_entered = false, b_finished = false;
    std::exception_ptr failure;
    auto capture_failure = [&] {
        std::lock_guard<std::mutex> lock(gate);
        failure = std::current_exception(); b_finished = true; changed.notify_all();
    };
    std::thread a([&] {
        try {
            const std::vector<int32_t> input{1, 2, 3};
            auto lease = directory.lookup(ContentPrefixDigests(input), 7, 2, false, proof(input));
            check(lease.candidates.size() == 1 && lease.candidates[0] == source.snapshot, "A retained source");
            {
                std::unique_lock<std::mutex> lock(gate);
                a_entered = true; changed.notify_all();
                changed.wait(lock, [&] { return b_finished; });
            }
            auto published = directory.publish(descriptor(input), std::make_shared<Native>(input, 11));
            check(bool(published), "older-started A must publish after B");
            check(lease.candidates[0]->native->tokens == std::vector<int32_t>({1, 2}), "A did not mutate source");
        } catch (...) { capture_failure(); }
    });
    std::thread b([&] {
        try {
            {
                std::unique_lock<std::mutex> lock(gate);
                changed.wait(lock, [&] { return a_entered || failure; });
                if (failure) return;
            }
            const std::vector<int32_t> input{1, 2, 4};
            auto lease = directory.lookup(ContentPrefixDigests(input), 7, 2, false, proof(input));
            check(lease.candidates.size() == 1, "A in flight cannot consume source for B");
            auto published = directory.publish(descriptor(input), std::make_shared<Native>(input, 12));
            check(bool(published), "B published independent branch");
            { std::lock_guard<std::mutex> lock(gate); b_finished = true; changed.notify_all(); }
        } catch (...) { capture_failure(); }
    });
    a.join(); b.join();
    if (failure) std::rethrow_exception(failure);
    check(directory.stats().entries == 3, "root and both branches coexist");

    const std::vector<int32_t> input{1, 2, 3, 99};
    auto found = directory.lookup(ContentPrefixDigests(input), 7, input.size(), true, proof(input));
    check(found.candidates.size() == 2 && found.candidates[0]->native->state == 11, "find A and shared root");
    check(directory.lookup(ContentPrefixDigests(input), 8, input.size(), true, proof(input)).candidates.empty(),
        "different compatible-model domain is not a candidate");
    check(directory.lookup(ContentPrefixDigests(input), 7, input.size(), false, proof(input, false)).candidates.empty(),
        "tool continuation must not gain cold-history discovery");
    check(directory.lookup(ContentPrefixDigests(input), 7, 1, true, proof(input)).candidates.empty(), "restore limit enforced");

    // Same content and frontier may describe distinct native selection/replay
    // states. They stay separate until the adapter proves a compatible state.
    auto variant = directory.publish(descriptor({1, 2, 3}), std::make_shared<Native>(std::vector<int32_t>{1, 2, 3}, 13));
    check(bool(variant) && directory.stats().entries == 4, "equal content cannot collapse native variants");
    auto exact = directory.lookup(ContentPrefixDigests(input), 7, 3, true,
        [&](const Directory::Snapshot& snapshot) {
            return ContentSnapshotProof{snapshot.native->state == 11 && proof(input)(snapshot).compatible, true};
        }, 1);
    check(exact.candidates.size() == 1 && exact.candidates[0]->native->state == 11,
        "ineligible variant must not exhaust returned candidate budget");

    // Deliberately publish a digest collision: native proof, not the digest,
    // decides whether the object is restorable.
    auto collision = directory.publish(descriptor({1, 2, 3}), std::make_shared<Native>(std::vector<int32_t>{9, 9, 9}, 14));
    found = directory.lookup(ContentPrefixDigests(input), 7, 3, true, proof(input));
    for (const auto& value : found.candidates) check(value->native->state != 14, "digest collision rejected by proof");
    check(found.content_matches == 4 && found.native_checks == 4, "all matching variants receive native proof");

    // Proof callbacks may query/reenter the directory; no native validation
    // runs while holding its metadata mutex.
    found = directory.lookup(ContentPrefixDigests(input), 7, 3, true,
        [&](const Directory::Snapshot& snapshot) {
            check(directory.stats().entries == 5, "native proof executes outside directory lock");
            return proof(input)(snapshot);
        });
    check(found.candidates.size() == 3, "reentrant native proof");
}

static void capacity_group_tests() {
    // Group 1's sole old state remains useful while group 2 publishes nearby
    // versions. A full directory first recycles one of the repeated versions.
    std::vector<ContentCacheSlot> slots{
        {true, false, 1, nullptr, 1}, {true, false, 2, nullptr, 2},
        {true, false, 3, nullptr, 2}, {true, false, 4, nullptr, 2}};
    auto plan = plan_content_group_publication(slots, 4, 2, 2);
    check(plan && *plan == 1 && plan.retired_slots == std::vector<size_t>({1}),
        "repeated versions must not displace the only state of another group");
    // A later-started branch does not own the group or suppress an earlier
    // publication. With room, both versions simply get independent slots.
    plan = plan_content_group_publication(slots, 8, 2, 2);
    check(plan && *plan == 4 && plan.retired_slots.empty(), "reverse completion publishes an independent version");
    // Recency of a group is the newest use among its versions, not the oldest
    // publication. Introducing group 3 must remove all of the cold group 1.
    plan = plan_content_group_publication(slots, 4, 2, 3);
    check(plan && *plan == 0 && plan.retired_slots == std::vector<size_t>({0}), "group count eviction is explicit");
    slots[0].last_used = 9;
    plan = plan_content_group_publication(slots, 4, 2, 3);
    check(plan && plan.retired_slots == std::vector<size_t>({1, 2, 3}), "full cold group retired together");
    slots[1].pinned = true;
    plan = plan_content_group_publication(slots, 4, 2, 3);
    check(plan && plan.retired_slots == std::vector<size_t>({0}), "one pinned version protects group-wide lifetime");
    slots[0].pinned = true;
    plan = plan_content_group_publication(slots, 4, 2, 3);
    check(!plan && plan.retired_slots.empty() && plan.reason == ContentPublicationReason::AllPinned,
        "failed group admission returns no partial evictions");
    check(!plan_content_group_publication(slots, 4, 0, 3), "zero group capacity is disabled");
    check(plan_content_group_publication(slots, 4, 2, 0).reason == ContentPublicationReason::Incomplete,
        "unknown incoming group cannot collapse anonymous records");
    slots = {{true, false, 1, nullptr, 0}, {true, false, 2, nullptr, 0}};
    plan = plan_content_group_publication(slots, 4, 2, 3);
    check(plan && plan.retired_slots == std::vector<size_t>({0}), "unknown retained groups remain distinct");

    // Physical pressure uses the same fairness even when directory slots and
    // group quotas have spare capacity. The caller retries until real bytes fit.
    slots = {{true, false, 1, nullptr, 1}, {true, false, 2, nullptr, 2},
             {true, false, 3, nullptr, 2}, {true, false, 4, nullptr, 2}};
    check(select_content_eviction(slots) == 1, "physical pressure protects the sole old group");
    slots[1].occupied = false;
    check(select_content_eviction(slots) == 2, "physical pressure recomputes remaining redundancy");
    slots[2].occupied = false;
    check(select_content_eviction(slots) == 0, "only unique versions remain: ordinary LRU");
    slots = {{true, true, 1, nullptr, 2}, {true, false, 9, nullptr, 2},
             {true, false, 0, nullptr, 1}};
    check(select_content_eviction(slots) == 1, "pinned sibling preserves group when redundant version is evicted");
    slots[1].pinned = true;
    check(select_content_eviction(slots) == 2, "never choose a pinned version");
    slots[2].pinned = true;
    check(!select_content_eviction(slots), "physical pressure refuses all pinned records");
    slots = {{true, false, 9, nullptr, 0}, {true, false, 8, nullptr, 0},
             {true, false, 1, nullptr, 3}, {false, false, 0, nullptr, 3}};
    check(select_content_eviction(slots) == 2, "unknown groups and empty slots do not create redundancy");
}

static void eviction_and_failure_tests() {
    std::atomic<unsigned> destroyed{0};
    Directory directory({1, 1 << 20});
    const std::vector<int32_t> input{5, 6, 7};
    auto source = directory.publish(descriptor({5, 6}), std::make_shared<Native>(std::vector<int32_t>{5, 6}, 20,
        [&] { ++destroyed; (void)directory.stats(); }));
    auto lease = directory.lookup(ContentPrefixDigests(input), 7, 2, false, proof(input)).candidates[0];
    source.snapshot.reset();
    auto next = directory.publish(descriptor(input), std::make_shared<Native>(input, 21));
    check(next.reason == ContentPublicationReason::Replaced && next.retired.size() == 1, "capacity eviction explicit");
    next.retired.clear();
    check(destroyed == 0 && lease->native->state == 20, "eviction retains outstanding reader");
    check(!directory.touch(lease), "retired lease cannot touch a reused directory slot");
    lease.reset();
    check(destroyed == 1, "last native release runs outside directory lock");

    auto invalid = descriptor({8, 9}); invalid.complete = false;
    check(directory.publish(invalid, std::make_shared<Native>(std::vector<int32_t>{8, 9}, 22)).reason ==
        ContentPublicationReason::Incomplete, "incomplete capture not published");
    auto huge = descriptor({8, 9}); huge.metadata_bytes = 1 << 21;
    check(directory.publish(huge, std::make_shared<Native>(std::vector<int32_t>{8, 9}, 23)).reason ==
        ContentPublicationReason::MetadataLimit, "metadata admission refusal");
    check(directory.stats().entries == 1, "failed publication preserves current directory");
    bool threw = false;
    try {
        directory.lookup(ContentPrefixDigests(input), 7, 3, true,
            [](const Directory::Snapshot&) -> ContentSnapshotProof { throw std::runtime_error("restore preparation failed"); });
    } catch (const std::runtime_error&) { threw = true; }
    check(threw && directory.stats().entries == 1, "failed restore/proof leaves source intact");
    auto retired = directory.clear();
    check(directory.stats().entries == 0 && retired.size() == 1 && retired[0] == next.snapshot,
        "clear returns retirements rather than consuming external leases");

    Directory disabled({0, 1 << 20});
    check(disabled.publish(descriptor(input), std::make_shared<Native>(input, 24)).reason ==
        ContentPublicationReason::Disabled, "zero retained-directory capacity");
}

static void deduplication_and_metadata_tests() {
    const std::vector<int32_t> input{1, 2, 3};
    auto native = std::make_shared<Native>(input, 30);
    auto info = descriptor(input); info.immutable_identity = native.get();
    Directory directory({8, 1 << 20});
    auto first = directory.publish(info, native);
    const auto bytes = directory.stats().metadata_bytes;
    auto second = directory.publish(info, native);
    check(second.reason == ContentPublicationReason::Deduplicated && first.snapshot == second.snapshot &&
        directory.stats().entries == 1 && directory.stats().metadata_bytes == bytes, "sealed identity dedup");
    auto earlier = descriptor({1, 2}); earlier.immutable_identity = native.get();
    auto different_frontier = directory.publish(earlier, native);
    check(different_frontier.reason == ContentPublicationReason::Inserted && directory.stats().entries == 2,
        "one allocation may back distinct exact boundaries");

    Directory bounded({8, bytes * 2});
    auto a = bounded.publish(descriptor({1}), std::make_shared<Native>(std::vector<int32_t>{1}, 31));
    auto b = bounded.publish(descriptor({2}), std::make_shared<Native>(std::vector<int32_t>{2}, 32));
    check(bounded.touch(a.snapshot), "touch active snapshot");
    auto c = bounded.publish(descriptor({3}), std::make_shared<Native>(std::vector<int32_t>{3}, 33));
    check(c.retired.size() == 1 && c.retired[0] == b.snapshot && bounded.stats().entries == 2 &&
        bounded.stats().metadata_bytes <= bytes * 2, "metadata budget evicts LRU by real use");

    ContentPrefixDigests plain(input), media;
    for (auto token : input) media.append(token, {123, 456});
    check(plain.at(3) != media.at(3), "opaque native input identities affect shortlist digest");
    ContentPrefixDigests appended; for (auto token : input) appended.append(token);
    check(plain.at(3) == appended.at(3) && plain.size() == 3, "incremental digest matches complete input");
}

int main() {
    policy_tests();
    branching_and_proof_tests();
    capacity_group_tests();
    eviction_and_failure_tests();
    deduplication_and_metadata_tests();
    std::cout << "content cache: concurrent publication, source leases, native proof, tool gates, budgets and failures passed\n";
}
