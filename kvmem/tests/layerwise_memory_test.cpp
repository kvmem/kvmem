#include "memory_test_support.hpp"
#include "kvmem/layerwise_memory.hpp"

static MemoryDescriptor layout() {
    MemoryDescriptor d;
    d.identity = {"layer-memory-test", "native-v1", "mixed-layer-model", "state-v1"};
    d.layout = {"native-plan-fixture", 1, 4, 128, 8, {
        {3, PayloadPlaneKind::Key, 0, 16, 8, 4, 0, 0, 2},
        {3, PayloadPlaneKind::Value, 16, 16, 8, 4, 0, 0, 2},
        {7, PayloadPlaneKind::Key, 40, 10, 8, 0, 0, 0, 2},
        {7, PayloadPlaneKind::Value, 56, 9, 8, 0, 0, 0, 2},
    }};
    d.capabilities.page_tokens = 2;
    d.capabilities.max_position = 4096;
    d.capabilities.native_payload = true;
    d.capabilities.sparse_view = true;
    d.capabilities.original_rope_positions = true;
    d.validate();
    return d;
}

static MemorySnapshot snapshot(const MemoryDescriptor & d, bool tail) {
    MemorySnapshot s;
    s.stamp = {true, {42, 3}, 5, 4, tail ? 18u : 16u, 9};
    for (uint64_t id = 0; id < (tail ? 5u : 4u); ++id)
        s.blocks.push_back({block(d, id, id == 4 ? 2 : 4), 1});
    s.resident = tail ? std::vector<uint64_t>{0, 1, 4} : std::vector<uint64_t>{0, 1, 2, 3};
    return s;
}

static std::vector<LayerWorkingSetSelection> selections(const MemorySnapshot & a, const MemorySnapshot & b) {
    const MemoryBudget budget{10, 2, 12, 5 * 128};
    return {{7, b.stamp, {1, 3, 4}, {4}, budget}, {3, a.stamp, {0, 2, 4}, {4}, budget}};
}

static void independent_layer_views() {
    const auto d = layout();
    auto a = snapshot(d, true), b = a;
    b.stamp.view_version = 8;
    b.resident = {2, 3, 4};
    const LayerwiseMemoryCatalog catalog(d, {{7, b}, {3, a}});
    const auto requested = selections(a, b);
    const auto plans = catalog.plan(requested);
    CHECK(plans.backend == d.identity && plans.layers.size() == 2);
    CHECK(plans.history.frontier == 18 && plans.history.view_version == 0);
    CHECK(plans.host_payload_bytes_required == 5 * d.layout.record_bytes);
    const auto & p = plans.layers[0];
    const auto & q = plans.layers[1];
    CHECK(p.model_layer == 3 && q.model_layer == 7);
    CHECK(p.plan.expected == a.stamp && q.plan.expected == b.stamp);
    CHECK(p.plan.next.stamp.view_version == 5 && q.plan.next.stamp.view_version == 9);
    CHECK(p.plan.next.visible_tokens == 10 && q.plan.next.visible_tokens == 10);
    CHECK(p.plan.next.blocks[2].compact_start == 8 && q.plan.next.blocks[2].compact_start == 8);
    CHECK(p.plan.restore.size() == 1 && p.plan.restore.front().id == 2);
    CHECK(q.plan.restore.size() == 1 && q.plan.restore.front().id == 1);
    CHECK(p.plan.evict.size() == 1 && p.plan.evict.front().id == 1);
    CHECK(q.plan.evict.size() == 1 && q.plan.evict.front().id == 2);
    CHECK(p.restore_bytes == 32 && q.restore_bytes == 19);
    CHECK(p.spill_bytes == 0 && q.spill_bytes == 0);
    CHECK(p.descriptor.layout.record_bytes == 32 && q.descriptor.layout.record_bytes == 32);
    CHECK(q.payload.regions()[0].plane.offset == 40 && q.payload.regions()[1].plane.offset == 56);
    CHECK(q.descriptor.layout.planes[0].offset == 0 && q.descriptor.layout.planes[1].offset == 16);
    CHECK(q.descriptor.layout.planes[0].head_count == 2 && q.descriptor.layout.planes[1].rank == 0);
    CHECK(catalog.snapshot(3).resident == a.resident && catalog.snapshot(7).resident == b.resident);
    CHECK(catalog.snapshot(3).stamp == a.stamp && catalog.snapshot(7).stamp == b.stamp);
}

static void canonical_archive_union() {
    const auto d = layout();
    auto s = snapshot(d, false);
    for (auto & b : s.blocks) b.host_version = 0;
    const LayerwiseMemoryCatalog catalog(d, {{3, s}, {7, s}});
    MemoryBudget budget{8, 0, 16, 2 * 128};
    std::vector<LayerWorkingSetSelection> selected{
        {3, s.stamp, {0, 1}, {}, budget}, {7, s.stamp, {2, 3}, {}, budget}};
    // Each layer needs two records; their union needs four, not two or eight.
    rejects(MemoryErrorCode::BudgetExceeded, [&] { catalog.plan(selected); });
    for (auto & selection : selected) selection.budget.host_payload_bytes = 4 * 128;
    const auto p = catalog.plan(selected);
    CHECK(p.host_payload_bytes_required == 4 * 128);
    CHECK(p.layers[0].spill_bytes == 64 && p.layers[1].spill_bytes == 38);
    selected[1].selected = {0, 1};
    const auto shared = catalog.plan(selected);
    CHECK(shared.host_payload_bytes_required == 2 * 128);
}

static void sequential_layer_preparation() {
    const auto d = layout();
    auto a = snapshot(d, true), b = a;
    b.resident = {2, 3, 4}; b.stamp.view_version = 8;
    const LayerwiseMemoryCatalog catalog(d, {{3, a}, {7, b}});
    const MemoryBudget budget{10, 2, 12, 5 * 128};
    // No next-layer Query or selection exists when preparing this layer.
    const auto first = catalog.plan_layer({3, a.stamp, {0, 2, 4}, {4}, budget});
    CHECK(first.model_layer == 3 && first.plan.restore.front().id == 2);
    CHECK(first.plan.next.blocks[1].block.id == 2);
    // The owner executes attention before it can supply the next selection.
    // This test checks planning boundaries only; it does not simulate a model.
    const std::vector<uint64_t> next_selection{0, 3, 4};
    const auto second = catalog.plan_layer({7, b.stamp, next_selection, {4}, budget});
    CHECK(second.model_layer == 7 && second.plan.restore.front().id == 0);
    CHECK(second.plan.next.blocks.front().block.id == 0);
    CHECK(first.plan.expected == a.stamp && second.plan.expected == b.stamp);
    CHECK(catalog.snapshot(3).resident == a.resident && catalog.snapshot(7).resident == b.resident);
    auto stale = b.stamp; ++stale.view_version;
    rejects(MemoryErrorCode::Stale, [&] { catalog.plan_layer({7, stale, next_selection, {4}, budget}); });
}

static void head_shards_share_one_layer_view() {
    auto d = layout();
    d.layout.planes.push_back({3, PayloadPlaneKind::Key, 80, 8, 8, 0, 1, 2, 1});
    d.layout.planes.push_back({3, PayloadPlaneKind::Value, 96, 8, 8, 0, 1, 2, 1});
    const auto s = snapshot(d, true);
    const LayerwiseMemoryCatalog catalog(d, {{3, s}, {7, s}});
    const auto plans = catalog.plan(selections(s, s));
    const auto & p = plans.layers.front();
    CHECK(p.descriptor.layout.planes.size() == 4 && p.descriptor.layout.record_bytes == 48);
    CHECK(p.payload.regions()[2].plane.rank == 1 && p.payload.regions()[2].plane.head_begin == 2);
    CHECK(p.payload.regions()[2].plane.offset == 80 && p.descriptor.layout.planes[2].offset == 32);
    const NativeKvLayerRegions shard(d.layout, 3, 1);
    CHECK(shard.plane_bytes() == 16 && shard.regions().size() == 2);
    CHECK(plans.host_payload_bytes_required == 5 * d.layout.record_bytes);
}

static void invalid_catalogs_and_plans() {
    const auto d = layout();
    const auto a = snapshot(d, true);
    auto b = a; b.resident = {2, 3, 4}; ++b.stamp.view_version;
    rejects(MemoryErrorCode::InvalidPlan, [&] { LayerwiseMemoryCatalog c(d, {}); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { LayerwiseMemoryCatalog c(d, {{3, a}, {3, b}}); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { LayerwiseMemoryCatalog c(d, {{4, a}}); });
    auto wrong = b; ++wrong.stamp.execution_history;
    rejects(MemoryErrorCode::Stale, [&] { LayerwiseMemoryCatalog c(d, {{3, a}, {7, wrong}}); });
    wrong = b; ++wrong.blocks[0].block.statistics_version;
    rejects(MemoryErrorCode::Stale, [&] { LayerwiseMemoryCatalog c(d, {{3, a}, {7, wrong}}); });
    wrong = b; wrong.blocks[0].host_version = 0;
    rejects(MemoryErrorCode::Stale, [&] { LayerwiseMemoryCatalog c(d, {{3, a}, {7, wrong}}); });
    wrong = b; wrong.resident = {2, 3};
    rejects(MemoryErrorCode::InvalidPlan, [&] { LayerwiseMemoryCatalog c(d, {{3, a}, {7, wrong}}); });
    const LayerwiseMemoryCatalog catalog(d, {{3, a}, {7, b}});
    rejects(MemoryErrorCode::InvalidPlan, [&] { catalog.snapshot(9); });
    auto selected = selections(a, b);
    selected.pop_back();
    rejects(MemoryErrorCode::InvalidPlan, [&] { catalog.plan(selected); });
    selected = selections(a, b); selected[1] = selected[0];
    rejects(MemoryErrorCode::InvalidPlan, [&] { catalog.plan(selected); });
    selected = selections(a, b); ++selected[1].expected.view_version;
    rejects(MemoryErrorCode::Stale, [&] { catalog.plan(selected); });
    selected = selections(a, b); selected[1].budget.host_payload_bytes += 128;
    rejects(MemoryErrorCode::InvalidPlan, [&] { catalog.plan(selected); });
    selected = selections(a, b); selected[1].selected = {0, 4};
    rejects(MemoryErrorCode::InvalidPlan, [&] { catalog.plan(selected); });
    selected = selections(a, b); selected[1].selected = {0, 2};
    rejects(MemoryErrorCode::InvalidPlan, [&] { catalog.plan(selected); }); // mandatory tail absent
    CHECK(catalog.snapshot(3).resident == a.resident && catalog.snapshot(7).resident == b.resident);
}

int main() {
    independent_layer_views();
    canonical_archive_union();
    sequential_layer_preparation();
    head_shards_share_one_layer_view();
    invalid_catalogs_and_plans();
    std::puts("layerwise memory: independent native views, shared archive accounting and accepted geometry passed");
}
