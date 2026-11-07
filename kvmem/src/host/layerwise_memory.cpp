#include "kvmem/layerwise_memory.hpp"
#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

namespace kvmem {
namespace {
[[noreturn]] void fail(MemoryErrorCode code, const char * message) {
    throw MemoryError(code, message);
}
bool same_history(const MemoryStamp & a, const MemoryStamp & b) {
    return a.valid == b.valid && a.session == b.session &&
        a.history_revision == b.history_revision && a.frontier == b.frontier &&
        a.execution_history == b.execution_history;
}
uint64_t multiply(uint64_t count, uint64_t bytes) {
    if (count && bytes > std::numeric_limits<uint64_t>::max() / count)
        fail(MemoryErrorCode::InvalidPlan, "layerwise payload size overflow");
    return count * bytes;
}
using Partial = std::tuple<uint64_t, uint32_t, uint64_t>;
std::vector<Partial> partials(const ExecutionView & view, uint32_t block_tokens) {
    std::vector<Partial> result;
    for (const auto & b : view.blocks)
        if (b.block.valid_tokens != block_tokens)
            result.emplace_back(b.block.original_start, b.block.valid_tokens, b.compact_start);
    return result;
}
} // namespace

LayerwiseMemoryCatalog::LayerwiseMemoryCatalog(MemoryDescriptor canonical,
                                             std::vector<LayerMemorySnapshot> layers)
    : canonical_(std::move(canonical)), layers_(std::move(layers)) {
    canonical_.validate();
    if (layers_.empty()) fail(MemoryErrorCode::InvalidPlan, "layerwise catalog needs target layers");
    std::sort(layers_.begin(), layers_.end(), [](const auto & a, const auto & b) {
        return a.model_layer < b.model_layer;
    });
    const auto & first = layers_.front().memory;
    const auto page = canonical_.capabilities.page_tokens;
    const auto maximum = std::numeric_limits<uint64_t>::max();
    const MemoryBudget validation{maximum - maximum % page, 0, maximum - maximum % page, maximum};
    std::optional<ExecutionView> resident_view;
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto & layer = layers_[i];
        if (i && layer.model_layer == layers_[i - 1].model_layer)
            fail(MemoryErrorCode::InvalidPlan, "duplicate target layer in memory catalog");
        NativeKvLayerRegions(canonical_.layout, layer.model_layer);
        // Reuse the native contract's catalog/residency validation without changing it.
        const auto validated = plan_working_set(canonical_, layer.memory, layer.memory.resident, {}, validation);
        if (resident_view && (validated.next.visible_tokens != resident_view->visible_tokens ||
            partials(validated.next, canonical_.layout.block_tokens) != partials(*resident_view, canonical_.layout.block_tokens)))
            fail(MemoryErrorCode::InvalidPlan, "target layers have inconsistent resident extent or partial-block geometry");
        if (!resident_view) resident_view = validated.next;
        if (!same_history(first.stamp, layer.memory.stamp) || first.blocks.size() != layer.memory.blocks.size())
            fail(MemoryErrorCode::Stale, "target layers do not describe the same accepted history");
        for (size_t b = 0; b < first.blocks.size(); ++b)
            if (!(first.blocks[b].block == layer.memory.blocks[b].block) ||
                first.blocks[b].host_version != layer.memory.blocks[b].host_version)
                fail(MemoryErrorCode::Stale, "target layer catalog differs from canonical native records");
    }
}

const MemorySnapshot & LayerwiseMemoryCatalog::snapshot(uint32_t model_layer) const {
    const auto found = std::lower_bound(layers_.begin(), layers_.end(), model_layer,
        [](const auto & item, uint32_t layer) { return item.model_layer < layer; });
    if (found == layers_.end() || found->model_layer != model_layer)
        fail(MemoryErrorCode::InvalidPlan, "unknown target layer in memory catalog");
    return found->memory;
}

LayerWorkingSetPlan LayerwiseMemoryCatalog::plan_layer(const LayerWorkingSetSelection & selection) const {
    const auto & state = snapshot(selection.model_layer);
    if (!(selection.expected == state.stamp))
        fail(MemoryErrorCode::Stale, "layerwise plan expected an obsolete layer view");
    auto native = plan_working_set(canonical_, state, selection.selected, selection.mandatory, selection.budget);
    NativeKvLayerRegions payload(canonical_.layout, selection.model_layer);
    auto descriptor = canonical_;
    descriptor.layout = payload.packed_layout();
    descriptor.validate();
    const auto restored = multiply(native.restore.size(), payload.plane_bytes());
    const auto spilled = multiply(native.spill.size(), payload.plane_bytes());
    return {selection.model_layer, std::move(descriptor), std::move(payload),
            std::move(native), restored, spilled};
}

LayerwiseWorkingSetPlan LayerwiseMemoryCatalog::plan(const std::vector<LayerWorkingSetSelection> & selections) const {
    if (selections.size() != layers_.size())
        fail(MemoryErrorCode::InvalidPlan, "layerwise plan must include every target layer");
    LayerwiseWorkingSetPlan result;
    result.backend = canonical_.identity;
    result.history = layers_.front().memory.stamp;
    result.history.view_version = 0;
    const auto host_limit = selections.front().budget.host_payload_bytes;
    std::set<uint32_t> seen;
    std::set<uint64_t> host_records;
    for (const auto & block : layers_.front().memory.blocks)
        if (block.host_version) host_records.insert(block.block.id);
    for (const auto & selection : selections) {
        if (!seen.insert(selection.model_layer).second || selection.budget.host_payload_bytes != host_limit)
            fail(MemoryErrorCode::InvalidPlan, "duplicate target layer or inconsistent shared archive budget");
        auto native = plan_layer(selection);
        if (!result.layers.empty()) {
            const auto & reference = result.layers.front().plan.next;
            if (native.plan.next.visible_tokens != reference.visible_tokens ||
                partials(native.plan.next, canonical_.layout.block_tokens) != partials(reference, canonical_.layout.block_tokens))
                fail(MemoryErrorCode::InvalidPlan, "layerwise selection changed native extent or partial-block geometry");
        }
        for (const auto & block : native.plan.spill) host_records.insert(block.id);
        result.layers.push_back(std::move(native));
    }
    std::sort(result.layers.begin(), result.layers.end(), [](const auto & a, const auto & b) {
        return a.model_layer < b.model_layer;
    });
    result.host_payload_bytes_required = multiply(host_records.size(), canonical_.layout.record_bytes);
    if (result.host_payload_bytes_required > host_limit)
        fail(MemoryErrorCode::BudgetExceeded, "layerwise canonical archive exceeds its shared budget");
    return result;
}

} // namespace kvmem
