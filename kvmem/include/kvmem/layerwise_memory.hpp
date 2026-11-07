#pragma once

#include "memory_contract.hpp"
#include "native_kv_payload.hpp"

namespace kvmem {

struct LayerMemorySnapshot {
    uint32_t model_layer = 0;
    MemorySnapshot memory;
};

struct LayerWorkingSetSelection {
    uint32_t model_layer = 0;
    MemoryStamp expected;
    std::vector<uint64_t> selected;
    std::vector<uint64_t> mandatory;
    // host_payload_bytes is one shared canonical archive limit in every entry.
    MemoryBudget budget;
};

struct LayerWorkingSetPlan {
    uint32_t model_layer;
    MemoryDescriptor descriptor; // aligned per-layer staging, all rank/head planes
    NativeKvLayerRegions payload; // exact offsets into the canonical source record
    WorkingSetPlan plan; // canonical archive accounting; descriptor is the layer transfer layout
    uint64_t restore_bytes = 0, spill_bytes = 0; // native plane bytes, excluding staging padding
};

struct LayerwiseWorkingSetPlan {
    BackendIdentity backend;
    // Accepted history identity only; per-layer expected/view stamps are in layers.
    MemoryStamp history;
    std::vector<LayerWorkingSetPlan> layers;
    // Union of canonical full-record IDs. Do not sum layers' archive requirements.
    uint64_t host_payload_bytes_required = 0;
};

// Immutable pure planning over one canonical history. Each target layer owns its
// own resident set and view version. All head shards of a layer use that view.
// Native owners reserve, transfer, fence, and publish separately.
class LayerwiseMemoryCatalog {
public:
    LayerwiseMemoryCatalog(MemoryDescriptor canonical, std::vector<LayerMemorySnapshot>);
    const MemoryDescriptor & descriptor() const noexcept { return canonical_; }
    const std::vector<LayerMemorySnapshot> & layers() const noexcept { return layers_; }
    const MemorySnapshot & snapshot(uint32_t model_layer) const;
    // Prepare only the current layer after its Q exists. This does not require
    // another layer's future selection. The Query owner first publishes sealed
    // canonical history and requires complete current Host backing before any
    // layer swap; generic native planning still permits full-record spills.
    LayerWorkingSetPlan plan_layer(const LayerWorkingSetSelection &) const;
    // Bulk validation for already known selections, not causal Query execution.
    LayerwiseWorkingSetPlan plan(const std::vector<LayerWorkingSetSelection> &) const;
private:
    MemoryDescriptor canonical_;
    std::vector<LayerMemorySnapshot> layers_;
};

} // namespace kvmem
