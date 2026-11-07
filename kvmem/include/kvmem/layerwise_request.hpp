#pragma once

#include "layerwise_memory.hpp"
#include "layerwise_selection.hpp"
#include <memory>

namespace kvmem {

struct LayerwiseRequestLayer {
    uint32_t model_layer = 0;
    LayerwiseQueryGeometry query;
    MemoryBudget budget;
    std::vector<uint32_t> mandatory; // canonical chronological policy ordinals
};

enum class LayerwiseRequestPhase {
    BetweenLayers, Projecting, Restoring, Attending, AwaitingAcceptance, Accepted, Aborted
};

// An engine-safe-boundary transaction shared by native executors. It connects
// real projection reports to the qualified scorer/picker and native layer plan.
// It does NOT execute tensors, copy GPU bytes, fence streams, or publish state.
// Those operations stay with the Program/adapter and must precede their reports.
// The index is borrowed under the owner's immutable source lease until all
// layers finish. No global probe or prior Query tensor is an input to this API.
class LayerwiseQueryRequest {
public:
    using SourceRevision = LayerwiseQuerySelection::SourceRevision;
    LayerwiseQueryRequest(LayerwiseQueryIdentity, LayerwiseMemoryCatalog,
        const KvMemStore&, const RawKvStore&, std::vector<LayerwiseRequestLayer>,
        uint64_t native_history_tokens, SourceRevision);
    LayerwiseQueryRequest(const LayerwiseQueryRequest&) = delete;
    LayerwiseQueryRequest& operator=(const LayerwiseQueryRequest&) = delete;

    LayerwiseRequestPhase phase() const noexcept { return phase_; }
    const LayerwiseQueryIdentity& identity() const noexcept { return identity_; }
    LayerwiseQueryWork work() const noexcept { return progress_.work(); }
    void begin_layer(const LayerwiseQueryIdentity&, uint32_t model_layer);
    // Full native evaluation tile. Only its intersection with the explicit
    // user Query enters scoring; template/header/trailer rows remain unscored.
    void projected(const LayerwiseQueryIdentity&, uint64_t begin, CurrentLayerQuery);
    void projected_native(const LayerwiseQueryIdentity&, uint64_t begin,
                          NativeCurrentLayerQuery, const LayerwiseQuerySelection::NativeScorer&);
    const LayerWorkingSetPlan& prepare_restore(const LayerwiseQueryIdentity&);
    const LayerwiseHistorySelection& selection() const;
    // Read the actual post-fence native view, not a guessed successful result.
    void restored(const LayerwiseQueryIdentity&, const ExecutionView& native_view);
    void attended(const LayerwiseQueryIdentity&, uint64_t begin, uint64_t count);
    void complete_layer(const LayerwiseQueryIdentity&);

    // Caller supplies successful native acceptance AFTER all target layers,
    // non-attention states and final model work have completed. These snapshots
    // include the appended evaluation suffix and any qualified final Pressure.
    // The source lease may now be released: no old index is read here.
    void accepted(const LayerwiseQueryIdentity&, std::vector<LayerMemorySnapshot>);
    const std::vector<LayerMemorySnapshot>& accepted_layers() const;
    // Native owner separately drains its in-flight work and restores/invalidates
    // its checkpoint. No metadata-only abort can prove native rollback.
    void abort() noexcept;
private:
    void check(const LayerwiseQueryIdentity&) const;
    void source();
    void require(LayerwiseRequestPhase) const;
    const LayerwiseRequestLayer& current() const;
    LayerwiseQueryIdentity identity_;
    LayerwiseMemoryCatalog catalog_;
    KvMemStore policy_;
    const RawKvStore& keys_;
    std::vector<LayerwiseRequestLayer> specs_;
    uint64_t native_history_tokens_;
    SourceRevision revision_;
    LayerwiseQueryProgress progress_;
    LayerwiseRequestPhase phase_ = LayerwiseRequestPhase::BetweenLayers;
    size_t next_ = 0;
    std::unique_ptr<LayerwiseQuerySelection> scoring_;
    std::unique_ptr<LayerWorkingSetPlan> plan_;
    std::unique_ptr<LayerwiseHistorySelection> selection_;
    std::vector<ExecutionView> completed_views_;
    std::vector<LayerMemorySnapshot> accepted_;
};

} // namespace kvmem
