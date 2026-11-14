#pragma once

#include "layerwise_selection.hpp"
#include "host_kv_storage.hpp"
#include <memory>

namespace kvmem {

struct LayerwiseProbeHistorySelection {
    LayerwiseProbeHistorySelection() = default;
    LayerwiseProbeHistorySelection(LayerwiseProbeHistorySelection&&) noexcept = default;
    LayerwiseProbeHistorySelection& operator=(LayerwiseProbeHistorySelection&&) noexcept = default;
    LayerwiseProbeHistorySelection(const LayerwiseProbeHistorySelection&) = delete;
    LayerwiseProbeHistorySelection& operator=(const LayerwiseProbeHistorySelection&) = delete;
    QueryScoreFrame frame;
    std::vector<double> scores;
    std::vector<uint32_t> fixed;
    std::vector<uint32_t> selected;
    uint64_t historical_tokens = 0;
    uint64_t charged_tokens = 0;
private:
    friend class LayerwiseProbeSelection;
    // A returned result can outlive its accumulator. Its bounded metadata
    // stays charged until both owners retire. Move-only ownership prevents
    // uncharged copies of the retained vectors.
    std::shared_ptr<HostKvReservation> metadata_;
};

// Pinned upstream72a81cc's private-probe global Mean-K reduction for MTP.
// Each append supplies the private TARGET forward's normalized pre-RoPE Q,
// never Q from the later real fresh-layer forward. The owner independently
// preserves the real resident operand, exact model state and accepted boundary,
// restores/fences the private state, and restores draft-owned archive stripes.
// This class owns scoring/selection only; it cannot prove native rollback.
//
// Tiles arrive token-major: every specified target in its native causal order
// for [begin,begin+rows), then the next Query token tile. Only the true Query
// intersection is appended. All L*T rows share one frozen page denominator.
// No picker runs before every target covers the entire Query. Native callbacks
// return raw sums over token rows / mean query heads; this class averages L.
// Metadata is O(pages+layers), explicitly admitted before any owned copies.
class LayerwiseProbeSelection {
public:
    using SourceRevision = LayerwiseQuerySelection::SourceRevision;
    using NativeScorer = LayerwiseQuerySelection::NativeScorer;
    static uint64_t metadata_bound(const KvMemStore&, uint32_t target_layers, bool half_pages = false);
    LayerwiseProbeSelection(const KvMemStore&, const RawKvStore&,
        LayerwiseQueryIdentity, const std::vector<LayerwiseQueryGeometry>& target_order,
        SourceRevision, HostKvStorage& metadata,
        const std::vector<uint32_t>& mandatory = {});
    LayerwiseProbeSelection(const LayerwiseProbeSelection&) = delete;
    LayerwiseProbeSelection& operator=(const LayerwiseProbeSelection&) = delete;
    LayerwiseSelectionPhase phase() const noexcept { return phase_; }
    uint64_t scored_rows() const noexcept { return next_ - identity_.query.begin; }
    uint32_t next_layer() const noexcept { return layer_cursor_; }
    uint64_t metadata_bytes() const noexcept { return metadata_bytes_; }
    void append(const LayerwiseQueryIdentity&, uint64_t begin, CurrentLayerQuery);
    void append_native(const LayerwiseQueryIdentity&, uint64_t begin,
                       NativeCurrentLayerQuery, const NativeScorer&);
    LayerwiseProbeHistorySelection finish(const LayerwiseQueryIdentity&);
    void abort() noexcept { phase_ = LayerwiseSelectionPhase::Aborted; }
private:
    void check_identity(const LayerwiseQueryIdentity&) const;
    void check_source();
    void check_tile(const LayerwiseQueryIdentity&, uint64_t, uint32_t rows,
                    uint32_t layer, uint32_t query_heads, uint32_t kv_heads,
                    uint32_t head_dim, float scale) const;
    void accumulate(uint32_t rows, const std::vector<double>&);
    uint64_t metadata_bytes_ = 0;
    std::shared_ptr<HostKvReservation> metadata_;
    KvMemStore policy_;
    const RawKvStore& keys_;
    LayerwiseQueryIdentity identity_;
    std::vector<LayerwiseQueryGeometry> geometry_;
    SourceRevision revision_;
    LayerwiseProbeHistorySelection result_;
    uint64_t next_ = 0;
    uint32_t layer_cursor_ = 0, group_rows_ = 0;
    LayerwiseSelectionPhase phase_ = LayerwiseSelectionPhase::Collecting;
};

} // namespace kvmem
