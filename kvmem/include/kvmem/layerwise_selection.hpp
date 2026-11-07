#pragma once

#include "layerwise_scoring.hpp"
#include "layerwise_query.hpp"
#include "native_layer_query.hpp"
#include "kvmem_store.hpp"
#include <functional>

namespace kvmem {

struct LayerwiseHistorySelection {
    uint32_t layer = 0; // RawKvStore statistics ordinal, not a physical GPU slot
    QueryScoreFrame frame;
    std::vector<double> scores;
    std::vector<uint32_t> fixed;
    std::vector<uint32_t> selected; // chronological logical block IDs
    uint64_t historical_tokens = 0;
    uint64_t charged_tokens = 0;
};

// Pure host reference of upstream72a81cc's per-layer selection. The policy is
// the complete canonical, sealed prefix before the mutable Query suffix. Its
// current runtime selection budget is frozen by the owner through publication.
// Scoring uses only this forward's target-layer Q. Fixed bands and ranking are
// delegated to the existing native picker; no shared retrieval scores, LRFU
// rounds, graph state, residency or payloads are modified.
LayerwiseHistorySelection select_layerwise_history(const KvMemStore& policy,
    const RawKvStore& keys, CurrentLayerQuery query,
    const std::vector<uint32_t>& mandatory = {});

struct LayerwiseQueryGeometry {
    uint32_t layer = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    float attention_scale = 0;
};

enum class LayerwiseSelectionPhase { Collecting, Complete, Aborted };

// Accumulates raw scores from all contiguous Query tiles against ONE frozen
// denominator, then picks once. This is the tiled equivalent of upstream's
// full-Query layer scorer; it never chooses pages independently per tile.
//
// The policy and frame are copied. RawKvStore is borrowed, NOT made immutable
// by a const reference: the owner must hold a source/index lease, prevent writes
// during each call, and expose its exact frozen history stamp via revision.
// Any index mutation must advance that stamp. The callback keeps the proof's
// owner alive and must refer to this index, not an unrelated resident view.
class LayerwiseQuerySelection {
public:
    using SourceRevision = std::function<MemoryStamp()>;
    using NativeScorer = std::function<std::vector<double>(const RawKvStore&,
        NativeCurrentLayerQuery, const QueryScoreFrame&)>;
    LayerwiseQuerySelection(const KvMemStore& policy, const RawKvStore& keys,
        LayerwiseQueryIdentity identity, LayerwiseQueryGeometry geometry,
        SourceRevision revision, const std::vector<uint32_t>& mandatory = {});
    LayerwiseQuerySelection(const LayerwiseQuerySelection&) = delete;
    LayerwiseQuerySelection& operator=(const LayerwiseQuerySelection&) = delete;
    LayerwiseSelectionPhase phase() const noexcept { return phase_; }
    uint64_t scored_rows() const noexcept { return next_ - identity_.query.begin; }
    const LayerwiseQueryIdentity& identity() const noexcept { return identity_; }
    void append(const LayerwiseQueryIdentity&, uint64_t begin, CurrentLayerQuery);
    // Scorer sees exactly this transaction's frozen denominator. It must score
    // the supplied fresh native rows with the qualified Mean-K formula; native
    // stream completion must precede return. No external score cache is input.
    void append_native(const LayerwiseQueryIdentity&, uint64_t begin,
                       NativeCurrentLayerQuery, const NativeScorer&);
    LayerwiseHistorySelection finish(const LayerwiseQueryIdentity&);
    void abort() noexcept { phase_ = LayerwiseSelectionPhase::Aborted; }
private:
    void check_identity(const LayerwiseQueryIdentity&) const;
    void check_source();
    void check_tile(const LayerwiseQueryIdentity&, uint64_t begin, uint32_t rows,
                    uint32_t layer, uint32_t query_heads, uint32_t kv_heads,
                    uint32_t head_dim, float attention_scale) const;
    void accumulate(uint32_t rows, const std::vector<double>&);
    KvMemStore policy_;
    const RawKvStore& keys_;
    LayerwiseQueryIdentity identity_;
    LayerwiseQueryGeometry geometry_;
    SourceRevision revision_;
    LayerwiseHistorySelection result_;
    uint64_t next_ = 0;
    LayerwiseSelectionPhase phase_ = LayerwiseSelectionPhase::Collecting;
};

} // namespace kvmem
