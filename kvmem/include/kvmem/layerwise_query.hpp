#pragma once

#include "memory_contract.hpp"
#include "query_policy.hpp"
#include <vector>

namespace kvmem {

// Native owners report completed work at real execution/fence boundaries.
// This guard owns no tensor, transfer, checkpoint, or policy. In particular,
// reporting progress is not a substitute for executing or draining that work.
struct LayerwiseQueryIdentity {
    MemoryStamp history;
    uint64_t request = 0;
    QueryRange evaluation; // frozen prefix -> end of public input; owner proves block alignment
    QueryRange query;      // proven user rows within evaluation
};

// Same nonzero logical identity fields required by plan_working_set. A zero
// view_version is allowed for the initial native view.
inline bool valid_layerwise_query_identity(const LayerwiseQueryIdentity& i) noexcept {
    return i.history.valid && i.history.session.id && i.history.session.incarnation &&
        i.history.history_revision && i.history.execution_history && i.request &&
        !i.evaluation.empty() && !i.query.empty() &&
        i.evaluation.begin == i.history.frontier &&
        i.query.begin >= i.evaluation.begin && i.query.end <= i.evaluation.end;
}

enum class LayerwiseQueryPhase {
    BetweenLayers, Projection, Selection, Restore, Attention, Complete, Aborted
};

struct LayerwiseQueryWork {
    uint64_t projected_rows = 0;
    uint64_t attended_rows = 0;
    uint32_t completed_layers = 0;
};

class LayerwiseQueryProgress {
public:
    LayerwiseQueryProgress(LayerwiseQueryIdentity, std::vector<uint32_t> model_layers);
    LayerwiseQueryPhase phase() const noexcept { return phase_; }
    const LayerwiseQueryIdentity& identity() const noexcept { return identity_; }
    LayerwiseQueryWork work() const noexcept { return work_; }

    void begin_layer(const LayerwiseQueryIdentity&, uint32_t model_layer);
    void projected(const LayerwiseQueryIdentity&, uint32_t model_layer,
                   uint64_t begin, uint64_t count);
    // All evaluation Q rows must exist before a selection can start. Scoring
    // consumes only identity().query and never the following assistant output.
    void begin_restore(const LayerwiseQueryIdentity&, uint32_t model_layer);
    // Call only after native KV/scale writes and the consuming-stream fence.
    // Retain upstream72a81cc's equal physical historical extent. Independently
    // variable extents require separate algorithm qualification, not this guard.
    void restored(const LayerwiseQueryIdentity&, uint32_t model_layer,
                  uint64_t historical_tokens);
    void attended(const LayerwiseQueryIdentity&, uint32_t model_layer,
                  uint64_t begin, uint64_t count);
    // Includes completion of this layer's post-attention/residual/FFN work.
    void complete_layer(const LayerwiseQueryIdentity&, uint32_t model_layer);
    // Native owner must drain/cancel transfers and restore/invalidate its state
    // separately. Abort permanently closes this progress instance.
    void abort() noexcept;
private:
    void check(const LayerwiseQueryIdentity&) const;
    void layer(uint32_t) const;
    void require(LayerwiseQueryPhase) const;
    void span(uint64_t begin, uint64_t count, uint64_t& cursor, uint64_t& total);
    LayerwiseQueryIdentity identity_;
    std::vector<uint32_t> layers_;
    LayerwiseQueryPhase phase_ = LayerwiseQueryPhase::BetweenLayers;
    LayerwiseQueryWork work_;
    size_t next_ = 0;
    uint64_t projection_ = 0, attention_ = 0, history_tokens_ = 0;
};

} // namespace kvmem
