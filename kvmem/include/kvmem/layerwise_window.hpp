#pragma once

#include "memory_contract.hpp"
#include "query_policy.hpp"
#include <optional>

namespace kvmem {

enum class LayerwiseWindowPolicy { Pressure, KeepSelected, SemanticChunk };
enum class LayerwiseWindowAction { None, Query, Pressure, Chunk, Exhausted };

// Trusted ordinary request metadata, finalized before native allocation. Prompt
// end is the full request's original-coordinate endpoint, never this tile's end.
// Query/replay describe the one atomic public evaluation at its aligned origin;
// native execution may tile it after reserving the whole span, without Pressure.
struct LayerwiseRequestEnvelope {
    uint64_t prompt_end = 0;
    uint32_t block_tokens = 0;
    uint64_t generation_tokens = 0, output_tokens = 0;
    bool rolling_active_window = false;
    LayerwiseWindowPolicy prefill_window = LayerwiseWindowPolicy::Pressure;
    std::optional<QueryRange> query, replay;
    bool reserve_query_output = false;
    uint64_t probe_tokens = 0, pressure_probe_tokens = 0;
    bool has_query_mandatory = false;
};
struct LayerwiseWindowState {
    uint64_t history = 0; // complete accepted original-coordinate frontier
    uint64_t visible_tokens = 0; // actual native length, including partial tail
};

// Literal 72a81cc RequestPlan action/admit/selection_budget. This pure policy
// neither executes native work nor proves the supplied prompt/prefix identity.
// Owners must use the same trusted envelope for every tile, inspect action
// before allocation, and rerun admission after a completed native selection.
class LayerwiseWindowPlan {
public:
    LayerwiseWindowPlan(LayerwiseRequestEnvelope, uint64_t capacity);
    LayerwiseWindowAction action(LayerwiseWindowState) const;
    uint64_t admit(LayerwiseWindowState, uint64_t desired) const;
    uint64_t selection_budget(uint64_t history) const;
    std::optional<uint64_t> prefill_checkpoint() const;
    uint64_t generation_checkpoint() const;
    int64_t prefill_boundary(uint64_t history) const;
    std::optional<QueryRange> query_evaluation() const noexcept { return evaluation_; }
    const LayerwiseRequestEnvelope & envelope() const noexcept { return envelope_; }
    uint64_t capacity() const noexcept { return capacity_; }
private:
    uint64_t output_reserve() const noexcept;
    bool generation_pressure(LayerwiseWindowState) const;
    void validate(LayerwiseWindowState) const;
    LayerwiseRequestEnvelope envelope_;
    uint64_t capacity_;
    std::optional<QueryRange> evaluation_;
};

} // namespace kvmem
