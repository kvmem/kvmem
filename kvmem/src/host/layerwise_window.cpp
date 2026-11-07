#include "kvmem/layerwise_window.hpp"
#include <algorithm>
#include <limits>

namespace kvmem {
namespace {
[[noreturn]] void invalid(const char * message) { throw MemoryError(MemoryErrorCode::InvalidPlan,message); }
[[noreturn]] void capacity_fail(const char * message) { throw MemoryError(MemoryErrorCode::BudgetExceeded,message); }
// Native token domains are bounded; signed intermediate boundaries must never
// wrap when the reserved Query leaves no room for an ordinary input tile.
constexpr uint64_t coordinate_limit=uint64_t(std::numeric_limits<int64_t>::max())/4;
}

LayerwiseWindowPlan::LayerwiseWindowPlan(LayerwiseRequestEnvelope envelope,uint64_t capacity_tokens)
    :envelope_(std::move(envelope)),capacity_(capacity_tokens) {
    const auto & e=envelope_;
    if (!e.prompt_end || !e.block_tokens || !capacity_ || capacity_>coordinate_limit ||
        e.prompt_end>coordinate_limit || e.generation_tokens>coordinate_limit || e.output_tokens>coordinate_limit ||
        e.probe_tokens>coordinate_limit || e.pressure_probe_tokens>coordinate_limit ||
        e.generation_tokens>=capacity_ || e.block_tokens>capacity_-e.generation_tokens ||
        (e.prefill_window!=LayerwiseWindowPolicy::Pressure && e.prefill_window!=LayerwiseWindowPolicy::KeepSelected &&
         e.prefill_window!=LayerwiseWindowPolicy::SemanticChunk))
        invalid("invalid trusted native request envelope or capacity");
    const auto range=[&](const std::optional<QueryRange> & r) {
        return !r || (!r->empty() && r->end<=e.prompt_end);
    };
    if (!range(e.query) || !range(e.replay) || (e.replay && !e.query))
        invalid("invalid original-coordinate Query/replay envelope");
    if (e.query) {
        const auto first=e.replay && e.replay->begin<e.query->begin ? e.replay->begin:e.query->begin;
        evaluation_=QueryRange{first/e.block_tokens*e.block_tokens,e.replay ? e.replay->end:e.query->end};
        if (evaluation_->end<e.query->end || evaluation_->begin>e.query->begin)
            invalid("complete Query is outside its atomic native evaluation");
        const auto rows=evaluation_->end-evaluation_->begin;
        if (rows>e.generation_tokens || e.block_tokens>e.generation_tokens-rows ||
            e.probe_tokens>e.generation_tokens-rows-e.block_tokens)
            capacity_fail("complete Query must leave the reference native pressure headroom");
    }
}

void LayerwiseWindowPlan::validate(LayerwiseWindowState state) const {
    if (state.history>coordinate_limit || state.visible_tokens>state.history || state.visible_tokens>capacity_)
        invalid("native working-set extent is not a complete accepted prefix");
}
uint64_t LayerwiseWindowPlan::output_reserve() const noexcept {
    return envelope_.rolling_active_window ? 0:envelope_.output_tokens;
}
uint64_t LayerwiseWindowPlan::generation_checkpoint() const {
    auto point=(envelope_.prompt_end-1)/envelope_.block_tokens*envelope_.block_tokens;
    if (evaluation_ && point<evaluation_->end) point=evaluation_->begin;
    return point;
}
bool LayerwiseWindowPlan::generation_pressure(LayerwiseWindowState state) const {
    // prompt_tokens-offset == visible_tokens + prompt_tokens-history.
    return output_reserve() && state.history<envelope_.prompt_end &&
        state.history==generation_checkpoint() &&
        state.visible_tokens+envelope_.prompt_end-state.history+output_reserve()>capacity_-envelope_.block_tokens;
}
uint64_t LayerwiseWindowPlan::selection_budget(uint64_t history) const {
    if (history>coordinate_limit) invalid("selection history exceeds native coordinates");
    const auto & e=envelope_;
    uint64_t budget=capacity_-e.generation_tokens;
    if (!history && e.prompt_end+output_reserve()<=capacity_-e.block_tokens) return budget;
    const bool query_reserve=e.reserve_query_output && evaluation_ && history==evaluation_->begin;
    if (output_reserve() && (query_reserve || history==generation_checkpoint())) {
        if (history>e.prompt_end) invalid("final prompt reservation precedes selection history");
        const auto reserved=output_reserve()+(e.prompt_end-history)+e.block_tokens;
        if (reserved>=capacity_) capacity_fail("complete Query and output reserve exceed active capacity");
        budget=std::min(budget,(capacity_-reserved)/e.block_tokens*e.block_tokens);
    }
    if (!budget) capacity_fail("complete Query and output reserve exceed active capacity");
    return budget;
}
std::optional<uint64_t> LayerwiseWindowPlan::prefill_checkpoint() const {
    const auto & e=envelope_;
    if (evaluation_) return evaluation_->begin;
    if (e.prefill_window==LayerwiseWindowPolicy::SemanticChunk || e.has_query_mandatory) return {};
    // Exact qw3 branch_safe_checkpoint(prompt_end,0,block_tokens). The short
    // four-token tail guard remains even though only aligned results are used.
    uint64_t point=0;
    if (e.prompt_end>e.block_tokens) {
        const auto last=(e.prompt_end-1)/e.block_tokens*e.block_tokens;
        point=last>=e.block_tokens ? last-e.block_tokens:0;
    } else point=e.prompt_end>4 ? e.prompt_end-4:0;
    return point && point%e.block_tokens==0 ? std::optional<uint64_t>(point):std::nullopt;
}
int64_t LayerwiseWindowPlan::prefill_boundary(uint64_t history) const {
    if (history>coordinate_limit) invalid("prefill history exceeds native coordinates");
    const auto & e=envelope_;
    auto boundary=int64_t(capacity_)-(e.prefill_window==LayerwiseWindowPolicy::KeepSelected ? 1:2)*int64_t(e.block_tokens)-
        int64_t(e.pressure_probe_tokens);
    if (evaluation_ && history<evaluation_->begin) {
        const auto room=int64_t(capacity_)-int64_t(evaluation_->end-evaluation_->begin)-int64_t(e.block_tokens)-int64_t(e.probe_tokens);
        // Constructor proves rows+block+probe<=G<C, so floor is nonnegative.
        boundary=std::min(boundary,room/int64_t(e.block_tokens)*int64_t(e.block_tokens));
    }
    return boundary;
}
LayerwiseWindowAction LayerwiseWindowPlan::action(LayerwiseWindowState state) const {
    validate(state);
    if (evaluation_ && state.history==evaluation_->begin) return LayerwiseWindowAction::Query;
    if (state.history%envelope_.block_tokens) return LayerwiseWindowAction::None;
    if (generation_pressure(state)) return LayerwiseWindowAction::Pressure;
    if (envelope_.prefill_window==LayerwiseWindowPolicy::SemanticChunk && state.history<envelope_.prompt_end)
        return LayerwiseWindowAction::Chunk;
    const auto boundary=state.history<envelope_.prompt_end ? prefill_boundary(state.history):
        int64_t(capacity_-envelope_.block_tokens);
    if (int64_t(state.visible_tokens)>=boundary) {
        const bool rolling_decode=envelope_.rolling_active_window && state.history>=envelope_.prompt_end;
        return rolling_decode || envelope_.prefill_window!=LayerwiseWindowPolicy::KeepSelected ?
            LayerwiseWindowAction::Pressure:LayerwiseWindowAction::Exhausted;
    }
    return LayerwiseWindowAction::None;
}
uint64_t LayerwiseWindowPlan::admit(LayerwiseWindowState state,uint64_t desired) const {
    validate(state);
    if (desired>coordinate_limit) invalid("native requested tile exceeds coordinate range");
    const auto & e=envelope_;
    if (state.history>=e.prompt_end) return std::min(desired,capacity_-state.visible_tokens);
    const auto checkpoint=prefill_checkpoint();
    if (checkpoint && state.history<*checkpoint) desired=std::min(desired,*checkpoint-state.history);
    const auto generation=generation_checkpoint();
    if (output_reserve() && state.history<generation &&
        state.visible_tokens+e.prompt_end-state.history+output_reserve()>capacity_-e.block_tokens)
        desired=std::min(desired,generation-state.history);
    if (evaluation_) {
        if (state.history<evaluation_->begin) desired=std::min(desired,evaluation_->begin-state.history);
        else if (state.history==evaluation_->begin) {
            const auto rows=evaluation_->end-evaluation_->begin;
            if (rows>capacity_-state.visible_tokens)
                capacity_fail("atomic native Query has no complete reserved physical extent");
            return desired>=rows ? rows:0;
        } else if (state.history<evaluation_->end) invalid("explicit Query was split or its boundary was skipped");
    }
    const auto boundary=prefill_boundary(state.history);
    desired=int64_t(state.visible_tokens)<boundary ?
        std::min(desired,uint64_t(boundary)-state.visible_tokens):std::min<uint64_t>(desired,e.block_tokens);
    auto tokens=std::min({desired,capacity_-state.visible_tokens,e.prompt_end-state.history});
    if (state.history+tokens<e.prompt_end) {
        const auto end=(state.history+tokens)/e.block_tokens*e.block_tokens;
        tokens=end>state.history ? end-state.history:0;
    }
    return tokens;
}

} // namespace kvmem
