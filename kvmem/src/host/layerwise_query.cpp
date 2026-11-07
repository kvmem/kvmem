#include "kvmem/layerwise_query.hpp"
#include <algorithm>
#include <limits>

namespace kvmem {
namespace {
bool same_history(const MemoryStamp& a, const MemoryStamp& b) {
    return a.valid == b.valid && a.session.id == b.session.id &&
        a.session.incarnation == b.session.incarnation &&
        a.history_revision == b.history_revision && a.view_version == b.view_version &&
        a.frontier == b.frontier && a.execution_history == b.execution_history;
}
bool same_range(QueryRange a, QueryRange b) { return a.begin == b.begin && a.end == b.end; }
[[noreturn]] void invalid(const char* text) {
    throw MemoryError(MemoryErrorCode::InvalidPlan, text);
}
}

LayerwiseQueryProgress::LayerwiseQueryProgress(LayerwiseQueryIdentity identity,
                                             std::vector<uint32_t> layers)
    : identity_(std::move(identity)), layers_(std::move(layers)) {
    const auto& i = identity_;
    if (!valid_layerwise_query_identity(i) || layers_.empty() ||
        layers_.size() > std::numeric_limits<uint32_t>::max() ||
        !std::is_sorted(layers_.begin(), layers_.end()) ||
        std::adjacent_find(layers_.begin(), layers_.end()) != layers_.end())
        invalid("invalid layerwise Query identity, boundary, or model layer order");
    const auto rows = i.evaluation.end - i.evaluation.begin;
    if (rows > UINT64_MAX / layers_.size())
        invalid("layerwise Query work counter overflow");
}

void LayerwiseQueryProgress::check(const LayerwiseQueryIdentity& i) const {
    if (!same_history(i.history, identity_.history) || i.request != identity_.request ||
        !same_range(i.evaluation, identity_.evaluation) || !same_range(i.query, identity_.query))
        throw MemoryError(MemoryErrorCode::Stale, "foreign or stale layerwise Query progress");
}
void LayerwiseQueryProgress::layer(uint32_t l) const {
    if (next_ >= layers_.size() || l != layers_[next_])
        invalid("layerwise Query executed outside causal model layer order");
}
void LayerwiseQueryProgress::require(LayerwiseQueryPhase p) const {
    if (phase_ != p) invalid("invalid layerwise Query execution phase");
}
void LayerwiseQueryProgress::begin_layer(const LayerwiseQueryIdentity& i, uint32_t l) {
    check(i); require(LayerwiseQueryPhase::BetweenLayers); layer(l);
    projection_ = attention_ = identity_.evaluation.begin;
    phase_ = LayerwiseQueryPhase::Projection;
}
void LayerwiseQueryProgress::span(uint64_t begin, uint64_t count,
                                uint64_t& cursor, uint64_t& total) {
    if (!count || begin != cursor || count > identity_.evaluation.end - cursor)
        invalid("layerwise Query row gap, duplicate, or out-of-range extent");
    cursor += count; total += count;
}
void LayerwiseQueryProgress::projected(const LayerwiseQueryIdentity& i, uint32_t l,
                                     uint64_t begin, uint64_t count) {
    check(i); require(LayerwiseQueryPhase::Projection); layer(l);
    span(begin, count, projection_, work_.projected_rows);
    if (projection_ == identity_.evaluation.end) phase_ = LayerwiseQueryPhase::Selection;
}
void LayerwiseQueryProgress::begin_restore(const LayerwiseQueryIdentity& i, uint32_t l) {
    check(i); require(LayerwiseQueryPhase::Selection); layer(l);
    phase_ = LayerwiseQueryPhase::Restore;
}
void LayerwiseQueryProgress::restored(const LayerwiseQueryIdentity& i, uint32_t l,
                                    uint64_t tokens) {
    check(i); require(LayerwiseQueryPhase::Restore); layer(l);
    if (tokens > identity_.history.frontier)
        invalid("layerwise Query restored more tokens than the frozen history");
    if (next_ && tokens != history_tokens_)
        invalid("layerwise Query changed the qualified native causal length");
    history_tokens_ = tokens;
    phase_ = LayerwiseQueryPhase::Attention;
}
void LayerwiseQueryProgress::attended(const LayerwiseQueryIdentity& i, uint32_t l,
                                    uint64_t begin, uint64_t count) {
    check(i); require(LayerwiseQueryPhase::Attention); layer(l);
    span(begin, count, attention_, work_.attended_rows);
}
void LayerwiseQueryProgress::complete_layer(const LayerwiseQueryIdentity& i, uint32_t l) {
    check(i); require(LayerwiseQueryPhase::Attention); layer(l);
    if (attention_ != identity_.evaluation.end)
        invalid("layerwise Query layer completed before all causal rows");
    ++next_; ++work_.completed_layers;
    phase_ = next_ == layers_.size() ? LayerwiseQueryPhase::Complete
                                    : LayerwiseQueryPhase::BetweenLayers;
}
void LayerwiseQueryProgress::abort() noexcept { phase_ = LayerwiseQueryPhase::Aborted; }
} // namespace kvmem
