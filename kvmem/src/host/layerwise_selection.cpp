#include "kvmem/layerwise_selection.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/layerwise_options.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kvmem {
namespace {
LayerwiseHistorySelection prepare_selection(const KvMemStore& policy,
    const RawKvStore& keys, uint32_t layer,
    const std::vector<uint32_t>& mandatory) {
    const auto& cfg = policy.config();
    const auto block = cfg.block_tokens;
    const auto budget_tokens = policy.select_budget_tokens();
    if (!block || !budget_tokens || budget_tokens % block ||
        keys.config().block_tokens != block || policy.total_tokens() % block ||
        uint64_t(policy.block_count()) * block != policy.total_tokens()) {
        throw std::invalid_argument("layerwise selection requires an aligned canonical prefix and budget");
    }
    if (cfg.select_method != KvMemMethod::Retrieval ||
        cfg.select_policy != KvMemSelectPolicy::TopK || cfg.lrfu ||
        cfg.retrieval_method != KvMemRetrievalMethod::MeanK || cfg.n_subblocks != 1 ||
        cfg.semantic_expansion != KvMemSemanticExpansion::None) {
        throw std::invalid_argument("layerwise selection requires unfused plain Mean-K TopK");
    }
    const auto count = policy.block_count();
    const auto budget = policy.budget_blocks();
    validate_layerwise_selection_bands({cfg.sink_blocks,cfg.recent_blocks},block,budget_tokens);
    for (auto id : mandatory) {
        if (id >= count) throw std::invalid_argument("layerwise mandatory block is outside the frozen prefix");
    }
    for (uint32_t id = 0; id < count; ++id) {
        const auto& record = policy.blocks()[id];
        if (record.block_id != id || record.orig_pos_start != uint64_t(id)*block ||
            record.n_tokens != block) {
            throw std::invalid_argument("layerwise policy is not the canonical sealed logical index");
        }
    }
    LayerwiseHistorySelection result;
    result.layer = layer;
    result.fixed = policy.query_fixed_blocks(mandatory);
    result.frame = query_score_frame(policy.total_tokens(), block, budget,
        cfg.sink_blocks, cfg.recent_blocks, result.fixed);
    return result;
}

void pick_selection(const KvMemStore& policy, LayerwiseHistorySelection& result) {
    const auto count = policy.block_count();
    const auto budget = policy.budget_blocks();
    std::vector<uint32_t> candidates(count);
    std::iota(candidates.begin(),candidates.end(),0u);
    // Upstream's chronological newer-first tie rule is identical to the native
    // picker's higher-ID rule for this validated canonical logical directory.
    // The picked set stays chronological; score_mask and fixed are independent.
    result.selected = policy.pick_scored_blocks(candidates,result.scores,result.fixed,budget);
    result.historical_tokens = 0;
    for (auto id : result.selected) result.historical_tokens += policy.blocks()[id].n_tokens;
    result.charged_tokens = uint64_t(result.selected.size()) * policy.config().block_tokens;
}
} // namespace

LayerwiseHistorySelection select_layerwise_history(const KvMemStore& policy,
    const RawKvStore& keys, CurrentLayerQuery query,
    const std::vector<uint32_t>& mandatory) {
    // The pinned picker knows only bands and explicit request pins. Legacy
    // media metadata promotes the newest group and expands scored groups;
    // those are not layerwise selection rules. Owners pass exact image pins.
    auto plain_policy=policy;
    plain_policy.set_media_ranges({});
    auto result = prepare_selection(plain_policy,keys,query.layer,mandatory);
    result.scores = layerwise_mean_key_scores(keys,query,result.frame);
    pick_selection(plain_policy,result);
    return result;
}

LayerwiseQuerySelection::LayerwiseQuerySelection(const KvMemStore& policy,
    const RawKvStore& keys, LayerwiseQueryIdentity identity,
    LayerwiseQueryGeometry geometry, SourceRevision revision,
    const std::vector<uint32_t>& mandatory)
    : policy_(policy), keys_(keys), identity_(std::move(identity)), geometry_(geometry),
      revision_(std::move(revision)), next_(identity_.query.begin) {
    if (!identity_.history.valid || !identity_.history.session.id ||
        !identity_.history.session.incarnation || !identity_.history.history_revision ||
        !identity_.history.execution_history || !identity_.request || identity_.query.empty() ||
        identity_.evaluation.empty() || identity_.evaluation.begin != identity_.history.frontier ||
        identity_.query.begin < identity_.evaluation.begin || identity_.query.end > identity_.evaluation.end ||
        !revision_ || identity_.history.frontier != policy_.total_tokens() ||
        geometry.layer >= keys_.config().n_layer || !geometry.query_heads ||
        !geometry.kv_heads || !geometry.head_dim || geometry.query_heads % geometry.kv_heads ||
        uint64_t(geometry.kv_heads)*geometry.head_dim != keys_.config().n_embd_k ||
        !std::isfinite(geometry.attention_scale) || geometry.attention_scale < 0) {
        throw MemoryError(MemoryErrorCode::InvalidPlan,"invalid tiled layerwise Query source or geometry");
    }
    policy_.set_media_ranges({});
    check_source();
    result_ = prepare_selection(policy_,keys_,geometry.layer,mandatory);
    result_.scores.assign(result_.frame.starts.size(),0.);
    check_source();
}

void LayerwiseQuerySelection::check_identity(const LayerwiseQueryIdentity& other) const {
    if (!(other.history == identity_.history) || other.request != identity_.request ||
        other.evaluation.begin != identity_.evaluation.begin ||
        other.evaluation.end != identity_.evaluation.end ||
        other.query.begin != identity_.query.begin || other.query.end != identity_.query.end) {
        throw MemoryError(MemoryErrorCode::Stale,"foreign or stale tiled layerwise Query identity");
    }
    if (phase_ != LayerwiseSelectionPhase::Collecting)
        throw MemoryError(MemoryErrorCode::InvalidPlan,"tiled layerwise Query selection is closed");
}

void LayerwiseQuerySelection::check_source() {
    if (!(revision_() == identity_.history)) {
        phase_ = LayerwiseSelectionPhase::Aborted;
        throw MemoryError(MemoryErrorCode::Stale,"tiled layerwise Query index revision changed");
    }
}

void LayerwiseQuerySelection::append(const LayerwiseQueryIdentity& identity,
    uint64_t begin, CurrentLayerQuery tile) {
    check_tile(identity,begin,tile.rows,tile.layer,tile.query_heads,tile.kv_heads,
               tile.head_dim,tile.attention_scale);
    check_source();
    const auto scores = layerwise_mean_key_scores(keys_,tile,result_.frame);
    check_source();
    accumulate(tile.rows,scores);
}

void LayerwiseQuerySelection::check_tile(const LayerwiseQueryIdentity& identity,
    uint64_t begin, uint32_t rows, uint32_t layer, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, float attention_scale) const {
    check_identity(identity);
    if (begin != next_ || !rows || uint64_t(rows) > identity_.query.end - next_ ||
        layer != geometry_.layer || query_heads != geometry_.query_heads ||
        kv_heads != geometry_.kv_heads || head_dim != geometry_.head_dim ||
        attention_scale != geometry_.attention_scale) {
        throw MemoryError(MemoryErrorCode::InvalidPlan,"tiled layerwise Query rows or geometry changed");
    }
}

void LayerwiseQuerySelection::append_native(const LayerwiseQueryIdentity& identity,
    uint64_t begin, NativeCurrentLayerQuery tile, const NativeScorer& scorer) {
    check_tile(identity,begin,tile.rows,tile.layer,tile.query_heads,tile.kv_heads,
               tile.head_dim,tile.attention_scale);
    validate_native_layer_query(tile);
    if (!scorer) throw MemoryError(MemoryErrorCode::InvalidPlan,"native Query scorer is missing");
    check_source();
    const auto scores = scorer(keys_,tile,result_.frame);
    check_source();
    accumulate(tile.rows,scores);
}

void LayerwiseQuerySelection::accumulate(uint32_t rows, const std::vector<double>& scores) {
    if (scores.size() != result_.scores.size())
        throw MemoryError(MemoryErrorCode::InvalidPlan,"native Query score extent differs from frozen frame");
    for (size_t i=0;i<scores.size();++i)
        if (!std::isfinite(scores[i]) || scores[i] < 0 || (!result_.frame.score_mask[i] && scores[i] != 0))
            throw MemoryError(MemoryErrorCode::InvalidPlan,"invalid raw native Query page mass");
    // No Query-token mean: upstream sums tile masses over the same denominator.
    for (size_t i=0;i<scores.size();++i) result_.scores[i] += scores[i];
    next_ += rows;
}

LayerwiseHistorySelection LayerwiseQuerySelection::finish(const LayerwiseQueryIdentity& identity) {
    check_identity(identity);
    if (next_ != identity_.query.end)
        throw MemoryError(MemoryErrorCode::Pending,"complete Query scores are required before selection");
    check_source();
    auto result = result_;
    pick_selection(policy_,result);
    check_source();
    phase_ = LayerwiseSelectionPhase::Complete;
    return result;
}

} // namespace kvmem
