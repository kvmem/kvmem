#include "kvmem/layerwise_selection.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/layerwise_options.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kvmem {
bool layerwise_score_geometry(uint32_t page_tokens, uint32_t mean_tokens) {
    return mean_tokens && (page_tokens == mean_tokens || uint64_t(mean_tokens) * 2 == page_tokens);
}

std::size_t layerwise_score_slots(const QueryScoreFrame& pages, bool keep_halves) {
    if (keep_halves && pages.starts.size() > SIZE_MAX/2)
        throw std::length_error("layerwise half-page score count overflows");
    return pages.starts.size() * (keep_halves ? 2 : 1);
}

std::vector<double> layerwise_max_page_scores(const std::vector<double>& halves) {
    if (halves.size()%2) throw std::invalid_argument("incomplete layerwise half-page pair");
    std::vector<double> pages(halves.size()/2);
    for (std::size_t i=0; i<pages.size(); ++i)
        pages[i] = std::max(halves[2*i], halves[2*i+1]);
    return pages;
}

std::vector<double> layerwise_tile_scores(const QueryScoreFrame& pages, uint32_t page_tokens,
    uint32_t mean_tokens, bool keep_halves, const std::function<std::vector<double>(const QueryScoreFrame&)>& score) {
    if (!layerwise_score_geometry(page_tokens, mean_tokens))
        throw std::invalid_argument("incompatible layerwise Mean-K and transport page sizes");
    if (mean_tokens == page_tokens) {
        if (keep_halves) throw std::invalid_argument("half-page max requires two Mean-K records per page");
        return score(pages);
    }

    const auto count = pages.starts.size();
    if (pages.counts.size() != count || pages.score_mask.size() != count ||
        pages.transport_mask.size() != count || count > SIZE_MAX / 2)
        throw std::invalid_argument("invalid layerwise transport page frame");
    QueryScoreFrame halves;
    halves.accepted_through = pages.accepted_through;
    halves.starts.reserve(2 * count);
    halves.counts.assign(2 * count, mean_tokens);
    halves.score_mask.reserve(2 * count);
    halves.transport_mask.reserve(2 * count);
    for (size_t i = 0; i < count; ++i) {
        const auto start = pages.starts[i];
        if (pages.counts[i] != page_tokens || start % page_tokens ||
            start > pages.accepted_through || page_tokens > pages.accepted_through - start)
            throw std::invalid_argument("layerwise half-page scores require aligned sealed transport pages");
        halves.starts.push_back(start);
        halves.starts.push_back(start + mean_tokens);
        halves.score_mask.insert(halves.score_mask.end(), 2, pages.score_mask[i]);
        halves.transport_mask.insert(halves.transport_mask.end(), 2, pages.transport_mask[i]);
    }
    auto mass = score(halves);
    if (mass.size() != halves.starts.size())
        throw std::invalid_argument("incomplete layerwise half-page probability vector");
    // Legacy sum is linear: collapse this tile in place, retaining its
    // original floating-point order. Max must wait until all tiles arrive.
    for (size_t i = 0; i < count; ++i) {
        const double left = mass[2*i], right = mass[2*i+1];
        const double sum = left + right;
        if (left < 0 || right < 0 || !std::isfinite(sum) || (!pages.score_mask[i] && sum != 0))
            throw std::invalid_argument("invalid layerwise half-page probability mass");
        if (!keep_halves) mass[i] = sum;
    }
    if (!keep_halves) mass.resize(count);
    return mass;
}

std::vector<double> layerwise_page_scores(const QueryScoreFrame& pages, uint32_t page_tokens,
    uint32_t mean_tokens, const std::function<std::vector<double>(const QueryScoreFrame&)>& score) {
    return layerwise_tile_scores(pages, page_tokens, mean_tokens, false, score);
}

namespace {
LayerwiseHistorySelection prepare_selection(const KvMemStore& policy,
    const RawKvStore& keys, uint32_t layer,
    const std::vector<uint32_t>& mandatory) {
    const auto& cfg = policy.config();
    const auto block = cfg.block_tokens;
    const auto budget_tokens = policy.select_budget_tokens();
    if (!block || !budget_tokens || budget_tokens % block ||
        !layerwise_score_geometry(block, keys.config().block_tokens) || policy.total_tokens() % block ||
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
    const bool keep_halves = policy.config().layerwise_half_max;
    result.scores = layerwise_tile_scores(result.frame, policy.config().block_tokens, keys.config().block_tokens, keep_halves,
        [&](const auto& frame) { return layerwise_mean_key_scores(keys,query,frame); });
    if (keep_halves) result.scores = layerwise_max_page_scores(result.scores);
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
    result_.scores.assign(layerwise_score_slots(result_.frame, policy_.config().layerwise_half_max),0.);
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
    const auto scores = layerwise_tile_scores(result_.frame, policy_.config().block_tokens, keys_.config().block_tokens,
        policy_.config().layerwise_half_max,
        [&](const auto& frame) { return layerwise_mean_key_scores(keys_,tile,frame); });
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
    const auto scores = layerwise_tile_scores(result_.frame, policy_.config().block_tokens, keys_.config().block_tokens,
        policy_.config().layerwise_half_max,
        [&](const auto& frame) { return scorer(keys_,tile,frame); });
    check_source();
    accumulate(tile.rows,scores);
}

void LayerwiseQuerySelection::accumulate(uint32_t rows, const std::vector<double>& scores) {
    if (scores.size() != result_.scores.size())
        throw MemoryError(MemoryErrorCode::InvalidPlan,"native Query score extent differs from frozen frame");
    const std::size_t slots_per_page = policy_.config().layerwise_half_max ? 2 : 1;
    for (size_t i=0;i<scores.size();++i)
        if (!std::isfinite(scores[i]) || scores[i] < 0 ||
            !std::isfinite(result_.scores[i]+scores[i]) ||
            (!result_.frame.score_mask[i/slots_per_page] && scores[i] != 0))
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
    if (policy_.config().layerwise_half_max)
        result.scores = layerwise_max_page_scores(result.scores);
    pick_selection(policy_,result);
    check_source();
    phase_ = LayerwiseSelectionPhase::Complete;
    return result;
}

} // namespace kvmem
