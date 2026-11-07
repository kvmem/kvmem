#include "kvmem/layerwise_probe_selection.hpp"
#include "kvmem/layerwise_options.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

namespace kvmem {
namespace {
HostKvReservation reserve_metadata(HostKvStorage& owner,uint64_t bytes) {
    if (!owner.capacity_bytes() || owner.capacity_bytes()==UINT64_MAX)
        throw MemoryError(MemoryErrorCode::InvalidPlan,"probe selection requires a finite metadata account");
    return owner.reserve(bytes);
}
}

uint64_t LayerwiseProbeSelection::metadata_bound(const KvMemStore& policy,uint32_t layers) {
    if (!layers) throw MemoryError(MemoryErrorCode::InvalidPlan,"probe selection has no target layers");
    // Owned policy copy; all frame/fixed/score/result vectors; picker candidate,
    // scorer-return and atomic accumulation scratch. The copied policy vector
    // capacities cannot exceed the source capacities. No Q payload is owned.
    const uint64_t extra=sizeof(LayerwiseProbeSelection)+4096+
        uint64_t(policy.block_count())*128+uint64_t(layers)*(sizeof(LayerwiseQueryGeometry)+64);
    if (policy.allocated_bytes()>UINT64_MAX-extra)
        throw MemoryError(MemoryErrorCode::BudgetExceeded,"probe metadata extent overflow");
    return policy.allocated_bytes()+extra;
}

LayerwiseProbeSelection::LayerwiseProbeSelection(const KvMemStore& policy,const RawKvStore& keys,
    LayerwiseQueryIdentity identity,const std::vector<LayerwiseQueryGeometry>& geometry,
    SourceRevision revision,HostKvStorage& metadata,const std::vector<uint32_t>& mandatory)
    :metadata_bytes_(metadata_bound(policy,uint32_t(geometry.size()))),
     metadata_(std::make_shared<HostKvReservation>(reserve_metadata(metadata,metadata_bytes_))),policy_(policy),keys_(keys),
     identity_(std::move(identity)),geometry_(geometry),revision_(std::move(revision)),
     next_(identity_.query.begin) {
    if (!identity_.history.valid || !identity_.history.session.id || !identity_.history.session.incarnation ||
        !identity_.history.history_revision || !identity_.history.execution_history || !identity_.request ||
        identity_.query.empty() || identity_.evaluation.empty() ||
        identity_.evaluation.begin!=identity_.history.frontier || identity_.query.begin<identity_.evaluation.begin ||
        identity_.query.end>identity_.evaluation.end || identity_.history.frontier!=policy_.total_tokens() || !revision_)
        throw MemoryError(MemoryErrorCode::InvalidPlan,"invalid private-probe Query identity");
    const auto& cfg=policy_.config();const auto block=cfg.block_tokens;
    const auto budget=policy_.select_budget_tokens();
    if (!block || !budget || budget%block || keys_.config().block_tokens!=block ||
        policy_.total_tokens()%block || uint64_t(policy_.block_count())*block!=policy_.total_tokens() ||
        cfg.select_method!=KvMemMethod::Retrieval || cfg.select_policy!=KvMemSelectPolicy::TopK || cfg.lrfu ||
        cfg.retrieval_method!=KvMemRetrievalMethod::MeanK || cfg.n_subblocks!=1 ||
        cfg.semantic_expansion!=KvMemSemanticExpansion::None)
        throw MemoryError(MemoryErrorCode::InvalidPlan,"probe selection requires a sealed plain Mean-K TopK index");
    validate_layerwise_selection_bands({cfg.sink_blocks,cfg.recent_blocks},block,budget);
    for (uint32_t id=0;id<policy_.block_count();++id) {
        const auto& record=policy_.blocks()[id];
        if (record.block_id!=id || record.orig_pos_start!=uint64_t(id)*block || record.n_tokens!=block)
            throw MemoryError(MemoryErrorCode::InvalidPlan,"probe policy is not the canonical sealed prefix");
    }
    for (auto id:mandatory)if(id>=policy_.block_count())
        throw MemoryError(MemoryErrorCode::InvalidPlan,"probe mandatory block is outside accepted history");
    std::set<uint32_t> seen;
    for (const auto& g:geometry_) {
        const auto& first=geometry_.front();
        if (!seen.insert(g.layer).second || g.layer>=keys_.config().n_layer ||
            !g.query_heads || !g.kv_heads || !g.head_dim || g.query_heads%g.kv_heads ||
            uint64_t(g.kv_heads)*g.head_dim!=keys_.config().n_embd_k ||
            !std::isfinite(g.attention_scale) || g.attention_scale<0 ||
            g.query_heads!=first.query_heads || g.kv_heads!=first.kv_heads ||
            g.head_dim!=first.head_dim || g.attention_scale!=first.attention_scale)
            throw MemoryError(MemoryErrorCode::InvalidPlan,"private-probe target tensor geometry differs from upstream LTHD");
        for (uint32_t id=0;id<policy_.block_count();++id)
            if (keys_.mean_k_tokens(id,g.layer)!=block)
                throw MemoryError(MemoryErrorCode::InvalidPlan,"probe target Mean-K does not match the accepted frame");
    }
    // New layerwise paths use only explicit request pins plus exact fixed
    // bands; legacy media promotion/expansion is not part of this picker.
    policy_.set_media_ranges({});
    check_source();
    result_.fixed=policy_.query_fixed_blocks(mandatory);
    result_.frame=query_score_frame(policy_.total_tokens(),block,policy_.budget_blocks(),
        cfg.sink_blocks,cfg.recent_blocks,result_.fixed);
    result_.scores.assign(result_.frame.starts.size(),0.);
    result_.metadata_=metadata_;
    check_source();
}

void LayerwiseProbeSelection::check_identity(const LayerwiseQueryIdentity& other) const {
    if (!(other.history==identity_.history) || other.request!=identity_.request ||
        other.query.begin!=identity_.query.begin || other.query.end!=identity_.query.end ||
        other.evaluation.begin!=identity_.evaluation.begin || other.evaluation.end!=identity_.evaluation.end)
        throw MemoryError(MemoryErrorCode::Stale,"foreign or stale private-probe Query identity");
    if (phase_!=LayerwiseSelectionPhase::Collecting)
        throw MemoryError(MemoryErrorCode::InvalidPlan,"private-probe selection is closed");
}
void LayerwiseProbeSelection::check_source() {
    if (!(revision_()==identity_.history)) {
        abort();throw MemoryError(MemoryErrorCode::Stale,"private-probe accepted index changed");
    }
}
void LayerwiseProbeSelection::check_tile(const LayerwiseQueryIdentity& identity,uint64_t begin,uint32_t rows,
    uint32_t layer,uint32_t qheads,uint32_t kheads,uint32_t dim,float scale) const {
    check_identity(identity);
    const auto& g=geometry_[layer_cursor_];
    if (begin!=next_ || !rows || uint64_t(rows)>identity_.query.end-next_ ||
        (layer_cursor_ && rows!=group_rows_) || layer!=g.layer || qheads!=g.query_heads ||
        kheads!=g.kv_heads || dim!=g.head_dim || scale!=g.attention_scale)
        throw MemoryError(MemoryErrorCode::InvalidPlan,"private-probe tile lost token-major target coverage");
}
void LayerwiseProbeSelection::append(const LayerwiseQueryIdentity& identity,uint64_t begin,CurrentLayerQuery tile) {
    check_tile(identity,begin,tile.rows,tile.layer,tile.query_heads,tile.kv_heads,tile.head_dim,tile.attention_scale);
    check_source();const auto scores=layerwise_mean_key_scores(keys_,tile,result_.frame);check_source();
    accumulate(tile.rows,scores);
}
void LayerwiseProbeSelection::append_native(const LayerwiseQueryIdentity& identity,uint64_t begin,
    NativeCurrentLayerQuery tile,const NativeScorer& scorer) {
    check_tile(identity,begin,tile.rows,tile.layer,tile.query_heads,tile.kv_heads,tile.head_dim,tile.attention_scale);
    validate_native_layer_query(tile);
    if (!scorer)throw MemoryError(MemoryErrorCode::InvalidPlan,"private-probe native scorer is missing");
    check_source();const auto scores=scorer(keys_,tile,result_.frame);check_source();accumulate(tile.rows,scores);
}
void LayerwiseProbeSelection::accumulate(uint32_t rows,const std::vector<double>& scores) {
    if (scores.size()!=result_.scores.size())
        throw MemoryError(MemoryErrorCode::InvalidPlan,"private-probe mass shape differs from frozen frame");
    auto accumulated=result_.scores;
    for (size_t i=0;i<scores.size();++i) {
        if (!std::isfinite(scores[i]) || scores[i]<0 || (!result_.frame.score_mask[i] && scores[i]!=0))
            throw MemoryError(MemoryErrorCode::InvalidPlan,"invalid private-probe raw page mass");
        // Existing scorer already averages heads and sums tokens. Upstream's
        // remaining reduction is the uniform TARGET-layer mean, exactly once.
        accumulated[i]+=scores[i]/double(geometry_.size());
        if (!std::isfinite(accumulated[i]))
            throw MemoryError(MemoryErrorCode::InvalidPlan,"private-probe accumulated mass overflow");
    }
    result_.scores=std::move(accumulated);
    if (!layer_cursor_)group_rows_=rows;
    if (++layer_cursor_==geometry_.size()) {layer_cursor_=0;next_+=rows;group_rows_=0;}
}
LayerwiseProbeHistorySelection LayerwiseProbeSelection::finish(const LayerwiseQueryIdentity& identity) {
    check_identity(identity);
    if (layer_cursor_ || next_!=identity_.query.end)
        throw MemoryError(MemoryErrorCode::Pending,"all private-probe Query targets are required before selection");
    check_source();
    std::vector<uint32_t> candidates(policy_.block_count());std::iota(candidates.begin(),candidates.end(),0u);
    auto selected=policy_.pick_scored_blocks(candidates,result_.scores,result_.fixed,policy_.budget_blocks());
    uint64_t historical=0;for(auto id:selected)historical+=policy_.blocks()[id].n_tokens;
    check_source();result_.selected=std::move(selected);result_.historical_tokens=historical;
    result_.charged_tokens=uint64_t(result_.selected.size())*policy_.config().block_tokens;
    phase_=LayerwiseSelectionPhase::Complete;return std::move(result_);
}
} // namespace kvmem
