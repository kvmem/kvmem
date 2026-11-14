#include "kvmem/layerwise_request.hpp"
#include "kvmem/raw_kv_store.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace kvmem {
namespace {
[[noreturn]] void fail(MemoryErrorCode c, const char* text) { throw MemoryError(c, text); }
bool same_history(const MemoryStamp& a, const MemoryStamp& b) {
    return a.valid && b.valid && a.session == b.session &&
        a.history_revision == b.history_revision && a.frontier == b.frontier &&
        a.execution_history == b.execution_history;
}
std::vector<uint32_t> layer_ids(const std::vector<LayerwiseRequestLayer>& specs) {
    std::vector<uint32_t> ids;
    for (const auto& spec : specs) ids.push_back(spec.model_layer);
    return ids;
}
bool same_view(const ExecutionView& a, const ExecutionView& b) {
    if (!(a.stamp == b.stamp) || a.visible_tokens != b.visible_tokens || a.blocks.size() != b.blocks.size())
        return false;
    for (size_t i = 0; i < a.blocks.size(); ++i)
        if (!(a.blocks[i].block == b.blocks[i].block) || a.blocks[i].compact_start != b.blocks[i].compact_start)
            return false;
    return true;
}
}

LayerwiseQueryRequest::LayerwiseQueryRequest(LayerwiseQueryIdentity identity,
    LayerwiseMemoryCatalog catalog, const KvMemStore& policy, const RawKvStore& keys,
    std::vector<LayerwiseRequestLayer> specs, uint64_t tokens, SourceRevision revision)
    : identity_(std::move(identity)), catalog_(std::move(catalog)), policy_(policy), keys_(keys),
      specs_(std::move(specs)), native_history_tokens_(tokens), revision_(std::move(revision)),
      progress_(identity_, layer_ids(specs_)) {
    const auto block = catalog_.descriptor().layout.block_tokens;
    if (!revision_ || specs_.size() != catalog_.layers().size() ||
        identity_.history.frontier != policy_.total_tokens() || policy_.config().block_tokens != block ||
        identity_.history.frontier % block || tokens % block || tokens > identity_.history.frontier ||
        !layerwise_score_geometry(block, keys_.config().block_tokens))
        fail(MemoryErrorCode::InvalidPlan,"invalid native layerwise Query admission");
    source();
    completed_views_.reserve(specs_.size());
    std::set<uint32_t> statistics;
    for (const auto& spec : specs_) {
        const auto& snapshot = catalog_.snapshot(spec.model_layer);
        if (!same_history(snapshot.stamp, identity_.history) ||
            spec.budget.selected_tokens != policy_.select_budget_tokens() ||
            tokens > spec.budget.selected_tokens || spec.query.layer >= keys_.config().n_layer ||
            !statistics.insert(spec.query.layer).second)
            fail(MemoryErrorCode::InvalidPlan,"layerwise Query layer identity or budget mismatch");
        // Publish-sealed is a real admission requirement. Replacing just one
        // stripe must never establish a complete all-layer Host record.
        for (const auto& item : snapshot.blocks)
            if (item.host_version != item.block.content_version)
                fail(MemoryErrorCode::Pending,"layerwise Query needs complete sealed canonical history");
    }
}
void LayerwiseQueryRequest::check(const LayerwiseQueryIdentity& i) const {
    if (!(i.history == identity_.history) || i.request != identity_.request ||
        i.evaluation.begin != identity_.evaluation.begin || i.evaluation.end != identity_.evaluation.end ||
        i.query.begin != identity_.query.begin || i.query.end != identity_.query.end)
        fail(MemoryErrorCode::Stale,"foreign layerwise Query owner");
    if (phase_ == LayerwiseRequestPhase::Aborted || phase_ == LayerwiseRequestPhase::Accepted)
        fail(MemoryErrorCode::InvalidPlan,"layerwise Query request is closed");
}
void LayerwiseQueryRequest::source() {
    if (!(revision_() == identity_.history)) {
        abort(); fail(MemoryErrorCode::Stale,"native layerwise Query index changed");
    }
}
void LayerwiseQueryRequest::require(LayerwiseRequestPhase phase) const {
    if (phase_ != phase) fail(MemoryErrorCode::InvalidPlan,"invalid native layerwise Query phase");
}
const LayerwiseRequestLayer& LayerwiseQueryRequest::current() const {
    if (next_ >= specs_.size()) fail(MemoryErrorCode::InvalidPlan,"layerwise Query has no current layer");
    return specs_[next_];
}
void LayerwiseQueryRequest::begin_layer(const LayerwiseQueryIdentity& i, uint32_t model_layer) {
    check(i); require(LayerwiseRequestPhase::BetweenLayers);
    const auto& spec = current();
    if (model_layer != spec.model_layer) fail(MemoryErrorCode::InvalidPlan,"future native layer Query requested");
    source();
    auto scoring = std::make_unique<LayerwiseQuerySelection>(policy_,keys_,identity_,spec.query,
                                                           revision_,spec.mandatory);
    progress_.begin_layer(i,model_layer);
    scoring_ = std::move(scoring);
    phase_ = LayerwiseRequestPhase::Projecting;
}
void LayerwiseQueryRequest::projected(const LayerwiseQueryIdentity& i, uint64_t begin, CurrentLayerQuery q) {
    check(i); require(LayerwiseRequestPhase::Projecting);
    const auto& spec = current();
    const auto& geometry = spec.query;
    const uint64_t width = uint64_t(q.query_heads)*q.head_dim;
    if (!q.data || !q.rows || q.layer != geometry.layer || q.query_heads != geometry.query_heads ||
        q.kv_heads != geometry.kv_heads || q.head_dim != geometry.head_dim ||
        q.attention_scale != geometry.attention_scale ||
        width > std::numeric_limits<size_t>::max()/sizeof(float)/q.rows ||
        begin < identity_.evaluation.begin || begin > identity_.evaluation.end ||
        q.rows > identity_.evaluation.end-begin)
        fail(MemoryErrorCode::InvalidPlan,"native projected rows differ from admitted layer geometry");
    try {
        // Validate exact evaluation coverage before allowing duplicate Q rows
        // into the accumulator. Any later failure closes this native attempt.
        progress_.projected(i,spec.model_layer,begin,q.rows);
        const auto first = std::max(begin,identity_.query.begin);
        const auto last = std::min(begin+q.rows,identity_.query.end);
        if (first < last) {
            q.data += size_t(first-begin)*size_t(width);
            q.rows = uint32_t(last-first);
            scoring_->append(i,first,q);
        }
    } catch (...) { abort(); throw; }
}
void LayerwiseQueryRequest::projected_native(const LayerwiseQueryIdentity& i, uint64_t begin,
    NativeCurrentLayerQuery q, const LayerwiseQuerySelection::NativeScorer& scorer) {
    check(i); require(LayerwiseRequestPhase::Projecting);
    const auto& spec = current();
    const auto& geometry = spec.query;
    validate_native_layer_query(q);
    if (!scorer || q.layer != geometry.layer || q.query_heads != geometry.query_heads ||
        q.kv_heads != geometry.kv_heads || q.head_dim != geometry.head_dim ||
        q.attention_scale != geometry.attention_scale ||
        begin < identity_.evaluation.begin || begin > identity_.evaluation.end ||
        q.rows > identity_.evaluation.end-begin)
        fail(MemoryErrorCode::InvalidPlan,"native projected rows differ from admitted layer geometry");
    try {
        progress_.projected(i,spec.model_layer,begin,q.rows);
        const auto first = std::max(begin,identity_.query.begin);
        const auto last = std::min(begin+q.rows,identity_.query.end);
        if (first < last)
            scoring_->append_native(i,first,native_layer_query_rows(q,uint32_t(first-begin),uint32_t(last-first)),scorer);
    } catch (...) { abort(); throw; }
}

const LayerWorkingSetPlan& LayerwiseQueryRequest::prepare_restore(const LayerwiseQueryIdentity& i) {
    check(i); require(LayerwiseRequestPhase::Projecting);
    if (progress_.phase() != LayerwiseQueryPhase::Selection)
        fail(MemoryErrorCode::Pending,"native full-Query projection is incomplete");
    const auto& spec = current();
    try {
        auto selected = std::make_unique<LayerwiseHistorySelection>(scoring_->finish(i));
        if (selected->historical_tokens != native_history_tokens_)
            fail(MemoryErrorCode::InvalidPlan,"layerwise selection changed admitted native history extent");
        const auto& snapshot = catalog_.snapshot(spec.model_layer);
        LayerWorkingSetSelection selection{spec.model_layer,snapshot.stamp,{},{},spec.budget};
        for (auto id : selected->selected) selection.selected.push_back(snapshot.blocks.at(id).block.id);
        for (auto id : selected->fixed) selection.mandatory.push_back(snapshot.blocks.at(id).block.id);
        auto plan = std::make_unique<LayerWorkingSetPlan>(catalog_.plan_layer(selection));
        if (!plan->plan.spill.empty())
            fail(MemoryErrorCode::InvalidPlan,"sealed Query history unexpectedly requires a partial canonical spill");
        progress_.begin_restore(i,spec.model_layer);
        selection_ = std::move(selected); plan_ = std::move(plan);
        phase_ = LayerwiseRequestPhase::Restoring;
        return *plan_;
    } catch (...) { abort(); throw; }
}
const LayerwiseHistorySelection& LayerwiseQueryRequest::selection() const {
    if (!selection_ || (phase_ != LayerwiseRequestPhase::Restoring && phase_ != LayerwiseRequestPhase::Attending))
        fail(MemoryErrorCode::Pending,"native layer selection is not available");
    return *selection_;
}
void LayerwiseQueryRequest::restored(const LayerwiseQueryIdentity& i, const ExecutionView& view) {
    check(i); require(LayerwiseRequestPhase::Restoring);
    try {
        source();
        if (!same_view(view,plan_->plan.next))
            fail(MemoryErrorCode::Stale,"native post-fence layer view differs from its plan");
        progress_.restored(i,current().model_layer,view.visible_tokens);
        phase_ = LayerwiseRequestPhase::Attending;
    } catch (...) { abort(); throw; }
}
void LayerwiseQueryRequest::attended(const LayerwiseQueryIdentity& i, uint64_t begin, uint64_t count) {
    check(i); require(LayerwiseRequestPhase::Attending);
    try { progress_.attended(i,current().model_layer,begin,count); }
    catch (...) { abort(); throw; }
}
void LayerwiseQueryRequest::complete_layer(const LayerwiseQueryIdentity& i) {
    check(i); require(LayerwiseRequestPhase::Attending);
    try {
        source(); progress_.complete_layer(i,current().model_layer);
        completed_views_.push_back(std::move(plan_->plan.next));
        ++next_; scoring_.reset(); plan_.reset(); selection_.reset();
        phase_ = next_ == specs_.size() ? LayerwiseRequestPhase::AwaitingAcceptance
                                      : LayerwiseRequestPhase::BetweenLayers;
    } catch (...) { abort(); throw; }
}
void LayerwiseQueryRequest::accepted(const LayerwiseQueryIdentity& i,
                                    std::vector<LayerMemorySnapshot> snapshots) {
    check(i); require(LayerwiseRequestPhase::AwaitingAcceptance);
    if (snapshots.size() != specs_.size()) fail(MemoryErrorCode::InvalidPlan,"incomplete native layer acceptance");
    // Reuse complete catalog/extent/partial checks, including native identity,
    // instead of inventing a token-only checkpoint compatibility shortcut.
    LayerwiseMemoryCatalog accepted(catalog_.descriptor(),std::move(snapshots));
    for (size_t layer = 0; layer < specs_.size(); ++layer) {
        const auto& spec = specs_[layer];
        const auto& state = accepted.snapshot(spec.model_layer);
        const auto& previous = catalog_.snapshot(spec.model_layer);
        if (!(state.stamp.session == identity_.history.session) ||
            state.stamp.frontier != identity_.evaluation.end ||
            state.stamp.history_revision <= identity_.history.history_revision ||
            state.blocks.size() < previous.blocks.size())
            fail(MemoryErrorCode::Stale,"native Query acceptance lacks exact suffix/state history");
        for (size_t b = 0; b < previous.blocks.size(); ++b) {
            const auto& old = previous.blocks[b].block;
            const auto& now = state.blocks[b].block;
            if (!(old.session == now.session) || old.id != now.id ||
                old.original_start != now.original_start || old.valid_tokens != now.valid_tokens ||
                old.content_version != now.content_version || now.statistics_version < old.statistics_version)
                fail(MemoryErrorCode::Stale,"native Query acceptance rewrote frozen canonical content");
        }
        const MemoryBudget budget{spec.budget.device_tokens,0,spec.budget.device_tokens,
                                  spec.budget.host_payload_bytes};
        plan_working_set(catalog_.descriptor(),state,state.resident,{},budget);
        // Ordinary final Pressure may discard residents, but cannot resurrect
        // cold pre-Query history which this target layer never restored.
        std::set<uint64_t> allowed;
        for (const auto& block : completed_views_.at(layer).blocks) allowed.insert(block.block.id);
        for (size_t b = previous.blocks.size(); b < state.blocks.size(); ++b)
            allowed.insert(state.blocks[b].block.id);
        for (auto id : state.resident)
            if (!allowed.count(id))
                fail(MemoryErrorCode::InvalidPlan,"native Query acceptance resurrected unselected cold history");
    }
    accepted_ = accepted.layers();
    phase_ = LayerwiseRequestPhase::Accepted;
}
const std::vector<LayerMemorySnapshot>& LayerwiseQueryRequest::accepted_layers() const {
    if (phase_ != LayerwiseRequestPhase::Accepted)
        fail(MemoryErrorCode::Pending,"native Query has not committed its complete state");
    return accepted_;
}
void LayerwiseQueryRequest::abort() noexcept {
    if (scoring_) scoring_->abort();
    progress_.abort(); phase_ = LayerwiseRequestPhase::Aborted;
    plan_.reset(); selection_.reset(); completed_views_.clear(); accepted_.clear();
}

} // namespace kvmem
