#include "statistics_observer.hpp"
#include "pool_geometry.hpp"
#include "rolling_pressure.hpp"
#include "kvmem/mean_key_retrieval.hpp"
#include "mem/kv.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace kvmem::radiance {
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
uint32_t positive(int64_t n) {
    require(n > 0 && n <= INT32_MAX, "statistics geometry outside supported range");
    return uint32_t(n);
}
int failure(const char* where, const std::exception& error) {
    RAD_ERR("KVMem statistics %s: %s", where, error.what());
    return RAD_E_STATE;
}
}
int StatisticsObserver::pool_token_limit(const rad::Config& cfg, const rad::Program& p, int64_t* out) noexcept {
    try {
        *out = 0;
        if (!options_.budget_tokens) return RAD_OK;
        require(cfg.tp == 1 && cfg.max_seqs == 1 && !cfg.prefix_cache && !cfg.kld(),
                "bounded residency requires TP1, one sequence and no engine prefix cache");
        std::vector<uint32_t> pages;
        for (size_t g = 1; g < p.kv_groups.size(); ++g) {
            if (p.kv_groups[g].decl.kind == RAD_KV_LINEAR || p.kv_groups[g].decl.kind == RAD_KV_CONV) continue;
            if (p.kv_groups[g].decl.kind == RAD_KV_WINDOW && p.drafter.kind == RAD_DRAFT_BLOCK) continue;
            require(p.kv_groups[g].decl.kind == RAD_KV_FULL && !p.kv_groups[g].zero,
                    "bounded residency currently supports FULL-attention groups only");
            pages.push_back(positive(p.kv_groups[g].block_size));
        }
        const auto geometry = pool_geometry(options_.block_tokens, options_.budget_tokens, positive(cfg.max_tok), pages);
        // Native refs stay shared until pressure. Host demotion then makes old
        // pages recyclable without a permanent second view-sized GPU reserve.
        *out = leased_pool_tokens(geometry, options_.budget_tokens, options_.replay_tokens);
        RAD_INFO("KVMEM_POOL_LIMIT resident_tokens=%lld logical_context=%lld",
                 (long long)*out, (long long)(cfg.max_ctx > 0 ? cfg.max_ctx : p.meta.n_ctx_train));
        RAD_INFO("KVMEM_POOL_MODE native_recycle_v2 budget=%u output_reserve=%u",
                 options_.budget_tokens, options_.replay_tokens);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("pool", e); }
}
int StatisticsObserver::configure(const rad::Config& cfg, rad::CtxDesc& desc) noexcept {
    try {
        require(!capture_ && desc.program && desc.rank == 0 && desc.world_size == 1 && cfg.tp == 1,
                "initial statistics bridge requires TP1");
        require(cfg.max_seqs == 1 && !cfg.prefix_cache && !cfg.kld(),
                "use max-num-seqs 1 and no-prefix-cache");
        const auto& p = *desc.program;
        require(cfg.n_spec == 0 || p.drafter.kind == RAD_DRAFT_BLOCK || p.drafter.kind == RAD_DRAFT_SERIAL,
                "speculation bridge needs a native block or serial drafter");
        serial_drafter_ = cfg.n_spec > 0 && p.drafter.kind == RAD_DRAFT_SERIAL;
        require(!p.pooled_buf && !p.encoder.modalities && !p.audio_encoder.modalities,
                "initial statistics bridge supports text generation only");
        max_position_ = positive(cfg.max_ctx > 0 ? cfg.max_ctx : p.meta.n_ctx_train);
        std::vector<StatisticsLayer> layers;
        uint64_t block_tokens = positive(options_.block_tokens);
        for (size_t group = 1; group < p.kv_groups.size(); ++group) {
            const auto& g = p.kv_groups[group];
            if (g.decl.kind == RAD_KV_LINEAR || g.decl.kind == RAD_KV_CONV) continue;
            if (g.decl.kind == RAD_KV_WINDOW) {
                require(p.drafter.kind == RAD_DRAFT_BLOCK && !g.layers.empty() &&
                    std::all_of(g.layers.begin(), g.layers.end(), [&](int l) { return l >= p.meta.n_layers; }),
                    "windowed trunk attention needs its own memory policy");
                continue; // Drafter's native rolling cache is not retrieval history.
            }
            require(g.decl.kind == RAD_KV_FULL && !g.zero && !g.layers.empty(),
                    "initial statistics bridge requires populated FULL-attention groups");
            const uint32_t native_block = positive(g.block_size);
            block_tokens = (block_tokens / std::gcd(block_tokens, uint64_t(native_block))) * native_block;
            require(block_tokens <= uint64_t(INT32_MAX), "selection block size overflow");
            const auto hd = positive(g.decl.head_dim);
            const auto kh = positive(g.decl.n_head_kv);
            require(hd == positive(p.meta.head_dim) && kh == positive(p.meta.n_head_kv),
                    "per-group heads differ from the model metadata");
            for (int layer : g.layers) {
                // MTP shares these physical pages, but its Q/K do not score
                // target-model retrieval. groups_ below keeps ALL payload layers.
                if (layer < p.meta.n_layers)
                    layers.push_back({layer, rad_kvgroup(group), 0, hd, positive(p.meta.n_head), kh});
                else require(serial_drafter_, "unexpected non-trunk FULL KV layer");
            }
        }
        require(!layers.empty(), "no attention layers declared");
        std::sort(layers.begin(), layers.end(), [](const auto& a, const auto& b) { return a.layer < b.layer; });
        for (size_t i = 0; i < layers.size(); ++i) layers[i].ordinal = uint32_t(i);
        const uint64_t kw = uint64_t(layers[0].head_dim) * layers[0].kv_heads;
        require(kw <= UINT32_MAX, "mean-K width overflow");
        store_config_.n_layer = uint32_t(layers.size());
        store_config_.n_embd_k = store_config_.n_embd_v = uint32_t(kw);
        store_config_.block_tokens = uint32_t(block_tokens);
        require(!options_.budget_tokens || (options_.budget_tokens % block_tokens == 0 &&
                options_.budget_tokens >= 2 * block_tokens), "selection budget needs whole blocks and room for sink + tail");
        if (options_.budget_tokens) {
            keep_ = resolve_kvmem_keep_allocation(uint32_t(block_tokens), options_.budget_tokens,
                -1, -1, options_.sink_tokens, options_.recent_tokens);
            RAD_INFO("KVMEM_KEEP sink_tokens=%u recent_tokens=%u sink_blocks=%u recent_blocks=%u budget=%u",
                keep_.sink_effective_tokens, keep_.recent_effective_tokens,
                keep_.sink_blocks, keep_.recent_blocks, options_.budget_tokens);
        }
        device_ = desc.device; max_rows_ = positive(cfg.max_tok);
        head_dim_ = layers[0].head_dim; query_heads_ = layers[0].query_heads; kv_heads_ = layers[0].kv_heads;
        groups_ = p.kv_groups;
        identity_ = {"radiance", KVMEM_RADIANCE_UPSTREAM_REVISION, cfg.model, "full-native-pages-v1"};
        capture_ = std::make_unique<StatisticsCapture>(desc.device, layers, positive(cfg.max_tok));
        desc.kv_observer = StatisticsCapture::observe;
        desc.kv_observer_data = capture_.get();
        RAD_INFO("KVMem statistics: %zu attention layers, selection block %u, max rows %lld, selection budget %u",
                 layers.size(), store_config_.block_tokens, (long long)cfg.max_tok, options_.budget_tokens);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("configure", e); }
}
int StatisticsObserver::bind_rank(rad::RankResources resources) noexcept {
    try {
        require(resources.lifetime && resources.ctx && resources.kv && !resources_.lifetime,
                "missing/duplicate native rank binding");
        resources_ = std::move(resources);
        device_tokens_ = UINT64_MAX;
        for (const auto& g : resources_.kv->plans()) {
            if (g.kind < 0) continue;
            if (g.stateful()) {
                require(!g.zero && g.state_copies >= 1, "state retention requires a committed slot");
                continue;
            }
            if (g.kind == RAD_KV_WINDOW) continue;
            require(g.kind == RAD_KV_FULL && !g.zero && g.block_size > 0, "unsupported native rank geometry");
            device_tokens_ = std::min(device_tokens_, uint64_t(g.n_blocks) * uint64_t(g.block_size));
        }
        if (options_.budget_tokens) {
            // Prefix can leave GPU under pressure; bounded append slack remains.
            const uint64_t reserve = uint64_t(max_rows_) + store_config_.block_tokens;
            require(device_tokens_ >= uint64_t(options_.budget_tokens) + 2 * reserve,
                    "KV pool too small for selection, append and transactional scratch");
        }
        uint64_t carved = 0;
        for (const auto& g : resources_.kv->plans()) carved += uint64_t(resources_.kv->pool_bytes_carved(g.index));
        RAD_INFO("KVMEM_POOL resident_tokens=%llu carved_bytes=%llu logical_context=%u",
                 (unsigned long long)device_tokens_, (unsigned long long)carved, max_position_);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("bind", e); }
}
void StatisticsObserver::clear_session() {
    std::atomic_store(&cached_identity_, std::shared_ptr<const CachedIdentity>{});
    prefix_.reset(); prefix_mean_.clear();
    checkpoint_.reset(); memory_session_.reset(); memory_.reset(); policy_.reset();
    store_.reset(); session_ = {}; active_request_ = 0;
    query_total_.clear(); query_rows_.clear(); query_counts_.clear();
}
int StatisticsObserver::resume(const rad::StepPlan& plan, int64_t* resumed, std::string* refusal) noexcept {
    *resumed = -1; refusal->clear();
    try {
        if (plan.e.size() != 1) return RAD_OK;
        const auto& e = plan.e[0];
        if (!e.is_prefill || e.ctx_len != 0) return RAD_OK;
        if (active_request_ == e.seq) return RAD_OK; // same request's unsampled replay
        require(e.req && !e.dev && !e.n_spec && !active_request_, "invalid fresh request boundary");
        const auto identity = std::atomic_load(&cached_identity_);
        if (prefix_ && identity && identity->matches(*e.req) && fresh_reserve_error(*e.req).empty()) {
            require(store_ && memory_ && query_frozen_, "incomplete retained session");
            require(rad_stream_sync(resources_.ctx->stream()) >= 0, "drain new state clears before resume");
            // The current plan reserved, but has NOT evaluated, its first batch.
            // Discard that reservation; the engine rebuilds the plan after restore.
            require(resources_.kv->rollback(e.seq, 0) >= 0, "release unissued initial reservation");
            prefix_->restore(*resources_.kv, e.seq, e.req->prompt, device_, resources_.ctx->stream());
            memory_->rebind_sequence(e.seq, prefix_->host_bytes() != 0);
            frontier_ = uint32_t(prefix_->frontier());
            prefix_.reset(); prefix_mean_.clear();
            std::atomic_store(&cached_identity_, std::shared_ptr<const CachedIdentity>{});
            active_request_ = e.seq;
            memory_session_ = std::make_unique<MemorySession>(memory_);
            replay_done_ = true; replaying_ = false; checkpoint_.reset();
            steps_ = reselections_ = spilled_ = restored_ = 0;
            selected_frontier_ = frontier_; selected_visible_ = 0; decode_reselections_ = 0;
            const auto resumed_view = memory_->snapshot();
            for (const auto& item : resumed_view.blocks)
                if (std::find(resumed_view.resident.begin(), resumed_view.resident.end(), item.block.id) != resumed_view.resident.end())
                    selected_visible_ += item.block.valid_tokens;
            // Reuse the unchanged query; appended tool history is processed in
            // bounded chunks. Only generation is pinned in its declared reserve.
            if (frontier_ > options_.budget_tokens) reselect();
            *resumed = frontier_;
            RAD_INFO("KVMEM_PREFIX_RESUME request=%llu session=%llu frontier=%u exact_prefix=1 same_query=1",
                (unsigned long long)e.seq, (unsigned long long)session_.id, frontier_);
            return RAD_OK;
        }
        clear_session();
        *refusal = fresh_reserve_error(*e.req);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("resume", e); }
}
int StatisticsObserver::begin(const rad::StepPlan& plan) noexcept {
    try {
        require(capture_ && plan.e.size() == 1 && plan.draft_pass == 0,
                "unsupported statistics step");
        const auto& e = plan.e[0];
        require(!e.dev && e.seq && e.ctx_len >= 0 && e.n_tokens > 0 &&
                e.n_spec >= 0 && e.n_spec < e.n_tokens && (!e.is_prefill || e.n_spec == 0),
                "statistics step requires exact host positions and a valid verify prefix");
        if (!store_) {
            require(e.ctx_len == 0 && next_incarnation_ != 0, "new statistics session must start at position zero");
            session_ = {e.seq, next_incarnation_++};
            active_request_ = e.seq;
            store_ = std::make_unique<RawKvStore>(store_config_);
            frontier_ = 0; steps_ = 0; query_total_.clear(); query_rows_.clear();
            query_counts_.clear(); query_frozen_ = false;
            checkpoint_.reset(); replaying_ = replay_done_ = false; content_version_ = 0;
            require(e.req, "statistics session has no frontend request identity");
            query_begin_ = e.req->query_begin; query_end_ = e.req->query_end;
            require((query_begin_ == -1 && query_end_ == -1) ||
                (query_begin_ >= 0 && query_end_ > query_begin_ && size_t(query_end_) <= e.req->prompt.size()),
                "invalid frontend query span");
            RAD_INFO("KVMEM_QUERY_SPAN session=%llu begin=%d end=%d proven=%d",
                (unsigned long long)session_.id, query_begin_, query_end_, int(query_begin_ >= 0));
            reselections_ = spilled_ = restored_ = 0;
            selected_frontier_ = selected_visible_ = decode_reselections_ = 0;
            if (options_.budget_tokens) {
                KvMemStoreConfig config;
                config.block_tokens = store_config_.block_tokens;
                config.select_budget = config.prefill_budget = options_.budget_tokens;
                config.gen_budget = max_rows_ + store_config_.block_tokens;
                config.sink_blocks = keep_.sink_blocks;
                config.recent_blocks = keep_.recent_blocks;
                policy_ = std::make_unique<KvMemStore>(config);
            }
        }
        require(active_request_ == e.seq && uint32_t(e.ctx_len) == frontier_, "statistics session/frontier diverged");
        require(frontier_ <= max_position_ && uint32_t(e.n_tokens) <= max_position_ - frontier_,
                "statistics step exceeds context limit");
        if (policy_ && !checkpoint_ && !replay_done_ && query_begin_ >= 0 &&
            rolling::query_checkpoint_due(frontier_, uint64_t(e.n_tokens), uint32_t(query_begin_), store_config_.block_tokens)) {
            require(frontier_ % store_config_.block_tokens == 0,
                    "query checkpoint requires a block-aligned chunk boundary");
            require(fresh_reserve_error(*e.req).empty(), "output and verify step exceed the generation reserve");
            checkpoint_ = std::make_unique<StateCheckpoint>();
            checkpoint_->save(*resources_.kv, active_request_, device_, resources_.ctx->stream());
            replay_start_ = frontier_;
            RAD_INFO("KVMEM_QUERY_CHECKPOINT session=%llu frontier=%u state_bytes=%zu",
                (unsigned long long)session_.id, replay_start_, checkpoint_->bytes());
        }
        prefill_collected_ = deferred_selection_ = false;
        capture_->begin(session_, frontier_, uint32_t(e.n_tokens),
            query_begin_ < 0 || query_frozen_ ? UINT32_MAX : uint32_t(query_begin_),
            query_end_ < 0 || query_frozen_ ? UINT32_MAX : uint32_t(query_end_));
        return RAD_OK;
    } catch (const std::exception& e) { return failure("begin", e); }
}
std::string StatisticsObserver::fresh_reserve_error(const rad::Request& request) const {
    if (!options_.budget_tokens) return {};
    if (options_.rolling_generation) {
        if (request.max_tokens <= 0) return "positive output limit required";
        if (uint64_t(options_.replay_tokens) <= uint64_t(max_rows_) + store_config_.block_tokens)
            return "rolling reserve must exceed one batch plus block";
        // The native HTTP layer still clamps total prompt + output to context.
        return {};
    }
    if (!output_fits_reserve(request.max_tokens, max_rows_, options_.replay_tokens))
        return "max_tokens and one verify step exceed the configured --kvmem-replay-tokens reserve";
    return {};
}
std::string StatisticsObserver::validate_request(const rad::Request& request) const {
    return fresh_reserve_error(request);
}
void StatisticsObserver::retain_prefix(const rad::StepEntry& e) {
    if (!policy_ || query_begin_ < 0 || (!replay_done_ && !replaying_) || !e.req || e.req->media || prefix_) return;
    if (e.req->history_end == 0) return; // No positive native checkpoint after drafter refill/alignment.
    const auto end = e.req->history_end > 0 ? size_t(e.req->history_end) : e.req->prompt.size();
    if (frontier_ != end) return;
    // Assistant-prefill continuations may have no proven history_end (-1),
    // leaving the full prompt's partial tail as the fallback end. Such a tail
    // is valid inference input, but cannot be retained as a full-block view.
    // Skip only this optional cache snapshot; never round away evaluated rows
    // or relax NativeMemory's exact restoration invariant.
    if (!rolling::retainable_prefix_boundary(frontier_, end, store_config_.block_tokens)) {
        RAD_INFO("KVMEM_PREFIX_SKIP request=%llu frontier=%u reason=partial-block",
            (unsigned long long)active_request_, frontier_);
        return;
    }
    require(e.is_prefill, "prefix checkpoint must name an evaluated prefill boundary");
    checkpoint_.reset(); // obsolete replay snapshot; keep only one retained state
    sync_memory();
    auto identity = std::make_shared<CachedIdentity>();
    identity->tokens.assign(e.req->prompt.begin(), e.req->prompt.begin() + end);
    identity->query_begin = query_begin_; identity->query_end = query_end_;
    // Alignment can put the checkpoint inside the last user span. Match its
    // ENTIRE content, including the part that will be recomputed after restore.
    identity->query_tokens.assign(e.req->prompt.begin() + query_begin_, e.req->prompt.begin() + query_end_);
    auto mean = store_->mean_checkpoint(frontier_);
    auto lease = std::make_unique<PrefixLease>(resources_.lifetime, *resources_.kv, active_request_,
        identity->tokens, device_, resources_.ctx->stream());
    memory_->retain_view();
    prefix_ = std::move(lease); prefix_mean_ = std::move(mean);
    std::atomic_store(&cached_identity_, std::shared_ptr<const CachedIdentity>(std::move(identity)));
    RAD_INFO("KVMEM_PREFIX_RETAIN request=%llu frontier=%u state_and_pages=1",
        (unsigned long long)active_request_, frontier_);
}
int StatisticsObserver::after_forward(const rad::StepPlan& plan, int64_t* rewind) noexcept {
    *rewind = -1;
    rad::Scheduler::StepResult accepted{};
    accepted.n_seq = int(plan.e.size()); // split path is prefill-only, with no draft rows
    const int rc = collect(plan, accepted);
    if (rc < 0) return rc;
    try {
        // Capture completion only fences Q/K copies, not later GDN/output ops.
        require(rad_stream_sync(resources_.ctx->stream()) >= 0, "drain model before replay/sampling");
        const auto& e = plan.e[0];
        if (!policy_ || !e.is_prefill) return RAD_OK;
        if (!e.produces_token) {
            // A small untrimmed input needs no probe replay. Longer inputs
            // checkpoint only during the corrected replay, never the probe.
            if (!replay_done_ && !replaying_ && e.req->prompt.size() <= options_.budget_tokens)
                replay_done_ = true;
            if (!serial_drafter_) retain_prefix(e);
            return RAD_OK;
        }
        if (!query_frozen_) return RAD_OK;
        if (replaying_) {
            replaying_ = false; replay_done_ = true;
            RAD_INFO("KVMEM_QUERY_REPLAY_DONE session=%llu begin=%u end=%u before_sampler=1",
                (unsigned long long)session_.id, replay_start_, frontier_);
            checkpoint_.reset();
        } else if (!replay_done_ && frontier_ > options_.budget_tokens) {
            require(bool(checkpoint_), "missing query state checkpoint");
            // The unsampled probe never issues a serial history pass. Its new
            // rows are discarded below, not archived. Old rows already contain
            // both target and drafter KV from completed steps.
            if (serial_drafter_)
                require(resources_.kv->rollback(e.seq, frontier_) >= 0, "trim unsampled serial probe");
            sync_memory();
            memory_session_.reset();
            memory_->rewind(replay_start_);
            memory_session_ = std::make_unique<MemorySession>(memory_);
            checkpoint_->restore(*resources_.kv, active_request_, device_, resources_.ctx->stream());
            store_->truncate_to(replay_start_);
            policy_->truncate_to(replay_start_);
            const uint32_t probe_end = frontier_;
            frontier_ = replay_start_;
            std::fill(query_rows_.begin(), query_rows_.end(), frontier_);
            // Replay the complete dependent suffix with frozen query scores.
            // Native pages are archived/reselected between bounded prefill
            // chunks; generation is pinned only after the final chunk.
            if (frontier_) reselect();
            replaying_ = true;
            *rewind = replay_start_;
            RAD_INFO("KVMEM_QUERY_REPLAY session=%llu begin=%u end=%u before_sampler=1",
                (unsigned long long)session_.id, replay_start_, probe_end);
        } else replay_done_ = true;
        if (*rewind < 0 && !serial_drafter_) retain_prefix(e);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("replay", e); }
}
int StatisticsObserver::collect(const rad::StepPlan& plan, const rad::Scheduler::StepResult& result) noexcept {
    try {
        require(plan.e.size() == 1 && result.n_seq == 1 && plan.e[0].seq == active_request_,
                "statistics collection identity mismatch");
        if (plan.e[0].is_prefill && prefill_collected_) {
            // Second callback: collect_step has fenced native serial history
            // and proposals. Publish pages only now, with all layer payloads
            // initialized. Block drafters use their separate native window.
            if (serial_drafter_) {
                require(resources_.kv->rollback(plan.e[0].seq, frontier_) >= 0,
                        "trim serial lookahead before page publication");
                if (deferred_selection_) reselect();
                retain_prefix(plan.e[0]);
            }
            return RAD_OK;
        }
        capture_->seal();
        static const bool wait_trace = std::getenv("KVMEM_CAPTURE_WAIT_TRACE") != nullptr;
        const auto wait_start = std::chrono::steady_clock::now();
        unsigned sleeps = 0;
        const auto deadline = wait_start + std::chrono::seconds(30);
        while (true) {
            const auto state = capture_->poll();
            if (state.status == TransferStatus::Complete) break;
            require(state.status == TransferStatus::Pending, "GPU statistics transfer failed");
            require(std::chrono::steady_clock::now() < deadline, "GPU statistics transfer timed out");
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            ++sleeps;
        }
        if (wait_trace) std::fprintf(stderr, "KVMEM_CAPTURE_WAIT prefill=%d rows=%lld sleeps=%u wait_us=%.3f\n",
            int(plan.e[0].is_prefill), (long long)plan.e[0].n_tokens, sleeps,
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-wait_start).count());
        const auto& entry = plan.e[0];
        const int accepted = result.n_accepted ? result.n_accepted[0] : 0;
        require(entry.is_prefill || (accepted >= 0 && accepted <= entry.n_spec), "invalid draft acceptance");
        const uint32_t rows = entry.is_prefill ? uint32_t(entry.n_tokens) : uint32_t(1 + accepted);
        // A verified trunk prefix is the only new history. Rejected proposals
        // never enter mean-K or the archive. Native scheduler rollback is
        // idempotent and will observe the same accepted frontier at commit.
        // The native block drafter may reserve a full verify block ahead of
        // the next trunk step. A final PREFILL chunk can be shorter than that
        // block too; allocated lookahead is not evaluated/accepted history.
        if (!(serial_drafter_ && entry.is_prefill))
            require(resources_.kv->rollback(entry.seq, entry.ctx_len + rows) >= 0,
                    "rollback unaccepted native KV before publishing statistics");
        auto query = capture_->commit(*store_, session_, rows,
            query_begin_ < 0 || query_frozen_ ? UINT32_MAX : uint32_t(query_begin_),
            query_end_ < 0 || query_frozen_ ? UINT32_MAX : uint32_t(query_end_));
        if (query_total_.empty()) {
            query_total_.resize(query.sums.size());
            query_rows_.resize(query.token_counts.size());
            query_counts_.resize(query.token_counts.size());
        }
        require(query_total_.size() == query.sums.size(), "query geometry changed during session");
        for (size_t i = 0; i < query.sums.size(); ++i) {
            require(std::isfinite(query.sums[i]), "non-finite Q statistics");
            query_total_[i] += query.sums[i];
        }
        for (size_t i = 0; i < query.token_counts.size(); ++i) {
            query_rows_[i] += rows; // seal() proved all attention layers were captured
            query_counts_[i] += query.token_counts[i];
        }
        frontier_ += rows;
        if (!query_frozen_ && query_begin_ >= 0 && frontier_ >= uint32_t(query_end_)) {
            for (auto n : query_counts_) require(n == uint32_t(query_end_ - query_begin_), "query span coverage incomplete");
            query_frozen_ = true;
            RAD_INFO("KVMEM_QUERY_FROZEN session=%llu begin=%d end=%d tokens=%d",
                (unsigned long long)session_.id, query_begin_, query_end_, query_end_ - query_begin_);
        }
        ++steps_;
        if (policy_) {
            policy_->register_append(rows);
            // Intermediate chunks reclaim on block boundaries. A prompt's last
            // chunk must select before generation even when its tail is partial.
            const bool final_probe = plan.e[0].is_prefill && plan.e[0].produces_token && query_begin_ >= 0;
            if (entry.is_prefill && (!final_probe || replaying_ || replay_done_) &&
                (frontier_ % store_config_.block_tokens == 0 ||
                 (plan.e[0].is_prefill && plan.e[0].produces_token)) && frontier_ > options_.budget_tokens) {
                if (serial_drafter_) deferred_selection_ = true;
                else reselect();
            }
        }
        if (policy_ && options_.rolling_generation && !entry.is_prefill) {
            require(frontier_ >= selected_frontier_, "rolling frontier moved backwards");
            const auto resident = selected_visible_ + frontier_ - selected_frontier_;
            if (rolling::needs_pressure(resident, options_.budget_tokens, options_.replay_tokens,
                                        max_rows_, store_config_.block_tokens)) {
                // collect_step has fenced the serial drafter. Rejected target/
                // draft lookahead was rolled back above; archive accepted rows only.
                reselect(true);
            }
        }
        prefill_collected_ = entry.is_prefill;
        return RAD_OK;
    } catch (const std::exception& e) { return failure("collect", e); }
}
void StatisticsObserver::sync_memory() {
    const uint32_t block = store_config_.block_tokens;
    std::vector<BlockDescriptor> added;
    const uint64_t old = memory_ ? memory_->stamp().frontier : 0;
    if (memory_ && old == frontier_) return;
    require(content_version_ != UINT64_MAX, "KV content version exhausted");
    ++content_version_;
    for (uint64_t pos = old / block * block; pos < frontier_; pos += block) {
        added.push_back({session_, pos / block, pos, uint32_t(std::min<uint64_t>(block, frontier_ - pos)),
                         content_version_, content_version_});
    }
    if (!memory_) {
        memory_ = std::make_shared<NativeMemory>(MemoryBinding{resources_.lifetime, resources_.kv,
            resources_.ctx->stream(), device_, 1}, groups_, identity_,
            MemoryStamp{true, session_, 1, 0, frontier_, 1}, added, block, max_position_, options_.host_bytes);
        memory_session_ = std::make_unique<MemorySession>(memory_);
    } else memory_->accept_append(frontier_, added);
}
void StatisticsObserver::reselect(bool decode_pressure) {
    sync_memory();
    const uint32_t block = store_config_.block_tokens;
    std::vector<uint32_t> mandatory{(frontier_ - 1) / block};
    std::vector<uint32_t> selected;
    const bool pressure = !query_frozen_;
    if (decode_pressure) {
        const auto snapshot = memory_->snapshot();
        std::vector<uint32_t> candidates;
        std::vector<double> scores;
        std::vector<double> query_scores;
        if (query_frozen_) {
            const std::vector<float> frozen(query_total_.begin(), query_total_.end());
            query_scores = mean_key_retrieval_scores(*store_, (frontier_ + block - 1) / block,
                {frozen.data(), query_counts_.data(), store_config_.n_layer,
                 head_dim_, query_heads_, kv_heads_});
        }
        const uint64_t end_block = (uint64_t(frontier_) + block - 1) / block;
        const auto recent_begin = end_block > keep_.recent_blocks ? end_block - keep_.recent_blocks : 0;
        for (auto id : snapshot.resident) {
            require(id < end_block && id <= UINT32_MAX, "invalid accepted resident block");
            candidates.push_back(uint32_t(id));
            // Upstream ranks nonnegative masses. Offset preserves the legacy
            // cosine order exactly; it is not an attention-probability claim.
            scores.push_back(query_frozen_ ? std::max(0.0, query_scores.at(id) + 1.0) : 0.0);
            if (id < keep_.sink_blocks || id >= recent_begin) mandatory.push_back(uint32_t(id));
        }
        selected = rolling::pressure_selection(candidates, scores, mandatory, options_.budget_tokens / block);
        for (auto id : selected)
            require(std::find(snapshot.resident.begin(), snapshot.resident.end(), id) != snapshot.resident.end(),
                    "DecodePressure must not recall cold KV");
    } else if (pressure) selected = policy_->pick_prefill_pressure_blocks(mandatory);
    else {
        const std::vector<float> frozen(query_total_.begin(), query_total_.end());
        policy_->set_retrieval_scores(mean_key_retrieval_scores(*store_, (frontier_ + block - 1) / block,
            {frozen.data(), query_counts_.data(), store_config_.n_layer,
             head_dim_, query_heads_, kv_heads_}));
        selected = policy_->pick_topk_blocks(mandatory);
    }
    // Tail is also in the recent band (and may be in sink for a small view).
    // The selector canonicalizes its own output, but MemorySession receives
    // this separate mandatory list and deliberately rejects duplicate IDs.
    rolling::canonicalize_mandatory(mandatory);
    const std::vector<uint64_t> chosen(selected.begin(), selected.end()), required(mandatory.begin(), mandatory.end());
    auto prefix_bytes = prefix_ ? prefix_->host_bytes() : 0;
    require(prefix_bytes <= options_.host_bytes, "prefix exceeds shared host budget");
    MemoryBudget budget{options_.budget_tokens, uint64_t(max_rows_) + block,
                        device_tokens_, options_.host_bytes - prefix_bytes};
    if (prefix_ && !prefix_bytes && memory_->needs_unleased_pages(chosen, required, budget)) {
        require(memory_->archive_bytes() <= options_.host_bytes, "archive exceeds host budget");
        prefix_->demote(device_, resources_.ctx->stream(), options_.host_bytes - memory_->archive_bytes());
        prefix_bytes = prefix_->host_bytes();
        budget.host_payload_bytes = options_.host_bytes - prefix_bytes;
        RAD_INFO("KVMEM_PREFIX_DEMOTE request=%llu frontier=%zu host_bytes=%zu pressure_only=1",
            (unsigned long long)active_request_, prefix_->frontier(), prefix_->host_bytes());
    }
    const auto& selection = memory_session_->prepare(memory_->stamp(), chosen, required, budget);
    const size_t spill = selection.spill.size(), restore = selection.restore.size();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (memory_session_->phase() != MemoryPhase::Ready) {
        require(std::chrono::steady_clock::now() < deadline, "native working-set transfer timed out");
        memory_session_->advance();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    const auto view = memory_session_->publish();
    ++reselections_; spilled_ += spill; restored_ += restore;
    selected_frontier_ = frontier_; selected_visible_ = view.visible_tokens;
    if (decode_pressure) {
        require(restore == 0, "DecodePressure unexpectedly restored cold KV");
        ++decode_reselections_;
        RAD_INFO("KVMEM_DECODE_PRESSURE round=%llu frontier=%u visible=%llu budget=%u reserve=%u spill=%zu restore=%zu scorer=legacy-mean-key accepted_only=1",
            (unsigned long long)decode_reselections_, frontier_, (unsigned long long)view.visible_tokens,
            options_.budget_tokens, options_.replay_tokens, spill, restore);
    }
    RAD_INFO("KVMEM_RESELECT session=%llu frontier=%u visible=%llu spill=%zu restore=%zu mode=%s original_positions=1",
             (unsigned long long)session_.id, frontier_, (unsigned long long)view.visible_tokens,
             spill, restore, decode_pressure ? "decode-pressure-mean-key" : (pressure ? "prefill-pressure" : "mean-key"));
}
void StatisticsObserver::retire(uint64_t id) noexcept {
    if (!store_ || active_request_ != id) return;
    try {
        std::vector<float> mean(store_config_.n_embd_k);
        double checksum = 0;
        bool complete = true;
        for (uint32_t layer = 0; layer < store_config_.n_layer; ++layer) {
            complete = complete && query_rows_.at(layer) == frontier_;
            for (uint32_t block = 0; block < (frontier_ + store_config_.block_tokens - 1) / store_config_.block_tokens; ++block) {
                store_->mean_k(block, layer, mean.data());
                for (float v : mean) { complete = complete && std::isfinite(v); checksum += std::abs(v); }
            }
        }
        RAD_INFO("KVMEM_STATISTICS request=%llu session=%llu incarnation=%llu tokens=%u steps=%llu layers=%u complete=%d mean_k_l1=%.9g bytes=%zu",
                 (unsigned long long)active_request_, (unsigned long long)session_.id, (unsigned long long)session_.incarnation, frontier_,
                 (unsigned long long)steps_, store_config_.n_layer, int(complete), checksum, store_->allocated_bytes());
        if (policy_) RAD_INFO("KVMEM_SELECTION_SUMMARY session=%llu rounds=%llu spill=%llu restore=%llu",
            (unsigned long long)session_.id, (unsigned long long)reselections_,
            (unsigned long long)spilled_, (unsigned long long)restored_);
        if (memory_) {
            const auto reuse = memory_->reuse_stats();
            RAD_INFO("KVMEM_PAGE_REUSE transactions=%llu pages=%llu journal_peak_bytes=%llu rollbacks=%llu",
                (unsigned long long)reuse.transactions, (unsigned long long)reuse.recycled_pages,
                (unsigned long long)reuse.journal_bytes_peak, (unsigned long long)reuse.rollbacks);
        }
    } catch (const std::exception& e) { failure("retire", e); }
    if (prefix_ && query_frozen_) {
        try {
            frontier_ = uint32_t(prefix_->frontier());
            store_->truncate_to(frontier_);
            store_->restore_mean_checkpoint(frontier_, prefix_mean_);
            policy_->truncate_to(frontier_);
            std::fill(query_rows_.begin(), query_rows_.end(), frontier_);
            checkpoint_.reset(); memory_session_.reset(); active_request_ = 0;
            return;
        } catch (const std::exception& e) { failure("retained prefix", e); }
    }
    clear_session();
}
void StatisticsObserver::shutdown() noexcept {
    if (capture_) (void)capture_->discard();
    clear_session(); resources_ = {}; capture_.reset();
}
} // namespace kvmem::radiance
