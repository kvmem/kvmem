#include "statistics_observer.hpp"
#include "pool_geometry.hpp"
#include "checkpoint_handoff.hpp"
#include "rolling_pressure.hpp"
#include "kvmem/mean_key_retrieval.hpp"
#include "mem/kv.h"
#include "build/startup.h"
#include "platform/module.h"
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
// Radiance's scheduler builds shared batches on rank 0's device. Adapter
// operations may visit every rank, but must return the caller to that device
// on success, capacity refusal, cancellation and exceptional cleanup alike.
struct SchedulerDeviceReturn {
    int device;
    ~SchedulerDeviceReturn(){if(device>=0 && rad_dev_set(device)<0)RAD_ERR("KVMem could not restore scheduler device %d",device);}
};
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
uint32_t positive(int64_t n) {
    require(n > 0 && n <= INT32_MAX, "statistics geometry outside supported range");
    return uint32_t(n);
}
int failure(const char* where, const std::exception& error) {
    RAD_ERR("KVMem statistics %s: %s", where, error.what());
    if(dynamic_cast<const StorageFailure*>(&error))return RAD_E_IO;
    if (const auto* memory=dynamic_cast<const MemoryError*>(&error);
        memory && memory->code()==MemoryErrorCode::BudgetExceeded) return RAD_E_FULL;
    return RAD_E_STATE;
}
}
int StatisticsObserver::pool_token_limit(const rad::Config& cfg, const rad::Program& p, int64_t* out) noexcept {
    try {
        *out = 0;
        if (!options_.budget_tokens) return RAD_OK;
        require((cfg.tp == 1 || cfg.tp == 2) && cfg.max_seqs == 1 && !cfg.prefix_cache && !cfg.kld(),
                "bounded residency requires TP1/TP2, one controller per sequence and no engine prefix cache");
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
        require(desc.program && (cfg.tp==1 || cfg.tp==2) && desc.world_size==cfg.tp &&
            desc.rank>=0 && size_t(desc.rank)==ranks_.size(),"statistics rank configuration is incomplete/out of order");
        auto rank=std::make_unique<Rank>();
        rank->device=desc.device;world_size_=uint32_t(cfg.tp);
        require(cfg.max_seqs == 1 && !cfg.prefix_cache && !cfg.kld(),
                "use max-num-seqs 1 and no-prefix-cache");
        const auto& p = *desc.program;
        require(p.meta.n_head>0 && p.meta.n_head_kv>0 && p.meta.n_head%cfg.tp==0 && p.meta.n_head_kv%cfg.tp==0,
            "TP statistics require evenly sharded query and KV heads");
        require(cfg.n_spec == 0 || p.drafter.kind == RAD_DRAFT_BLOCK || p.drafter.kind == RAD_DRAFT_SERIAL,
                "speculation bridge needs a native block or serial drafter");
        serial_drafter_ = cfg.n_spec > 0 && p.drafter.kind == RAD_DRAFT_SERIAL;
        require(cfg.draft_window!=rad::DraftWindow::Adaptive || serial_drafter_,
                "adaptive KVMem drafting requires an active native serial drafter");
        require(!p.pooled_buf && !p.audio_encoder.modalities,
                "KVMem supports text/image generation; pooled embeddings and audio are not supported");
        require(!p.encoder.modalities || rad::mm::decoder_available(),
                "image generation requires a Radiance build with RAD_WITH_FFMPEG=ON");
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
            const auto hd = positive(p.meta.head_dim);
            const auto packing = cfg.kv_cache_dtype == "fp8_v4" ? KvPacking::Fp8V4 :
                                 cfg.kv_cache_dtype == "k6_v4" ? KvPacking::K6V4 : KvPacking::Plain;
            require(2 * positive(g.decl.head_dim) == kv_row_elements(hd, packing) &&
                    (packing == KvPacking::Plain || (hd == 256 && g.decl.dtype == RAD_F8E4M3)),
                    "native KV row layout differs from configured packing");
            const auto kh = positive(g.decl.n_head_kv);
            require(hd == positive(p.meta.head_dim) && uint64_t(kh)*world_size_ == positive(p.meta.n_head_kv),
                    "per-group heads differ from the model metadata");
            for (int layer : g.layers) {
                // MTP shares these physical pages, but its Q/K do not score
                // target-model retrieval. groups_ below keeps ALL payload layers.
                if (layer < p.meta.n_layers)
                    layers.push_back({layer, rad_kvgroup(group), 0, hd, positive(p.meta.n_head/cfg.tp), kh, packing});
                else require(serial_drafter_, "unexpected non-trunk FULL KV layer");
            }
        }
        require(!layers.empty(), "no attention layers declared");
        std::sort(layers.begin(), layers.end(), [](const auto& a, const auto& b) { return a.layer < b.layer; });
        for (size_t i = 0; i < layers.size(); ++i) layers[i].ordinal = uint32_t(i);
        for(const auto& l:layers)rank->layers.push_back(l.layer);
        if(!ranks_.empty())require(rank->layers==ranks_[0]->layers &&
            layers[0].head_dim==head_dim_ && layers[0].query_heads*world_size_==query_heads_ &&
            layers[0].kv_heads*world_size_==kv_heads_,"rank statistics model geometry differs");
        const uint64_t kw = uint64_t(layers[0].head_dim) * layers[0].kv_heads * world_size_;
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
        if(ranks_.empty())device_ = desc.device;
        max_rows_ = positive(cfg.max_tok);
        head_dim_ = layers[0].head_dim; query_heads_ = layers[0].query_heads*world_size_; kv_heads_ = layers[0].kv_heads*world_size_;
        rank->query_heads=layers[0].query_heads;rank->groups = p.kv_groups;
        if (cfg.n_spec > 0) {
            for (const auto& op : p.ops) if (op.op == "gdn_recurrent_update")
                for (const auto& band : op.bands) {
                    const auto* row = band.dom[RAD_DOMAIN_DEVICE].row;
                    if (!row) continue;
                    const auto* plugin = rad::rad_tools_registry().plugin(row->plugin);
                    const auto fn = plugin ? reinterpret_cast<GdnMaterialize>(
                        rad::platform::module_symbol(plugin->handle, "rad_gdn_materialize_v1")) : nullptr;
                    require(fn && (!rank->materialize_gdn || fn == rank->materialize_gdn),
                            "selected recurrent producer lacks a compatible accepted-state materializer");
                    rank->materialize_gdn = fn;
                }
        }
        identity_ = {"radiance", KVMEM_RADIANCE_UPSTREAM_REVISION, cfg.model, "full-native-pages-v2:" + cfg.kv_cache_dtype + ":" + cfg.state_cache_dtype + "/tp"+std::to_string(world_size_)};
        rank->capture = std::make_unique<StatisticsCapture>(desc.device, layers, positive(cfg.max_tok));
        if (options_.budget_tokens) {
            rank->attention = std::make_unique<AttentionCapture>(desc.device, layers, max_rows_,
                (max_position_ + store_config_.block_tokens - 1) / store_config_.block_tokens);
            if(ranks_.empty())query_gpu_ = std::make_unique<PageSoftmaxCuda>(device_);
        }
        desc.kv_observer = [](void* data, int layer, rad_kvgroup group, int64_t hd,
            const RadTensor& q, const RadTensor& k, const RadBatch& batch, RadStream stream) {
            return StatisticsCapture::observe(static_cast<Rank*>(data)->capture.get(),
                layer, group, hd, q, k, batch, stream);
        };
        desc.attention_observer = [](void* data, int layer, rad_kvgroup group, int64_t hd,
            const RadTensor& q, const RadTensor& kv, const RadBatch& batch, RadStream stream) {
            auto& rank = *static_cast<Rank*>(data);
            return rank.attention ? rank.attention->observe(layer,group,hd,q,kv,batch,stream) : RAD_OK;
        };
        desc.kv_observer_data = rank.get();
        ranks_.push_back(std::move(rank));
        RAD_INFO("KVMem statistics: rank=%d/%u %zu attention layers, selection block %u, max rows %lld, selection budget %u",
                 desc.rank,world_size_,layers.size(), store_config_.block_tokens, (long long)cfg.max_tok, options_.budget_tokens);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("configure", e); }
}
int StatisticsObserver::bind_rank(rad::RankResources resources) noexcept {
    try {
        require(resources.lifetime && resources.ctx && resources.kv,
                "missing/duplicate native rank binding");
        const int index=resources.ctx->rank();
        require(index>=0 && size_t(index)<ranks_.size() && resources.ctx->world_size()==int(world_size_) &&
            !ranks_[index]->resources.lifetime,"native rank binding identity mismatch");
        auto& rank=*ranks_[index];rank.resources=std::move(resources);
        for (const auto& g : rank.resources.kv->plans()) {
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
        for (const auto& g : rank.resources.kv->plans()) carved += uint64_t(rank.resources.kv->pool_bytes_carved(g.index));
        RAD_INFO("KVMEM_POOL rank=%d resident_tokens=%llu carved_bytes=%llu logical_context=%u",
                 index,(unsigned long long)device_tokens_, (unsigned long long)carved, max_position_);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("bind", e); }
}
void StatisticsObserver::drain_ranks(const char* reason) {
    require(ranks_.size()==world_size_,"missing native rank");
    for(auto& rank:ranks_)require(rank->resources.ctx && rad_dev_set(rank->device)>=0 &&
        rad_stream_sync(rank->resources.ctx->stream())>=0,reason);
}
void StatisticsObserver::rollback_ranks(uint64_t sequence,uint64_t frontier,const char* reason) {
    require(ranks_.size()==world_size_,"missing native rank");
    for(auto& rank:ranks_)require(rank->resources.kv && rad_dev_set(rank->device)>=0 &&
        rank->resources.kv->rollback(sequence,frontier)>=0,reason);
}
void StatisticsObserver::verify_rank_view(uint64_t sequence) const {
    require(!ranks_.empty() && ranks_.size()==world_size_,"incomplete rank view");
    const auto& leader=*ranks_[0]->resources.kv;
    for(size_t i=1;i<ranks_.size();++i) {
        const auto& follower=*ranks_[i]->resources.kv;
        require(follower.seq_tokens(sequence)==leader.seq_tokens(sequence),"rank accepted frontiers differ");
        for(const auto& group:leader.plans()) {
            if(group.kind==RAD_KV_FULL)require(follower.block_table(sequence,group.index)==leader.block_table(sequence,group.index) &&
                follower.execution_offset(sequence,group.index)==leader.execution_offset(sequence,group.index),
                "rank physical page indices differ from the shared native batch");
            else if(group.stateful())require(follower.state_slot(sequence,group.index)==leader.state_slot(sequence,group.index),
                "rank recurrent slots differ from the shared native batch");
        }
    }
}
void StatisticsObserver::clear_session() {
    if(finishing_id_) {
        std::erase_if(cache_->finishing,[&](const auto& entry){return entry.id==finishing_id_;});
        finishing_id_=0;
    }
    pending_finish_.reset();
    pending_input_.reset(); input_parking_.reset(); input_.reset(); fallback_.reset();
    update_input_charge(0);
    draining_prefix_failure_=false;draining_selection_failure_=false;
    pending_cache_=0;
    pending_prefix_.reset();prefix_payload_.reset();
    pending_selection_.reset();
    std::atomic_store(&cached_identity_, std::shared_ptr<const CachedIdentity>{});
    prefix_.reset(); prefix_mean_.clear();
    checkpoint_.reset(); memory_session_.reset(); memory_.reset(); policy_.reset();
    update_archive_charge();
    store_.reset(); session_ = {}; active_request_ = 0; images_.clear();
    query_total_.clear(); query_rows_.clear(); query_counts_.clear(); query_tokens_.clear();
    recall_.reset(); checkpoint_recall_.reset(); prefix_recall_.reset();
    accepted_view_ = {};
}
uint64_t StatisticsObserver::CacheState::sharing_credit(const RetainedSession* extra) const {
    SharedPayloadAccounting out;
    const auto saved=[&](auto&& self,const RetainedSession& point)->void {
        if(point.memory)point.memory->account_archive(out);
        if(point.prefix)point.prefix->account(out);
        if(point.input)self(self,*point.input);
        if(point.fallback)self(self,*point.fallback);
    };
    for(const auto& point:sessions)saved(saved,point);
    // Parking workers mutate only their prefix payload. Its logical reservation
    // stays charged in full until the scheduler publishes the completed result.
    for(const auto& pending:parking)saved(saved,pending->saved);
    for(const auto* owner:observers) {
        if(owner->archive_charge_ && owner->memory_)owner->memory_->account_archive(out);
        if(owner->input_charge_) {
            if(owner->input_)saved(saved,*owner->input_);
            if(owner->fallback_)saved(saved,*owner->fallback_);
            if(owner->input_parking_)saved(saved,owner->input_parking_->saved);
        }
    }
    if(extra)saved(saved,*extra);
    return out.credit();
}
uint64_t StatisticsObserver::history_allowance(const RetainedSession* extra) const {
    const auto limit=options_.history_bytes(),credit=cache_->sharing_credit(extra);
    require(credit<=UINT64_MAX-limit,"shared history allowance overflow");
    return limit+credit;
}
void StatisticsObserver::update_archive_charge() {
    require(cache_->active_archive_bytes>=archive_charge_,"invalid active archive accounting");
    cache_->active_archive_bytes-=archive_charge_;
    archive_charge_=memory_ ? memory_->archive_bytes() : 0;
    cache_->active_archive_bytes+=archive_charge_;
    update_input_charge(input_extra_bytes());
}
void StatisticsObserver::update_input_charge(uint64_t bytes) {
    require(cache_->active_checkpoint_bytes>=input_charge_, "invalid input checkpoint accounting");
    cache_->active_checkpoint_bytes-=input_charge_;
    input_charge_=bytes;
    cache_->active_checkpoint_bytes+=input_charge_;
}
uint64_t StatisticsObserver::retained_bytes(const RetainedSession& saved) {
    require(saved.prefix && saved.memory && saved.store && saved.policy && saved.recall &&
        saved.recall_snapshot && saved.identity, "incomplete checkpoint accounting");
    return saved.prefix->page_bytes()+saved.prefix->state_bytes()+saved.memory->archive_bytes()+
        saved.store->allocated_bytes()+saved.policy->allocated_bytes()+saved.recall->allocated_bytes()+
        saved.recall_snapshot->graph.bytes()+saved.recall_snapshot->feedback.bytes()+
        saved.mean.size()*sizeof(float)+saved.identity->tokens.size()*sizeof(int32_t)+
        saved.identity->images.size()*sizeof(ImageIdentity);
}
uint64_t StatisticsObserver::input_extra_bytes() const {
    uint64_t total=0;
    for(const auto* saved:{input_.get(),fallback_.get(),input_parking_ ? &input_parking_->saved : nullptr}) {
        if(!saved)continue;
        const auto bytes=saved->bytes;
        require(bytes<=UINT64_MAX-total,"input checkpoint charge overflow");total+=bytes;
    }
    return total;
}
StatisticsObserver::RetainedSession* StatisticsObserver::select_checkpoint(RetainedSession& saved,
                                                                         const rad::Request& request, bool shared) {
    std::vector<PrefixCandidate> candidates;
    auto add=[&](RetainedSession& point,CheckpointPurpose purpose) {
        const auto& identity=*point.identity;
        candidates.push_back({&identity.tokens,&point,purpose,(shared ? identity.scope!=request.cache_owner : identity.scope==request.cache_owner) &&
            image_prefix_matches(identity.images,request.media.get(),request.prompt.size(),identity.tokens.size())});
    };
    add(saved,CheckpointPurpose::Endpoint);
    if(saved.input)add(*saved.input,CheckpointPurpose::ReuseAnchor);
    if(saved.fallback)add(*saved.fallback,CheckpointPurpose::InputFallback);
    return const_cast<RetainedSession*>(static_cast<const RetainedSession*>(select_prefix_checkpoint(
        request.prompt,candidates,request.query_begin<0 ? UINT64_MAX : uint64_t(request.query_begin))));
}
void StatisticsObserver::retain_input(const rad::StepEntry& entry) {
    if(!policy_ || !options_.cache_sessions || !entry.is_prefill || !entry.req ||
       pending_input_ || input_parking_) return;
    const bool fallback=entry.req->history_fallback_end>0 && frontier_==uint64_t(entry.req->history_fallback_end) &&
        entry.req->history_fallback_end!=entry.req->history_reuse_end;
    const auto target=fallback ? entry.req->history_fallback_end : entry.req->history_reuse_end;
    if(target<=0 || (fallback ? bool(fallback_) : bool(input_)))return;
    const auto end=uint64_t(target);
    if(frontier_!=end || !rolling::retainable_prefix_boundary(frontier_,end,store_config_.block_tokens))return;
    for(const auto& image:images_)if(uint64_t(image.begin)<end && end<uint64_t(image.end)) {
        RAD_INFO("KVMEM_INPUT_SKIP session=%llu reason=image_boundary",(unsigned long long)session_.id);return;
    }
    // A probe is not committed history. Tool continuation has no Query, and
    // must not inherit an old one merely to obtain an input checkpoint.
    if(query_begin_>=0 && !replay_done_ && !replaying_ && entry.req->prompt.size()>options_.budget_tokens &&
       !(fallback && frontier_<=uint64_t(query_begin_)))return;
    auto identity=std::make_shared<CachedIdentity>();identity->fallback=fallback;
    identity->tokens.assign(entry.req->prompt.begin(),entry.req->prompt.begin()+end);
    identity->images=image_prefix(images_,end);identity->scope=entry.req->cache_owner;
    pending_input_=std::move(identity);
    (void)poll_input();
}
void StatisticsObserver::capture_input(std::shared_ptr<const CachedIdentity> identity) {
    SchedulerDeviceReturn device_return{device_};
    drain_ranks("drain input checkpoint producers");
    require(identity && identity->tokens.size()==frontier_ && !(identity->fallback ? bool(fallback_) : bool(input_)) && !input_parking_ && !pending_selection_,
        "input checkpoint lost its drained boundary");
    sync_memory();
    require(next_incarnation_!=UINT64_MAX,"input checkpoint incarnation exhausted");
    auto saved=std::make_unique<RetainedSession>();
    saved->session={session_.id,next_incarnation_++};saved->content_version=content_version_;saved->identity=std::move(identity);
    saved->prefix=std::make_unique<RankPrefix>();
    for(auto& rank:ranks_) {
        auto lease=std::make_unique<PrefixLease>(rank->resources.lifetime,*rank->resources.kv,active_request_,
            saved->identity->tokens,rank->device,rank->resources.ctx->stream());
        saved->prefix->add_rank(rank->device,rank->resources.ctx->stream(),*rank->resources.kv,std::move(lease));
    }
    saved->mean=store_->mean_checkpoint(frontier_);
    saved->store=store_->fork_mean_prefix(frontier_);
    saved->policy=std::make_unique<KvMemStore>(*policy_);
    saved->recall=std::make_unique<RecallPolicy>(options_,store_config_.block_tokens,graph_budget_);
    saved->recall_snapshot=std::make_unique<RecallSnapshot>(recall_->checkpoint());
    saved->recall->restore(*saved->recall_snapshot);
    memory_->retain_view();saved->memory=memory_->fork_retained(saved->session);
    saved->bytes=retained_bytes(*saved);
    const auto extra=saved->bytes;
    const auto active=cache_->active_archive_bytes+cache_->active_checkpoint_bytes;
    if(active>history_allowance(saved.get()) || extra>history_allowance(saved.get())-active) {
        RAD_INFO("KVMEM_INPUT_SKIP session=%llu reason=host_budget",(unsigned long long)session_.id);return;
    }
    auto fits=[&]{const auto limit=history_allowance(saved.get());return active<=limit && extra<=limit-active && cache_->bytes<=limit-active-extra;};
    while(!cache_->sessions.empty() && !fits())evict_session();
    if(!fits()) {
        RAD_INFO("KVMEM_INPUT_SKIP session=%llu reason=pending_cache_capacity",(unsigned long long)session_.id);return;
    }
    // No extra persistent GPU view: copy once, release page references before
    // the next append/Pressure, then let the bounded storage worker seal it.
    saved->prefix->demote(saved->prefix->page_bytes());
    if(asynchronous_ && cache_->payload && cache_->io) {
        input_parking_=std::make_unique<PendingPark>();
        input_parking_->work=std::make_shared<ParkWork>();input_parking_->work->storage=cache_->payload;
        input_parking_->work->prefix=std::move(saved->prefix);
        input_parking_->saved=std::move(*saved);
    } else {
        if(cache_->payload)saved->prefix->park(cache_->payload,cache_->io);
        (saved->identity->fallback ? fallback_ : input_)=std::move(saved);
    }
    update_input_charge(input_extra_bytes());
    RAD_INFO("KVMEM_INPUT_RETAIN request=%llu frontier=%u query=%d state_and_pages=1 extra_bytes=%llu",
        (unsigned long long)active_request_,frontier_,query_begin_>=0,(unsigned long long)extra);
}
bool StatisticsObserver::poll_input() {
    if(pending_input_) {
        if(pending_selection_)return false;
        auto identity=std::move(pending_input_);
        try {capture_input(std::move(identity));}
        catch(const MemoryError& error) {
            if(error.code()!=MemoryErrorCode::BudgetExceeded)throw;
            RAD_INFO("KVMEM_INPUT_SKIP session=%llu reason=storage_budget",(unsigned long long)session_.id);
        } catch(const std::length_error&) {
            RAD_INFO("KVMEM_INPUT_SKIP session=%llu reason=metadata_budget",(unsigned long long)session_.id);
        } catch(const std::bad_alloc&) {
            RAD_INFO("KVMEM_INPUT_SKIP session=%llu reason=allocation",(unsigned long long)session_.id);
        }
    }
    if(!input_parking_)return true;
    auto& pending=*input_parking_;
    try {
        if(!pending.ticket) {
            const auto work=pending.work;
            pending.ticket=cache_->io->submit(work->prefix->host_bytes()+work->prefix->state_bytes(),[work]{work->run();});
        }
        if(!pending.ticket.ready())return false;
        pending.ticket.wait();pending.saved.prefix=std::move(pending.work->prefix);
        const auto ready_frontier=pending.saved.identity->tokens.size();
        auto ready=std::make_unique<RetainedSession>(std::move(pending.saved));
        publish_retained_checkpoint(std::move(ready),input_,fallback_);
        input_parking_.reset();
        RAD_INFO("KVMEM_INPUT_READY request=%llu frontier=%zu",(unsigned long long)active_request_,ready_frontier);
    } catch(const StorageFailure& error) {
        input_parking_.reset();update_input_charge(input_extra_bytes());
        RAD_ERR("KVMem optional input parking dropped: %s",error.what());
    } catch(const MemoryError& error) {
        if(error.code()==MemoryErrorCode::Pending)return false;
        if(error.code()!=MemoryErrorCode::BudgetExceeded)throw;
        input_parking_.reset();update_input_charge(input_extra_bytes());
        RAD_INFO("KVMEM_INPUT_SKIP session=%llu reason=io_budget",(unsigned long long)session_.id);
    } catch(const std::bad_alloc&) {
        input_parking_.reset();update_input_charge(input_extra_bytes());
        RAD_INFO("KVMEM_INPUT_SKIP session=%llu reason=io_allocation",(unsigned long long)session_.id);
    }
    return true;
}
void StatisticsObserver::promote_input() {
    require(input_ && !input_parking_ && !pending_selection_,"input fallback is incomplete");
    auto saved=std::move(input_);memory_session_.reset();
    session_=saved->session;content_version_=saved->content_version;
    prefix_=std::move(saved->prefix);prefix_mean_=std::move(saved->mean);
    store_=std::move(saved->store);policy_=std::move(saved->policy);memory_=std::move(saved->memory);
    recall_=std::move(saved->recall);prefix_recall_=std::move(saved->recall_snapshot);
    std::atomic_store(&cached_identity_,saved->identity);update_archive_charge();
}
std::unique_ptr<StatisticsObserver::RetainedSession> StatisticsObserver::carry_input(
        const RetainedSession& saved,const RetainedSession& selected,const rad::Request& request,bool fallback) {
    // A retry or short append can restore AT/AFTER its next input boundary.
    // There will be no forward callback there to recapture it. Keep the proven
    // earlier state, otherwise a second retry loses reuse when cache_sessions=1.
    if(request.history_end>0 && uint64_t(request.history_end)>selected.identity->tokens.size())return {};
    if(fallback && !saved.fallback)return {};
    const auto* point=fallback ? saved.fallback.get() : saved.input ? saved.input.get() : &selected;
    if(!point->identity->prefix_matches(request) ||
       (request.query_begin>=0 && uint64_t(request.query_begin)<point->identity->tokens.size()))return {};
    require(next_incarnation_!=UINT64_MAX,"carried checkpoint incarnation exhausted");
    auto next=std::make_unique<RetainedSession>();
    try {
        next->session={request.id,next_incarnation_++};next->content_version=point->content_version;
        auto identity=std::make_shared<CachedIdentity>(*point->identity);identity->scope=request.cache_owner;
        next->identity=std::move(identity);next->prefix=point->prefix->fork();next->mean=point->mean;
        next->store=point->store->fork_mean_prefix(uint32_t(point->identity->tokens.size()));
        next->policy=std::make_unique<KvMemStore>(*point->policy);
        next->memory=point->memory->fork_retained(next->session);
        next->recall=std::make_unique<RecallPolicy>(options_,store_config_.block_tokens,graph_budget_);
        next->recall_snapshot=std::make_unique<RecallSnapshot>(*point->recall_snapshot);
        next->recall->restore(*next->recall_snapshot);next->bytes=retained_bytes(*next);
        return next;
    } catch(const MemoryError& error) {if(error.code()!=MemoryErrorCode::BudgetExceeded)throw;}
      catch(const std::length_error&) {} catch(const std::bad_alloc&) {}
    return {};
}
void StatisticsObserver::admit_carried_input(std::unique_ptr<RetainedSession> saved) {
    if(!saved)return;
    require(memory_ && !(saved->identity->fallback ? bool(fallback_) : bool(input_)) && !input_parking_,"carried checkpoint needs an active owner");
    const auto extra=saved->bytes;
    const auto active=cache_->active_archive_bytes+cache_->active_checkpoint_bytes;
    auto fits=[&]{const auto limit=history_allowance(saved.get());return active<=limit && extra<=limit-active && cache_->bytes<=limit-active-extra;};
    while(!cache_->sessions.empty() && !fits())evict_session();
    if(!fits())return;
    const auto session=saved->session.id,frontier=saved->identity->tokens.size();
    (saved->identity->fallback ? fallback_ : input_)=std::move(saved);update_input_charge(input_extra_bytes());
    RAD_INFO("KVMEM_INPUT_CARRY request=%llu frontier=%zu immutable_payload_shared=1",
        (unsigned long long)session,frontier);
}
void StatisticsObserver::evict_session() {
    require(!cache_->sessions.empty() && cache_->bytes >= cache_->sessions.front().bytes, "invalid cache accounting");
    RAD_INFO("KVMEM_SESSION_EVICT session=%llu bytes=%llu policy=lru",
        (unsigned long long)cache_->sessions.front().session.id, (unsigned long long)cache_->sessions.front().bytes);
    cache_->bytes -= cache_->sessions.front().bytes;
    cache_->sessions.pop_front();
    ++cache_->evictions;
}
void StatisticsObserver::park_session() {
    if (!prefix_ || !prefix_recall_ || !options_.cache_sessions) { clear_session(); return; }
    const auto identity = std::atomic_load(&cached_identity_);
    require(identity && memory_ && store_ && policy_ && recall_, "incomplete session cache entry");
    // Count inactive KV, recurrent state and statistical metadata against one
    // retained-host allowance. Model/runtime workspace is not a cache payload.
    const uint64_t fixed = input_extra_bytes() + prefix_->state_bytes() + memory_->archive_bytes() + store_->allocated_bytes() +
        policy_->allocated_bytes() + recall_->allocated_bytes() + prefix_recall_->graph.bytes() +
        prefix_recall_->feedback.bytes() + prefix_mean_.size()*sizeof(float) + identity->tokens.size()*sizeof(int32_t) +
        identity->images.size()*sizeof(ImageIdentity);
    const uint64_t pages = prefix_->page_bytes();
    const uint64_t other=cache_->active_archive_bytes-archive_charge_+cache_->active_checkpoint_bytes-input_charge_;
    require(other<=history_allowance(),"other active archives exceed host allowance");
    auto available=[&]{const auto limit=history_allowance();return limit>other ? limit-other : 0;};
    if (fixed > available() || pages > available() - fixed) {
        RAD_INFO("KVMEM_SESSION_SKIP session=%llu reason=host_budget required_bytes=%llu limit=%llu",
            (unsigned long long)session_.id, (unsigned long long)(fixed+pages), (unsigned long long)options_.history_bytes());
        clear_session(); return;
    }
    const uint64_t bytes = fixed + pages;
    auto fits=[&]{return bytes<=available() && cache_->bytes<=available()-bytes;};
    while (!cache_->sessions.empty() && (cache_->count() >= options_.cache_sessions || !fits()))
        evict_session();
    if(cache_->count()>=options_.cache_sessions || !fits() || cache_->next_park==UINT64_MAX) {
        RAD_INFO("KVMEM_SESSION_SKIP session=%llu reason=pending_cache_capacity",(unsigned long long)session_.id);
        clear_session();return;
    }
    // An inactive conversation never pins GPU pages needed by another one.
    prefix_->demote(pages);
    std::unique_ptr<PendingPark> parking;
    if(asynchronous_ && cache_->payload && cache_->io) {
        parking=std::make_unique<PendingPark>();parking->id=finishing_id_ ? finishing_id_ : cache_->next_park++;
        parking->work=std::make_shared<ParkWork>();parking->work->storage=cache_->payload;
        const uint64_t charge=prefix_->host_bytes()+prefix_->state_bytes();
        parking->work->prefix=std::move(prefix_);
        try {
            const auto work=parking->work;parking->ticket=cache_->io->submit(charge,[work]{work->run();});
        } catch(const MemoryError& error) {
            if(error.code()!=MemoryErrorCode::Pending && error.code()!=MemoryErrorCode::BudgetExceeded)throw;
            RAD_INFO("KVMEM_SESSION_SKIP session=%llu reason=io_queue_capacity",(unsigned long long)session_.id);
            clear_session();return;
        }
    } else if(cache_->payload)prefix_->park(cache_->payload,cache_->io);
    RetainedSession saved;
    saved.session=session_; saved.content_version=content_version_; saved.bytes=bytes; saved.identity=identity;
    saved.prefix=std::move(prefix_); saved.mean=std::move(prefix_mean_); saved.store=std::move(store_);
    saved.policy=std::move(policy_); saved.memory=std::move(memory_); saved.recall=std::move(recall_);
    saved.input=std::move(input_);saved.fallback=std::move(fallback_);
    update_archive_charge();
    saved.recall_snapshot=std::move(prefix_recall_);
    if(parking) {
        parking->saved=std::move(saved);cache_->parking.push_back(std::move(parking));
    } else cache_->sessions.push_back(std::move(saved));
    cache_->bytes+=bytes;
    RAD_INFO("KVMEM_SESSION_CACHE entries=%zu logical_bytes=%llu limit=%llu gpu_pages=0 pending=%zu",
        cache_->sessions.size(),(unsigned long long)cache_->bytes,(unsigned long long)options_.history_bytes(),cache_->parking.size());
    clear_session();
}
void StatisticsObserver::activate_session(const rad::Request& request) {
    require(!active_request_ && !store_, "cache activation requires an idle observer");
    auto best=cache_->sessions.end();
    RetainedSession* point=nullptr;
    for(unsigned pass=0;pass<2 && !point && !pending_cache_;++pass) {
    const bool shared=pass!=0;
    if(shared && !(request.query_begin>=0 && request.query_end>request.query_begin))break;
    for (auto it=cache_->sessions.begin(); it!=cache_->sessions.end(); ++it) {
        auto* candidate=select_checkpoint(*it,request,shared);
        if(candidate && (!point || candidate->identity->tokens.size()>point->identity->tokens.size())) {
            best=it;point=candidate;
        }
    }
    size_t longest=point ? point->identity->tokens.size() : 0;
    for(const auto& pending:cache_->parking) {
        auto* candidate=select_checkpoint(pending->saved,request,shared);
        if(candidate && candidate->identity->tokens.size()>longest) {
            pending_cache_=pending->id;longest=candidate->identity->tokens.size();
        }
    }
    for(const auto& pending:cache_->finishing) {
        for(const auto& identity:{pending.identity,pending.input_identity,pending.fallback_identity}) {
            if(identity && identity->tokens.size()>longest &&
               (shared ? identity->scope!=request.cache_owner : identity->scope==request.cache_owner) && identity->prefix_matches(request) &&
               (request.query_begin<0 || uint64_t(request.query_begin)>=identity->tokens.size())) {
                pending_cache_=pending.id;longest=identity->tokens.size();
            }
        }
    }
    }
    if(pending_cache_)return;
    if (best==cache_->sessions.end()) return;
    const auto& previous=best->identity->tokens;
    const auto n=std::min(previous.size(),request.prompt.size());
    const auto lcp=std::mismatch(previous.begin(),previous.begin()+n,request.prompt.begin()).first-previous.begin();
    RAD_INFO("KVMEM_CHECKPOINT_SELECT request=%llu purpose=%s frontier=%zu lcp=%zu",
        (unsigned long long)request.id,point==&*best ? "terminal" : "input",point->identity->tokens.size(),size_t(lcp));
    auto carried=carry_input(*best,*point,request);
    auto carried_fallback=carry_input(*best,*point,request,true);
    const auto active=cache_->active_archive_bytes+cache_->active_checkpoint_bytes;
    {
        RetainedSession branch;
        try {
            branch.session={request.id,next_incarnation_};branch.content_version=point->content_version;
            branch.prefix=point->prefix->fork();branch.mean=point->mean;
            auto identity=std::make_shared<CachedIdentity>(*point->identity);identity->scope=request.cache_owner;
            branch.identity=std::move(identity);
            branch.store=point->store->fork_mean_prefix(uint32_t(point->prefix->frontier()));
            branch.policy=std::make_unique<KvMemStore>(*point->policy);
            branch.memory=point->memory->fork_retained(branch.session);
            branch.recall=std::make_unique<RecallPolicy>(options_,options_.block_tokens,graph_budget_);
            branch.recall->restore(*point->recall_snapshot);
            branch.recall_snapshot=std::make_unique<RecallSnapshot>(*point->recall_snapshot);
            branch.bytes=retained_bytes(branch);
            const auto total=history_allowance(&branch);
            if(active>total || cache_->bytes>total-active || branch.bytes>total-active-cache_->bytes)
                branch={};
        } catch(const MemoryError& error) {
            if(error.code()!=MemoryErrorCode::BudgetExceeded)throw;
            branch={}; // release the incomplete fork before the move fallback
        } catch(const std::length_error&) {
            branch={}; // bounded graph metadata cannot admit another copy
        } catch(const std::bad_alloc&) {
            branch={}; // this optional optimization never requires a new copy
        }
        if(branch.memory) {
            ++next_incarnation_;++cache_->hits;++cache_->forks;
            // A private fork also protects an own continuation; only a
            // different or anonymous owner is cross-session reuse.
            if(request.cache_owner.empty() || point->identity->scope!=request.cache_owner)++cache_->cross_owner_forks;
            RAD_INFO("KVMEM_SESSION_FORK source=%llu branch=%llu frontier=%zu immutable_payload_shared=1",
                (unsigned long long)point->session.id,(unsigned long long)branch.session.id,point->prefix->frontier());
            session_=branch.session;content_version_=branch.content_version;
            prefix_=std::move(branch.prefix);prefix_mean_=std::move(branch.mean);store_=std::move(branch.store);
            policy_=std::move(branch.policy);memory_=std::move(branch.memory);recall_=std::move(branch.recall);
            prefix_recall_=std::move(branch.recall_snapshot);update_archive_charge();
            std::atomic_store(&cached_identity_,branch.identity);
            std::rotate(best,std::next(best),cache_->sessions.end());
            admit_carried_input(std::move(carried));admit_carried_input(std::move(carried_fallback));
            return;
        }
    }
    if(point->identity->scope!=request.cache_owner || request.cache_owner.empty()) {
        RAD_INFO("KVMEM_SESSION_SKIP request=%llu reason=fork_capacity source_preserved=1",(unsigned long long)request.id);
        return;
    }
    ++cache_->hits;
    require(cache_->bytes>=best->bytes,"invalid cache hit accounting");
    cache_->bytes-=best->bytes;
    session_=point->session; content_version_=point->content_version;
    prefix_=std::move(point->prefix); prefix_mean_=std::move(point->mean); store_=std::move(point->store);
    policy_=std::move(point->policy); memory_=std::move(point->memory); recall_=std::move(point->recall);
    update_archive_charge();
    prefix_recall_=std::move(point->recall_snapshot);
    std::atomic_store(&cached_identity_,point->identity);
    cache_->sessions.erase(best);
    admit_carried_input(std::move(carried));admit_carried_input(std::move(carried_fallback));
}
int StatisticsObserver::resume(const rad::StepPlan& plan, int64_t* resumed, std::string* refusal) noexcept {
    SchedulerDeviceReturn device_return{device_};
    *resumed = -1; refusal->clear();
    try {
        if (plan.e.size() != 1) return RAD_OK;
        const auto& e = plan.e[0];
        if (!e.is_prefill || e.ctx_len != 0) return RAD_OK;
        if (active_request_ == e.seq) return RAD_OK; // same request's unsampled replay
        require(e.req && !e.dev && !e.n_spec && !active_request_, "invalid fresh request boundary");
        if (!store_) activate_session(*e.req);
        if(pending_cache_) {
            drain_ranks("drain new state before cache wait");
            rollback_ranks(e.seq,0,"release unissued cache-wait reservation");
            *resumed=0;return RAD_OK;
        }
        const auto identity = std::atomic_load(&cached_identity_);
        const auto next_query = e.req->query_begin >= 0 ? bounded_query(e.req->query_begin,
            e.req->query_end-e.req->query_begin,e.req->prompt.size(),options_.query_tokens) : QueryRange{};
        if (prefix_ && identity && identity->matches(*e.req) &&
            (next_query.empty() || next_query.begin >= prefix_->frontier()) && fresh_reserve_error(*e.req).empty()) {
            require(store_ && memory_ && recall_ && prefix_recall_, "incomplete retained session");
            drain_ranks("drain new state clears before resume");
            // The current plan reserved, but has NOT evaluated, its first batch.
            // Discard that reservation; the engine rebuilds the plan after restore.
            rollback_ranks(e.seq,0,"release unissued initial reservation");
            if(asynchronous_ && cache_->io && !prefix_payload_) {
                if(!pending_prefix_) {
                    auto read=std::make_unique<PrefixRead>();read->work=std::make_shared<PrefixReadWork>();
                    read->work->source=prefix_->fork();read->deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                    pending_prefix_=std::move(read);
                }
                if(!poll_prefix()){*resumed=0;return RAD_OK;} // retire this unissued plan and wait
            }
            prefix_->restore(e.seq,e.req->prompt,prefix_payload_.get());
            verify_rank_view(e.seq);
            prefix_payload_.reset();
            memory_->rebind_sequence(e.seq, prefix_->host_bytes() != 0);
            frontier_ = uint32_t(prefix_->frontier());
            prefix_.reset(); prefix_mean_.clear();
            std::atomic_store(&cached_identity_, std::shared_ptr<const CachedIdentity>{});
            active_request_ = e.seq;
            memory_session_ = std::make_unique<MemorySession>(memory_);
            recall_->restore(*prefix_recall_); prefix_recall_.reset();
            start_request(*e.req);
            steps_ = reselections_ = spilled_ = restored_ = 0;
            selected_frontier_ = frontier_; selected_visible_ = 0; decode_reselections_ = 0;
            const auto resumed_view = memory_->snapshot();
            accepted_view_ = resumed_view;
            for (const auto& item : resumed_view.blocks)
                if (std::find(resumed_view.resident.begin(), resumed_view.resident.end(), item.block.id) != resumed_view.resident.end())
                    selected_visible_ += item.block.valid_tokens;
            // A proven new user Query is captured afresh. Ordinary continuation
            // can only reclaim its accepted resident working set.
            if (rolling::needs_pressure(selected_visible_, options_.budget_tokens, options_.replay_tokens,
                                        max_rows_, store_config_.block_tokens))
                reselect(SelectionReason::PrefillPressure);
            *resumed = frontier_;
            RAD_INFO("KVMEM_PREFIX_RESUME request=%llu session=%llu frontier=%u exact_prefix=1",
                (unsigned long long)e.seq, (unsigned long long)session_.id, frontier_);
            return RAD_OK;
        }
        clear_session();
        *refusal = fresh_reserve_error(*e.req);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("resume", e); }
}
void StatisticsObserver::start_request(const rad::Request& request) {
    RAD_INFO("KVMEM_HISTORY_BOUNDARY request=%llu prompt=%zu history_end=%d",
        (unsigned long long)request.id,request.prompt.size(),request.history_end);
    images_=image_identity(request.media.get(),request.prompt.size());
    image_budget_check(images_,store_config_.block_tokens,options_.budget_tokens,keep_.sink_blocks,keep_.recent_blocks);
    require((request.query_begin == -1 && request.query_end == -1) ||
        (request.query_begin >= 0 && request.query_end > request.query_begin && size_t(request.query_end) <= request.prompt.size()),
        "invalid frontend query span");
    const auto q = request.query_begin >= 0 ? bounded_query(request.query_begin,
        request.query_end-request.query_begin,request.prompt.size(),options_.query_tokens) : QueryRange{};
    query_begin_ = q.empty() ? -1 : int32_t(q.begin); query_end_ = q.empty() ? -1 : int32_t(q.end);
    query_total_.clear();query_counts_.clear();query_tokens_.clear();
    query_rows_.assign(store_config_.n_layer,frontier_);
    query_frozen_=false;replaying_=false;replay_done_=q.empty();
    checkpoint_.reset();checkpoint_recall_.reset();replay_start_=0;
    recall_->start_request(!q.empty(),request.sp.reasoning.starts_open);
    std::vector<GraphMessageSpan> messages;
    uint64_t end = 0;
    for (const auto& span : request.message_spans) {
        require(span.first >= 0 && uint64_t(span.first) == end && span.second > span.first &&
            size_t(span.second) <= request.prompt.size(), "invalid proven message boundary");
        messages.push_back({uint64_t(span.first),uint64_t(span.second),true});end=span.second;
    }
    if (!messages.empty()) messages.push_back({end,uint64_t(request.prompt.size()),false});
    recall_->set_messages(messages);
    RAD_INFO("KVMEM_QUERY_SPAN session=%llu begin=%d end=%d proven=%d limit=%u",
        (unsigned long long)session_.id,query_begin_,query_end_,int(!q.empty()),options_.query_tokens);
    if(!images_.empty())RAD_INFO("KVMEM_IMAGE_LAYOUT session=%llu images=%zu required_pages=%zu content_identity=128bit native_rope=1",
        (unsigned long long)session_.id,images_.size(),image_required(images_,request.prompt.size(),store_config_.block_tokens).size());
}
PageQueryView StatisticsObserver::query_view(std::vector<const float*>& pointers) const {
    require(query_frozen_ && query_tokens_.size()==store_config_.n_layer,"missing frozen Query tokens");
    for (size_t i=0;i<query_tokens_.size();++i) {
        require(query_tokens_[i].size()==uint64_t(query_counts_[i])*query_heads_*head_dim_,"incomplete bounded Query rows");
        pointers.push_back(query_tokens_[i].data());
    }
    return {nullptr,query_counts_.data(),store_config_.n_layer,uint32_t(query_end_-query_begin_),
        query_heads_,kv_heads_,head_dim_,pointers.data()};
}
void StatisticsObserver::begin_attention(const rad::StepEntry& entry) {
    const auto block=store_config_.block_tokens;
    const auto pages=attention_pages(memory_ ? &accepted_view_ : nullptr,frontier_,uint32_t(entry.n_tokens),block);
    const uint32_t compact=pages.back().compact+pages.back().count;
    exposed_.clear();for(const auto& page:pages)exposed_.push_back(page.start);
    step_tokens_.clear();
    if (entry.is_prefill) sample_rows_=prefill_sample_rows(frontier_,entry.n_tokens,
        recall_->feedback().prefill_end,compact-entry.n_tokens,
        uint64_t(options_.budget_tokens)+options_.replay_tokens,1024,32,block);
    else {
        require(entry.req && frontier_>=entry.req->prompt.size(),"decode token history missing");
        const auto output_index=frontier_-entry.req->prompt.size();
        require(output_index<entry.req->output.size(),"target has not consumed the emitted tail");
        step_tokens_.push_back(entry.req->output[output_index]);
        require(!entry.n_spec || (entry.draft && entry.n_draft>=entry.n_spec),"verify proposal tokens missing");
        for(int i=0;i<entry.n_spec;++i)step_tokens_.push_back(entry.draft[i]);
        sample_rows_=accepted_output_samples(recall_->output(),step_tokens_.data(),entry.n_tokens,
            entry.req->sp.reasoning.open,entry.req->sp.reasoning.close).rows;
    }
    for(auto& rank:ranks_)rank->attention->begin(frontier_,sample_rows_,pages,rank->resources.ctx->stream());
}
int StatisticsObserver::begin(const rad::StepPlan& plan) noexcept {
    SchedulerDeviceReturn device_return{device_};
    try {
        require(ranks_.size()==world_size_ && !ranks_.empty() && plan.e.size() == 1 && plan.draft_pass == 0,
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
            recall_ = std::make_unique<RecallPolicy>(options_,store_config_.block_tokens,graph_budget_);
            start_request(*e.req);
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
            frontier_ <= uint32_t(query_begin_) &&
            (rolling::query_checkpoint_due(frontier_, uint64_t(e.n_tokens), uint32_t(query_begin_), store_config_.block_tokens) ||
             frontier_ / store_config_.block_tokens == uint32_t(query_begin_) / store_config_.block_tokens)) {
            require(fresh_reserve_error(*e.req).empty(), "output and verify step exceed the generation reserve");
            checkpoint_ = std::make_unique<RankCheckpoint>();
            for(auto& rank:ranks_)checkpoint_->save_rank(*rank->resources.kv,active_request_,rank->device,rank->resources.ctx->stream());
            checkpoint_mean_ = store_->mean_checkpoint(frontier_);
            replay_start_ = frontier_;
            checkpoint_recall_ = std::make_unique<RecallSnapshot>(recall_->checkpoint());
            RAD_INFO("KVMEM_QUERY_CHECKPOINT session=%llu frontier=%u state_bytes=%zu",
                (unsigned long long)session_.id, replay_start_, checkpoint_->bytes());
        }
        prefill_collected_ = deferred_selection_ = false;
        for(auto& rank:ranks_)rank->capture->begin(session_, frontier_, uint32_t(e.n_tokens),
            query_begin_ < 0 || query_frozen_ ? UINT32_MAX : uint32_t(query_begin_),
            query_end_ < 0 || query_frozen_ ? UINT32_MAX : uint32_t(query_end_));
        if (options_.budget_tokens) begin_attention(e);
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
    try {
        const auto images=image_identity(request.media.get(),request.prompt.size());
        image_budget_check(images,store_config_.block_tokens,options_.budget_tokens,keep_.sink_blocks,keep_.recent_blocks);
    } catch(const std::exception& e){return e.what();}
    return fresh_reserve_error(request);
}
int StatisticsObserver::after_forward(const rad::StepPlan& plan, int64_t* rewind) noexcept {
    SchedulerDeviceReturn device_return{device_};
    *rewind = -1;
    rad::Scheduler::StepResult accepted{};
    accepted.n_seq = int(plan.e.size()); // split path is prefill-only, with no draft rows
    const int rc = collect(plan, accepted);
    if (rc < 0) return rc;
    try {
        // Capture completion only fences Q/K copies, not later GDN/output ops.
        drain_ranks("drain model before replay/sampling");
        const auto& e = plan.e[0];
        if (!policy_ || !e.is_prefill) return RAD_OK;
        if (!e.produces_token) {
            // A small untrimmed input needs no probe replay. Longer inputs
            // checkpoint only during the corrected replay, never the probe.
            if (!replay_done_ && !replaying_ && e.req->prompt.size() <= options_.budget_tokens)
                replay_done_ = true;
            if(!serial_drafter_)retain_input(e);
            return RAD_OK;
        }
        if (!query_frozen_) {
            if(!serial_drafter_)retain_input(e);
            return RAD_OK;
        }
        if (replaying_) {
            replaying_ = false; replay_done_ = true;
            RAD_INFO("KVMEM_QUERY_REPLAY_DONE session=%llu begin=%u end=%u before_sampler=1",
                (unsigned long long)session_.id, replay_start_, frontier_);
            checkpoint_.reset();
            checkpoint_recall_.reset();
        } else if (!replay_done_ && frontier_ > options_.budget_tokens) {
            const auto admission = query_replay_plan({uint64_t(query_begin_),uint64_t(query_end_)},
                replay_start_,frontier_,options_.query_replay_tokens,store_config_.block_tokens,
                options_.budget_tokens/store_config_.block_tokens,
                device_tokens_/store_config_.block_tokens,max_rows_);
            if (!checkpoint_ || !checkpoint_recall_ || !replay_start_ || !admission.admitted) {
                replay_done_ = true;
                checkpoint_.reset(); checkpoint_recall_.reset();
                RAD_INFO("KVMEM_QUERY_REPLAY_SKIP session=%llu reason=%s first_pass_preserved=1",
                    (unsigned long long)session_.id,admission.admitted ? "no_history_checkpoint" : admission.reason);
                return RAD_OK;
            }
            // Prepare the score frame and statistical branch before changing any
            // native KV/state. Probe suffix pages never enter the recall domain.
            auto next = std::make_unique<RecallPolicy>(*recall_);
            next->restore(*checkpoint_recall_);
            const auto block = store_config_.block_tokens;
            auto required = query_prefix_required(replay_start_,block,keep_.sink_blocks,keep_.recent_blocks,
                image_required(images_,replay_start_,block));
            const auto frame = query_score_frame(replay_start_,block,options_.budget_tokens/block,
                keep_.sink_blocks,keep_.recent_blocks,required);
            std::vector<const float*> pointers;
            const auto q = query_view(pointers);
            // A reused output may end inside a selection block. The probe has
            // appended Query rows to that block: score only its frozen history.
            const auto probe_mean = store_->mean_checkpoint(replay_start_);
            store_->restore_mean_checkpoint(replay_start_, checkpoint_mean_);
            std::vector<double> accelerated;
            std::vector<uint32_t> chosen;
            try {
                if (options_.scorer == QueryScorer::PageSoftmax)
                    accelerated = query_gpu_->score(*store_,q,frame,static_cast<GqaScoreAggregation>(options_.aggregation));
                chosen = next->query(*store_,q,frame,required,options_.budget_tokens/block,
                    options_.scorer == QueryScorer::PageSoftmax ? &accelerated : nullptr);
            } catch (...) {
                store_->restore_mean_checkpoint(replay_start_, probe_mean);
                throw;
            }
            if (serial_drafter_)
                rollback_ranks(e.seq,frontier_,"trim unsampled serial probe");
            sync_memory();
            memory_session_.reset();
            memory_->rewind(replay_start_);
            update_archive_charge();
            require(content_version_ != UINT64_MAX, "replay content version exhausted");
            ++content_version_; // rewind may assign a new version to a partial tail
            memory_session_ = std::make_unique<MemorySession>(memory_);
            checkpoint_->restore(active_request_);
            store_->truncate_to(replay_start_);
            store_->restore_mean_checkpoint(replay_start_, checkpoint_mean_);
            policy_->truncate_to(replay_start_);
            const uint32_t probe_end = frontier_;
            frontier_ = replay_start_;
            std::fill(query_rows_.begin(), query_rows_.end(), frontier_);
            recall_ = std::move(next);
            publish_selection(chosen,required,SelectionReason::Query);
            recall_->commit_query();
            replaying_ = true;
            *rewind = replay_start_;
            RAD_INFO("KVMEM_QUERY_REPLAY session=%llu begin=%u end=%u scorer=%s fixed_selection=1 before_sampler=1",
                (unsigned long long)session_.id, replay_start_, probe_end,query_scorer_name(options_.scorer));
        } else replay_done_ = true;
        if(*rewind<0 && !serial_drafter_)retain_input(e);
        return RAD_OK;
    } catch (const std::exception& e) { return failure("replay", e); }
}
int StatisticsObserver::collect(const rad::StepPlan& plan, const rad::Scheduler::StepResult& result) noexcept {
    SchedulerDeviceReturn device_return{device_};
    try {
        require(plan.e.size() == 1 && result.n_seq == 1 && plan.e[0].seq == active_request_,
                "statistics collection identity mismatch");
        if (plan.e[0].is_prefill && prefill_collected_) {
            // Second callback: collect_step has fenced native serial history
            // and proposals. Publish pages only now, with all layer payloads
            // initialized. Block drafters use their separate native window.
            if (serial_drafter_) {
                rollback_ranks(plan.e[0].seq,frontier_,"trim serial lookahead before page publication");
                if (deferred_selection_) reselect(SelectionReason::PrefillPressure);
                retain_input(plan.e[0]);
            }
            return RAD_OK;
        }
        for(auto& rank:ranks_)rank->capture->seal();
        static const bool wait_trace = std::getenv("KVMEM_CAPTURE_WAIT_TRACE") != nullptr;
        const auto wait_start = std::chrono::steady_clock::now();
        unsigned sleeps = 0;
        const auto deadline = wait_start + std::chrono::seconds(30);
        while (true) {
            bool complete=true;
            for(auto& rank:ranks_) {
                const auto state=rank->capture->poll();
                require(state.status!=TransferStatus::Failed,"GPU statistics transfer failed");
                complete=complete && state.status==TransferStatus::Complete;
            }
            if(complete)break;
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
            rollback_ranks(entry.seq,entry.ctx_len+rows,"rollback unaccepted native KV before publishing statistics");
        std::vector<AcceptedStatistics> shards;
        for(auto& rank:ranks_)shards.push_back(rank->capture->take(session_, rows,
            query_begin_ < 0 || query_frozen_ ? UINT32_MAX : uint32_t(query_begin_),
            query_end_ < 0 || query_frozen_ ? UINT32_MAX : uint32_t(query_end_)));
        auto frame=merge_rank_statistics(shards);
        require(frame.key_width==store_config_.n_embd_k && frame.query_width==query_heads_*head_dim_,
            "assembled rank head geometry changed");
        for(size_t l=0;l<frame.keys.size();++l)
            store_->write_layer_mean_k(frame.first,frame.rows,uint32_t(l),frame.keys[l].data());
        auto query=std::move(frame.query);
        if (query_total_.empty()) {
            query_total_.resize(query.sums.size());
            query_rows_.resize(query.token_counts.size());
            query_counts_.resize(query.token_counts.size());
            query_tokens_.resize(query.token_counts.size());
        }
        require(query_total_.size() == query.sums.size(), "query geometry changed during session");
        for (size_t i = 0; i < query.sums.size(); ++i) {
            require(std::isfinite(query.sums[i]), "non-finite Q statistics");
            query_total_[i] += query.sums[i];
        }
        for (size_t i = 0; i < query.token_counts.size(); ++i) {
            query_rows_[i] += rows; // seal() proved all attention layers were captured
            query_counts_[i] += query.token_counts[i];
            query_tokens_[i].insert(query_tokens_[i].end(),query.tokens[i].begin(),query.tokens[i].end());
        }
        if (options_.budget_tokens) {
            auto output = recall_->output();
            auto samples = sample_rows_;
            if (!entry.is_prefill) {
                const auto plan = accepted_output_samples(output,step_tokens_.data(),rows,
                    entry.req->sp.reasoning.open,entry.req->sp.reasoning.close);
                output = plan.state; samples = plan.rows;
            }
            std::vector<std::vector<AcceptedAttentionRow>> rank_attention;
            std::vector<uint32_t> rank_heads;
            for(auto& rank:ranks_) {
                rank_attention.push_back(rank->attention->collect(rows,samples));rank_heads.push_back(rank->query_heads);
            }
            const auto observations=merge_rank_attention(rank_attention,rank_heads);
            auto exposed = exposed_;
            exposed.erase(std::remove_if(exposed.begin(),exposed.end(),[&](auto p){return p>=uint64_t(frontier_)+rows;}),exposed.end());
            recall_->accept(frontier_,rows,entry.is_prefill,observations,exposed,output);
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
            const auto resident = selected_visible_ + frontier_ - selected_frontier_;
            if (entry.is_prefill && !replaying_ && (!final_probe || replay_done_) &&
                rolling::needs_pressure(resident,options_.budget_tokens,options_.replay_tokens,max_rows_,store_config_.block_tokens)) {
                if (serial_drafter_) deferred_selection_ = true;
                else reselect(SelectionReason::PrefillPressure);
            }
        }
        if (policy_ && options_.rolling_generation && !entry.is_prefill) {
            require(frontier_ >= selected_frontier_, "rolling frontier moved backwards");
            const auto resident = selected_visible_ + frontier_ - selected_frontier_;
            if (rolling::needs_pressure(resident, options_.budget_tokens, options_.replay_tokens,
                                        max_rows_, store_config_.block_tokens)) {
                // collect_step has fenced the serial drafter. Rejected target/
                // draft lookahead was rolled back above; archive accepted rows only.
                reselect(SelectionReason::DecodePressure);
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
        std::vector<std::shared_ptr<NativeMemory>> memories;
        for(size_t i=0;i<ranks_.size();++i) {
            auto& rank=*ranks_[i];auto identity=identity_;identity.state_schema+="/rank"+std::to_string(i);
            memories.push_back(std::make_shared<NativeMemory>(MemoryBinding{rank.resources.lifetime,rank.resources.kv,
                rank.resources.ctx->stream(),rank.device,1},rank.groups,identity,
                MemoryStamp{true,session_,1,0,frontier_,1},added,block,max_position_,options_.history_bytes(),cache_->payload,cache_->io));
        }
        memory_=std::make_shared<RankNativeMemory>(std::move(memories),identity_);
        memory_session_ = std::make_unique<MemorySession>(memory_);
    } else memory_->accept_append(frontier_, added);
    accepted_view_ = memory_->snapshot();
    verify_rank_view(active_request_);
}
void StatisticsObserver::reselect(SelectionReason reason) {
    require(reason != SelectionReason::Query && !replaying_, "Pressure during fixed Query replay");
    sync_memory();
    const auto snapshot = memory_->snapshot();
    const uint32_t block = store_config_.block_tokens;
    const auto end = (uint64_t(frontier_) + block - 1) / block;
    const auto recent = end > keep_.recent_blocks ? end - keep_.recent_blocks : 0;
    std::vector<uint32_t> candidates, mandatory=image_required(images_,frontier_,block);
    mandatory.push_back((frontier_-1)/block);
    for (auto id : snapshot.resident) {
        require(id < end && id <= UINT32_MAX,"invalid accepted resident page");
        candidates.push_back(uint32_t(id));
        if (id < keep_.sink_blocks || id >= recent) mandatory.push_back(uint32_t(id));
    }
    const auto selected = recall_->pressure(reason,candidates,mandatory,options_.budget_tokens/block);
    publish_selection(selected,std::move(mandatory),reason);
}
void StatisticsObserver::publish_selection(const std::vector<uint32_t>& selected,
                                           std::vector<uint32_t> mandatory, SelectionReason reason) {
    require(!pending_selection_,"a selection is already waiting on storage");
    sync_memory();
    const auto block = store_config_.block_tokens;
    // Tail is also in the recent band (and may be in sink for a small view).
    // The selector canonicalizes its own output, but MemorySession receives
    // this separate mandatory list and deliberately rejects duplicate IDs.
    rolling::canonicalize_mandatory(mandatory);
    const std::vector<uint64_t> chosen(selected.begin(), selected.end()), required(mandatory.begin(), mandatory.end());
    // Only completed, inactive sessions retain leases; active pages can be recycled.
    require(!prefix_, "active pressure cannot borrow a retained session lease");
    const auto other=cache_->active_archive_bytes-archive_charge_+cache_->active_checkpoint_bytes;
    require(other<=history_allowance(),"active archives exceed shared host allowance");
    auto available=[&]{const auto limit=history_allowance();return limit>other ? limit-other : 0;};
    while (!cache_->sessions.empty() && cache_->bytes>available()) evict_session();
    if(other>history_allowance() || cache_->bytes>available())
        throw MemoryError(MemoryErrorCode::BudgetExceeded,"pending cache reservations exhaust history allowance");
    MemoryBudget budget{options_.budget_tokens, uint64_t(max_rows_) + block,
                        device_tokens_, available()-cache_->bytes};
    for (;;) {
        try { (void)plan_working_set(memory_->descriptor(),memory_->snapshot(),chosen,required,budget); break; }
        catch (const MemoryError& error) {
            if (error.code()!=MemoryErrorCode::BudgetExceeded)throw;
            if(!cache_->sessions.empty())evict_session();
            else if(input_ || fallback_) {
                if(input_)input_.reset();else fallback_.reset();
                update_input_charge(input_extra_bytes());
                RAD_INFO("KVMEM_INPUT_DROP session=%llu reason=pressure_budget",(unsigned long long)session_.id);
            } else throw;
            const auto occupied=cache_->bytes+cache_->active_archive_bytes-archive_charge_+cache_->active_checkpoint_bytes;
            const auto limit=history_allowance();
            budget.host_payload_bytes=limit>occupied ? limit-occupied : 0;
        }
    }
    const auto& selection = memory_session_->prepare(memory_->stamp(), chosen, required, budget);
    // Reserve the future logical archive charge before yielding to another
    // sequence; otherwise two pending spills could spend the same allowance.
    require(selection.host_payload_bytes_required>=archive_charge_,"pending archive charge shrank unexpectedly");
    cache_->active_archive_bytes+=selection.host_payload_bytes_required-archive_charge_;
    archive_charge_=selection.host_payload_bytes_required;
    pending_selection_=PendingSelection{reason,selection.spill.size(),selection.restore.size(),
        std::chrono::steady_clock::now()+std::chrono::seconds(30),memory_};
    (void)poll_selection(!asynchronous_);
}
bool StatisticsObserver::poll_prefix() {
    if(!pending_prefix_)return true;
    auto& read=*pending_prefix_;
    if(std::chrono::steady_clock::now()>read.deadline)throw StorageFailure("prefix read timed out");
    if(!read.ticket) {
        const auto work=read.work;
        const uint64_t bytes=work->source->host_bytes()+work->source->state_bytes();
        try {read.ticket=cache_->io->submit(bytes,[work]{work->run();});}
        catch(const MemoryError& error){if(error.code()==MemoryErrorCode::Pending)return false;throw;}
    }
    if(!read.ticket.ready())return false;
    read.ticket.wait();prefix_payload_=std::make_unique<RankPrefix::Payload>(std::move(read.work->payload));
    pending_prefix_.reset();return true;
}
void StatisticsObserver::cancel_selection() noexcept {
    if(!pending_selection_)return;
    draining_selection_failure_=true;
    pending_selection_->storage->cancel_queued_storage();
}
int StatisticsObserver::poll_pending_selection() noexcept {
    SchedulerDeviceReturn device_return{device_};
    try {
        if(draining_selection_failure_) {
            if(!pending_selection_->storage->storage_quiescent()) {
                if(std::chrono::steady_clock::now()>=pending_selection_->deadline &&
                   !pending_selection_->storage->cpu_io_pending())
                    throw std::runtime_error("device transfer timed out during storage drain");
                return RAD_OK;
            }
            const auto result=memory_session_->abort();
            if(result==AbortResult::SessionInvalidated)
                throw std::runtime_error("device view invalidated during storage drain");
            clear_session();return RAD_OK;
        }
        if(draining_prefix_failure_) {
            if(!pending_prefix_->ticket.ready())return RAD_OK;
            try{pending_prefix_->ticket.wait();}
            catch(const std::exception& error){RAD_ERR("KVMem retired prefix I/O: %s",error.what());}
            clear_session();return RAD_OK;
        }
        if(pending_prefix_ && std::chrono::steady_clock::now()>pending_prefix_->deadline &&
           pending_prefix_->ticket && !pending_prefix_->ticket.ready()) {
            // Reject the request now; a started CPU read still owns its sources.
            // Keep that bounded ownership until ready instead of blocking in a
            // ticket destructor on the engine thread while other requests run.
            draining_prefix_failure_=true;
            pending_prefix_->ticket.cancel(); // A queued read can release its slot immediately.
            RAD_ERR("KVMem prefix read timed out; draining owned CPU I/O asynchronously");
            return RAD_E_IO;
        }
        if(pending_cache_ && !cache_->pending(pending_cache_))pending_cache_=0;
        (void)poll_prefix();
        if(poll_selection() && poll_input() && pending_finish_) {
            capture_terminal(*pending_finish_);pending_finish_.reset();
        }
        return RAD_OK;
    }
    catch(const std::exception& error){return failure("pending selection",error);}
}
bool StatisticsObserver::poll_selection(bool wait) {
    if(!pending_selection_)return true;
    while(memory_session_->phase()!=MemoryPhase::Ready) {
        const auto& storage=pending_selection_->storage;
        if(std::chrono::steady_clock::now()>=pending_selection_->deadline) {
            if(storage->awaiting_storage() || storage->storage_error()) {
                cancel_selection();
                throw StorageFailure("active working-set storage timed out; draining asynchronously");
            }
            throw std::runtime_error("native working-set device transfer timed out");
        }
        const auto before=memory_session_->phase();memory_session_->advance();
        if(const auto error=storage->storage_error()) {
            cancel_selection();
            std::rethrow_exception(error);
        }
        if(memory_session_->phase()==before) {
            if(!wait)return false;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    const auto reason=pending_selection_->reason;
    const auto spill=pending_selection_->spill,restore=pending_selection_->restore;
    const auto view = memory_session_->publish();
    verify_rank_view(active_request_);
    pending_selection_.reset();
    update_archive_charge();
    accepted_view_ = memory_->snapshot();
    const auto required_images=image_required(images_,frontier_,store_config_.block_tokens);
    for(const auto page:required_images)
        require(std::find(accepted_view_.resident.begin(),accepted_view_.resident.end(),page)!=accepted_view_.resident.end(),
            "published selection lost an accepted image page");
    if(!required_images.empty())RAD_INFO("KVMEM_IMAGE_PAGES session=%llu frontier=%u required=%zu resident=%zu verified=1",
        (unsigned long long)session_.id,frontier_,required_images.size(),accepted_view_.resident.size());
    ++reselections_; spilled_ += spill; restored_ += restore;
    selected_frontier_ = frontier_; selected_visible_ = view.visible_tokens;
    if (reason != SelectionReason::Query) {
        require(restore == 0, "Pressure unexpectedly restored cold KV");
    }
    if (reason == SelectionReason::DecodePressure) {
        ++decode_reselections_;
        RAD_INFO("KVMEM_DECODE_PRESSURE round=%llu frontier=%u visible=%llu budget=%u reserve=%u spill=%zu restore=%zu scorer=accepted-attention accepted_only=1",
            (unsigned long long)decode_reselections_, frontier_, (unsigned long long)view.visible_tokens,
            options_.budget_tokens, options_.replay_tokens, spill, restore);
    }
    RAD_INFO("KVMEM_RESELECT session=%llu frontier=%u visible=%llu spill=%zu restore=%zu mode=%s original_positions=1",
             (unsigned long long)session_.id, frontier_, (unsigned long long)view.visible_tokens,
             spill, restore, reason == SelectionReason::Query ? query_scorer_name(options_.scorer) :
                (reason == SelectionReason::DecodePressure ? "decode-pressure-attention" : "prefill-pressure-attention"));
    return true;
}
int StatisticsObserver::finish(const rad::StepEntry& e, const rad::Request& request) noexcept {
    SchedulerDeviceReturn device_return{device_};
    try {
        if (!policy_ || !frontier_ || !options_.cache_sessions) return RAD_OK;
        require(request.id == active_request_ && e.seq == active_request_ &&
                request.n_computed == frontier_, "terminal state is not the accepted frontier");
        auto identity=std::make_shared<CachedIdentity>();
        identity->tokens=request.prompt;
        identity->tokens.insert(identity->tokens.end(),request.output.begin(),request.output.end());
        require(identity->tokens.size()>=frontier_,"terminal KV extends beyond emitted history");
        identity->tokens.resize(frontier_); // exclude the sampled, unevaluated tail
        identity->images=image_prefix(images_,frontier_);
        identity->scope=request.cache_owner;
        auto terminal=std::make_unique<TerminalSnapshot>();terminal->identity=std::move(identity);
        terminal->request=request.id;terminal->output_tokens=request.output.size();
        terminal->accepted_rows=e.is_prefill ? 0u : uint32_t(frontier_-e.ctx_len);
        if(!poll_selection(!asynchronous_) || !poll_input()) {
            require(cache_->next_park!=UINT64_MAX,"cache ticket identity exhausted");
            const auto id=cache_->next_park++;
            const auto input_identity=input_ ? input_->identity : input_parking_ ? input_parking_->saved.identity : pending_input_;
            cache_->finishing.push_back({id,terminal->identity,input_identity,fallback_ ? fallback_->identity : nullptr});finishing_id_=id;
            pending_finish_=std::move(terminal);
            RAD_INFO("KVMEM_SESSION_FINISH_PENDING session=%llu",(unsigned long long)session_.id);
            return RAD_OK;
        }
        capture_terminal(*terminal);return RAD_OK;
    } catch (const std::exception& error) { return failure("terminal snapshot", error); }
}
void StatisticsObserver::capture_terminal(const TerminalSnapshot& terminal) {
        require(terminal.request==active_request_ && terminal.identity->tokens.size()==frontier_,
            "deferred terminal frontier changed");
        sync_memory();
        uint64_t native_bytes=0;
        for(const auto& rank:ranks_)for (const auto& p : rank->resources.kv->plans()) {
            if (p.stateful()) native_bytes+=uint64_t(p.bytes_per_state);
            else if (p.kind==RAD_KV_FULL)
                native_bytes+=rank->resources.kv->block_table(active_request_,p.index).size()*uint64_t(p.bytes_per_block);
        }
        const uint64_t needed=input_extra_bytes()+native_bytes+memory_->archive_bytes()+store_->allocated_bytes()+
            policy_->allocated_bytes()+2*recall_->allocated_bytes()+terminal.identity->tokens.size()*sizeof(int32_t)+
            terminal.identity->images.size()*sizeof(ImageIdentity)+
            uint64_t(store_config_.n_layer)*(store_config_.n_embd_k+1)*sizeof(float);
        const auto other=cache_->active_archive_bytes-archive_charge_+cache_->active_checkpoint_bytes-input_charge_;
        require(other<=history_allowance(),"active archive quota exceeded");
        auto available=[&]{const auto limit=history_allowance();return limit>other ? limit-other : 0;};
        if (needed>available()) {
            RAD_INFO("KVMEM_SESSION_SKIP session=%llu reason=host_budget required_bytes=%llu limit=%llu",
                (unsigned long long)session_.id,(unsigned long long)needed,(unsigned long long)options_.history_bytes());
            return;
        }
        auto fits=[&]{return needed<=available() && cache_->bytes<=available()-needed;};
        while (!cache_->sessions.empty() && (!fits() || cache_->count()>=options_.cache_sessions))
            evict_session();
        if(!fits() || cache_->count()>=options_.cache_sessions) {
            RAD_INFO("KVMEM_SESSION_SKIP session=%llu reason=pending_cache_capacity",(unsigned long long)session_.id);
            return;
        }
        auto lease=std::make_unique<RankPrefix>();
        for(auto& rank:ranks_) {
            const AcceptedState accepted{&rank->groups,rank->materialize_gdn,terminal.accepted_rows};
            auto part=std::make_unique<PrefixLease>(rank->resources.lifetime,*rank->resources.kv,
                active_request_,terminal.identity->tokens,rank->device,rank->resources.ctx->stream(),accepted);
            lease->add_rank(rank->device,rank->resources.ctx->stream(),*rank->resources.kv,std::move(part));
        }
        auto mean = store_->mean_checkpoint(frontier_);
        auto recall = std::make_unique<RecallSnapshot>(recall_->checkpoint());
        memory_->retain_view();
        prefix_ = std::move(lease);
        prefix_mean_ = std::move(mean);
        prefix_recall_ = std::move(recall);
        checkpoint_.reset(); checkpoint_recall_.reset();
        std::atomic_store(&cached_identity_, terminal.identity);
        RAD_INFO("KVMEM_SESSION_RETAIN request=%llu frontier=%u output_tokens=%zu accepted_rows=%u partial_tail=%u state_canonical=1",
            (unsigned long long)active_request_, frontier_, terminal.output_tokens,
            terminal.accepted_rows, frontier_ % store_config_.block_tokens);
}
void StatisticsObserver::retire(uint64_t id) noexcept {
    SchedulerDeviceReturn device_return{device_};
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
    if(!prefix_ && (input_ || fallback_)) {
        try {if(!input_)input_=std::move(fallback_);promote_input();}
        catch(const std::exception& e){failure("input fallback",e);clear_session();return;}
    }
    if (prefix_ && prefix_recall_) {
        try {
            frontier_ = uint32_t(prefix_->frontier());
            store_->truncate_to(frontier_);
            store_->restore_mean_checkpoint(frontier_, prefix_mean_);
            policy_->truncate_to(frontier_);
            recall_->restore(*prefix_recall_);
            std::fill(query_rows_.begin(), query_rows_.end(), frontier_);
            checkpoint_.reset(); memory_session_.reset(); active_request_ = 0;
            park_session();
            return;
        } catch (const std::exception& e) { failure("retained prefix", e); }
    }
    clear_session();
}
void StatisticsObserver::shutdown() noexcept {
    SchedulerDeviceReturn device_return{device_};
    for(auto& rank:ranks_) {
        if(rank->capture)(void)rank->capture->discard();
        if(rank->attention)rank->attention->discard();
    }
    clear_session();
    if (cache_.use_count()==1) { cache_->parking.clear();cache_->sessions.clear(); cache_->bytes=0; }
    query_gpu_.reset();ranks_.clear();world_size_=0;
}
} // namespace kvmem::radiance
