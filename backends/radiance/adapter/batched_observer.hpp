#pragma once
#include "execution_options.hpp"
#include "statistics_observer.hpp"
#include "observation_slice.hpp"
#include "request_monitor.hpp"
#include "native_pages.hpp"
#include <array>

namespace kvmem::radiance {
// One controller per active sequence. Native decode stays batched; Query probe
// and replay use fair, isolated prefill steps so no other sequence is rewound.
// Host archive/cache and graph quotas are shared by every controller.
class BatchedStatisticsObserver final : public rad::StepObserver {
    struct Slot {
        uint64_t request=0;
        bool retiring=false;
        bool reuse_observed=false, shared_restore=false;
        uint64_t planned_reuse=0;
        std::unique_ptr<StatisticsObserver> observer;
        struct Capture {
            rad::CtxDesc::KvObserver keys=nullptr, attention=nullptr;
            void* data=nullptr;
        };
        std::vector<Capture> ranks;
    };
    struct Frame { size_t slot; int64_t offset; rad::StepPlan plan; };
    SelectionOptions options_;
    std::vector<Slot> slots_;
    struct RankDispatch {BatchedStatisticsObserver* owner;size_t rank;};
    std::vector<std::unique_ptr<RankDispatch>> rank_dispatch_;
    int mirror_rank_metadata() noexcept {
        if(slots_.empty())return RAD_E_STATE;
        const auto& ranks=slots_[0].observer->ranks_;
        if(ranks.size()!=slots_[0].observer->world_size_ || ranks.empty())return RAD_E_STATE;
        for(size_t i=1;i<ranks.size();++i) {
            const int status=ranks[i]->resources.kv->mirror_adapter_metadata(*ranks[0]->resources.kv);
            if(status<0){RAD_ERR("KVMem rank %zu metadata mirror failed",i);return status;}
        }
        return RAD_OK;
    }
    std::vector<Frame> frames_;
    struct Refusal { uint64_t id; int status; };
    std::array<Refusal,8> capacity_refusals_{};
    size_t refusal_count_=0;
    std::shared_ptr<StatisticsObserver::CacheState> cache_=std::make_shared<StatisticsObserver::CacheState>();
    std::shared_ptr<GraphStatisticsBudget> graph_=std::make_shared<GraphStatisticsBudget>(256u<<20);
    size_t peak_batch_=0;
    RequestMonitor monitor_;
    nlohmann::json runtime_=nlohmann::json::object();
    std::shared_ptr<const nlohmann::json> cache_snapshot_;
    void publish_monitor() noexcept {
        try {
            const auto io=cache_->payload ? cache_->payload->io_stats() : SpillIoStats{};
            const auto sharing_credit=cache_->sharing_credit();
            auto value=std::make_shared<nlohmann::json>(nlohmann::json{
                {"retained_sessions",cache_->groups()},{"retained_records",cache_->sessions.size()},
                {"retained_records_limit",options_.effective_cache_records()},{"cache_retention_revision",2},
                {"retained_recovery_points",cache_->recovery_points()},{"planned_history_hits",cache_->hits},
                {"history_hits",cache_->actual_own_hits+cache_->actual_shared_hits},
                {"own_history_hits",cache_->actual_own_hits},{"shared_history_hits",cache_->actual_shared_hits},
                {"history_misses",cache_->actual_misses},
                {"parking_sessions",cache_->parking.size()},{"cache_park_errors",cache_->park_errors},
                {"finishing_sessions",cache_->finishing.size()},
                {"history_forks",cache_->forks},
                {"cache_min_tokens",options_.cache_min_tokens},{"cache_candidate_tokens",cache_->candidate_tokens},
                {"cache_below_minimum",cache_->below_minimum},
                {"history_retention_evictions",cache_->retention_evictions},
                {"history_capacity_evictions",cache_->evictions-cache_->retention_evictions},
                {"io_pending_sequences",std::count_if(slots_.begin(),slots_.end(),[](const Slot& s){return s.observer->pending_io();})},
                {"history_evictions",cache_->evictions},{"host_payload_bytes",cache_->payload ? cache_->payload->charged_bytes() : 0},
                {"host_payload_budget_bytes",options_.host_bytes},{"disk_payload_bytes",cache_->payload ? cache_->payload->disk_bytes() : 0},
                {"disk_payload_budget_bytes",options_.disk_bytes},{"selected_tokens",options_.budget_tokens},
                {"disk_read_bytes",io.read_bytes},{"disk_written_bytes",io.written_bytes},
                {"disk_reads",io.reads},{"disk_writes",io.writes},{"disk_errors",io.errors},
                {"checkpoint_kv_capture_bytes",cache_->capture_bytes},{"checkpoint_kv_logical_bytes",cache_->capture_logical_bytes},
                {"retained_logical_bytes",cache_->bytes},{"active_archive_bytes",cache_->active_archive_bytes},
                {"active_checkpoint_bytes",cache_->active_checkpoint_bytes},
                {"shared_payload_credit_bytes",sharing_credit},
                {"accounted_cache_bytes",cache_->bytes+cache_->active_archive_bytes+cache_->active_checkpoint_bytes-sharing_credit},
                {"reserve_tokens",options_.replay_tokens},{"max_concurrency",slots_.size()}});
            value->update(runtime_);
            int64_t played=0,recorded=0,audited=0,unkeyed=0,refused=0;
            bool tape_enabled=false;
            // Each slot references the same rank Ctxs. Count each rank once.
            if (!slots_.empty()) for (const auto& rank:slots_.front().observer->ranks_) {
                const auto* ctx=rank->resources.ctx;
                if (!ctx) continue;
                const auto counts=ctx->pass_counts();
                played+=counts.played;recorded+=counts.recorded;audited+=counts.audited;
                unkeyed+=counts.issued_unkeyed;refused+=counts.issued_refused;
                tape_enabled=tape_enabled || ctx->records_passes();
            }
            (*value)["execution"]={{"revision",1},{"tape_enabled",tape_enabled},
                {"passes_played",played},{"passes_recorded",recorded},{"passes_audited",audited},
                {"passes_unkeyed",unkeyed},{"passes_refused",refused},
                {"early_statistics_collection",early_statistics_collection()},
                {"cross_step_run_ahead",false},{"transfer_completion_fence","event"},
                {"borrowed_spill_capture",borrow_spill_capture()}};
            std::atomic_store(&cache_snapshot_,std::shared_ptr<const nlohmann::json>(std::move(value)));
        } catch(...) { RAD_ERR("KVMem cache monitor metadata allocation failed"); }
    }
    size_t slot_for(uint64_t request, bool create=false) {
        for (size_t i=0;i<slots_.size();++i) if (slots_[i].request==request) return i;
        if (create) for (size_t i=0;i<slots_.size();++i) if (!slots_[i].request) {
            slots_[i].request=request; slots_[i].reuse_observed=false;
            slots_[i].shared_restore=false;slots_[i].planned_reuse=0;return i;
        }
        throw std::runtime_error("missing per-sequence statistics controller");
    }
    static rad::StepPlan single(const rad::StepPlan& plan, const rad::StepEntry& entry) {
        rad::StepPlan one;
        one.e.push_back(entry); one.n_tok=entry.n_tokens; one.draft_pass=plan.draft_pass;
        one.phase=entry.is_prefill ? RAD_PHASE_PREFILL : RAD_PHASE_DECODE;
        one.n_spec=entry.n_spec; one.max_q_len=entry.n_tokens; one.max_ctx_len=entry.ctx_len;
        one.n_seq_decode=entry.is_prefill ? 0 : 1; one.n_tok_decode=entry.is_prefill ? 0 : entry.n_tokens;
        return one;
    }
    int observe(size_t rank,bool attention, int layer, rad_kvgroup group, int64_t dim,
                const RadTensor& q, const RadTensor& k, const RadBatch& batch, RadStream stream) noexcept {
        try {
            if (batch.draft_pass || frames_.size()!=size_t(batch.n_seq)) return RAD_E_STATE;
            int64_t end=0;
            for (size_t i=0;i<frames_.size();++i) {
                auto& frame=frames_[i]; auto& slot=slots_[frame.slot];
                if (frame.offset!=end) return RAD_E_STATE;
                const auto count=frame.plan.e[0].n_tokens;
                ObservationBatch view(batch,int64_t(i),frame.offset,count);
                const auto qs=observation_rows(q,frame.offset,count);
                const auto ks=attention ? k : observation_rows(k,frame.offset,count);
                if(rank>=slot.ranks.size())return RAD_E_STATE;
                const auto& capture=slot.ranks[rank];
                const auto callback=attention ? capture.attention : capture.keys;
                if (callback) {
                    const int status=callback(capture.data,layer,group,dim,qs,ks,view.batch,stream);
                    if (status<0) return status;
                }
                end+=count;
            }
            return end==batch.n_tok ? RAD_OK : RAD_E_STATE;
        } catch (const std::exception& error) { RAD_ERR("KVMem batch capture: %s",error.what()); return RAD_E_STATE; }
    }
public:
    bool early_collect() const noexcept override { return early_statistics_collection(); }
    explicit BatchedStatisticsObserver(SelectionOptions options={}) : options_(options) {}
    bool holds_sequence(uint64_t request) const noexcept override {
        for(const auto& slot:slots_)if(slot.request==request)return slot.observer->pending_io();
        return false;
    }
    int poll_pending() noexcept override {
        try {
        bool updated=false;
        for(auto it=cache_->parking.begin();it!=cache_->parking.end();) {
            if(!(*it)->ticket.ready()){++it;continue;}
            auto pending=std::move(*it);it=cache_->parking.erase(it);updated=true;
            const auto bytes=pending->saved.bytes;
            try {
                pending->ticket.wait();pending->work->prefix->finish_park();pending->saved.prefix=std::move(pending->work->prefix);
                const auto session=pending->saved.session.id;
                if(!cache_->publish(std::move(pending->saved),options_.cache_sessions,options_.effective_cache_records(),options_.cache_min_tokens,true)) {
                    if(cache_->bytes<bytes)throw std::runtime_error("invalid pending cache charge");
                    cache_->bytes-=bytes;
                    RAD_INFO("KVMEM_SESSION_SKIP session=%llu reason=directory_capacity source_preserved=1",(unsigned long long)session);
                    continue;
                }
                RAD_INFO("KVMEM_SESSION_PARK_READY session=%llu entries=%zu pending=%zu",
                    (unsigned long long)session,cache_->sessions.size(),cache_->parking.size());
            } catch(const std::exception& error) {
                if(cache_->bytes<bytes)throw std::runtime_error("invalid pending cache charge");
                cache_->bytes-=bytes;++cache_->park_errors;
                RAD_ERR("KVMem optional cache parking dropped: %s",error.what());
            }
        }
        for(auto& slot:slots_)if(slot.observer->pending_io()) {
            updated=true;
            const int status=slot.observer->poll_pending_selection();
            if(status==RAD_E_FULL || status==RAD_E_IO) {
                if(!slot.retiring) {
                    if(refusal_count_>=capacity_refusals_.size())return RAD_E_STATE;
                    capacity_refusals_[refusal_count_++]={slot.request,status};
                }
                if(!slot.observer->draining_storage_failure())slot.observer->clear_session();
            } else if(status<0)return status;
            if(slot.retiring && !slot.observer->pending_io()) {
                if(!slot.observer->active_request_)slot.observer->clear_session();
                else slot.observer->retire(slot.request);
                slot.request=0;slot.retiring=false;
            }
        }
        if(updated)publish_monitor();return RAD_OK;
        } catch(const std::exception& error){RAD_ERR("KVMem pending cache: %s",error.what());return RAD_E_STATE;}
    }
    void request_metadata(const rad::Request& request,int64_t age,int64_t first) noexcept override {
        monitor_.update(request,age,first);
    }
    std::string monitor_snapshot() const override {
        auto data=monitor_.snapshot();data["backend"]="radiance";
        const auto cache=std::atomic_load(&cache_snapshot_);
        data["kvmem"]=cache ? *cache : nlohmann::json::object();
        return data.dump();
    }
    bool retain_history_boundary() const noexcept override { return options_.budget_tokens>0; }
    bool serial_prefill() const noexcept override { return true; }
    int pool_token_limit(const rad::Config& cfg,const rad::Program& program,int64_t* out) noexcept override {
        if (cfg.max_seqs<1 || cfg.max_seqs>8 || (cfg.tp!=1 && cfg.tp!=2)) return RAD_E_UNSUPPORTED;
        auto single_cfg=cfg;single_cfg.max_seqs=1;
        StatisticsObserver planner(options_);
        int status=planner.pool_token_limit(single_cfg,program,out);
        if (status<0 || *out>INT64_MAX/cfg.max_seqs) return status<0 ? status : RAD_E_INVAL;
        *out*=cfg.max_seqs;
        RAD_INFO("KVMEM_BATCH_POOL sequences=%lld total_tokens=%lld per_sequence_budget=%u",
            (long long)cfg.max_seqs,(long long)*out,options_.budget_tokens);
        return RAD_OK;
    }
    int configure(const rad::Config& cfg,rad::CtxDesc& desc) noexcept override {
        try {
            if (cfg.max_seqs<1 || cfg.max_seqs>8 || desc.rank<0 || size_t(desc.rank)!=rank_dispatch_.size()) return RAD_E_INVAL;
            if(!options_.host_bytes || bool(options_.disk_bytes)!=!options_.disk_path.empty())return RAD_E_INVAL;
            (void)options_.history_bytes();
            if(desc.rank==0) {
                cache_->payload=std::make_shared<HostKvStorage>(options_.host_bytes,options_.disk_bytes ?
                    std::make_shared<SpillFile>(std::filesystem::u8path(options_.disk_path),options_.disk_bytes) : nullptr);
                for(int64_t i=0;i<cfg.max_seqs;++i) {
                    Slot slot;slot.observer=std::make_unique<StatisticsObserver>(options_);
                    slot.observer->asynchronous_=true;
                    slot.observer->cache_->observers.erase(slot.observer.get());
                    slot.observer->cache_=cache_;cache_->observers.insert(slot.observer.get());
                    slot.observer->graph_budget_=graph_;
                    slots_.push_back(std::move(slot));
                }
            }
            if(slots_.size()!=size_t(cfg.max_seqs))return RAD_E_INVAL;
            auto one=cfg;one.max_seqs=1;
            for (auto& slot:slots_) {
                auto config=desc;
                const auto status=slot.observer->configure(one,config); if (status<0) return status;
                slot.ranks.push_back({config.kv_observer,config.attention_observer,config.kv_observer_data});
            }
            auto dispatch=std::make_unique<RankDispatch>(RankDispatch{this,size_t(desc.rank)});
            desc.kv_observer=[](void* data,int layer,rad_kvgroup group,int64_t dim,const RadTensor& q,
                                const RadTensor& k,const RadBatch& batch,RadStream stream) {
                auto& rank=*static_cast<RankDispatch*>(data);
                return rank.owner->observe(rank.rank,false,layer,group,dim,q,k,batch,stream);
            };
            desc.attention_observer=[](void* data,int layer,rad_kvgroup group,int64_t dim,const RadTensor& q,
                                       const RadTensor& k,const RadBatch& batch,RadStream stream) {
                auto& rank=*static_cast<RankDispatch*>(data);
                return rank.owner->observe(rank.rank,true,layer,group,dim,q,k,batch,stream);
            };
            desc.kv_observer_data=dispatch.get();rank_dispatch_.push_back(std::move(dispatch));
            runtime_={{"enabled",options_.budget_tokens>0},{"scoring",{
                {"scorer",options_.scorer==QueryScorer::PageSoftmax ? "page-softmax" : "meanq-cosine"},
                {"agg",options_.aggregation==0 ? "mean" : "max"},{"lrfu",options_.lrfu},
                {"lambda",options_.lrfu_lambda},{"top_frac",options_.lrfu_top_fraction},
                {"graph_alpha",options_.graph_alpha},{"query_max_tokens",options_.query_tokens},
                {"query_replay_max_tokens",options_.query_replay_tokens},{"block_tokens",options_.block_tokens}}},
                {"kv_type",cfg.kv_cache_dtype},{"state_cache_type",cfg.state_cache_dtype},
                {"block_tokens",options_.block_tokens},{"prefill_tokens",cfg.max_tok},
                {"vram_reserve_mib",cfg.gpu_headroom_mib},{"draft_policy",rad::draft_window_name(cfg.draft_window)},
                {"sink_tokens",slots_.front().observer->keep_.sink_blocks*options_.block_tokens},
                {"recent_tokens",slots_.front().observer->keep_.recent_blocks*options_.block_tokens},
                {"query_scorer",options_.scorer==QueryScorer::PageSoftmax ? "page-softmax" : "meanq-cosine"},
                {"graph_alpha",options_.graph_alpha},{"query_max_tokens",options_.query_tokens},
                {"query_replay_max_tokens",options_.query_replay_tokens},{"rolling_generation",options_.rolling_generation},
                {"capabilities",{{"disk_budget_bytes",options_.disk_bytes},{"ssd_enabled",options_.disk_bytes>0},
                    {"retained_sessions_limit",options_.cache_sessions},{"retained_records_limit",options_.effective_cache_records()},
                    {"snapshot_storage_revision",1},{"incremental_checkpoint_capture",true},
                    {"restore_staging","full_selected_view"},{"max_active_requests",cfg.max_seqs}}}};
            publish_monitor();
            return RAD_OK;
        } catch (const std::exception& error) { RAD_ERR("KVMem batch configure: %s",error.what()); return RAD_E_STATE; }
    }
    int bind_rank(rad::RankResources resources) noexcept override {
        try {
        for (auto& slot:slots_) { const int status=slot.observer->bind_rank(resources);if(status<0)return status; }
        const auto& first=*slots_.front().observer;
        if(options_.budget_tokens && first.ranks_.size()==first.world_size_) {
            uint64_t page=0,state=0;
            for(const auto& rank:first.ranks_) {
                if(!rank->resources.kv)throw std::runtime_error("storage configured before all rank bindings");
                NativePages geometry(rank->resources.kv->plans(),rank->groups,first.store_config_.block_tokens,first.identity_.state_schema);
                if(geometry.layout().record_bytes>UINT64_MAX-page)throw std::runtime_error("rank page size overflow");
                page+=geometry.layout().record_bytes;
                for(const auto& group:rank->resources.kv->plans())if(group.stateful()) {
                    if(uint64_t(group.bytes_per_state)>UINT64_MAX-state)throw std::runtime_error("rank state size overflow");
                    state+=uint64_t(group.bytes_per_state);
                }
            }
            if(!page || page>options_.host_bytes)throw MemoryError(MemoryErrorCode::BudgetExceeded,"H must hold at least one complete native KV record");
            const uint64_t blocks=(uint64_t(options_.budget_tokens)+options_.replay_tokens+options_.block_tokens-1)/options_.block_tokens+2;
            if(blocks>UINT64_MAX/4/page || state>UINT64_MAX/2 ||
                blocks*4*page>UINT64_MAX-2*state || (blocks*4*page+2*state)>UINT64_MAX/slots_.size())
                throw MemoryError(MemoryErrorCode::BudgetExceeded,"storage staging size overflow");
            const auto bytes=(blocks*4*page+2*state)*slots_.size();
            cache_->io=std::make_shared<StorageIoQueue>(2*slots_.size()+2,bytes,std::min<size_t>(2,slots_.size()));
            RAD_INFO("KVMEM_STORAGE ram_bytes=%llu disk_bytes=%llu io_queue_bytes=%llu shared_sessions=1",
                (unsigned long long)options_.host_bytes,(unsigned long long)options_.disk_bytes,(unsigned long long)bytes);
        }
        return RAD_OK;
        } catch(const std::exception& error){RAD_ERR("KVMem shared storage: %s",error.what());return RAD_E_INVAL;}
    }
    std::string validate_request(const rad::Request& request) const override {
        return slots_.empty() ? "statistics controllers are not initialized" : slots_[0].observer->validate_request(request);
    }
    int resume(const rad::StepPlan& plan,int64_t* frontier,std::string* refusal) noexcept override {
        *frontier=-1;refusal->clear();
        try {
            if (plan.e.size()!=1 || !plan.e[0].is_prefill) return RAD_OK;
            const int mirrored=mirror_rank_metadata();if(mirrored<0)return mirrored;
            auto& slot=slots_[slot_for(plan.e[0].seq,true)];
            const bool fresh=!slot.reuse_observed && plan.e[0].ctx_len==0 && slot.observer->active_request_!=plan.e[0].seq;
            const auto before_forks=cache_->cross_owner_forks;
            const auto status=slot.observer->resume(plan,frontier,refusal);
            if(fresh) {
                slot.shared_restore=slot.shared_restore || cache_->cross_owner_forks>before_forks;
                if(slot.observer->prefix_)slot.planned_reuse=std::max<uint64_t>(slot.planned_reuse,slot.observer->prefix_->frontier());
                if(*frontier>0)slot.planned_reuse=std::max<uint64_t>(slot.planned_reuse,uint64_t(*frontier));
                const bool waiting=slot.observer->pending_cache_ || slot.observer->pending_prefix_;
                if(status<0 || !refusal->empty() || *frontier>0 || !waiting) {
                    const auto actual=*frontier>0 ? uint64_t(*frontier) : 0;
                    monitor_.reuse(plan.e[0].seq,slot.planned_reuse,actual,slot.shared_restore);
                    if(!actual)++cache_->actual_misses;
                    else if(slot.shared_restore)++cache_->actual_shared_hits;
                    else ++cache_->actual_own_hits;
                    slot.reuse_observed=true;publish_monitor();
                }
            }
            if((status==RAD_E_FULL || status==RAD_E_IO) && !slot.observer->draining_storage_failure())slot.observer->clear_session();
            return status;
        } catch(const std::exception& error) {RAD_ERR("KVMem batch resume: %s",error.what());return RAD_E_STATE;}
    }
    int begin(const rad::StepPlan& plan) noexcept override {
        try {
            const int mirrored=mirror_rank_metadata();if(mirrored<0)return mirrored;
            frames_.clear();int64_t offset=0;
            for (const auto& entry:plan.e) {
                if (entry.is_prefill && plan.e.size()!=1) return RAD_E_STATE;
                const auto i=slot_for(entry.seq,true);
                frames_.push_back({i,offset,single(plan,entry)});offset+=entry.n_tokens;
                const auto status=slots_[i].observer->begin(frames_.back().plan);if(status<0)return status;
            }
            if (offset!=plan.n_tok) return RAD_E_STATE;
            if (plan.e.size()>peak_batch_) {
                peak_batch_=plan.e.size();RAD_INFO("KVMEM_BATCH_ACTIVE sequences=%zu tokens=%lld",peak_batch_,(long long)offset);
            }
            return RAD_OK;
        } catch(const std::exception& error){RAD_ERR("KVMem batch begin: %s",error.what());return RAD_E_STATE;}
    }
    bool split_sampling(const rad::StepPlan& plan) const noexcept override {
        return plan.e.size()==1 && plan.e[0].is_prefill;
    }
    int after_forward(const rad::StepPlan& plan,int64_t* rewind) noexcept override {
        *rewind=-1;
        if (frames_.size()!=1 || !split_sampling(plan)) return RAD_E_STATE;
        return slots_[frames_[0].slot].observer->after_forward(frames_[0].plan,rewind);
    }
    int collect(const rad::StepPlan& plan,const rad::Scheduler::StepResult& result) noexcept override {
        try {
        if (frames_.size()!=plan.e.size() || result.n_seq!=int(frames_.size())) return RAD_E_STATE;
        for (size_t i=0;i<frames_.size();++i) {
            auto one=result;one.n_seq=1;
            if (one.n_accepted) one.n_accepted+=i;
            if (one.token) one.token+=i;
            if (one.eos) one.eos+=i;
            if (one.first_row) one.first_row+=i;
            auto& slot=slots_[frames_[i].slot];
            const auto status=slot.observer->collect(frames_[i].plan,one);
            if(status==RAD_E_FULL || status==RAD_E_IO) {
                if(refusal_count_>=capacity_refusals_.size())return RAD_E_STATE;
                capacity_refusals_[refusal_count_++]={slot.request,status};
                if(!slot.observer->draining_storage_failure())slot.observer->clear_session(); // retain in-flight storage ownership
            } else if(status<0)return status;
        }
        publish_monitor();
        return RAD_OK;
        } catch(const std::exception& error) {
            RAD_ERR("KVMem batch collect: %s",error.what());return RAD_E_STATE;
        }
    }
    uint64_t take_request_refusal(int* status) noexcept override {
        if(!refusal_count_)return 0;
        const auto refusal=capacity_refusals_[--refusal_count_];*status=refusal.status;return refusal.id;
    }
    int finish(const rad::StepEntry& entry,const rad::Request& request) noexcept override {
        try { return slots_[slot_for(entry.seq)].observer->finish(entry,request); }
        catch(const std::exception& error){RAD_ERR("KVMem batch finish: %s",error.what());return RAD_E_STATE;}
    }
    void retire(uint64_t request) noexcept override {
        for(auto& slot:slots_)if(slot.request==request) {
            if(slot.observer->pending_cache_) {
                // The cache job owns its source independently. Cancelling a
                // waiter need not pin a new native sequence until that job ends.
                slot.observer->clear_session();slot.request=0;slot.retiring=false;publish_monitor();return;
            }
            // The sink is already closed. Keep ownership while a started CPU
            // transfer drains; other sequences can still be scheduled meanwhile.
            if(slot.observer->pending_io()){
                if(!slot.observer->pending_finish_) {
                    if(slot.observer->pending_prefix_) {
                        slot.observer->draining_prefix_failure_=true;
                        if(slot.observer->pending_prefix_->ticket)
                            try{slot.observer->pending_prefix_->ticket.cancel();}catch(...){}
                    } else slot.observer->cancel_selection();
                }
                slot.retiring=true;return;
            }
            if(!slot.observer->active_request_)slot.observer->clear_session();
            else slot.observer->retire(request);
            slot.request=0;slot.retiring=false;publish_monitor();return;
        }
    }
    void shutdown() noexcept override {
        for(auto& slot:slots_)slot.observer->shutdown();
        frames_.clear();slots_.clear();rank_dispatch_.clear();cache_->parking.clear();cache_->sessions.clear();
        cache_->bytes=0;cache_->active_archive_bytes=0;cache_->active_checkpoint_bytes=0;
    }
};
} // namespace kvmem::radiance
