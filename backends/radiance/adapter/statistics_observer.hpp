#pragma once
#include "execution_options.hpp"
#include "statistics_capture.hpp"
#include "runtime/step_observer.h"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/kvmem_store.hpp"
#include "rank_native_memory.hpp"
#include "state_checkpoint.hpp"
#include "rank_prefix.hpp"
#include "prefix_directory.hpp"
#include "query_identity.hpp"
#include "kvmem/checkpoint_policy.hpp"
#include "image_contract.hpp"
#include "canonical_input.hpp"
#include "attention_capture.hpp"
#include "recall_policy.hpp"
#include "kvmem/page_softmax_cuda.hpp"
#include <atomic>
#include <algorithm>
#include <deque>
#include <chrono>
#include <set>

namespace kvmem::radiance {
class BatchedStatisticsObserver;
struct SelectionOptions : RecallOptions {
    std::string drafter_mode = "auto";
    uint32_t max_draft_tokens = 7;
    bool rolling_generation = true;
    uint32_t budget_tokens = 0; // zero: observation-only probe
    uint32_t block_tokens = 128;
    uint64_t host_bytes = 256ull * 1024 * 1024;
    uint64_t disk_bytes = 0;
    std::string disk_path;
    uint64_t history_bytes() const {
        if(disk_bytes>UINT64_MAX-host_bytes)throw std::invalid_argument("history storage capacity overflow");
        return host_bytes+disk_bytes;
    }
    uint32_t cache_sessions = 8; // content groups, not checkpoint count
    uint32_t cache_records = 0; // zero follows cache_sessions; never a hidden floor
    uint32_t effective_cache_records() const noexcept {return cache_sessions ? (cache_records ? cache_records : cache_sessions) : 0;}
    uint32_t replay_tokens = 1024; // generation/verify reserve; prefill is blockwise
    int64_t sink_tokens = -1;   // shared KVMem automatic allocation
    int64_t recent_tokens = -1; // explicit nonnegative values override auto
};
// Engine-lifecycle bridge. Owns bounded pre-RoPE capture and per-session mean-K
// statistics. With a nonzero selection budget, composes the common KVMem policy
// and native memory transactions to publish original-position sparse KV views.
// Exact serial continuations retain native pages and matching recurrent state.
// Drafter proposals remain engine-owned; only verified trunk rows enter
// the shared statistics. BatchedStatisticsObserver multiplexes these controllers;
// the engine-owned prefix cache is disabled in favor of the shared KVMem cache.
class StatisticsObserver final : public rad::StepObserver {
public:
    bool early_collect() const noexcept override { return early_statistics_collection(); }
    explicit StatisticsObserver(SelectionOptions options = {}) : options_(options) {cache_->observers.insert(this);}
    ~StatisticsObserver() override {cache_->observers.erase(this);}
    bool retain_history_boundary() const noexcept override { return options_.budget_tokens > 0; }
    int pool_token_limit(const rad::Config&, const rad::Program&, int64_t*) noexcept override;
    int configure(const rad::Config&, rad::CtxDesc&) noexcept override;
    int bind_rank(rad::RankResources) noexcept override;
    int resume(const rad::StepPlan&, int64_t*, std::string*) noexcept override;
    int begin(const rad::StepPlan&) noexcept override;
    bool split_sampling(const rad::StepPlan& plan) const noexcept override {
        return plan.e.size() == 1 && plan.e[0].is_prefill;
    }
    std::string validate_request(const rad::Request&) const override;
    int after_forward(const rad::StepPlan&, int64_t*) noexcept override;
    int collect(const rad::StepPlan&, const rad::Scheduler::StepResult&) noexcept override;
    int finish(const rad::StepEntry&, const rad::Request&) noexcept override;
    void retire(uint64_t) noexcept override;
    void shutdown() noexcept override;
private:
    friend class BatchedStatisticsObserver;
    struct Rank {
        int device=-1;
        uint32_t query_heads=0;
        rad::RankResources resources;
        GdnMaterialize materialize_gdn=nullptr;
        std::vector<rad::KVGroupInfo> groups;
        std::vector<int> layers;
        std::unique_ptr<StatisticsCapture> capture;
        std::unique_ptr<AttentionCapture> attention;
    };
    std::vector<std::unique_ptr<Rank>> ranks_;
    uint32_t world_size_=0;
    void drain_ranks(const char* reason);
    void rollback_ranks(uint64_t sequence,uint64_t frontier,const char* reason);
    void verify_rank_view(uint64_t sequence) const;
    std::unique_ptr<PageSoftmaxCuda> query_gpu_;
    std::unique_ptr<RecallPolicy> recall_;
    std::unique_ptr<RecallSnapshot> checkpoint_recall_, prefix_recall_;
    std::shared_ptr<GraphStatisticsBudget> graph_budget_ = std::make_shared<GraphStatisticsBudget>(256u << 20);
    std::vector<std::vector<float>> query_tokens_;
    std::vector<int32_t> step_tokens_;
    std::vector<uint32_t> sample_rows_;
    std::vector<uint64_t> exposed_;
    MemorySnapshot accepted_view_; // frozen before the scheduler reserves the next batch
    std::unique_ptr<RawKvStore> store_;
    RawKvStoreConfig store_config_;
    SessionIdentity session_{};
    uint64_t active_request_ = 0, cache_group_ = 0;
    struct CachedIdentity {
        std::vector<int32_t> tokens;
        ImageIdentities images;
        std::string scope;
        uint64_t input_end=0; // Native input boundary, before generated output.
        std::shared_ptr<const NativeCanonicalInput<ImageIdentity>> canonical;
        bool fallback = false;
        std::vector<int32_t> query_tokens;
        int32_t query_begin = -1, query_end = -1;
        bool prefix_matches(const rad::Request& r) const {
            return r.prompt.size() > tokens.size() &&
                std::equal(tokens.begin(), tokens.end(), r.prompt.begin()) &&
                image_prefix_matches(images,r.media.get(),r.prompt.size(),tokens.size());
        }
        bool continues_input(const rad::Request& r) const {
            const auto n=std::min(tokens.size(),r.prompt.size());
            const auto common=size_t(std::mismatch(tokens.begin(),tokens.begin()+n,r.prompt.begin()).first-tokens.begin());
            return input_end>0 && checkpoint_anonymous_continuation(common,input_end,tokens.size()) &&
                image_prefix_matches(images,r.media.get(),r.prompt.size(),std::min<uint64_t>(common,input_end));
        }
        bool continues_group_input(const rad::Request& r) const {
            return canonical && !canonical->tokens.empty() && r.prompt.size()>=canonical->tokens.size() &&
                std::equal(canonical->tokens.begin(),canonical->tokens.end(),r.prompt.begin()) &&
                image_prefix_matches(canonical->images,r.media.get(),r.prompt.size(),canonical->tokens.size());
        }
        // Client labels do not own execution state. Exact native content and
        // media positions remain required by the continuation path.
        bool matches(const rad::Request& r) const {return prefix_matches(r);}
    };
    ImageIdentities images_; // immutable layout of the active native request
    std::shared_ptr<const CachedIdentity> cached_identity_; // atomic immutable HTTP snapshot
    std::unique_ptr<RankPrefix> prefix_;
    std::vector<float> prefix_mean_;
    struct RetainedSession {
        SessionIdentity session;
        uint64_t content_version = 0, bytes = 0, group = 0; // Directory fairness metadata, never native identity.
        std::shared_ptr<const CachedIdentity> identity;
        std::unique_ptr<RankPrefix> prefix;
        std::vector<float> mean;
        std::unique_ptr<RawKvStore> store;
        std::unique_ptr<KvMemStore> policy;
        std::shared_ptr<RankNativeMemory> memory;
        std::unique_ptr<RecallPolicy> recall;
        std::unique_ptr<RecallSnapshot> recall_snapshot;
        // One LRU conversation, two native restore points. Immutable archive
        // records are shared; mutable catalogs and statistics are independent.
        std::unique_ptr<RetainedSession> input, fallback;
    };
    std::unique_ptr<RetainedSession> input_, fallback_;
    struct ParkWork {
        std::unique_ptr<RankPrefix> prefix;
        std::shared_ptr<HostKvStorage> storage;
        void run(){storage_operation([&]{prefix->park(storage,{});});}
    };
    struct PendingPark {
        uint64_t id=0;
        RetainedSession saved;
        std::shared_ptr<ParkWork> work;
        StorageTicket ticket;
        ~PendingPark(){if(ticket){ticket.cancel();try{ticket.wait();}catch(...){}}}
    };
    struct CacheState {
        std::shared_ptr<HostKvStorage> payload;
        std::shared_ptr<StorageIoQueue> io;
        uint64_t hits=0,evictions=0,forks=0,cross_owner_forks=0;
        uint64_t capture_bytes=0,capture_logical_bytes=0;
        uint64_t actual_own_hits=0,actual_shared_hits=0,actual_misses=0;
        std::deque<RetainedSession> sessions; // LRU, owns no live native sequence
        std::vector<std::unique_ptr<PendingPark>> parking;
        struct Finishing {uint64_t id;std::shared_ptr<const CachedIdentity> identity, input_identity, fallback_identity;};
        std::vector<Finishing> finishing; // bounded by active native sequence count
        uint64_t next_park=1,park_errors=0,next_group=1;
        uint64_t allocate_group() {
            if(!next_group || next_group==UINT64_MAX)throw std::overflow_error("native cache group identity exhausted");
            return next_group++;
        }
        size_t recovery_points() const noexcept {
            size_t n=0;for(const auto& saved:sessions)n+=1+bool(saved.input)+bool(saved.fallback);return n;
        }
        size_t count() const noexcept {return sessions.size()+parking.size();}
        size_t groups() const {
            std::set<uint64_t> origins;
            for(const auto& saved:sessions)origins.insert(saved.group ? saved.group : saved.session.id);
            return origins.size();
        }
        bool pending(uint64_t id) const noexcept {
            return std::any_of(parking.begin(),parking.end(),[&](const auto& x){return x->id==id;}) ||
                std::any_of(finishing.begin(),finishing.end(),[&](const auto& x){return x.id==id;});
        }
        bool publish(RetainedSession&&,size_t max_groups,size_t max_records,bool already_charged=false);
        uint64_t bytes = 0, active_archive_bytes = 0, active_checkpoint_bytes = 0;
        std::set<const StatisticsObserver*> observers; // scheduler-thread ownership only
        uint64_t sharing_credit(const RetainedSession* extra=nullptr,const RankPrefix* incoming=nullptr) const;
    };
    std::shared_ptr<CacheState> cache_ = std::make_shared<CacheState>();
    uint64_t archive_charge_ = 0;
    uint64_t input_charge_ = 0;
    uint64_t history_allowance(const RetainedSession* extra=nullptr,const RankPrefix* incoming=nullptr) const;
    uint64_t next_incarnation_ = 1, steps_ = 0;
    uint32_t frontier_ = 0, max_position_ = 0;
    std::vector<double> query_total_;
    std::vector<uint64_t> query_rows_;
    std::vector<uint32_t> query_counts_;
    int32_t query_begin_ = -1, query_end_ = -1;
    bool query_frozen_ = false;
    // Native serial heads write their own layer in the trunk's FULL KV group
    // AFTER the target pass. Do not spill/lease those pages until it finishes.
    bool serial_drafter_ = false;
    bool defer_prefill_storage_ = false;
    uint32_t block_drafter_window_ = 0;
    bool prefill_collected_ = false, deferred_selection_ = false;
    bool replaying_ = false, replay_done_ = false;
    uint32_t replay_start_ = 0;
    uint64_t content_version_ = 0;
    std::unique_ptr<RankCheckpoint> checkpoint_;
    std::vector<float> checkpoint_mean_;
    SelectionOptions options_;
    KvMemKeepAllocation keep_;
    BackendIdentity identity_;
    int device_ = -1;
    uint32_t max_rows_ = 0, head_dim_ = 0, query_heads_ = 0, kv_heads_ = 0;
    uint64_t device_tokens_ = UINT64_MAX, reselections_ = 0, spilled_ = 0, restored_ = 0;
    std::unique_ptr<KvMemStore> policy_;
    std::shared_ptr<RankNativeMemory> memory_;
    std::unique_ptr<MemorySession> memory_session_;
    struct PendingSelection {
        SelectionReason reason;
        size_t spill,restore;
        std::chrono::steady_clock::time_point deadline;
        std::shared_ptr<RankMemory> storage;
    };
    std::optional<PendingSelection> pending_selection_;
    bool draining_selection_failure_=false;
    bool draining_storage_failure() const noexcept {return draining_prefix_failure_ || draining_selection_failure_;}
    void cancel_selection() noexcept;
    bool asynchronous_=false; // enabled only by a wrapper implementing scheduler holds
    struct PrefixReadWork {
        std::unique_ptr<RankPrefix> source;
        RankPrefix::Payload payload;
        void run(){storage_operation([&]{payload=source->load_cpu();});}
    };
    struct PrefixRead {
        std::shared_ptr<PrefixReadWork> work;
        StorageTicket ticket;
        std::chrono::steady_clock::time_point deadline;
        ~PrefixRead(){if(ticket){ticket.cancel();try{ticket.wait();}catch(...){}}}
    };
    std::unique_ptr<PrefixRead> pending_prefix_;
    bool draining_prefix_failure_=false;
    std::unique_ptr<RankPrefix::Payload> prefix_payload_;
    struct TerminalSnapshot {
        std::shared_ptr<const CachedIdentity> identity;
        uint64_t request=0;
        uint32_t accepted_rows=0;
        size_t output_tokens=0;
    };
    std::unique_ptr<TerminalSnapshot> pending_finish_;
    std::shared_ptr<const CachedIdentity> pending_input_;
    std::unique_ptr<PendingPark> input_parking_;
    uint64_t finishing_id_=0;
    void capture_terminal(const TerminalSnapshot&);
    void retain_input(const rad::StepEntry&);
    void capture_input(std::shared_ptr<const CachedIdentity>);
    bool poll_input();
    void update_input_charge(uint64_t);
    uint64_t input_extra_bytes() const;
    void promote_input();
    std::unique_ptr<RetainedSession> carry_input(const RetainedSession&, const RetainedSession&, const rad::Request&,bool fallback=false);
    void admit_carried_input(std::unique_ptr<RetainedSession>);
    static uint64_t retained_bytes(const RetainedSession&);
    RetainedSession* select_checkpoint(RetainedSession&, const rad::Request&) const;
    uint64_t restore_limit(const rad::Request&) const;
    uint64_t pending_cache_=0;
    bool pending_io() const noexcept {return pending_cache_ || bool(pending_prefix_) || bool(pending_selection_) || bool(pending_finish_) || bool(pending_input_) || bool(input_parking_);}
    bool poll_prefix();
    bool poll_selection(bool wait=false);
    int poll_pending_selection() noexcept;
    uint64_t selected_frontier_ = 0, selected_visible_ = 0, decode_reselections_ = 0;
    void reselect(SelectionReason);
    void publish_selection(const std::vector<uint32_t>&, std::vector<uint32_t>, SelectionReason);
    void start_request(const rad::Request&);
    PageQueryView query_view(std::vector<const float*>&) const;
    void begin_attention(const rad::StepEntry&);
    void sync_memory();
    void clear_session();
    void park_session();
    void evict_session();
    void activate_session(const rad::Request&);
    void update_archive_charge();
    std::string fresh_reserve_error(const rad::Request&) const;
};
} // namespace kvmem::radiance
