#pragma once
#include "statistics_capture.hpp"
#include "runtime/step_observer.h"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/kvmem_store.hpp"
#include "native_memory.hpp"
#include "state_checkpoint.hpp"
#include "prefix_lease.hpp"
#include "query_identity.hpp"
#include <atomic>
#include <algorithm>

namespace kvmem::radiance {
struct SelectionOptions {
    bool rolling_generation = false; // opt-in bounded DecodePressure backport
    uint32_t budget_tokens = 0; // zero: observation-only probe
    uint32_t block_tokens = 128;
    uint64_t host_bytes = 256ull * 1024 * 1024;
    uint32_t replay_tokens = 1024; // generation/verify reserve; prefill is blockwise
    int64_t sink_tokens = -1;   // shared KVMem automatic allocation
    int64_t recent_tokens = -1; // explicit nonnegative values override auto
};
// Engine-lifecycle bridge. Owns bounded pre-RoPE capture and per-session mean-K
// statistics. With a nonzero selection budget, composes the common KVMem policy
// and native memory transactions to publish original-position sparse KV views.
// Exact serial continuations retain native pages and matching recurrent state.
// Drafter proposals remain engine-owned; only verified trunk rows enter
// the shared statistics. Concurrency and engine-owned prefix cache are rejected.
class StatisticsObserver final : public rad::StepObserver {
public:
    explicit StatisticsObserver(SelectionOptions options = {}) : options_(options) {}
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
    void retire(uint64_t) noexcept override;
    void shutdown() noexcept override;
private:
    std::unique_ptr<StatisticsCapture> capture_;
    std::unique_ptr<RawKvStore> store_;
    RawKvStoreConfig store_config_;
    SessionIdentity session_{};
    uint64_t active_request_ = 0;
    struct CachedIdentity {
        std::vector<int32_t> tokens;
        std::vector<int32_t> query_tokens;
        int32_t query_begin = -1, query_end = -1;
        bool matches(const rad::Request& r) const {
            return !r.media && matches_query_identity(query_tokens, query_begin, query_end,
                                                     r.prompt, r.query_begin, r.query_end) &&
                r.prompt.size() > tokens.size() &&
                std::equal(tokens.begin(), tokens.end(), r.prompt.begin());
        }
    };
    std::shared_ptr<const CachedIdentity> cached_identity_; // atomic immutable HTTP snapshot
    std::unique_ptr<PrefixLease> prefix_;
    std::vector<float> prefix_mean_;
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
    bool prefill_collected_ = false, deferred_selection_ = false;
    bool replaying_ = false, replay_done_ = false;
    uint32_t replay_start_ = 0;
    uint64_t content_version_ = 0;
    std::unique_ptr<StateCheckpoint> checkpoint_;
    SelectionOptions options_;
    KvMemKeepAllocation keep_;
    rad::RankResources resources_;
    std::vector<rad::KVGroupInfo> groups_;
    BackendIdentity identity_;
    int device_ = -1;
    uint32_t max_rows_ = 0, head_dim_ = 0, query_heads_ = 0, kv_heads_ = 0;
    uint64_t device_tokens_ = 0, reselections_ = 0, spilled_ = 0, restored_ = 0;
    std::unique_ptr<KvMemStore> policy_;
    std::shared_ptr<NativeMemory> memory_;
    std::unique_ptr<MemorySession> memory_session_;
    uint64_t selected_frontier_ = 0, selected_visible_ = 0, decode_reselections_ = 0;
    void reselect(bool decode_pressure = false);
    void sync_memory();
    void clear_session();
    void retain_prefix(const rad::StepEntry&);
    std::string fresh_reserve_error(const rad::Request&) const;
};
} // namespace kvmem::radiance
