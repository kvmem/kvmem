#pragma once
#include "kvmem/historical_kv_policy.hpp"
#include "kvmem/historical_kv_page_features.hpp"
#include <array>
#include <mutex>
#include <string>

namespace kvmem {
inline constexpr uint32_t historical_kv_max_source_pages = 128;
inline constexpr uint32_t historical_kv_max_witness_pages = 8;
inline constexpr uint32_t historical_kv_duplicate_context_tokens = 8;

// One bit per result page for each full source page. A digit-bearing run and
// its local context must appear within a single retained witness page.
std::vector<uint8_t> historical_kv_literal_witnesses(const std::vector<int32_t>& source,
    const std::vector<uint8_t>& source_digits, const std::vector<int32_t>& result,
    uint32_t page_tokens);

// Evidence/retained-result design adapted from kvmem/kvmem-vllm@a0240f7c.
// The application supplies completion and real observation evidence. This is
// not an automatic thinking/summary similarity estimator. Publication/binding
// and boundary commit belong to the same serialized native request owner.
class HistoricalKvStepEvidence {
public:
    struct Config {
        uint32_t min_observations = 3, min_cold_observations = 3;
        double max_importance = 0.1;
        uint32_t max_steps = 64;
        uint32_t source_floor_tokens = 0; // permanent sink is not a retireable source
        // In this mode application spans are scopes, never deletion approvals.
        bool feature_gate = false;
        double similarity = 0.9;
        // Experimental content decision: actual output attention and Mean-K
        // similarity only. Literal masks do not veto; native lifetime guards stay.
        bool relaxed_attention_redundancy = false;
        uint32_t max_source_pages = historical_kv_max_source_pages;
        uint32_t max_result_pages = historical_kv_max_witness_pages;
        bool charge_result_pages = true;
        bool observe_only = false; // Evaluate proposals without changing recall or pressure eligibility.
    };
    struct Update {
        std::string id, kind;
        uint32_t source_begin = 0, source_end = 0, result_begin = 0, result_end = 0;
        uint64_t observation_id = 0;
        double importance = 1.0;
        bool cold = false, complete = false;
        // Derived from accepted tokens by the backend. Empty retains the
        // feature-only behavior for producers without a literal guard.
        std::vector<uint8_t> witness_masks = {};
        // Verified position of the copy of source_begin in a later tool response.
        uint32_t duplicate_begin = 0;
    };
    HistoricalKvStepEvidence(uint32_t logical_pages, uint32_t page_tokens, Config config);
    // Validates an entire batch before changing anything. Equal observation IDs
    // are idempotent only for identical data; older/conflicting reports fail.
    void publish(const std::vector<Update>& updates, uint32_t accepted_tokens);
    void reset_bindings();
    void bind(uint32_t page, HistoricalKvHandle handle);
    struct RetainedWitness {
        uint32_t page = 0, result_end = 0; // Zero end means no approved witness.
        uint32_t page_count = 1;
    };
    // The serialized boundary owner calls this only after repository retirement.
    // Pending CPU readers do not make logically retired KV selectable again.
    RetainedWitness retired(uint32_t page, HistoricalKvHandle handle) noexcept;
    void truncate_to(uint32_t accepted_tokens);
    bool result_page(uint32_t page) const;
    // Pressure can reclaim advisory sources and witnesses. Only owner-published
    // marks and their retained witnesses are excluded from the fallback.
    bool capacity_reclaimable(const HistoricalKvBlockMeta&) const noexcept;
    // OR the retained-result protection bit into an existing owner bitmap.
    // One locked range walk avoids one mutex and one span scan per history page.
    void mark_result_pages(std::vector<uint8_t>& flags, uint8_t bit) const;
    uint64_t result_protection_revision() const noexcept {
        return result_protection_revision_.load(std::memory_order_acquire);
    }
    bool eligible(uint32_t page) const;
    bool handoff_candidate(uint32_t page) const;
    bool selection_excluded(uint32_t page) const;
    bool permits_selection(uint32_t page, bool mandatory = false) const;
    // Only an owner-published mark supplies a witness. A natural checkpoint
    // restore may omit its source before the previous reader's Host release;
    // the caller must still validate witness coverage in that checkpoint.
    RetainedWitness selection_witness(uint32_t page) const;
    struct Stats {
        size_t tracked = 0;
        uint64_t completed = 0, evicted_evidence = 0, ignored_retired_updates = 0;
    };
    Stats stats() const;
    uint64_t revision(const HistoricalKvBlockMeta& block) const noexcept;
    // Worker-only, one bounded step at a time, with no GPU/model calls. The
    // expensive feature loop runs outside the boundary owner's evidence lock.
    void evaluate(const std::shared_ptr<HistoricalKvPageFeatures>&, const HistoricalKvSnapshot&);
    // Native serialized owner only. Publish an irreversible selection mark at
    // a safe boundary; physical retirement can wait for the old active reader.
    struct SelectionGuard {
        uint64_t revision;
        void* context;
        // Runs synchronously under the evidence lock on the owner thread.
        bool (*permits)(void*, uint32_t source, RetainedWitness witness) noexcept;
    };
    size_t commit_selection_boundary(const std::vector<uint8_t>& protections,
                                     uint8_t allowed_protections = 0,
                                     uint64_t protection_epoch = 0,
                                     const SelectionGuard* guard = nullptr);
    struct Status { std::string id, state; uint32_t selected = 0, retired = 0; };
    std::vector<Status> status() const;
    uint64_t feature_comparisons() const noexcept { return comparisons_.load(std::memory_order_relaxed); }
private:
    struct Selection {
        uint32_t page, witness;
        HistoricalKvHandle handle;
        uint32_t witness_pages = 1;
    };
    struct Mark {
        uint64_t revision = 0;
        RetainedWitness witness;
    };
    struct Step {
        Update value;
        uint64_t revision = 0;
        uint32_t observations = 0, cold = 0, result_bound = 0, source_remaining = 0;
        bool evaluated = false;
        std::vector<Selection> selected;
        std::array<HistoricalKvHandle, historical_kv_max_witness_pages> witnesses{};
        std::vector<uint32_t> missing_observations;
        uint32_t marked = 0;
        uint32_t retired_count = 0;
        uint64_t feature_epoch = 0;
        uint64_t observation_revision = 0;
        uint64_t checked_protection_epoch = 0;
        uint64_t checked_guard_revision = 0;
    };
    uint32_t first_full_page(uint32_t begin) const noexcept;
    bool source(const Step&, uint32_t page) const;
    bool result(const Step&, uint32_t page) const;
    void invalidate_proposals(Step&, uint32_t page);
    bool release_mark(uint32_t page) noexcept;
    bool eligible_locked(uint32_t page, bool source_bound = true, bool result_bound = true) const;
    const uint32_t pages_, tokens_;
    const Config config_;
    mutable std::mutex mutex_;
    uint64_t next_revision_ = 1;
    std::vector<Step> steps_;
    std::vector<HistoricalKvHandle> bindings_;
    std::vector<Mark> marks_;
    std::vector<uint16_t> witness_refs_;
    // Context-bounded tombstones survive directory and repository-slot reuse.
    std::vector<uint8_t> retired_pages_;
    uint64_t completed_ = 0, evicted_evidence_ = 0, ignored_retired_updates_ = 0;
    std::vector<Status> finished_; // bounded acknowledgments, not deletion evidence
    std::weak_ptr<HistoricalKvPageFeatures> features_;
    std::atomic<uint64_t> result_protection_revision_{0};
    std::atomic<uint64_t> comparisons_{0};
};

class HistoricalKvStatisticalPolicy final : public HistoricalKvPolicy {
public:
    explicit HistoricalKvStatisticalPolicy(std::shared_ptr<HistoricalKvStepEvidence> evidence,
        std::shared_ptr<HistoricalKvPageFeatures> features = {}, uint32_t pressure_capacity = 0);
    bool permits(const HistoricalKvBlockMeta&) const noexcept override;
    bool permits_intent(const HistoricalKvBlockMeta&, const HistoricalKvDeleteIntent&) const noexcept override;
    HistoricalKvStepResult step(const HistoricalKvSnapshot&, const HistoricalKvPressure&,
        HistoricalKvWorkBudget, HistoricalKvDecisionBuffer&) override;
    Metrics metrics() const noexcept override;
private:
    std::shared_ptr<HistoricalKvStepEvidence> evidence_;
    std::shared_ptr<HistoricalKvPageFeatures> features_;
    uint64_t incarnation_ = 0, round_ = 0;
    size_t cursor_ = 0;
    const uint32_t pressure_capacity_;
    bool pressure_mode_ = false, ranked_ = false;
    size_t capacity_cursor_ = 0;
    std::vector<size_t> capacity_candidates_;
    std::atomic<uint64_t> lossy_intents_{0};
    bool audit_ = false;
};
} // namespace kvmem
