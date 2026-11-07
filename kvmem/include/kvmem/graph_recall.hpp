#pragma once

#include "mean_key_retrieval.hpp"
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <utility>
#include <vector>

namespace kvmem {
class RawKvStore;

enum class SelectionReason : uint8_t { Query, PrefillPressure, DecodePressure };

// One accepted logical index. Physical slots never identify persistent graph nodes.
// The owner must freeze the index and picker budget until selection publishes.
struct QueryScoreFrame {
    std::vector<uint64_t> starts;
    std::vector<uint32_t> counts;
    std::vector<uint8_t> score_mask;
    std::vector<uint8_t> transport_mask;
    uint64_t accepted_through = 0;
};

QueryScoreFrame query_score_frame(uint64_t accepted_through, uint32_t block_tokens,
    uint32_t budget_blocks, uint32_t sink_blocks, uint32_t recent_blocks,
    const std::vector<uint32_t>& fixed_blocks);

// Post-normalization, pre-RoPE Q, layer/token/head/channel order. Rows beyond
// layer_rows[layer] are absent (e.g. recurrent layers), never zero-valued samples.
struct PageQueryView {
    const float* data = nullptr;
    const uint32_t* layer_rows = nullptr;
    uint32_t layers = 0, rows = 0, query_heads = 0, kv_heads = 0, head_dim = 0;
    const float* const* layer_data = nullptr; // optional disjoint per-layer storage
    float attention_scale = 0; // 0 selects 1/sqrt(head_dim); Gemma uses 1
};

// Explicit FP16 operand boundary shared by CPU qualification and CUDA staging.
float recall_fp16(float value);
std::vector<double> page_softmax_scores(const RawKvStore&, PageQueryView,
                                      const QueryScoreFrame&, GqaScoreAggregation = GqaScoreAggregation::Mean);
std::vector<double> mean_query_cosine_scores(const RawKvStore&, PageQueryView,
                                           uint32_t blocks, GqaScoreAggregation);

struct GraphEdge { uint64_t target = 0; double mass = 0; };
struct GraphRow {
    uint64_t source = 0, samples = 0;
    std::vector<GraphEdge> edges;
};
struct GraphMessageSpan { uint64_t begin = 0, end = 0; bool end_known = true; };
struct GraphPartial {
    uint64_t source = 0, samples = 0;
    std::vector<GraphEdge> mass;
};
class GraphBudgetLease;
// Each snapshot charges its private state. Shared immutable allocations are
// charged once in each budget that uses them, including common ancestors.
class GraphStatisticsBudget : public std::enable_shared_from_this<GraphStatisticsBudget> {
public:
    explicit GraphStatisticsBudget(size_t limit,std::shared_ptr<GraphStatisticsBudget> parent={})
        : limit_(limit),parent_(std::move(parent)) {}
    size_t used() const noexcept {return used_.load();}
private:
    friend class GraphBudgetLease;
    void acquire(size_t);
    void release(size_t bytes) noexcept {used_.fetch_sub(bytes);}
    size_t limit_;
    std::atomic<size_t> used_{0};
    std::shared_ptr<GraphStatisticsBudget> parent_;
};
struct GraphSnapshot {
private:
    std::shared_ptr<GraphStatisticsBudget> budget_;
    std::shared_ptr<GraphBudgetLease> lease_;
    size_t feedback_bytes_=0;
public:
    GraphSnapshot()=default;
    GraphSnapshot(const GraphSnapshot&);
    GraphSnapshot(GraphSnapshot&&) noexcept=default;
    GraphSnapshot& operator=(GraphSnapshot&&) noexcept;
    GraphSnapshot& operator=(const GraphSnapshot& other) {
        if(this!=&other) {GraphSnapshot copy(other);*this=std::move(copy);}return *this;
    }
    void set_budget(std::shared_ptr<GraphStatisticsBudget>);
    std::shared_ptr<GraphStatisticsBudget> budget() const {return budget_;}
    void reserve_extra(size_t);
    void include_feedback(size_t);
    void set_messages(const std::vector<GraphMessageSpan>&);
    void set_messages(std::shared_ptr<const std::vector<GraphMessageSpan>>);
    uint64_t accepted_through = 0, completed_through = 0, revision = 0, epoch = 0;
    uint32_t block_tokens = 64;
    // Immutable chunks avoid copying the whole history on accepted decode rows.
    std::vector<std::shared_ptr<const std::vector<GraphRow>>> chunks;
    GraphPartial partial;
    std::shared_ptr<const std::vector<GraphMessageSpan>> messages;
    size_t bytes() const;
};
struct AcceptedAttentionRow {
    uint64_t source_position = 0;
    std::vector<GraphEdge> mass; // complete active-window mass, including self
};
struct OutputSampleState { bool thinking = false; uint64_t eligible_tokens = 0; };
struct OutputSamplePlan { OutputSampleState state; std::vector<uint32_t> rows; };
OutputSamplePlan accepted_output_samples(OutputSampleState, const int32_t* tokens,
    uint32_t accepted, int32_t think_open, int32_t think_close, uint32_t interval = 16);
std::vector<uint32_t> prefill_sample_rows(uint64_t begin, uint32_t count, uint64_t last_end,
    uint64_t visible, uint64_t capacity, uint32_t interval = 1024, uint32_t stride = 32,
    uint32_t block_tokens = 64);

// Prepare is pure. A caller publishes only after the corresponding native KV
// acceptance and other statistical preparations have all succeeded.
GraphSnapshot prepare_graph_accept(const GraphSnapshot&, uint64_t begin, uint64_t end,
    const std::vector<AcceptedAttentionRow>&, size_t max_bytes, uint32_t topk = 4);
// One hop over current-query positive mass, with signed negative residuals kept
// at their source: min(score,0) + G(max(score,0)). Alpha zero is an exact bypass.
// Apply before LRFU preview; never propagate accumulated history again.
std::vector<double> propagate_query_scores(const std::vector<double>&,
    const QueryScoreFrame&, const GraphSnapshot&, double alpha = .2,
    double cross_message_weight = .25);

struct PrefillMass { double mass = 0, exposure = 0; };
struct AttentionFeedbackSnapshot {
    std::vector<std::pair<uint64_t, double>> output;
    std::vector<std::pair<uint64_t, double>> pending_output;
    std::vector<std::pair<uint64_t, PrefillMass>> prefill;
    uint64_t output_samples = 0, prefill_end = 0;
    size_t bytes() const noexcept {
        return output.capacity()*sizeof(output[0]) + pending_output.capacity()*sizeof(pending_output[0]) +
            prefill.capacity()*sizeof(prefill[0]);
    }
};

// Logical identities only. Snapshots also retain accepted pending output, so
// restoring a checkpoint cannot retain contributions from a discarded suffix.
class AttentionFeedback {
public:
    const AttentionFeedbackSnapshot& snapshot() const noexcept { return state_; }
    void restore(AttentionFeedbackSnapshot);
    void observe_output(const std::vector<GraphEdge>&);
    void finish_output(double decay = .5);
    void observe_prefill(uint64_t begin, uint64_t end, uint32_t accepted,
        const std::vector<AcceptedAttentionRow>&, const std::vector<uint64_t>& exposed,
        double half_life = 8192);
    std::vector<double> pressure_scores(const std::vector<uint64_t>& candidates,
                                       double decay = .5) const;
private:
    AttentionFeedbackSnapshot state_;
};

// Candidates are the frozen resident accepted working set. Required incoming
// pages belong in mandatory; they must never become recall candidates by accident.
std::vector<uint32_t> pressure_selection(const std::vector<uint32_t>& candidates,
    const std::vector<double>& scores, const std::vector<uint32_t>& mandatory,
    uint32_t budget_blocks);
} // namespace kvmem
