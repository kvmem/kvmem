#pragma once
#include "kvmem/graph_recall.hpp"
#include "kvmem/lrfu.hpp"
#include "kvmem/query_policy.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/retrieval_options.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace kvmem::radiance {
// Ordinary prefill samples against the resident Pressure threshold. A fixed
// Query replay cannot select again while rebuilding its suffix: its admission
// already proves that selected history + suffix + workspace fit the actual
// bounded device pool. Its sampling geometry uses that same physical bound.
inline std::vector<uint32_t> prefill_attention_samples(uint64_t begin, uint32_t rows,
        uint64_t last_end, uint64_t visible, uint64_t pressure_capacity,
        uint64_t device_capacity, bool fixed_query_replay, uint32_t block) {
    return prefill_sample_rows(begin, rows, last_end, visible,
        fixed_query_replay ? device_capacity : pressure_capacity, 1024, 32, block);
}

struct RecallOptions {
    QueryScorer scorer = QueryScorer::PageSoftmax;
    int aggregation = 0;
    double graph_alpha = .2;
    bool lrfu = false;
    double lrfu_lambda = .5, lrfu_top_fraction = .9;
    uint32_t query_tokens = default_query_tokens;
    uint32_t query_replay_tokens = default_query_replay_tokens;
};

// One atomic statistical checkpoint, paired by the executor with native KV and
// recurrent state. Copies charge graph/feedback storage to the same account.
struct RecallSnapshot {
    GraphSnapshot graph;
    AttentionFeedbackSnapshot feedback;
    OutputSampleState output;
    LrfuState lrfu;
};

class RecallPolicy {
public:
    RecallPolicy(RecallOptions options, uint32_t block,
                 std::shared_ptr<GraphStatisticsBudget> global = {})
        : options_(options), history_(options.lrfu_lambda, options.lrfu_top_fraction) {
        validate_query_scoring(options.scorer, options.aggregation, options.graph_alpha);
        if (!block || !options.query_tokens || !options.query_replay_tokens)
            throw std::invalid_argument("zero recall block/query/replay limit");
        graph_.block_tokens = block;
        graph_.set_budget(std::make_shared<GraphStatisticsBudget>(32u << 20, std::move(global)));
    }
    const GraphSnapshot& graph() const noexcept { return graph_; }
    const AttentionFeedbackSnapshot& feedback() const noexcept { return feedback_.snapshot(); }
    OutputSampleState output() const noexcept { return output_; }
    size_t allocated_bytes() const {
        return sizeof(*this) + graph_.bytes() + feedback_.snapshot().bytes() + history_.allocated_bytes();
    }
    RecallSnapshot checkpoint() const { return {graph_, feedback_.snapshot(), output_, history_.checkpoint()}; }
    void restore(RecallSnapshot value) {
        if (value.graph.block_tokens != graph_.block_tokens || value.feedback.prefill_end > value.graph.accepted_through)
            throw std::invalid_argument("recall checkpoint frontier/geometry mismatch");
        AttentionFeedback feedback; feedback.restore(std::move(value.feedback));
        LrfuHistory history(options_.lrfu_lambda, options_.lrfu_top_fraction); history.restore(value.lrfu);
        value.graph.include_feedback(feedback.snapshot().bytes() + history.allocated_bytes());
        graph_ = std::move(value.graph); feedback_ = std::move(feedback);
        history_ = std::move(history); output_ = value.output;
    }
    void start_request(bool new_query, bool thinking) {
        auto next = feedback_; next.finish_output();
        graph_.include_feedback(next.snapshot().bytes() + history_.allocated_bytes());
        feedback_ = std::move(next); output_ = {thinking, 0};
        if (new_query && options_.lrfu) history_.begin_round();
    }
    void set_messages(const std::vector<GraphMessageSpan>& messages) { graph_.set_messages(messages); }

    // Native acceptance has been proven by the caller. Rejected/draft rows must
    // not be present. The entire new snapshot is prepared before publication.
    void accept(uint64_t begin, uint32_t accepted, bool prefill,
                const std::vector<AcceptedAttentionRow>& rows,
                const std::vector<uint64_t>& exposed, OutputSampleState output) {
        if (begin != graph_.accepted_through || begin > UINT64_MAX - accepted)
            throw std::invalid_argument("recall acceptance is not a contiguous target prefix");
        auto graph = prepare_graph_accept(graph_, begin, begin + accepted, rows, 32u << 20);
        auto feedback = feedback_;
        if (prefill) feedback.observe_prefill(begin, begin + accepted, accepted, rows, exposed);
        else for (const auto& row : rows) feedback.observe_output(row.mass);
        graph.include_feedback(feedback.snapshot().bytes() + history_.allocated_bytes());
        graph_ = std::move(graph); feedback_ = std::move(feedback); output_ = output;
    }
    std::vector<uint32_t> pressure(SelectionReason reason, const std::vector<uint32_t>& resident,
                                   const std::vector<uint32_t>& required, uint32_t budget) const {
        if (reason == SelectionReason::Query) throw std::invalid_argument("Query is not resident Pressure");
        for (auto id : required) if (std::find(resident.begin(), resident.end(), id) == resident.end())
            throw std::invalid_argument("Pressure mandatory page is not accepted/resident");
        std::vector<uint64_t> starts;
        for (auto id : resident) {
            const auto start = uint64_t(id) * graph_.block_tokens;
            if (start >= graph_.accepted_through) throw std::invalid_argument("Pressure page is outside accepted prefix");
            starts.push_back(start);
        }
        return pressure_selection(resident, feedback_.pressure_scores(starts), required, budget);
    }
    std::vector<uint32_t> query(const RawKvStore& store, PageQueryView q,
                               const QueryScoreFrame& frame, const std::vector<uint32_t>& required,
                               uint32_t budget, const std::vector<double>* accelerated = nullptr) {
        if (frame.accepted_through != graph_.accepted_through || !budget)
            throw std::invalid_argument("Query must use the checkpoint's frozen history");
        const auto agg = static_cast<GqaScoreAggregation>(options_.aggregation);
        auto scores = accelerated ? *accelerated : (options_.scorer == QueryScorer::PageSoftmax
            ? page_softmax_scores(store, q, frame, agg)
            : mean_query_cosine_scores(store, q, uint32_t(frame.starts.size()), agg));
        scores = propagate_query_scores(scores, frame, graph_, options_.graph_alpha);
        if (options_.lrfu) {
            history_.preview(scores);
            for (size_t i = 0; i < scores.size(); ++i) scores[i] = history_.score(i);
        }
        std::vector<uint32_t> picked = required, rank(scores.size());
        std::sort(picked.begin(), picked.end());
        picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
        if (picked.size() > budget || (!picked.empty() && picked.back() >= scores.size()))
            throw std::invalid_argument("Query required pages exceed frozen history/budget");
        std::iota(rank.begin(), rank.end(), 0);
        std::sort(rank.begin(), rank.end(), [&](auto a, auto b) {
            return scores[a] != scores[b] ? scores[a] > scores[b] : a > b;
        });
        for (auto id : rank) {
            if (picked.size() == budget) break;
            if (std::find(picked.begin(), picked.end(), id) == picked.end()) picked.push_back(id);
        }
        std::sort(picked.begin(), picked.end());
        graph_.include_feedback(feedback_.snapshot().bytes() + history_.allocated_bytes());
        return picked;
    }
    void commit_query() noexcept { if (options_.lrfu) history_.commit(); }
private:
    RecallOptions options_;
    GraphSnapshot graph_;
    AttentionFeedback feedback_;
    OutputSampleState output_;
    LrfuHistory history_;
};
} // namespace kvmem::radiance
