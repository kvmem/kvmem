#pragma once

#include <cstdint>
#include <vector>

namespace kvmem {
class RawKvStore;
enum class GqaScoreAggregation : uint8_t { Mean = 0, Max = 1 };

// Borrowed post-RMSNorm/pre-RoPE sums. Layers match the statistics store's
// ordinals; each row contains query_heads * head_dim FP32 elements. A layer
// with no query tokens contributes no score. The caller owns prefix identity
// and commits only evaluated/accepted rows to the supplied statistics.
struct MeanKeyQueryView {
    const float* sums = nullptr;
    const uint32_t* token_counts = nullptr;
    uint32_t layers = 0;
    uint32_t head_dim = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    GqaScoreAggregation aggregation = GqaScoreAggregation::Mean;
};

// Mean mode averages Q over query rows and each GQA group before cosine.
// Max mode takes the largest per-Q-head cosine within each GQA group.
// Both average across KV heads and participating layers. FP32 dot/norm operations
// and epsilon 1e-6 preserve the existing llama product adapter's host policy.
// This is selection math only: it mutates no payload, frontier or residency.
std::vector<double> mean_key_retrieval_scores(const RawKvStore& keys, uint32_t blocks,
                                               MeanKeyQueryView query);

// Diversify independent query rankings by round-robin, skipping already emitted
// blocks. The caller still applies one shared top-k budget and mandatory union.
// One query (or one informative query) preserves its scores exactly. Otherwise,
// mandatory pages are excluded before interleaving. Constant-score queries are
// ignored; if none inform retrieval, rank by recency. Ties use block IDs.
// Interleaved ranks lie in (0,1]; no LRFU round is committed.
std::vector<double> diversify_retrieval_scores(const std::vector<std::vector<double>>& queries,
                                               const std::vector<uint32_t>& mandatory = {});
} // namespace kvmem
