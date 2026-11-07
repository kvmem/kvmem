#pragma once

#include "graph_recall.hpp"

namespace kvmem {

// Borrowed current-forward, post-normalization/pre-RoPE rows from exactly one
// target attention layer. `layer` is the RawKvStore statistics-layer ordinal;
// adapters resolve native model layer identities before constructing this view.
// The owner proves request/frontier identity and keeps both rows and the frozen
// score frame alive through scoring. No Query or scores are cached here.
struct CurrentLayerQuery {
    const float* data = nullptr; // contiguous [rows, query_heads, head_dim]
    uint32_t layer = 0;
    uint32_t rows = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    float attention_scale = 0; // zero selects 1/sqrt(head_dim)
};

// kvmem-vllm 72a81cc layerwise Mean-K: page softmax per Query row/head,
// sum Query rows, mean Query heads. Only frame.score_mask defines the softmax
// domain; mandatory/transport_mask never remove logits by themselves. Operands
// use the existing explicit FP16 boundary. Other layers contribute no samples.
// Every frame entry must have exactly its accepted count of target-layer
// Mean-K statistics, including fixed/masked entries; absent or future-tail
// statistics cannot stand in for the frozen accepted layer index.
// This pure scorer performs no graph propagation, LRFU fusion or selection.
std::vector<double> layerwise_mean_key_scores(const RawKvStore&,
    CurrentLayerQuery, const QueryScoreFrame&);

} // namespace kvmem
