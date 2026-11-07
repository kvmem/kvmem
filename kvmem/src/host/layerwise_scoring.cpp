#include "kvmem/layerwise_scoring.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace kvmem {

std::vector<double> layerwise_mean_key_scores(const RawKvStore& keys,
    CurrentLayerQuery query, const QueryScoreFrame& frame) {
    const auto layers = keys.config().n_layer;
    const uint64_t width = uint64_t(query.query_heads) * query.head_dim;
    if (!query.data || !query.rows || !query.query_heads || !query.kv_heads ||
        !query.head_dim || query.layer >= layers ||
        query.query_heads % query.kv_heads ||
        uint64_t(query.kv_heads) * query.head_dim != keys.config().n_embd_k ||
        width > std::numeric_limits<size_t>::max() / query.rows / sizeof(float)) {
        throw std::invalid_argument("layerwise Mean-K Query geometry mismatch");
    }
    if (!std::isfinite(query.attention_scale) || query.attention_scale < 0) {
        throw std::invalid_argument("layerwise Mean-K attention scale must be finite and nonnegative");
    }
    const auto blocks = frame.starts.size();
    const auto block_tokens = keys.config().block_tokens;
    if (!block_tokens || frame.counts.size() != blocks ||
        frame.score_mask.size() != blocks || frame.transport_mask.size() != blocks) {
        throw std::invalid_argument("layerwise Mean-K score frame geometry mismatch");
    }
    for (size_t i = 0; i < blocks; ++i) {
        const auto start = frame.starts[i];
        const auto count = frame.counts[i];
        if (!count || count > block_tokens || start % block_tokens ||
            start > frame.accepted_through || count > frame.accepted_through - start ||
            start / block_tokens > std::numeric_limits<uint32_t>::max() ||
            keys.mean_k_tokens(uint32_t(start / block_tokens), query.layer) != count) {
            throw std::invalid_argument("layerwise Mean-K target-layer statistics do not match accepted coverage");
        }
    }
    // Validate even an empty candidate domain. A missing domain is not a reason
    // to accept malformed native Query rows or silently replace their values.
    for (uint64_t i = 0; i < width * query.rows; ++i) {
        if (!std::isfinite(query.data[i])) {
            throw std::invalid_argument("layerwise Mean-K Query contains a nonfinite operand");
        }
    }
    std::vector<uint32_t> rows(layers, 0);
    std::vector<const float*> pointers(layers, nullptr);
    rows[query.layer] = query.rows;
    pointers[query.layer] = query.data;
    PageQueryView view;
    view.layer_rows = rows.data();
    view.layers = layers;
    view.rows = query.rows;
    view.query_heads = query.query_heads;
    view.kv_heads = query.kv_heads;
    view.head_dim = query.head_dim;
    view.layer_data = pointers.data();
    view.attention_scale = query.attention_scale;
    // Reuse the existing qualified page-softmax implementation with precisely
    // one active layer rather than introducing a second scoring formula.
    return page_softmax_scores(keys, view, frame, GqaScoreAggregation::Mean);
}

} // namespace kvmem
