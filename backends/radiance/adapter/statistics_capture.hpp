#pragma once
#include "kvmem/memory_contract.hpp"
#include "rad_runtime.h"
#include <memory>
#include <vector>
namespace kvmem { class RawKvStore; }
namespace kvmem::radiance {

struct StatisticsLayer {
    int layer;
    rad_kvgroup group;
    uint32_t ordinal; // shared RawKvStore attention-layer ordinal
    uint32_t head_dim, query_heads, kv_heads;
};
struct QueryStatistics {
    std::vector<float> sums; // [ordinal, query_heads * head_dim]
    std::vector<uint32_t> token_counts;
};

// Bounded correctness path for post-normalization/pre-RoPE observation. Captures
// one single-sequence trunk pass at a time, including a speculative verify.
// commit credits only its accepted prefix; a drafter pass is not a trunk pass.
// It never reads rotated K back as retrieval statistics.
// Copies are enqueued on the producer stream before RoPE can overwrite buffers.
class StatisticsCapture {
public:
    StatisticsCapture(int device, std::vector<StatisticsLayer>, uint32_t max_rows);
    ~StatisticsCapture();
    // Copy only the requested original-position query span. Empty spans still
    // collect K for every accepted row, without transferring unused Q to RAM.
    void begin(SessionIdentity, uint32_t original_start, uint32_t rows,
               uint32_t query_start = 0, uint32_t query_end = UINT32_MAX);
    static int observe(void*, int, rad_kvgroup, int64_t, const RadTensor& q,
                       const RadTensor& k, const RadBatch&, RadStream);
    void seal(); // after the architecture pass; all declared attention layers required
    TransferPoll poll();
    QueryStatistics commit(RawKvStore&, SessionIdentity, uint32_t accepted_rows,
                           uint32_t query_start, uint32_t query_end = UINT32_MAX); // original half-open span
    bool discard() noexcept;
private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace kvmem::radiance
