#pragma once
#include "graph_recall.hpp"
#include "native_layer_query.hpp"
#include <memory>

namespace kvmem {
// Owned by one adapter/Program, used at the quiescent Query selection boundary.
// Device storage is bounded to one layer and <=64 MiB of logits, not all L*T*H*B.
// No process-global stream, mutable buffer, model state or history is shared.
class PageSoftmaxCuda {
public:
    explicit PageSoftmaxCuda(int device);
    // Borrow a startup-planned workspace. score() never allocates device memory
    // in this mode; exceeding the declared geometry fails before a launch.
    static size_t workspace_bytes(uint32_t max_blocks,uint32_t query_heads,uint32_t kv_heads,uint32_t head_dim);
    PageSoftmaxCuda(int device,void* workspace,size_t bytes,uint32_t max_blocks,
        uint32_t query_heads,uint32_t kv_heads,uint32_t head_dim);
    ~PageSoftmaxCuda();
    PageSoftmaxCuda(const PageSoftmaxCuda&) = delete;
    PageSoftmaxCuda& operator=(const PageSoftmaxCuda&) = delete;
    std::vector<double> score(const RawKvStore&, PageQueryView, const QueryScoreFrame&, GqaScoreAggregation = GqaScoreAggregation::Mean);
    // Fresh single-layer native Q stays on its producing CUDA stream/device.
    // Reuses the qualified page-softmax kernels and explicit FP16 boundary;
    // transfers only canonical Mean-K operands and the resulting page masses.
    // The borrowed stream must be eager and remains owned by the Program.
    std::vector<double> score_current_layer(const RawKvStore&, NativeCurrentLayerQuery,
                                          const QueryScoreFrame&, void* producer_stream);
    size_t device_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kvmem
