#include "kvmem/layerwise_options.hpp"
#include "kvmem/kvmem_store.hpp"
#include "kvmem/memory_contract.hpp"

namespace kvmem {
LayerwiseKeepBands resolve_layerwise_keep_bands(uint32_t block,uint32_t startup,
    std::optional<uint32_t> sink,std::optional<uint32_t> recent) {
    if(!block || !startup || startup%block)
        throw std::invalid_argument("layerwise startup K must be a positive block multiple");
    // This native helper is byte-for-byte the pinned resolve function. Reuse
    // its token rounding, explicit-zero handling, overflow and fixed-sum guard.
    const auto resolved=resolve_kvmem_keep_allocation(block,startup,-1,-1,
        sink?int64_t(*sink):-1,recent?int64_t(*recent):-1);
    return {resolved.sink_blocks,resolved.recent_blocks};
}
void validate_layerwise_selection_bands(LayerwiseKeepBands bands,uint32_t block,uint64_t selected,bool pressure) {
    if(!block || !selected || selected%block || selected>UINT32_MAX)
        throw std::invalid_argument("layerwise selection B must be a positive native block multiple");
    const auto fixed=uint64_t(bands.sink_blocks)+(pressure?0:bands.recent_blocks);
    if(fixed>selected/block)
        throw MemoryError(MemoryErrorCode::BudgetExceeded,
            "layerwise selection B is smaller than the configured sink/recent allocation");
}
} // namespace kvmem
