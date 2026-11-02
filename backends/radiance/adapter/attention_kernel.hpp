#pragma once
#include "rad_runtime.h"
#include "attention_plan.hpp"
#include "packed_kv.hpp"
#include <cstdint>
namespace kvmem::radiance {
// Sorted logical pages and their offsets in the native compact attention view.
struct AttentionKernelArgs {
    RadTensor query, cache;
    const int32_t* table = nullptr;
    const uint32_t* rows = nullptr;
    const AttentionPage* pages = nullptr;
    uint32_t samples = 0, page_count = 0, query_heads = 0, head_dim = 0, original_begin = 0;
    float* logits = nullptr; // bounded [sample,head,page] logsumexp scratch
    float* masses = nullptr; // [sample,page], the only observation copied to host
    KvPacking packing = KvPacking::Plain;
};
int attention_page_masses(const AttentionKernelArgs&, RadStream);
} // namespace kvmem::radiance
