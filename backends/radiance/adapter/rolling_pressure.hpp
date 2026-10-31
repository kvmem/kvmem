#pragma once
// Calls the current KVMem core pressure_selection directly; no copied picker.
// Initial isolated qualification used the identical upstream picker body.
// Scores are supplied by this adapter's existing mean-key policy, NOT the new
// accepted-attention feedback implementation. Do not advertise full v0.19 parity.
#include "kvmem/graph_recall.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace kvmem::radiance::rolling {
inline void check(bool ok, const char* msg) { if (!ok) throw std::invalid_argument(msg); }
using ::kvmem::pressure_selection;
inline bool needs_pressure(uint64_t resident, uint32_t budget, uint32_t reserve,
                           uint32_t batch, uint32_t block) {
    const uint64_t safety = uint64_t(batch) + block;
    check(budget && block && reserve > safety, "rolling reserve must exceed one batch plus block");
    return resident > uint64_t(budget) + reserve - safety;
}
inline bool query_checkpoint_due(uint64_t frontier, uint64_t rows, uint64_t query_begin, uint32_t block) {
    check(block != 0, "zero selection block");
    // Native prefill may end on its 64-token quantum inside our 128-token
    // statistics block. Save before entering that block, not after its prefix.
    const uint64_t anchor = query_begin / block * block;
    return frontier <= anchor && rows > anchor - frontier;
}
inline void canonicalize_mandatory(std::vector<uint32_t>& blocks) {
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
}
inline bool retainable_prefix_boundary(uint64_t frontier, uint64_t end, uint32_t block) {
    check(block != 0, "zero prefix block");
    return frontier == end && frontier != 0 && frontier % block == 0;
}
} // namespace kvmem::radiance::rolling
