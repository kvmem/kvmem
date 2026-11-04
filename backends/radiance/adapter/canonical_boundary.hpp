#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace kvmem::radiance {
// The frontend proves ALL structured message prefixes against the exact final
// token/media rendering, or clears the entire vector. Unlike history_end, these
// spans are not rounded or changed by native scheduler admission. The last end
// includes the complete final message and excludes only its generation suffix.
// Completion/unknown/invalid metadata conservatively uses the full native input.
inline uint64_t canonical_input_boundary(size_t prompt_size,
        const std::vector<std::pair<int32_t,int32_t>>& spans) {
    uint64_t end=0;
    for(const auto& span:spans) {
        if(span.first<0 || uint64_t(span.first)!=end || span.second<=span.first ||
           uint64_t(span.second)>prompt_size)return prompt_size;
        end=uint64_t(span.second);
    }
    return spans.empty() ? prompt_size : end;
}
} // namespace kvmem::radiance
