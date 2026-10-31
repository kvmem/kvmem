#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace kvmem::radiance {
inline bool matches_query_identity(const std::vector<int32_t>& retained_query,
                                   int32_t begin, int32_t end,
                                   const std::vector<int32_t>& prompt,
                                   int32_t new_begin, int32_t new_end) {
    return begin >= 0 && end > begin && new_begin == begin && new_end == end &&
        size_t(end) <= prompt.size() && size_t(end - begin) == retained_query.size() &&
        std::equal(retained_query.begin(), retained_query.end(), prompt.begin() + begin);
}
}
