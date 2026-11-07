#pragma once

#include "layerwise_scoring.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kvmem {

enum class NativeQueryScalar { Float32, Float16, BFloat16 };

// Borrowed normalized, pre-RoPE Q produced by this native forward. Strides and
// extent are bytes, so a backend can expose its real tensor without a CPU copy
// or a layout conversion. The scorer owns device/stream validation and fences;
// request/index identity is proved by LayerwiseQueryRequest, not by a pointer.
struct NativeCurrentLayerQuery {
    const void* data = nullptr;
    size_t bytes = 0;
    uint32_t layer = 0, rows = 0, query_heads = 0, kv_heads = 0, head_dim = 0;
    float attention_scale = 0;
    NativeQueryScalar scalar = NativeQueryScalar::Float32;
    size_t row_stride = 0, head_stride = 0, dimension_stride = 0;
};

inline size_t native_query_scalar_bytes(NativeQueryScalar scalar) {
    switch (scalar) {
        case NativeQueryScalar::Float32: return 4;
        case NativeQueryScalar::Float16:
        case NativeQueryScalar::BFloat16: return 2;
    }
    throw std::invalid_argument("unknown native Query scalar");
}

inline void validate_native_layer_query(const NativeCurrentLayerQuery& q) {
    const auto element = native_query_scalar_bytes(q.scalar);
    if (!q.data || !q.rows || !q.query_heads || !q.kv_heads || !q.head_dim ||
        q.query_heads % q.kv_heads || !std::isfinite(q.attention_scale) || q.attention_scale < 0 ||
        reinterpret_cast<uintptr_t>(q.data) % element)
        throw std::invalid_argument("invalid native current-layer Query geometry");
    // Qualify positive non-overlapping native permutations, with optional
    // padding. No assumption that the token axis is the outermost axis.
    std::array<std::pair<size_t,uint32_t>,3> axes{{
        {q.row_stride,q.rows},{q.head_stride,q.query_heads},{q.dimension_stride,q.head_dim}}};
    std::sort(axes.begin(),axes.end());
    size_t span = element;
    for (const auto& axis : axes) {
        if (!axis.first || axis.first % element || (axis.second > 1 && axis.first < span) ||
            axis.first > (std::numeric_limits<size_t>::max()-span)/(axis.second-1 ? axis.second-1 : 1))
            throw std::invalid_argument("overlapping or overflowing native Query strides");
        span += axis.first * (axis.second-1);
    }
    if (span > q.bytes) throw std::invalid_argument("native Query exceeds its declared byte extent");
}

inline NativeCurrentLayerQuery native_layer_query_rows(NativeCurrentLayerQuery q,
                                                       uint32_t first, uint32_t rows) {
    validate_native_layer_query(q);
    if (first >= q.rows || !rows || rows > q.rows-first)
        throw std::invalid_argument("native Query row slice exceeds its tensor");
    const size_t offset = size_t(first)*q.row_stride;
    q.data = static_cast<const unsigned char*>(q.data)+offset;
    q.bytes -= offset;
    q.rows = rows;
    return q;
}
} // namespace kvmem
