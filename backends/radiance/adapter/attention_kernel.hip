#include <hip/hip_runtime.h>
#include "attention_kernel.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
namespace kvmem::radiance {
namespace {
__device__ float read_float(const void* data, uint32_t dtype, uint64_t i) {
    if (dtype == RAD_F32) return static_cast<const float*>(data)[i];
    if (dtype == RAD_F16) return __half2float(static_cast<const __half*>(data)[i]);
    if (dtype == RAD_BF16) return __uint_as_float(uint32_t(static_cast<const uint16_t*>(data)[i]) << 16);
    const auto v = static_cast<const uint8_t*>(data)[i];
    const auto magnitude = v & 127u;
    if (magnitude == 127) return NAN; // E4M3 finite-numbers encoding
    const int exponent = (v >> 3) & 15, mantissa = v & 7;
    const float x = exponent ? ldexpf(float(8 + mantissa), exponent - 10) : ldexpf(float(mantissa), -9);
    return v & 128 ? -x : x;
}
__device__ float read_key(const AttentionKernelArgs& a, uint64_t base, unsigned channel) {
    if (a.packing != KvPacking::K6V4) return read_float(a.cache.data, a.cache.dtype, base + channel);
    const auto* row = static_cast<const uint8_t*>(a.cache.data) + base;
    const auto* scale = reinterpret_cast<const __half*>(row + a.head_dim * 5 / 4);
    return float(packed_k6_code(row, a.head_dim, channel)) * __half2float(scale[channel / 32]);
}
__global__ void page_lse(AttentionKernelArgs a) {
    const unsigned page = blockIdx.x, head = blockIdx.y, sample = blockIdx.z;
    const auto p = a.pages[page];
    const unsigned row = a.rows[sample], lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned kv_head = head / (a.query_heads / unsigned(a.cache.shape[1]));
    const uint64_t qbase = uint64_t(row) * a.query.stride[0] + uint64_t(head) * a.head_dim;
    float maximum = -INFINITY, sum = 0;
    for (unsigned token = warp; token < p.count; token += 8) {
        if (p.start + token > uint64_t(a.original_begin) + row) continue;
        const unsigned compact = p.compact + token, native_block = unsigned(a.cache.shape[2]);
        const int physical = a.table[compact / native_block];
        float dot = 0;
        if (physical < 0 || physical >= a.cache.shape[0]) { dot = NAN; }
        else {
            const uint64_t kbase = uint64_t(physical) * a.cache.stride[0] +
                uint64_t(kv_head) * a.cache.stride[1] + (compact % native_block) * a.cache.stride[2];
            for (unsigned channel = lane; channel < a.head_dim; channel += 32)
                dot += read_float(a.query.data, a.query.dtype, qbase + channel) *
                       read_key(a, kbase, channel);
        }
        for (int delta = 16; delta; delta /= 2) dot += __shfl_down(dot, delta, 32);
        if (!lane) {
            dot *= rsqrtf(float(a.head_dim));
            if (!isfinite(dot)) { maximum = NAN; sum = NAN; continue; }
            const float next = fmaxf(maximum, dot);
            sum = sum * expf(maximum - next) + expf(dot - next); maximum = next;
        }
    }
    __shared__ float maxima[8], sums[8];
    if (!lane) { maxima[warp] = maximum; sums[warp] = sum; }
    __syncthreads();
    if (!threadIdx.x) {
        float m = -INFINITY, total = 0;
        for (unsigned w = 0; w < 8; ++w) if (!isfinite(sums[w])) {
            a.logits[(uint64_t(sample) * a.query_heads + head) * a.page_count + page] = NAN;
            return;
        }
        for (unsigned w = 0; w < 8; ++w) if (sums[w] > 0) m = fmaxf(m, maxima[w]);
        for (unsigned w = 0; w < 8; ++w) if (sums[w] > 0) total += sums[w] * expf(maxima[w] - m);
        a.logits[(uint64_t(sample) * a.query_heads + head) * a.page_count + page] =
            total > 0 ? m + logf(total) : -INFINITY;
    }
}
__global__ void normalize(AttentionKernelArgs a) {
    const unsigned head = blockIdx.x, sample = blockIdx.y, tid = threadIdx.x;
    auto* scores = a.logits + (uint64_t(sample) * a.query_heads + head) * a.page_count;
    __shared__ float reduce[256];
    float maximum = -INFINITY;
    for (unsigned p = tid; p < a.page_count; p += 256) maximum = fmaxf(maximum, scores[p]);
    reduce[tid] = maximum; __syncthreads();
    for (unsigned step = 128; step; step /= 2) {
        if (tid < step) reduce[tid] = fmaxf(reduce[tid], reduce[tid + step]); __syncthreads();
    }
    maximum = reduce[0]; __syncthreads();
    float sum = 0;
    for (unsigned p = tid; p < a.page_count; p += 256) sum += expf(scores[p] - maximum);
    reduce[tid] = sum; __syncthreads();
    for (unsigned step = 128; step; step /= 2) {
        if (tid < step) reduce[tid] += reduce[tid + step]; __syncthreads();
    }
    sum = reduce[0];
    for (unsigned p = tid; p < a.page_count; p += 256) scores[p] = expf(scores[p] - maximum) / sum;
}
__global__ void average_heads(AttentionKernelArgs a) {
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x, sample = blockIdx.y;
    if (p >= a.page_count) return;
    float mass = 0;
    for (unsigned head = 0; head < a.query_heads; ++head)
        mass += a.logits[(uint64_t(sample) * a.query_heads + head) * a.page_count + p];
    a.masses[uint64_t(sample) * a.page_count + p] = mass / a.query_heads;
}
}
int attention_page_masses(const AttentionKernelArgs& a, RadStream stream) {
    if (a.packing != KvPacking::Plain && a.packing != KvPacking::Fp8V4 && a.packing != KvPacking::K6V4) return RAD_E_INVAL;
    if (a.packing != KvPacking::Plain && (a.head_dim % 32 || a.cache.dtype != RAD_F8E4M3)) return RAD_E_INVAL;
    const auto row_elements = kv_row_elements(a.head_dim, a.packing);
    const auto floating = [](uint32_t t) { return t == RAD_F32 || t == RAD_F16 || t == RAD_BF16; };
    if (!stream || !a.samples || !a.page_count || !a.rows || !a.pages || !a.table || !a.logits || !a.masses ||
        !a.query.data || !a.cache.data || !floating(a.query.dtype) ||
        (!floating(a.cache.dtype) && a.cache.dtype != RAD_F8E4M3) || a.cache.rank != 4 ||
        !a.head_dim || a.cache.shape[1] <= 0 || a.query_heads % a.cache.shape[1] ||
        a.cache.shape[2] <= 0 || a.cache.shape[3] != row_elements || a.cache.stride[3] != 1 ||
        a.cache.stride[2] != row_elements || a.query.stride[0] < int64_t(a.query_heads) * a.head_dim)
        return RAD_E_INVAL;
    const auto s = reinterpret_cast<hipStream_t>(stream);
    hipLaunchKernelGGL(page_lse, dim3(a.page_count, a.query_heads, a.samples), dim3(256), 0, s, a);
    if (hipGetLastError() != hipSuccess) return RAD_E_STATE;
    hipLaunchKernelGGL(normalize, dim3(a.query_heads, a.samples), dim3(256), 0, s, a);
    if (hipGetLastError() != hipSuccess) return RAD_E_STATE;
    hipLaunchKernelGGL(average_heads, dim3((a.page_count + 255) / 256, a.samples), dim3(256), 0, s, a);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_STATE;
}
} // namespace kvmem::radiance
