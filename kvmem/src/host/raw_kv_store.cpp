#include "kvmem/raw_kv_store.hpp"
#include "kvmem/snapshot.hpp"
#include "kvmem/snapshot_buffer.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace kvmem {
namespace {

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if ((x & 0x7fffffffu) == 0) {
        return static_cast<uint16_t>(sign);
    }
    if (((x >> 23) & 0xffu) == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u | (man ? 0x200u : 0));
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        man |= 0x800000u;
        const uint32_t t = static_cast<uint32_t>(14 - exp);
        uint32_t half = man >> t;
        const uint32_t rem = man & ((1u << t) - 1u);
        if (rem > (1u << (t - 1)) || (rem == (1u << (t - 1)) && (half & 1u))) {
            half += 1;
        }
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t half = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
        half += 1;
    }
    return static_cast<uint16_t>(sign | half);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3ffu;
            out = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

void pack_f32(const float * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f32_to_f16(src[i]);
    }
}

void unpack_f16(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f16_to_f32(src[i]);
    }
}

} // namespace

RawKvStore::RawKvStore(RawKvStoreConfig cfg) : cfg_(std::move(cfg)),
    payload_storage_(cfg_.payload_storage ? cfg_.payload_storage : std::make_shared<HostKvStorage>()) {}

RawKvStore::~RawKvStore() = default;

uint64_t RawKvStore::k_row_bytes() const {
    return cfg_.k_row_bytes ? cfg_.k_row_bytes
                            : static_cast<uint64_t>(cfg_.n_embd_k) * sizeof(uint16_t);
}

bool RawKvStore::k_is_f16() const {
    return k_row_bytes() == static_cast<uint64_t>(cfg_.n_embd_k) * sizeof(uint16_t);
}

void RawKvStore::reserve_packed(uint32_t block_id, const std::vector<uint32_t> & layers) {
    ensure_blocks(block_id + 1);
    for (uint32_t il : layers) {
        if (il >= cfg_.n_layer) throw std::invalid_argument("invalid packed layer reservation");
        auto & lb = blocks_[block_id].layers[il];
        if (lb.k_gpu.empty()) lb.k_gpu = payload_storage_->allocate(cfg_.k_gpu_row_bytes * cfg_.block_tokens);
        if (lb.v_gpu.empty()) lb.v_gpu = payload_storage_->allocate(cfg_.v_gpu_row_bytes * cfg_.block_tokens);
    }
}

void RawKvStore::ensure_blocks(uint32_t block_count) {
    if (blocks_.size() >= block_count) {
        return;
    }
    const size_t old = blocks_.size();
    blocks_.resize(block_count);
    for (size_t i = old; i < blocks_.size(); ++i) {
        blocks_[i].layers.resize(cfg_.n_layer);
    }
}

void RawKvStore::capture_mean_f16(LayerBlk & lb) const {
    if (!k_is_f16() || lb.k.empty() || cfg_.n_embd_k == 0) {
        return;
    }
    const uint32_t nt = lb.n_tokens;
    if (nt == 0) {
        return;
    }
    const uint64_t row = k_row_bytes();
    lb.k_sum.assign(cfg_.n_embd_k, 0.0f);
    for (uint32_t t = 0; t < nt; ++t) {
        const auto * src = reinterpret_cast<const uint16_t *>(
                lb.k.data() + static_cast<size_t>(t) * static_cast<size_t>(row));
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            lb.k_sum[d] += f16_to_f32(src[d]);
        }
    }
    lb.mean_tokens = nt;
}

void RawKvStore::add_mean_f32(LayerBlk & lb, uint32_t off, uint32_t take,
                              const float * k) {
    if (!k || take == 0 || cfg_.n_embd_k == 0) {
        return;
    }
    if (lb.k_sum.size() != cfg_.n_embd_k) {
        lb.k_sum.assign(cfg_.n_embd_k, 0.0f);
    }
    if (off == 0) {
        std::fill(lb.k_sum.begin(), lb.k_sum.end(), 0.0f);
    }
    for (uint32_t t = 0; t < take; ++t) {
        const float * row = k + t * cfg_.n_embd_k;
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            lb.k_sum[d] += row[d];
        }
    }
    const uint32_t nt = off == 0 ? take : std::max(lb.mean_tokens, off + take);
    lb.mean_tokens = nt;
}

void RawKvStore::write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                                    const float * k, const float * v) {
    std::unique_lock<std::mutex> lk(mu_);
    if (n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);

        LayerBlk & lb = blocks_[bid].layers[il];
        if (k && k_is_f16()) {
            const uint64_t row = k_row_bytes();
            const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
            if (lb.k.size() < need) {
                lb.k.assign(need, 0);

            }
            pack_f32(k + done * cfg_.n_embd_k,
                     reinterpret_cast<uint16_t *>(
                             lb.k.data() + static_cast<size_t>(off) * static_cast<size_t>(row)),
                     static_cast<size_t>(take) * cfg_.n_embd_k);
            add_mean_f32(lb, off, take, k + done * cfg_.n_embd_k);
        }
        if (v) {
            lb.v_gpu.reset();

            const size_t need = static_cast<size_t>(bt) * cfg_.n_embd_v;
            if (lb.v.size() < need) {
                lb.v.assign(need, 0);

            }
            pack_f32(v + done * cfg_.n_embd_v,
                     lb.v.data() + off * cfg_.n_embd_v,
                     static_cast<size_t>(take) * cfg_.n_embd_v);
            lb.v_gpu_fmt = false;
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);

        done += take;
    }
}

void RawKvStore::write_layer_tokens_f16(uint32_t pos0, uint32_t n, uint32_t il,
                                        const uint16_t * k, const uint16_t * v) {
    std::unique_lock<std::mutex> lk(mu_);
    if (n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);

        LayerBlk & lb = blocks_[bid].layers[il];
        if (k && k_is_f16()) {
            const uint64_t row = k_row_bytes();
            const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
            if (lb.k.size() < need) {
                lb.k.assign(need, 0);

            }
            std::memcpy(lb.k.data() + static_cast<size_t>(off) * static_cast<size_t>(row),
                        k + done * cfg_.n_embd_k,
                        static_cast<size_t>(take) * static_cast<size_t>(row));
        }
        if (v) {
            lb.v_gpu.reset();

            const size_t need = static_cast<size_t>(bt) * cfg_.n_embd_v;
            if (lb.v.size() < need) {
                lb.v.assign(need, 0);

            }
            std::memcpy(lb.v.data() + off * cfg_.n_embd_v,
                        v + done * cfg_.n_embd_v,
                        static_cast<size_t>(take) * cfg_.n_embd_v * sizeof(uint16_t));
            lb.v_gpu_fmt = false;
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        if (k && k_is_f16()) {
            capture_mean_f16(lb);
        }

        done += take;
    }
}

void RawKvStore::write_layer_k_rows(uint32_t pos0, uint32_t n, uint32_t il,
                                    const uint8_t * k, const float * k_f32) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!k || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        k_row_bytes() == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    const uint64_t row = k_row_bytes();
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);

        LayerBlk & lb = blocks_[bid].layers[il];
        const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
        if (lb.k.size() < need) {
            lb.k.assign(need, 0);

        }
        std::memcpy(lb.k.data() + static_cast<size_t>(off) * static_cast<size_t>(row),
                    k + static_cast<size_t>(done) * static_cast<size_t>(row),
                    static_cast<size_t>(take) * static_cast<size_t>(row));
        if (k_f32) {
            add_mean_f32(lb, off, take, k_f32 + done * cfg_.n_embd_k);
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);

        done += take;
    }
}

void RawKvStore::write_layer_v_gpu(uint32_t pos0, uint32_t n, uint32_t il,
                                   const uint8_t * v) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!v || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        cfg_.v_gpu_row_bytes == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    const uint64_t row = cfg_.v_gpu_row_bytes;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);

        LayerBlk & lb = blocks_[bid].layers[il];
        lb.v.clear();
        const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
        const size_t nbytes = static_cast<size_t>(take) * static_cast<size_t>(row);
        const uint8_t * src = v + static_cast<size_t>(done) * static_cast<size_t>(row);
        if (off == 0 && take == bt) {

            if (lb.v_gpu.size() != need) lb.v_gpu = payload_storage_->allocate(need, src);
            else lb.v_gpu.write(0, src, need);
        } else {
            if (lb.v_gpu.size() < need) {

                lb.v_gpu = payload_storage_->allocate(need);

            }
            std::memcpy(lb.v_gpu.writable_data() + static_cast<size_t>(off) * row, src, nbytes);
        }
        lb.v_gpu_fmt = true;
        lb.v_gpu_tokens = std::max(lb.v_gpu_tokens, off + take);
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        if (lb.v_gpu_tokens == bt) lb.v_gpu.seal();

        done += take;
    }
}

void RawKvStore::write_layer_k_gpu(uint32_t pos0, uint32_t n, uint32_t il,
                                   const uint8_t * k) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!k || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        cfg_.k_gpu_row_bytes == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    const uint64_t row = cfg_.k_gpu_row_bytes;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        LayerBlk & lb = blocks_[bid].layers[il];
        const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
        const size_t nbytes = static_cast<size_t>(take) * static_cast<size_t>(row);
        const uint8_t * src = k + static_cast<size_t>(done) * static_cast<size_t>(row);
        if (off == 0 && take == bt) {
            if (lb.k_gpu.size() != need) lb.k_gpu = payload_storage_->allocate(need, src);
            else lb.k_gpu.write(0, src, need);
        } else {
            if (lb.k_gpu.size() < need) {
                lb.k_gpu = payload_storage_->allocate(need);
            }
            std::memcpy(lb.k_gpu.writable_data() + static_cast<size_t>(off) * row, src, nbytes);
        }
        lb.k_gpu_fmt = true;
        lb.k_gpu_tokens = std::max(lb.k_gpu_tokens, off + take);
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        if (lb.k_gpu_tokens == bt) lb.k_gpu.seal();
        done += take;
    }
}

void RawKvStore::write_layer_mean_k(uint32_t pos0, uint32_t n, uint32_t il,
                                    const float * k) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!k || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        cfg_.n_embd_k == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        LayerBlk & lb = blocks_[bid].layers[il];
        add_mean_f32(lb, off, take, k + done * cfg_.n_embd_k);
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        done += take;
    }
}

void RawKvStore::write_layer_mean_sum(uint32_t pos0, uint32_t n, uint32_t il,
                                      const float * sum) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!sum || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        cfg_.n_embd_k == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    const uint32_t bid = pos0 / bt;
    const uint32_t off = pos0 % bt;
    const uint32_t take = std::min(n, bt - off);
    ensure_blocks(bid + 1);
    LayerBlk & lb = blocks_[bid].layers[il];
    if (lb.k_sum.size() != cfg_.n_embd_k) {
        lb.k_sum.assign(cfg_.n_embd_k, 0.0f);
    }
    if (off == 0) {
        std::fill(lb.k_sum.begin(), lb.k_sum.end(), 0.0f);
    }
    for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
        lb.k_sum[d] += sum[d];
    }
    lb.n_tokens = std::max(lb.n_tokens, off + take);
    lb.mean_tokens = off == 0 ? take : std::max(lb.mean_tokens, off + take);
}

bool RawKvStore::has_block(uint32_t block_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size()) {
        return false;
    }
    for (const auto & lb : blocks_[block_id].layers) {
        if (lb.n_tokens > 0 &&
            (!lb.k.empty() || lb.k_gpu_fmt ||
             (lb.mean_tokens > 0 && !lb.k_sum.empty()))) {
            return true;
        }
    }
    return false;
}

bool RawKvStore::has_k(uint32_t block_id, uint32_t il) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return !lb.k.empty();
}

bool RawKvStore::has_k_gpu(uint32_t block_id, uint32_t il, uint32_t n) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return lb.k_gpu_fmt && !lb.k_gpu.empty() && lb.k_gpu_tokens >= n;
}

bool RawKvStore::has_v(uint32_t block_id, uint32_t il) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return !lb.v.empty() || !lb.v_gpu.empty();
}

bool RawKvStore::has_v_gpu(uint32_t block_id, uint32_t il, uint32_t n) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return lb.v_gpu_fmt && lb.v_gpu_tokens >= n;
}

uint32_t RawKvStore::n_tokens(uint32_t block_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size()) {
        return 0;
    }
    uint32_t n = 0;
    for (const auto & lb : blocks_[block_id].layers) {
        n = std::max(n, lb.n_tokens);
    }
    return n;
}

bool RawKvStore::copy_k(uint32_t block_id, uint32_t il, float * out) const {
    if (!out || !k_is_f16()) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) return false;
    const auto & lb = blocks_[block_id].layers[il];
    if (lb.k.empty()) return false;
    unpack_f16(reinterpret_cast<const uint16_t *>(lb.k.data()), out,
               static_cast<size_t>(lb.n_tokens) * cfg_.n_embd_k);
    return true;
}

bool RawKvStore::copy_v(uint32_t block_id, uint32_t il, float * out) const {
    if (!out) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) return false;
    const auto & lb = blocks_[block_id].layers[il];
    if (lb.v.empty()) return false;
    unpack_f16(lb.v.data(), out, static_cast<size_t>(lb.n_tokens) * cfg_.n_embd_v);
    return true;
}

bool RawKvStore::copy_k_rows(uint32_t block_id, uint32_t il, uint8_t * out,
                             uint32_t n) const {
    if (!out || n == 0 || k_row_bytes() == 0) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) return false;
    const auto & lb = blocks_[block_id].layers[il];
    if (lb.k.empty()) return false;
    std::memcpy(out, lb.k.data(), static_cast<size_t>(std::min(n, lb.n_tokens)) * k_row_bytes());
    return true;
}

bool RawKvStore::copy_k_gpu(uint32_t block_id, uint32_t il, uint8_t * out,
                            uint32_t n) const {
    if (!out || n == 0 || cfg_.k_gpu_row_bytes == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (!lb.k_gpu_fmt || lb.k_gpu.empty() || lb.k_gpu_tokens < n) {
        return false;
    }
    const uint64_t row = cfg_.k_gpu_row_bytes;
    const uint32_t nt = n;
    if (lb.k_gpu.size() < static_cast<size_t>(nt) * static_cast<size_t>(row)) {
        return false;
    }
    lb.k_gpu.read(0, out, static_cast<size_t>(nt) * static_cast<size_t>(row));
    return true;
}

bool RawKvStore::copy_v_gpu(uint32_t block_id, uint32_t il, uint8_t * out,
                            uint32_t n) const {
    if (!out || n == 0 || cfg_.v_gpu_row_bytes == 0) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) return false;
    const auto & lb = blocks_[block_id].layers[il];
    if (!lb.v_gpu_fmt || lb.v_gpu_tokens < n || lb.v_gpu.empty()) return false;
    lb.v_gpu.read(0, out, static_cast<size_t>(n) * cfg_.v_gpu_row_bytes);
    return true;
}

void RawKvStore::mean_k(uint32_t block_id, uint32_t il, float * out) const {
    if (!out) return;
    std::lock_guard<std::mutex> lk(mu_);
    std::fill_n(out, cfg_.n_embd_k, 0.0f);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) return;
    const auto & lb = blocks_[block_id].layers[il];
    if (lb.mean_tokens && lb.k_sum.size() == cfg_.n_embd_k) {
        const float inv = 1.0f / static_cast<float>(lb.mean_tokens);
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) out[d] = lb.k_sum[d] * inv;
        return;
    }
    if (!k_is_f16() || lb.k.empty() || lb.n_tokens == 0 || cfg_.n_embd_k == 0) return;
    const auto row = k_row_bytes();
    for (uint32_t t = 0; t < lb.n_tokens; ++t) {
        const auto * src = reinterpret_cast<const uint16_t *>(lb.k.data() + static_cast<size_t>(t) * row);
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) out[d] += f16_to_f32(src[d]);
    }
    const float inv = 1.0f / static_cast<float>(lb.n_tokens);
    for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) out[d] *= inv;
}

size_t RawKvStore::bytes_k() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (const auto & b : blocks_) {
        for (const auto & lb : b.layers) {
            n += lb.k.size();
            n += lb.k_gpu.size();
            n += lb.k_sum.size() * sizeof(float);
        }
    }
    return n;
}

size_t RawKvStore::bytes_v() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (const auto & b : blocks_) {
        for (const auto & lb : b.layers) {
            n += lb.v.size() * sizeof(uint16_t);
            n += lb.v_gpu.size();
        }
    }
    return n;
}

void RawKvStore::truncate_to(uint32_t token_pos) {
    std::lock_guard<std::mutex> lk(mu_);
    if (token_pos == 0 || cfg_.block_tokens == 0) {
        blocks_.clear();
        return;
    }
    const uint32_t keep = (token_pos + cfg_.block_tokens - 1) / cfg_.block_tokens;
    if (blocks_.size() > keep) {
        blocks_.resize(keep);
    }
    const uint32_t tail = token_pos % cfg_.block_tokens;
    if (tail && blocks_.size() == keep) {
        for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
            auto & lb = blocks_.back().layers[il];
            lb.n_tokens = std::min(lb.n_tokens, tail);
            lb.k_gpu_tokens = std::min(lb.k_gpu_tokens, tail);
            lb.v_gpu_tokens = std::min(lb.v_gpu_tokens, tail);
            if (lb.mean_tokens > tail) {
                lb.k_sum.clear();
                lb.mean_tokens = 0;
                if (k_is_f16()) {

                    capture_mean_f16(lb);
                }
            }
        }
    }
}

void RawKvStore::invalidate_packed_from(uint32_t token_pos) {
    std::lock_guard<std::mutex> lk(mu_);
    for (uint32_t bid = token_pos / cfg_.block_tokens; bid < blocks_.size(); ++bid) {
        const uint32_t keep = bid == token_pos / cfg_.block_tokens ? token_pos % cfg_.block_tokens : 0;
        for (auto & lb : blocks_[bid].layers) {
            lb.k_gpu_tokens = std::min(lb.k_gpu_tokens, keep);
            lb.v_gpu_tokens = std::min(lb.v_gpu_tokens, keep);
        }
    }
}

std::vector<float> RawKvStore::mean_checkpoint(uint32_t token_pos) const {
    std::lock_guard<std::mutex> lk(mu_);
    const uint32_t bid = token_pos / cfg_.block_tokens;
    if (token_pos % cfg_.block_tokens == 0) return {};
    const size_t stride = 1 + cfg_.n_embd_k;
    std::vector<float> state(cfg_.n_layer * stride, 0);
    if (bid >= blocks_.size()) return state;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const auto & lb = blocks_[bid].layers[il];
        float * dst = state.data() + il * stride;
        dst[0] = lb.mean_tokens;
        if (!lb.k_sum.empty()) std::copy(lb.k_sum.begin(), lb.k_sum.end(), dst + 1);
    }
    return state;
}

void RawKvStore::restore_mean_checkpoint(uint32_t token_pos, const std::vector<float> & state) {
    if (state.empty()) return;
    std::lock_guard<std::mutex> lk(mu_);
    const size_t stride = 1 + cfg_.n_embd_k;
    if (state.size() != cfg_.n_layer * stride) throw std::runtime_error("invalid mean-K checkpoint");
    const uint32_t bid = token_pos / cfg_.block_tokens;
    ensure_blocks(bid + 1);
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        auto & lb = blocks_[bid].layers[il];
        const float * src = state.data() + il * stride;
        lb.mean_tokens = (uint32_t) src[0];
        if (lb.mean_tokens > 0) {
            lb.k_sum.assign(src + 1, src + stride);
        } else {
            lb.k_sum.clear();
        }
    }
}

void RawKvStore::clear() {
    std::unique_lock<std::mutex> lk(mu_);

    blocks_.clear();

}

size_t RawKvStore::allocated_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t bytes = sizeof(*this) + blocks_.capacity() * sizeof(BlockRaw);
    for (const auto & b : blocks_) {
        bytes += b.layers.capacity() * sizeof(LayerBlk);
        for (const auto & l : b.layers) {
            bytes += l.k.capacity() + l.v.capacity()*2 + l.k_gpu.resident_bytes() + l.v_gpu.resident_bytes();
            bytes += l.k_sum.capacity()*sizeof(float);
        }
    }
    return bytes;
}

uint64_t RawKvStore::capacity_bytes(uint32_t tokens, uint32_t populated_layers) const {
    const uint64_t blocks = (uint64_t(tokens) + cfg_.block_tokens - 1) / cfg_.block_tokens + 1;
    return sizeof(*this) + blocks * (2*sizeof(BlockRaw) + cfg_.n_layer*sizeof(LayerBlk) +
        std::min(populated_layers, cfg_.n_layer) * (cfg_.n_embd_k*sizeof(float) + uint64_t(cfg_.block_tokens) *
         (cfg_.k_gpu_row_bytes + cfg_.v_gpu_row_bytes)));
}
uint64_t RawKvStore::payload_bytes(bool disk_only) const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t bytes = 0;
    for (const auto & block : blocks_) for (const auto & layer : block.layers) {
        bytes += layer.k_gpu.size() + layer.v_gpu.size();
        if (disk_only) bytes -= layer.k_gpu.resident_bytes() + layer.v_gpu.resident_bytes();
    }
    return bytes;
}
uint64_t RawKvStore::payload_capacity_bytes(uint32_t tokens, uint32_t populated_layers) const {
    if (!cfg_.block_tokens) throw std::invalid_argument("native payload capacity requires a block size");
    const uint64_t blocks = (uint64_t(tokens) + cfg_.block_tokens - 1) / cfg_.block_tokens;
    return blocks * cfg_.block_tokens * std::min(populated_layers, cfg_.n_layer) *
        (cfg_.k_gpu_row_bytes + cfg_.v_gpu_row_bytes);
}

void RawKvStore::snapshot_buffers(std::vector<SnapshotBuffer> & buffers) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto & b : blocks_) for (auto & l : b.layers) {
        // All lengths, token counts, formats and runtime identity stay in RAM.
        // Empty allocations can be reclaimed too, without serializing capacity.
        if (l.k.capacity()) buffers.push_back(SnapshotBuffer::bind(l.k));
        if (l.v.capacity()) buffers.push_back(SnapshotBuffer::bind(l.v));
        if (payload_storage_->disk_capacity_bytes()) {
            // The shared records remain authoritative through idle retention.
            // Only state/statistics need the session attachment serializer.
            l.k_gpu.seal(); l.v_gpu.seal();
        } else {
            if (l.k_gpu.size()) buffers.push_back(SnapshotBuffer::bind(l.k_gpu, *payload_storage_));
            if (l.v_gpu.size()) buffers.push_back(SnapshotBuffer::bind(l.v_gpu, *payload_storage_));
        }
        if (l.k_sum.capacity()) buffers.push_back(SnapshotBuffer::bind(l.k_sum));
    }
}

std::unique_ptr<RawKvStore> RawKvStore::fork_mean_prefix(uint32_t token_pos) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto count = (uint64_t(token_pos) + cfg_.block_tokens - 1) / cfg_.block_tokens;
    if (count > blocks_.size()) throw std::logic_error("mean-K fork exceeds the retained prefix");
    auto fork = std::make_unique<RawKvStore>(cfg_);
    fork->blocks_.resize(static_cast<size_t>(count));
    for (size_t b = 0; b < count; ++b) {
        auto & dest = fork->blocks_[b].layers;
        dest.resize(cfg_.n_layer);
        for (size_t i = 0; i < dest.size(); ++i) {
            const auto & source = blocks_[b].layers.at(i);
            if (!source.k.empty() || !source.v.empty() || source.k_gpu || source.v_gpu)
                throw std::logic_error("mean-K fork cannot discard raw KV payload");
            dest[i].n_tokens = source.n_tokens;
            dest[i].mean_tokens = source.mean_tokens;
            dest[i].k_sum = source.k_sum;
        }
    }
    return fork;
}

std::unique_ptr<RawKvStore> RawKvStore::fork_prefix(uint32_t token_pos) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto count = (uint64_t(token_pos) + cfg_.block_tokens - 1) / cfg_.block_tokens;
    if (count > blocks_.size()) throw std::logic_error("native KV fork exceeds the retained prefix");
    auto branch = std::make_unique<RawKvStore>(cfg_);
    branch->blocks_.resize(size_t(count));
    for (size_t b = 0; b < count; ++b) {
        auto & target = branch->blocks_[b].layers;
        target.resize(cfg_.n_layer);
        for (size_t i = 0; i < target.size(); ++i) {
            auto & source = blocks_[b].layers.at(i);
            auto & dest = target[i];
            source.k_gpu.seal(); source.v_gpu.seal();
            dest.n_tokens = source.n_tokens;
            dest.k_gpu_tokens = source.k_gpu_tokens; dest.v_gpu_tokens = source.v_gpu_tokens;
            dest.mean_tokens = source.mean_tokens;
            dest.k = source.k; dest.v = source.v; dest.k_sum = source.k_sum;
            dest.k_gpu_fmt = source.k_gpu_fmt; dest.v_gpu_fmt = source.v_gpu_fmt;
            dest.k_gpu = source.k_gpu.fork(); dest.v_gpu = source.v_gpu.fork();
        }
    }
    // Counts and partial-tail means must come from the selected checkpoint.
    // Truncation only changes the branch; its first packed write detaches.
    branch->truncate_to(token_pos);
    return branch;
}

void RawKvStore::snapshot_write(SnapshotWriter & out) {
    std::lock_guard<std::mutex> lk(mu_);
    out.scalar(uint32_t(0x31564b52));
    out.scalar(cfg_.n_layer); out.scalar(cfg_.n_embd_k); out.scalar(cfg_.n_embd_v);
    out.scalar(cfg_.block_tokens); out.scalar(cfg_.k_row_bytes);
    out.scalar(cfg_.k_gpu_row_bytes); out.scalar(cfg_.v_gpu_row_bytes);
    out.scalar(uint64_t(blocks_.size()));
    // Per-block section length allows a future reader to build a disk index.
    for (const auto & b : blocks_) {
        uint64_t size = 0;
        for (const auto & l : b.layers) size += 4*4 + 2 + 5*8 + l.k.size() + l.v.size()*2 +
            l.k_gpu.size() + l.v_gpu.size() + l.k_sum.size()*4;
        out.scalar(size);
        for (const auto & l : b.layers) {
            out.scalar(l.n_tokens); out.scalar(l.k_gpu_tokens);
            out.scalar(l.v_gpu_tokens); out.scalar(l.mean_tokens);
            out.scalar(uint8_t(l.k_gpu_fmt)); out.scalar(uint8_t(l.v_gpu_fmt));
            out.vector(l.k); out.vector(l.v);
            const auto write_packed = [&](const HostKvRecord & record) {
                out.scalar(uint64_t(record.size()));
                if (!record) return;
                const auto lease = record.lease();
                if (lease.data()) { out.write(lease.data(), lease.size()); return; }
                uint8_t chunk[16384];
                for (size_t offset = 0; offset < lease.size();) {
                    const auto bytes = std::min(sizeof(chunk), lease.size() - offset);
                    lease.read(offset, chunk, bytes); out.write(chunk, bytes); offset += bytes;
                }
            };
            write_packed(l.k_gpu); write_packed(l.v_gpu);
            out.vector(l.k_sum);
        }
    }
}

void RawKvStore::snapshot_read(SnapshotReader & in, uint32_t max_blocks) {
    in.expect(uint32_t(0x31564b52));
    in.expect(cfg_.n_layer); in.expect(cfg_.n_embd_k); in.expect(cfg_.n_embd_v);
    in.expect(cfg_.block_tokens); in.expect(cfg_.k_row_bytes);
    in.expect(cfg_.k_gpu_row_bytes); in.expect(cfg_.v_gpu_row_bytes);
    const auto n = in.scalar<uint64_t>();
    if (n > uint64_t(max_blocks) + 1 || n > in.remaining()/8)
        throw std::runtime_error("invalid snapshot block count");
    auto read_packed = [&](uint64_t max_bytes) {
        const auto bytes = in.scalar<uint64_t>();
        if (bytes > max_bytes || bytes > SIZE_MAX || bytes > in.remaining())
            throw SnapshotCorrupt("invalid native KV snapshot length");
        auto record = payload_storage_->allocate(static_cast<size_t>(bytes));
        in.read(record.writable_data(), bytes);
        record.seal();
        return record;
    };
    std::vector<BlockRaw> blocks(static_cast<size_t>(n));
    for (auto & b : blocks) {
        const auto size = in.scalar<uint64_t>();
        const auto before = in.remaining();
        if (size > before) throw std::runtime_error("truncated snapshot block");
        b.layers.resize(cfg_.n_layer);
        for (auto & l : b.layers) {
            l.n_tokens = in.scalar<uint32_t>(); l.k_gpu_tokens = in.scalar<uint32_t>();
            l.v_gpu_tokens = in.scalar<uint32_t>(); l.mean_tokens = in.scalar<uint32_t>();
            if (std::max({l.n_tokens,l.k_gpu_tokens,l.v_gpu_tokens,l.mean_tokens}) > cfg_.block_tokens)
                throw std::runtime_error("invalid snapshot token count");
            const auto kfmt = in.scalar<uint8_t>(), vfmt = in.scalar<uint8_t>();
            if (kfmt > 1 || vfmt > 1) throw std::runtime_error("invalid snapshot KV format");
            l.k_gpu_fmt = kfmt; l.v_gpu_fmt = vfmt;
            l.k = in.vector<uint8_t>(uint64_t(cfg_.block_tokens)*k_row_bytes());
            l.v = in.vector<uint16_t>(uint64_t(cfg_.block_tokens)*cfg_.n_embd_v);
            l.k_gpu = read_packed(uint64_t(cfg_.block_tokens)*cfg_.k_gpu_row_bytes);
            l.v_gpu = read_packed(uint64_t(cfg_.block_tokens)*cfg_.v_gpu_row_bytes);
            l.k_sum = in.vector<float>(cfg_.n_embd_k);
            if ((l.k_gpu.size() < uint64_t(l.k_gpu_tokens)*cfg_.k_gpu_row_bytes) ||
                (l.v_gpu.size() < uint64_t(l.v_gpu_tokens)*cfg_.v_gpu_row_bytes) ||
                (l.mean_tokens && l.k_sum.size() != cfg_.n_embd_k))
                throw std::runtime_error("incomplete snapshot KV block");
        }
        if (before - in.remaining() != size) throw std::runtime_error("invalid snapshot block size");
    }
    std::lock_guard<std::mutex> lk(mu_);
    blocks_.swap(blocks);
}

} // namespace kvmem
