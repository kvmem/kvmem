// SPDX-License-Identifier: Apache-2.0
// hipfire KVMem adapter — C ABI implementation. See hipfire_kvmem_shim.h.

#include "hipfire_kvmem_shim.h"

#include <kvmem/host_kv_storage.hpp>
#include <kvmem/kvmem_store.hpp>
#include <kvmem/graph_recall.hpp>
#include <kvmem/mean_key_retrieval.hpp>
#include <kvmem/raw_kv_store.hpp>
#include <kvmem/spill_io.hpp>

#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_error;

void set_error(const char * what) noexcept {
    try {
        g_error = what ? what : "unknown error";
    } catch (...) {
        // A failing diagnostic must not escape the ABI boundary.
    }
}

void clear_error() noexcept {
    try {
        g_error.clear();
    } catch (...) {
    }
}

template <typename F>
int guarded(F && f, const char * where) noexcept {
    try {
        f();
        return HK_OK;
    } catch (const std::exception & e) {
        set_error((std::string(where) + ": " + e.what()).c_str());
        return HK_ERR;
    } catch (...) {
        set_error((std::string(where) + ": unknown C++ exception").c_str());
        return HK_ERR;
    }
}

std::vector<uint32_t> copy_ids(const uint32_t * ids, uint32_t n) {
    if (!ids || n == 0) return {};
    return std::vector<uint32_t>(ids, ids + n);
}

} // namespace

struct hk_storage {
    std::shared_ptr<kvmem::HostKvStorage> impl;
};

struct hk_raw_store {
    std::unique_ptr<kvmem::RawKvStore> impl;
};

struct hk_kv_store {
    std::unique_ptr<kvmem::KvMemStore> impl;
};

// ── diagnostics ──

extern "C" const char * hk_last_error(void) {
    return g_error.empty() ? nullptr : g_error.c_str();
}

extern "C" void hk_clear_error(void) {
    clear_error();
}

extern "C" const char * hk_version(void) {
    return "kvmem-hipfire-shim-1";
}

// ── host payload storage ──

extern "C" hk_storage * hk_storage_create(uint64_t capacity_bytes, const char * spill_dir,
                                          uint64_t spill_capacity_bytes) {
    clear_error();
    try {
        std::shared_ptr<kvmem::HostKvStorage> impl;
        if (spill_dir && *spill_dir) {
            auto file = std::make_shared<kvmem::SpillFile>(
                std::filesystem::path(spill_dir), spill_capacity_bytes);
            impl = std::make_shared<kvmem::HostKvStorage>(capacity_bytes, std::move(file));
        } else {
            impl = std::make_shared<kvmem::HostKvStorage>(capacity_bytes);
        }
        return new hk_storage{std::move(impl)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_storage_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_storage_destroy(hk_storage * s) {
    delete s;
}

extern "C" uint64_t hk_storage_charged_bytes(const hk_storage * s) {
    if (!s || !s->impl) return 0;
    return s->impl->charged_bytes();
}

extern "C" uint64_t hk_storage_disk_bytes(const hk_storage * s) {
    if (!s || !s->impl) return 0;
    return s->impl->disk_bytes();
}

extern "C" void hk_storage_trim(hk_storage * s, uint64_t resident_target) {
    if (!s || !s->impl) return;
    guarded([&] { s->impl->trim(resident_target); }, "hk_storage_trim");
}

// ── block statistics / packed KV store ──

extern "C" hk_raw_store * hk_raw_store_create(const hk_raw_store_config * cfg,
                                              hk_storage * storage) {
    clear_error();
    if (!cfg) {
        set_error("hk_raw_store_create: null config");
        return nullptr;
    }
    try {
        kvmem::RawKvStoreConfig c;
        c.n_layer = cfg->n_layer;
        c.n_embd_k = cfg->n_embd_k;
        c.n_embd_v = cfg->n_embd_v;
        c.block_tokens = cfg->block_tokens;
        c.k_row_bytes = cfg->k_row_bytes;
        c.k_gpu_row_bytes = cfg->k_gpu_row_bytes;
        c.v_gpu_row_bytes = cfg->v_gpu_row_bytes;
        if (storage && storage->impl) c.payload_storage = storage->impl;
        return new hk_raw_store{std::make_unique<kvmem::RawKvStore>(c)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_raw_store_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_raw_store_destroy(hk_raw_store * st) {
    delete st;
}

extern "C" void hk_raw_store_ensure_blocks(hk_raw_store * st, uint32_t block_count) {
    if (!st || !st->impl) return;
    guarded([&] { st->impl->ensure_blocks(block_count); }, "hk_raw_store_ensure_blocks");
}

extern "C" void hk_raw_store_write_layer_mean_k(hk_raw_store * st, uint32_t pos0,
                                                uint32_t n, uint32_t il, const float * k) {
    if (!st || !st->impl) return;
    guarded([&] { st->impl->write_layer_mean_k(pos0, n, il, k); },
            "hk_raw_store_write_layer_mean_k");
}

extern "C" void hk_raw_store_write_layer_k_gpu(hk_raw_store * st, uint32_t pos0, uint32_t n,
                                               uint32_t il, const uint8_t * k) {
    if (!st || !st->impl) return;
    guarded([&] { st->impl->write_layer_k_gpu(pos0, n, il, k); },
            "hk_raw_store_write_layer_k_gpu");
}

extern "C" void hk_raw_store_write_layer_v_gpu(hk_raw_store * st, uint32_t pos0, uint32_t n,
                                               uint32_t il, const uint8_t * v) {
    if (!st || !st->impl) return;
    guarded([&] { st->impl->write_layer_v_gpu(pos0, n, il, v); },
            "hk_raw_store_write_layer_v_gpu");
}

extern "C" int hk_raw_store_copy_k_gpu(const hk_raw_store * st, uint32_t block, uint32_t il,
                                       uint8_t * out, uint32_t n) {
    if (!st || !st->impl) return 0;
    return st->impl->copy_k_gpu(block, il, out, n) ? 1 : 0;
}

extern "C" int hk_raw_store_copy_v_gpu(const hk_raw_store * st, uint32_t block, uint32_t il,
                                       uint8_t * out, uint32_t n) {
    if (!st || !st->impl) return 0;
    return st->impl->copy_v_gpu(block, il, out, n) ? 1 : 0;
}

extern "C" void hk_raw_store_mean_k(const hk_raw_store * st, uint32_t block, uint32_t il,
                                    float * out) {
    if (!st || !st->impl || !out) return;
    st->impl->mean_k(block, il, out);
}

extern "C" uint32_t hk_raw_store_mean_k_tokens(const hk_raw_store * st, uint32_t block,
                                               uint32_t il) {
    if (!st || !st->impl) return 0;
    return st->impl->mean_k_tokens(block, il);
}

extern "C" uint32_t hk_raw_store_n_tokens(const hk_raw_store * st, uint32_t block) {
    if (!st || !st->impl) return 0;
    return st->impl->n_tokens(block);
}

extern "C" int hk_raw_store_has_k_gpu(const hk_raw_store * st, uint32_t block, uint32_t il,
                                      uint32_t n) {
    if (!st || !st->impl) return 0;
    return st->impl->has_k_gpu(block, il, n) ? 1 : 0;
}

extern "C" int hk_raw_store_has_v_gpu(const hk_raw_store * st, uint32_t block, uint32_t il,
                                      uint32_t n) {
    if (!st || !st->impl) return 0;
    return st->impl->has_v_gpu(block, il, n) ? 1 : 0;
}

extern "C" void hk_raw_store_truncate_to(hk_raw_store * st, uint32_t token_pos) {
    if (!st || !st->impl) return;
    guarded([&] { st->impl->truncate_to(token_pos); }, "hk_raw_store_truncate_to");
}

extern "C" void hk_raw_store_clear(hk_raw_store * st) {
    if (!st || !st->impl) return;
    guarded([&] { st->impl->clear(); }, "hk_raw_store_clear");
}

extern "C" uint64_t hk_raw_store_allocated_bytes(const hk_raw_store * st) {
    if (!st || !st->impl) return 0;
    return static_cast<uint64_t>(st->impl->allocated_bytes());
}

extern "C" uint64_t hk_raw_store_payload_bytes(const hk_raw_store * st) {
    if (!st || !st->impl) return 0;
    return st->impl->payload_bytes();
}

extern "C" size_t hk_raw_store_mean_checkpoint(const hk_raw_store * st, uint32_t token_pos,
                                               float * out, size_t capacity) {
    if (!st || !st->impl) return 0;
    try {
        const std::vector<float> state = st->impl->mean_checkpoint(token_pos);
        if (state.empty()) return 0;
        if (!out) return state.size(); // size query
        if (capacity < state.size()) {
            set_error("hk_raw_store_mean_checkpoint: output capacity too small");
            return 0;
        }
        std::memcpy(out, state.data(), state.size() * sizeof(float));
        return state.size();
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_raw_store_mean_checkpoint: unknown C++ exception");
    }
    return 0;
}

extern "C" void hk_raw_store_restore_mean_checkpoint(hk_raw_store * st, uint32_t token_pos,
                                                     const float * state, size_t len) {
    if (!st || !st->impl) return;
    guarded(
        [&] {
            std::vector<float> v;
            if (state && len) v.assign(state, state + len);
            st->impl->restore_mean_checkpoint(token_pos, v);
        },
        "hk_raw_store_restore_mean_checkpoint");
}

// ── block repository / selection ──

extern "C" hk_kv_store * hk_kv_store_create(const hk_kv_store_config * cfg) {
    clear_error();
    if (!cfg) {
        set_error("hk_kv_store_create: null config");
        return nullptr;
    }
    try {
        kvmem::KvMemStoreConfig c;
        c.block_tokens = cfg->block_tokens;
        c.select_budget = cfg->select_budget;
        c.prefill_budget = cfg->prefill_budget;
        c.gen_budget = cfg->gen_budget;
        c.sink_blocks = cfg->sink_blocks;
        c.recent_blocks = cfg->recent_blocks;
        c.select_method = static_cast<kvmem::KvMemMethod>(cfg->select_method);
        c.retrieval_method = static_cast<kvmem::KvMemRetrievalMethod>(cfg->retrieval_method);
        c.update_mode = static_cast<kvmem::KvMemUpdateMode>(cfg->update_mode);
        c.lrfu = cfg->lrfu != 0;
        c.lrfu_lambda = cfg->lrfu_lambda;
        c.lrfu_top_fraction = cfg->lrfu_top_fraction;
        c.estimated_gpu_block_capacity = cfg->estimated_gpu_block_capacity;
        c.gpu_resident_block_bytes = cfg->gpu_resident_block_bytes;
        c.estimated_block_bytes = cfg->estimated_block_bytes;
        c.cpu_tier_bytes = cfg->cpu_tier_bytes;
        c.gpu_memory_ratio = cfg->gpu_memory_ratio;
        c.gpu_high_watermark = cfg->gpu_high_watermark;
        c.gpu_low_watermark = cfg->gpu_low_watermark;
        return new hk_kv_store{std::make_unique<kvmem::KvMemStore>(c)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_kv_store_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_kv_store_destroy(hk_kv_store * ks) {
    delete ks;
}

extern "C" void hk_kv_store_register_append(hk_kv_store * ks, uint32_t n_new_tokens) {
    if (!ks || !ks->impl) return;
    guarded([&] { ks->impl->register_append(n_new_tokens); }, "hk_kv_store_register_append");
}

extern "C" void hk_kv_store_truncate_to(hk_kv_store * ks, uint32_t token_pos) {
    if (!ks || !ks->impl) return;
    guarded([&] { (void)ks->impl->truncate_to(token_pos); }, "hk_kv_store_truncate_to");
}

extern "C" uint32_t hk_kv_store_block_count(const hk_kv_store * ks) {
    if (!ks || !ks->impl) return 0;
    return ks->impl->block_count();
}

extern "C" uint32_t hk_kv_store_total_tokens(const hk_kv_store * ks) {
    if (!ks || !ks->impl) return 0;
    return ks->impl->total_tokens();
}

extern "C" uint32_t hk_kv_store_select_budget_tokens(const hk_kv_store * ks) {
    if (!ks || !ks->impl) return 0;
    return ks->impl->select_budget_tokens();
}

extern "C" void hk_kv_store_set_runtime_select_budget(hk_kv_store * ks, uint32_t tokens) {
    if (!ks || !ks->impl) return;
    guarded([&] { ks->impl->set_runtime_select_budget(tokens); },
            "hk_kv_store_set_runtime_select_budget");
}

namespace {

template <typename Fn>
uint32_t pick_and_copy(const hk_kv_store * ks, const char * where, const uint32_t * mandatory,
                       uint32_t n_mandatory, uint32_t * out, uint32_t capacity, Fn pick) {
    if (!ks || !ks->impl) {
        set_error((std::string(where) + ": null store").c_str());
        return UINT32_MAX;
    }
    try {
        const std::vector<uint32_t> ids = pick(copy_ids(mandatory, n_mandatory));
        if (out && capacity >= ids.size() && !ids.empty()) {
            std::memcpy(out, ids.data(), ids.size() * sizeof(uint32_t));
        } else if (out && capacity < ids.size()) {
            set_error((std::string(where) + ": output capacity too small").c_str());
        }
        return static_cast<uint32_t>(ids.size());
    } catch (const std::exception & e) {
        set_error((std::string(where) + ": " + e.what()).c_str());
    } catch (...) {
        set_error((std::string(where) + ": unknown C++ exception").c_str());
    }
    return UINT32_MAX;
}

} // namespace

extern "C" uint32_t hk_kv_store_pick_topk_blocks(const hk_kv_store * ks,
                                                 const uint32_t * mandatory, uint32_t n_mandatory,
                                                 uint32_t * out, uint32_t capacity) {
    return pick_and_copy(ks, "hk_kv_store_pick_topk_blocks", mandatory, n_mandatory, out,
                         capacity, [&](const std::vector<uint32_t> & m) {
                             return ks->impl->pick_topk_blocks(m);
                         });
}

extern "C" uint32_t hk_kv_store_pick_prefill_pressure_blocks(const hk_kv_store * ks,
                                                             const uint32_t * mandatory,
                                                             uint32_t n_mandatory,
                                                             uint32_t * out, uint32_t capacity) {
    return pick_and_copy(ks, "hk_kv_store_pick_prefill_pressure_blocks", mandatory, n_mandatory,
                         out, capacity, [&](const std::vector<uint32_t> & m) {
                             return ks->impl->pick_prefill_pressure_blocks(m);
                         });
}

extern "C" void hk_kv_store_set_retrieval_scores(hk_kv_store * ks, const double * scores,
                                                 uint32_t n) {
    if (!ks || !ks->impl) return;
    guarded(
        [&] {
            std::vector<double> v;
            if (scores && n) v.assign(scores, scores + n);
            ks->impl->set_retrieval_scores(v);
        },
        "hk_kv_store_set_retrieval_scores");
}

extern "C" void hk_kv_store_begin_retrieval_round(hk_kv_store * ks) {
    if (!ks || !ks->impl) return;
    guarded([&] { ks->impl->begin_retrieval_round(); }, "hk_kv_store_begin_retrieval_round");
}

extern "C" int hk_kv_store_commit_retrieval_round(hk_kv_store * ks) {
    if (!ks || !ks->impl) return 0;
    bool committed = false;
    guarded([&] { committed = ks->impl->commit_retrieval_round(); },
            "hk_kv_store_commit_retrieval_round");
    return committed ? 1 : 0;
}

extern "C" int hk_kv_store_prefill_needs_offload(const hk_kv_store * ks,
                                                 uint32_t resident_tokens,
                                                 uint32_t incoming_tokens,
                                                 uint32_t pool_tokens) {
    if (!ks || !ks->impl) return 0;
    return ks->impl->prefill_needs_offload(resident_tokens, incoming_tokens, pool_tokens) ? 1
                                                                                         : 0;
}

extern "C" uint32_t hk_kv_store_gpu_high_watermark_tokens(const hk_kv_store * ks,
                                                          uint32_t pool_tokens) {
    if (!ks || !ks->impl) return 0;
    return ks->impl->gpu_high_watermark_tokens(pool_tokens);
}

extern "C" uint32_t hk_kv_store_gpu_low_watermark_tokens(const hk_kv_store * ks,
                                                         uint32_t pool_tokens) {
    if (!ks || !ks->impl) return 0;
    return ks->impl->gpu_low_watermark_tokens(pool_tokens);
}

extern "C" uint64_t hk_kv_store_allocated_bytes(const hk_kv_store * ks) {
    if (!ks || !ks->impl) return 0;
    return static_cast<uint64_t>(ks->impl->allocated_bytes());
}

// ── query-conditioned Mean-K retrieval ──

extern "C" int hk_mean_key_retrieval_scores(const hk_raw_store * keys, uint32_t blocks,
                                            const float * sums, const uint32_t * token_counts,
                                            uint32_t layers, uint32_t head_dim,
                                            uint32_t query_heads, uint32_t kv_heads,
                                            uint32_t aggregation, double * out) {
    clear_error();
    if (!keys || !keys->impl || !sums || !token_counts || !out || layers == 0 ||
        head_dim == 0 || query_heads == 0 || kv_heads == 0) {
        set_error("hk_mean_key_retrieval_scores: invalid arguments");
        return HK_ERR;
    }
    if (aggregation > 1) {
        set_error("hk_mean_key_retrieval_scores: unknown aggregation");
        return HK_ERR;
    }
    try {
        kvmem::MeanKeyQueryView view;
        view.sums = sums;
        view.token_counts = token_counts;
        view.layers = layers;
        view.head_dim = head_dim;
        view.query_heads = query_heads;
        view.kv_heads = kv_heads;
        view.aggregation = static_cast<kvmem::GqaScoreAggregation>(aggregation);
        const std::vector<double> scores =
            kvmem::mean_key_retrieval_scores(*keys->impl, blocks, view);
        if (scores.size() != blocks) {
            set_error("hk_mean_key_retrieval_scores: score count mismatch");
            return HK_ERR;
        }
        std::memcpy(out, scores.data(), scores.size() * sizeof(double));
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_mean_key_retrieval_scores: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" int hk_page_softmax_scores(const hk_raw_store * keys, uint32_t blocks,
                                      uint64_t accepted_through, uint32_t block_tokens,
                                      uint32_t budget_blocks, uint32_t sink_blocks,
                                      uint32_t recent_blocks, const float * q_data,
                                      const uint32_t * layer_rows, uint32_t layers, uint32_t rows,
                                      uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
                                      float attention_scale, uint32_t aggregation, double * out) {
    clear_error();
    if (!keys || !keys->impl || !q_data || !layer_rows || !out || layers == 0 ||
        rows == 0 || head_dim == 0 || query_heads == 0 || kv_heads == 0 ||
        block_tokens == 0 || budget_blocks == 0) {
        set_error("hk_page_softmax_scores: invalid arguments");
        return HK_ERR;
    }
    if (aggregation > 1) {
        set_error("hk_page_softmax_scores: unknown aggregation");
        return HK_ERR;
    }
    try {
        const kvmem::QueryScoreFrame frame = kvmem::query_score_frame(
            accepted_through, block_tokens, budget_blocks, sink_blocks, recent_blocks, {});
        if (frame.starts.size() != blocks) {
            set_error("hk_page_softmax_scores: frame block count mismatch");
            return HK_ERR;
        }
        kvmem::PageQueryView view;
        view.data = q_data;
        view.layer_rows = layer_rows;
        view.layers = layers;
        view.rows = rows;
        view.query_heads = query_heads;
        view.kv_heads = kv_heads;
        view.head_dim = head_dim;
        view.attention_scale = attention_scale;
        const std::vector<double> scores = kvmem::page_softmax_scores(
            *keys->impl, view, frame, static_cast<kvmem::GqaScoreAggregation>(aggregation));
        if (scores.size() != blocks) {
            set_error("hk_page_softmax_scores: score count mismatch");
            return HK_ERR;
        }
        std::memcpy(out, scores.data(), scores.size() * sizeof(double));
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_page_softmax_scores: unknown C++ exception");
    }
    return HK_ERR;
}
