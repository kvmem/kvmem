// SPDX-License-Identifier: Apache-2.0
// hipfire KVMem adapter — C ABI implementation. See hipfire_kvmem_shim.h.

#include "hipfire_kvmem_shim.h"

#include <kvmem/host_kv_storage.hpp>
#include <kvmem/kvmem_store.hpp>
#include <kvmem/graph_recall.hpp>
#include <kvmem/mean_key_retrieval.hpp>
#include <kvmem/raw_kv_store.hpp>
#include <kvmem/spill_io.hpp>
#include <kvmem/memory_contract.hpp>
#include <kvmem/native_kv_payload.hpp>
#include <kvmem/layerwise_memory.hpp>
#include <kvmem/layerwise_request.hpp>
#include <kvmem/query_projection_tape.hpp>
#include <kvmem/layerwise_pressure.hpp>

#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <optional>
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

// ── layerwise native payload foundation ──

namespace {

kvmem::BackendIdentity backend_id(const hk_backend_identity * b) {
    if (!b || !b->backend || !b->revision || !b->model || !b->state_schema)
        throw std::invalid_argument("null backend identity field");
    return {b->backend, b->revision, b->model, b->state_schema};
}

kvmem::BlockDescriptor block_desc(const hk_block_descriptor * b) {
    if (!b) throw std::invalid_argument("null block descriptor");
    kvmem::BlockDescriptor d;
    d.session = {b->session.id, b->session.incarnation};
    d.id = b->id;
    d.original_start = b->original_start;
    d.valid_tokens = b->valid_tokens;
    d.content_version = b->content_version;
    d.statistics_version = b->statistics_version;
    return d;
}

void block_desc_to(const kvmem::BlockDescriptor & b, hk_block_descriptor * out) {
    out->session = {b.session.id, b.session.incarnation};
    out->id = b.id;
    out->original_start = b.original_start;
    out->valid_tokens = b.valid_tokens;
    out->content_version = b.content_version;
    out->statistics_version = b.statistics_version;
}

kvmem::PayloadPlane plane_from(const hk_payload_plane & p) {
    if (p.kind > HK_PLANE_PACKED_KV) throw std::invalid_argument("unknown payload plane kind");
    kvmem::PayloadPlane out;
    out.layer = p.layer;
    out.kind = static_cast<kvmem::PayloadPlaneKind>(p.kind);
    out.offset = p.offset;
    out.bytes = p.bytes;
    out.alignment = p.alignment;
    out.token_stride = p.token_stride;
    out.rank = p.rank;
    out.head_begin = p.head_begin;
    out.head_count = p.head_count;
    return out;
}

void plane_to(const kvmem::PayloadPlane & p, hk_payload_plane * out) {
    out->layer = p.layer;
    out->kind = static_cast<uint32_t>(p.kind);
    out->offset = p.offset;
    out->bytes = p.bytes;
    out->alignment = p.alignment;
    out->token_stride = p.token_stride;
    out->rank = p.rank;
    out->head_begin = p.head_begin;
    out->head_count = p.head_count;
}

kvmem::RawKvStoreConfig raw_config(const hk_raw_store_config * cfg) {
    kvmem::RawKvStoreConfig c;
    c.n_layer = cfg->n_layer;
    c.n_embd_k = cfg->n_embd_k;
    c.n_embd_v = cfg->n_embd_v;
    c.block_tokens = cfg->block_tokens;
    c.k_row_bytes = cfg->k_row_bytes;
    c.k_gpu_row_bytes = cfg->k_gpu_row_bytes;
    c.v_gpu_row_bytes = cfg->v_gpu_row_bytes;
    return c;
}

} // namespace

struct hk_payload_layout {
    kvmem::PayloadLayout impl;
};
struct hk_memory_descriptor {
    kvmem::MemoryDescriptor impl;
};
struct hk_kv_regions {
    std::unique_ptr<kvmem::NativeKvLayerRegions> impl;
};
struct hk_host_record {
    kvmem::HostKvRecord impl;
};
struct hk_host_source {
    kvmem::HostKvSource impl;
};
struct hk_native_payload {
    kvmem::NativeKvPayload impl;
};
struct hk_native_archive {
    std::unique_ptr<kvmem::NativeKvArchive> impl;
};

extern "C" hk_payload_layout * hk_payload_layout_from_raw(const hk_raw_store_config * cfg,
                                                          const char * format,
                                                          const uint32_t * layers,
                                                          uint32_t n_layers) {
    clear_error();
    if (!cfg || !format || (!layers && n_layers)) {
        set_error("hk_payload_layout_from_raw: invalid arguments");
        return nullptr;
    }
    try {
        auto layout = kvmem::raw_kv_payload_layout(raw_config(cfg), format, copy_ids(layers, n_layers));
        return new hk_payload_layout{std::move(layout)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_payload_layout_from_raw: unknown C++ exception");
    }
    return nullptr;
}

extern "C" hk_payload_layout * hk_payload_layout_create(const char * format, uint32_t version,
                                                        uint32_t block_tokens, uint64_t record_bytes,
                                                        uint64_t alignment,
                                                        const hk_payload_plane * planes,
                                                        uint32_t n_planes) {
    clear_error();
    if (!format || (!planes && n_planes)) {
        set_error("hk_payload_layout_create: invalid arguments");
        return nullptr;
    }
    try {
        kvmem::PayloadLayout l;
        l.format = format;
        l.version = version;
        l.block_tokens = block_tokens;
        l.record_bytes = record_bytes;
        l.alignment = alignment;
        for (uint32_t i = 0; i < n_planes; ++i) l.planes.push_back(plane_from(planes[i]));
        l.validate();
        return new hk_payload_layout{std::move(l)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_payload_layout_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_payload_layout_destroy(hk_payload_layout * l) { delete l; }

extern "C" uint64_t hk_payload_layout_record_bytes(const hk_payload_layout * l) {
    return l ? l->impl.record_bytes : 0;
}

extern "C" uint32_t hk_payload_layout_block_tokens(const hk_payload_layout * l) {
    return l ? l->impl.block_tokens : 0;
}

extern "C" uint32_t hk_payload_layout_plane_count(const hk_payload_layout * l) {
    return l ? static_cast<uint32_t>(l->impl.planes.size()) : 0;
}

extern "C" int hk_payload_layout_plane(const hk_payload_layout * l, uint32_t i,
                                       hk_payload_plane * out) {
    if (!l || !out || i >= l->impl.planes.size()) {
        set_error("hk_payload_layout_plane: invalid arguments");
        return HK_ERR;
    }
    plane_to(l->impl.planes[i], out);
    return HK_OK;
}

extern "C" hk_memory_descriptor * hk_memory_descriptor_create(const hk_backend_identity * identity,
                                                              const hk_payload_layout * layout,
                                                              const hk_memory_capabilities * cap) {
    clear_error();
    if (!identity || !layout || !cap) {
        set_error("hk_memory_descriptor_create: invalid arguments");
        return nullptr;
    }
    try {
        kvmem::MemoryDescriptor d;
        d.identity = backend_id(identity);
        d.layout = layout->impl;
        d.capabilities.page_tokens = cap->page_tokens;
        d.capabilities.max_position = cap->max_position;
        d.capabilities.native_payload = cap->native_payload != 0;
        d.capabilities.sparse_view = cap->sparse_view != 0;
        d.capabilities.original_rope_positions = cap->original_rope_positions != 0;
        d.capabilities.pre_rope_statistics = cap->pre_rope_statistics != 0;
        d.capabilities.state_checkpoint = cap->state_checkpoint != 0;
        d.capabilities.asynchronous_transfer = cap->asynchronous_transfer != 0;
        d.validate();
        return new hk_memory_descriptor{std::move(d)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_memory_descriptor_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_memory_descriptor_destroy(hk_memory_descriptor * d) { delete d; }

extern "C" hk_host_record * hk_host_record_allocate(hk_storage * s, const void * bytes, size_t len) {
    clear_error();
    if (!s || !s->impl || (!bytes && len)) {
        set_error("hk_host_record_allocate: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_host_record{s->impl->allocate(len, bytes)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_host_record_allocate: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_host_record_destroy(hk_host_record * r) { delete r; }

extern "C" void hk_host_record_seal(hk_host_record * r) {
    if (!r) return;
    guarded([&] { r->impl.seal(); }, "hk_host_record_seal");
}

extern "C" size_t hk_host_record_size(const hk_host_record * r) { return r ? r->impl.size() : 0; }

extern "C" int hk_host_record_read(const hk_host_record * r, size_t offset, void * dest,
                                   size_t bytes) {
    if (!r || (!dest && bytes)) {
        set_error("hk_host_record_read: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] { r->impl.read(offset, dest, bytes); }, "hk_host_record_read");
}

extern "C" hk_host_source * hk_host_record_source(const hk_host_record * r) {
    clear_error();
    if (!r) {
        set_error("hk_host_record_source: null record");
        return nullptr;
    }
    try {
        return new hk_host_source{r->impl.source()};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_host_record_source: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_host_source_destroy(hk_host_source * s) { delete s; }

extern "C" size_t hk_host_source_size(const hk_host_source * s) { return s ? s->impl.size() : 0; }

extern "C" hk_kv_regions * hk_kv_regions_create(const hk_payload_layout * layout, uint32_t layer,
                                                int has_rank, uint32_t rank) {
    clear_error();
    if (!layout) {
        set_error("hk_kv_regions_create: null layout");
        return nullptr;
    }
    try {
        std::optional<uint32_t> r;
        if (has_rank) r = rank;
        return new hk_kv_regions{
            std::make_unique<kvmem::NativeKvLayerRegions>(layout->impl, layer, r)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_kv_regions_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_kv_regions_destroy(hk_kv_regions * r) { delete r; }

extern "C" uint64_t hk_kv_regions_packed_bytes(const hk_kv_regions * r) {
    return (r && r->impl) ? r->impl->packed_bytes() : 0;
}

extern "C" uint64_t hk_kv_regions_plane_bytes(const hk_kv_regions * r) {
    return (r && r->impl) ? r->impl->plane_bytes() : 0;
}

extern "C" uint64_t hk_kv_regions_record_bytes(const hk_kv_regions * r) {
    return (r && r->impl) ? r->impl->record_bytes() : 0;
}

extern "C" uint32_t hk_kv_regions_region_count(const hk_kv_regions * r) {
    return (r && r->impl) ? static_cast<uint32_t>(r->impl->regions().size()) : 0;
}

extern "C" int hk_kv_regions_region(const hk_kv_regions * r, uint32_t i, hk_payload_plane * out_plane,
                                    uint64_t * out_packed_offset) {
    if (!r || !r->impl || !out_plane || !out_packed_offset || i >= r->impl->regions().size()) {
        set_error("hk_kv_regions_region: invalid arguments");
        return HK_ERR;
    }
    const auto & region = r->impl->regions()[i];
    plane_to(region.plane, out_plane);
    *out_packed_offset = region.packed_offset;
    return HK_OK;
}

extern "C" int hk_kv_regions_read(const hk_kv_regions * r, const hk_host_source * source,
                                  uint64_t packed_offset, void * dest, size_t bytes,
                                  size_t max_read_bytes) {
    if (!r || !r->impl || !source || (!dest && bytes)) {
        set_error("hk_kv_regions_read: invalid arguments");
        return HK_ERR;
    }
    return guarded(
        [&] { r->impl->read(source->impl, packed_offset, dest, bytes, max_read_bytes); },
        "hk_kv_regions_read");
}

extern "C" hk_native_payload * hk_native_payload_export(const hk_raw_store * raw,
                                                        const hk_backend_identity * backend,
                                                        const hk_payload_layout * layout,
                                                        const hk_block_descriptor * block) {
    clear_error();
    if (!raw || !raw->impl || !backend || !layout || !block) {
        set_error("hk_native_payload_export: invalid arguments");
        return nullptr;
    }
    try {
        auto p = kvmem::export_raw_kv_payload(*raw->impl, backend_id(backend), layout->impl,
                                              block_desc(block));
        return new hk_native_payload{std::move(p)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_native_payload_export: unknown C++ exception");
    }
    return nullptr;
}

extern "C" hk_native_payload * hk_native_payload_create(const hk_backend_identity * backend,
                                                        const hk_payload_layout * layout,
                                                        const hk_block_descriptor * block,
                                                        const void * bytes, size_t len) {
    clear_error();
    if (!backend || !layout || !block || (!bytes && len)) {
        set_error("hk_native_payload_create: invalid arguments");
        return nullptr;
    }
    try {
        std::vector<uint8_t> v;
        if (bytes && len) {
            const auto * b = static_cast<const uint8_t *>(bytes);
            v.assign(b, b + len);
        }
        kvmem::NativeKvPayload p(backend_id(backend), layout->impl, block_desc(block), std::move(v));
        return new hk_native_payload{std::move(p)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_native_payload_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_native_payload_destroy(hk_native_payload * p) { delete p; }

extern "C" const uint8_t * hk_native_payload_bytes(const hk_native_payload * p) {
    return p ? p->impl.bytes().data() : nullptr;
}

extern "C" size_t hk_native_payload_size(const hk_native_payload * p) {
    return p ? p->impl.bytes().size() : 0;
}

extern "C" int hk_native_payload_block(const hk_native_payload * p, hk_block_descriptor * out) {
    if (!p || !out) {
        set_error("hk_native_payload_block: invalid arguments");
        return HK_ERR;
    }
    block_desc_to(p->impl.block(), out);
    return HK_OK;
}

extern "C" int hk_native_payload_import(hk_raw_store * raw, const hk_backend_identity * backend,
                                        const hk_payload_layout * layout, const hk_native_payload * p) {
    if (!raw || !raw->impl || !backend || !layout || !p) {
        set_error("hk_native_payload_import: invalid arguments");
        return HK_ERR;
    }
    return guarded(
        [&] { kvmem::import_raw_kv_payload(*raw->impl, backend_id(backend), layout->impl, p->impl); },
        "hk_native_payload_import");
}

extern "C" hk_native_archive * hk_native_archive_create(const hk_backend_identity * backend,
                                                        hk_session_identity session,
                                                        const hk_payload_layout * layout,
                                                        uint64_t capacity_bytes) {
    clear_error();
    if (!backend || !layout) {
        set_error("hk_native_archive_create: invalid arguments");
        return nullptr;
    }
    try {
        kvmem::SessionIdentity s{session.id, session.incarnation};
        return new hk_native_archive{std::make_unique<kvmem::NativeKvArchive>(
            backend_id(backend), s, layout->impl, capacity_bytes)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_native_archive_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_native_archive_destroy(hk_native_archive * a) { delete a; }

extern "C" int hk_native_archive_put(hk_native_archive * a, const hk_native_payload * p) {
    if (!a || !a->impl || !p) {
        set_error("hk_native_archive_put: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] { a->impl->put(p->impl); }, "hk_native_archive_put");
}

extern "C" hk_native_payload * hk_native_archive_get(const hk_native_archive * a,
                                                     const hk_block_descriptor * expected) {
    clear_error();
    if (!a || !a->impl || !expected) {
        set_error("hk_native_archive_get: invalid arguments");
        return nullptr;
    }
    try {
        const auto & p = a->impl->get(block_desc(expected));
        return new hk_native_payload{p};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_native_archive_get: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_native_archive_erase(hk_native_archive * a, uint64_t block_id) {
    if (!a || !a->impl) return;
    guarded([&] { a->impl->erase(block_id); }, "hk_native_archive_erase");
}

extern "C" uint64_t hk_native_archive_bytes(const hk_native_archive * a) {
    return (a && a->impl) ? a->impl->bytes() : 0;
}

// ── layerwise catalog + planning ──
namespace {

kvmem::MemoryStamp stamp_from(const hk_memory_stamp & s) {
    kvmem::MemoryStamp out;
    out.valid = s.valid != 0;
    out.session = {s.session.id, s.session.incarnation};
    out.history_revision = s.history_revision;
    out.view_version = s.view_version;
    out.frontier = s.frontier;
    out.execution_history = s.execution_history;
    return out;
}

void stamp_to(const kvmem::MemoryStamp & s, hk_memory_stamp * out) {
    out->valid = s.valid ? 1u : 0u;
    out->session = {s.session.id, s.session.incarnation};
    out->history_revision = s.history_revision;
    out->view_version = s.view_version;
    out->frontier = s.frontier;
    out->execution_history = s.execution_history;
}

std::vector<uint64_t> copy_u64(const uint64_t * p, size_t n) {
    if (!p || !n) return {};
    return std::vector<uint64_t>(p, p + n);
}

kvmem::MemorySnapshot snapshot_from(const hk_memory_snapshot & s) {
    if (!s.blocks && s.block_count) throw std::invalid_argument("null snapshot blocks");
    if (!s.resident && s.resident_count) throw std::invalid_argument("null snapshot resident set");
    kvmem::MemorySnapshot out;
    out.stamp = stamp_from(s.stamp);
    for (size_t i = 0; i < s.block_count; ++i)
        out.blocks.push_back({block_desc(&s.blocks[i].block), s.blocks[i].host_version});
    out.resident = copy_u64(s.resident, s.resident_count);
    return out;
}

kvmem::MemoryBudget budget_from(const hk_memory_budget & b) {
    return {b.selected_tokens, b.reserve_tokens, b.device_tokens, b.host_payload_bytes};
}

kvmem::LayerWorkingSetSelection selection_from(const hk_layer_selection & s) {
    if (!s.selected && s.selected_count) throw std::invalid_argument("null selection blocks");
    if (!s.mandatory && s.mandatory_count) throw std::invalid_argument("null mandatory blocks");
    kvmem::LayerWorkingSetSelection out;
    out.model_layer = s.model_layer;
    out.expected = stamp_from(s.expected);
    out.selected = copy_u64(s.selected, s.selected_count);
    out.mandatory = copy_u64(s.mandatory, s.mandatory_count);
    out.budget = budget_from(s.budget);
    return out;
}

void view_block_to(const kvmem::ViewBlock & v, hk_view_block * out) {
    block_desc_to(v.block, &out->block);
    out->compact_start = v.compact_start;
}

} // namespace

struct hk_memory_catalog {
    std::unique_ptr<kvmem::LayerwiseMemoryCatalog> impl;
};
struct hk_layer_plan {
    kvmem::LayerWorkingSetPlan impl;
};
struct hk_layerwise_plan {
    kvmem::LayerwiseWorkingSetPlan impl;
};

extern "C" hk_memory_catalog * hk_memory_catalog_create(const hk_memory_descriptor * canonical,
                                                        const hk_layer_memory_snapshot * layers,
                                                        size_t n_layers) {
    clear_error();
    if (!canonical || (!layers && n_layers)) {
        set_error("hk_memory_catalog_create: invalid arguments");
        return nullptr;
    }
    try {
        std::vector<kvmem::LayerMemorySnapshot> snaps;
        for (size_t i = 0; i < n_layers; ++i)
            snaps.push_back({layers[i].model_layer, snapshot_from(layers[i].memory)});
        auto cat = std::make_unique<kvmem::LayerwiseMemoryCatalog>(canonical->impl, std::move(snaps));
        return new hk_memory_catalog{std::move(cat)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_memory_catalog_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_memory_catalog_destroy(hk_memory_catalog * c) { delete c; }

extern "C" size_t hk_memory_catalog_layer_count(const hk_memory_catalog * c) {
    return (c && c->impl) ? c->impl->layers().size() : 0;
}

extern "C" uint32_t hk_memory_catalog_model_layer(const hk_memory_catalog * c, size_t i) {
    if (!c || !c->impl || i >= c->impl->layers().size()) return 0;
    return c->impl->layers()[i].model_layer;
}

extern "C" int hk_memory_catalog_snapshot_stamp(const hk_memory_catalog * c, uint32_t model_layer,
                                                hk_memory_stamp * out) {
    if (!c || !c->impl || !out) {
        set_error("hk_memory_catalog_snapshot_stamp: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] { stamp_to(c->impl->snapshot(model_layer).stamp, out); },
                   "hk_memory_catalog_snapshot_stamp");
}

extern "C" size_t hk_memory_catalog_snapshot_block_count(const hk_memory_catalog * c,
                                                         uint32_t model_layer) {
    if (!c || !c->impl) return 0;
    try {
        return c->impl->snapshot(model_layer).blocks.size();
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_memory_catalog_snapshot_block_count: unknown C++ exception");
    }
    return 0;
}

extern "C" int hk_memory_catalog_snapshot_block(const hk_memory_catalog * c, uint32_t model_layer,
                                                size_t i, hk_block_copies * out) {
    if (!c || !c->impl || !out) {
        set_error("hk_memory_catalog_snapshot_block: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] {
        const auto & blocks = c->impl->snapshot(model_layer).blocks;
        if (i >= blocks.size()) throw std::out_of_range("snapshot block index");
        block_desc_to(blocks[i].block, &out->block);
        out->host_version = blocks[i].host_version;
    }, "hk_memory_catalog_snapshot_block");
}

extern "C" size_t hk_memory_catalog_snapshot_resident_count(const hk_memory_catalog * c,
                                                            uint32_t model_layer) {
    if (!c || !c->impl) return 0;
    try {
        return c->impl->snapshot(model_layer).resident.size();
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_memory_catalog_snapshot_resident_count: unknown C++ exception");
    }
    return 0;
}

extern "C" uint64_t hk_memory_catalog_snapshot_resident(const hk_memory_catalog * c,
                                                        uint32_t model_layer, size_t i) {
    if (!c || !c->impl) return 0;
    try {
        const auto & resident = c->impl->snapshot(model_layer).resident;
        if (i >= resident.size()) throw std::out_of_range("snapshot resident index");
        return resident[i];
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_memory_catalog_snapshot_resident: unknown C++ exception");
    }
    return 0;
}

extern "C" int hk_memory_catalog_plan_layer(const hk_memory_catalog * c,
                                            const hk_layer_selection * sel, hk_layer_plan ** out) {
    clear_error();
    if (!c || !c->impl || !sel || !out) {
        set_error("hk_memory_catalog_plan_layer: invalid arguments");
        return HK_ERR;
    }
    try {
        auto plan = c->impl->plan_layer(selection_from(*sel));
        *out = new hk_layer_plan{std::move(plan)};
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_memory_catalog_plan_layer: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" int hk_memory_catalog_plan(const hk_memory_catalog * c, const hk_layer_selection * sels,
                                      size_t n_sels, hk_layerwise_plan ** out) {
    clear_error();
    if (!c || !c->impl || (!sels && n_sels) || !out) {
        set_error("hk_memory_catalog_plan: invalid arguments");
        return HK_ERR;
    }
    try {
        std::vector<kvmem::LayerWorkingSetSelection> selections;
        for (size_t i = 0; i < n_sels; ++i) selections.push_back(selection_from(sels[i]));
        auto plan = c->impl->plan(selections);
        *out = new hk_layerwise_plan{std::move(plan)};
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_memory_catalog_plan: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" void hk_layer_plan_destroy(hk_layer_plan * p) { delete p; }

extern "C" uint32_t hk_layer_plan_model_layer(const hk_layer_plan * p) {
    return p ? p->impl.model_layer : 0;
}
extern "C" uint64_t hk_layer_plan_restore_bytes(const hk_layer_plan * p) {
    return p ? p->impl.restore_bytes : 0;
}
extern "C" uint64_t hk_layer_plan_spill_bytes(const hk_layer_plan * p) {
    return p ? p->impl.spill_bytes : 0;
}
extern "C" uint64_t hk_layer_plan_device_tokens_required(const hk_layer_plan * p) {
    return p ? p->impl.plan.device_tokens_required : 0;
}
extern "C" uint64_t hk_layer_plan_host_payload_bytes_required(const hk_layer_plan * p) {
    return p ? p->impl.plan.host_payload_bytes_required : 0;
}

extern "C" int hk_layer_plan_next_stamp(const hk_layer_plan * p, hk_memory_stamp * out) {
    if (!p || !out) {
        set_error("hk_layer_plan_next_stamp: invalid arguments");
        return HK_ERR;
    }
    stamp_to(p->impl.plan.next.stamp, out);
    return HK_OK;
}
extern "C" uint64_t hk_layer_plan_next_visible_tokens(const hk_layer_plan * p) {
    return p ? p->impl.plan.next.visible_tokens : 0;
}
extern "C" size_t hk_layer_plan_next_block_count(const hk_layer_plan * p) {
    return p ? p->impl.plan.next.blocks.size() : 0;
}
extern "C" int hk_layer_plan_next_block(const hk_layer_plan * p, size_t i, hk_view_block * out) {
    if (!p || !out || i >= p->impl.plan.next.blocks.size()) {
        set_error("hk_layer_plan_next_block: invalid arguments");
        return HK_ERR;
    }
    view_block_to(p->impl.plan.next.blocks[i], out);
    return HK_OK;
}

extern "C" size_t hk_layer_plan_restore_count(const hk_layer_plan * p) {
    return p ? p->impl.plan.restore.size() : 0;
}
extern "C" int hk_layer_plan_restore_block(const hk_layer_plan * p, size_t i,
                                           hk_block_descriptor * out) {
    if (!p || !out || i >= p->impl.plan.restore.size()) {
        set_error("hk_layer_plan_restore_block: invalid arguments");
        return HK_ERR;
    }
    block_desc_to(p->impl.plan.restore[i], out);
    return HK_OK;
}
extern "C" size_t hk_layer_plan_spill_count(const hk_layer_plan * p) {
    return p ? p->impl.plan.spill.size() : 0;
}
extern "C" int hk_layer_plan_spill_block(const hk_layer_plan * p, size_t i, hk_block_descriptor * out) {
    if (!p || !out || i >= p->impl.plan.spill.size()) {
        set_error("hk_layer_plan_spill_block: invalid arguments");
        return HK_ERR;
    }
    block_desc_to(p->impl.plan.spill[i], out);
    return HK_OK;
}
extern "C" size_t hk_layer_plan_evict_count(const hk_layer_plan * p) {
    return p ? p->impl.plan.evict.size() : 0;
}
extern "C" int hk_layer_plan_evict_block(const hk_layer_plan * p, size_t i, hk_block_descriptor * out) {
    if (!p || !out || i >= p->impl.plan.evict.size()) {
        set_error("hk_layer_plan_evict_block: invalid arguments");
        return HK_ERR;
    }
    block_desc_to(p->impl.plan.evict[i], out);
    return HK_OK;
}
extern "C" size_t hk_layer_plan_mandatory_count(const hk_layer_plan * p) {
    return p ? p->impl.plan.mandatory.size() : 0;
}
extern "C" uint64_t hk_layer_plan_mandatory(const hk_layer_plan * p, size_t i) {
    if (!p || i >= p->impl.plan.mandatory.size()) return 0;
    return p->impl.plan.mandatory[i];
}

extern "C" int hk_layer_plan_payload(const hk_layer_plan * p, hk_kv_regions ** out) {
    clear_error();
    if (!p || !out) {
        set_error("hk_layer_plan_payload: invalid arguments");
        return HK_ERR;
    }
    try {
        *out = new hk_kv_regions{std::make_unique<kvmem::NativeKvLayerRegions>(p->impl.payload)};
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_layer_plan_payload: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" void hk_layerwise_plan_destroy(hk_layerwise_plan * p) { delete p; }
extern "C" size_t hk_layerwise_plan_layer_count(const hk_layerwise_plan * p) {
    return p ? p->impl.layers.size() : 0;
}
extern "C" int hk_layerwise_plan_history(const hk_layerwise_plan * p, hk_memory_stamp * out) {
    if (!p || !out) {
        set_error("hk_layerwise_plan_history: invalid arguments");
        return HK_ERR;
    }
    stamp_to(p->impl.history, out);
    return HK_OK;
}
extern "C" uint64_t hk_layerwise_plan_host_payload_bytes_required(const hk_layerwise_plan * p) {
    return p ? p->impl.host_payload_bytes_required : 0;
}
extern "C" int hk_layerwise_plan_layer(const hk_layerwise_plan * p, size_t i, hk_layer_plan ** out) {
    clear_error();
    if (!p || !out || i >= p->impl.layers.size()) {
        set_error("hk_layerwise_plan_layer: invalid arguments");
        return HK_ERR;
    }
    try {
        *out = new hk_layer_plan{p->impl.layers[i]};
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_layerwise_plan_layer: unknown C++ exception");
    }
    return HK_ERR;
}

// ── layerwise Query request state machine ──
namespace {

kvmem::LayerwiseQueryIdentity identity_from(const hk_layerwise_identity * id) {
    if (!id) throw std::invalid_argument("null layerwise identity");
    kvmem::LayerwiseQueryIdentity out;
    out.history = stamp_from(id->history);
    out.request = id->request;
    out.evaluation = {id->evaluation.begin, id->evaluation.end};
    out.query = {id->query.begin, id->query.end};
    return out;
}

kvmem::LayerwiseRequestLayer spec_from(const hk_layer_request_spec & s) {
    if (!s.mandatory && s.mandatory_count) throw std::invalid_argument("null spec mandatory");
    kvmem::LayerwiseRequestLayer out;
    out.model_layer = s.model_layer;
    out.query = {s.query.layer, s.query.query_heads, s.query.kv_heads, s.query.head_dim,
                 s.query.attention_scale};
    out.budget = budget_from(s.budget);
    out.mandatory = copy_ids(s.mandatory, static_cast<uint32_t>(s.mandatory_count));
    return out;
}

kvmem::CurrentLayerQuery current_query_from(const hk_current_layer_query * q) {
    if (!q) throw std::invalid_argument("null current-layer query");
    kvmem::CurrentLayerQuery out;
    out.data = q->data;
    out.layer = q->layer;
    out.rows = q->rows;
    out.query_heads = q->query_heads;
    out.kv_heads = q->kv_heads;
    out.head_dim = q->head_dim;
    out.attention_scale = q->attention_scale;
    return out;
}

kvmem::ExecutionView execution_view_from(const hk_execution_view * v) {
    if (!v || (!v->blocks && v->block_count)) throw std::invalid_argument("null execution view");
    kvmem::ExecutionView out;
    out.stamp = stamp_from(v->stamp);
    for (size_t i = 0; i < v->block_count; ++i)
        out.blocks.push_back({block_desc(&v->blocks[i].block), v->blocks[i].compact_start});
    out.visible_tokens = v->visible_tokens;
    return out;
}

kvmem::NativeCurrentLayerQuery native_query_from(const hk_native_current_layer_query * q) {
    if (!q) throw std::invalid_argument("null native current-layer query");
    kvmem::NativeCurrentLayerQuery out;
    out.data = q->data;
    out.bytes = q->bytes;
    out.layer = q->layer;
    out.rows = q->rows;
    out.query_heads = q->query_heads;
    out.kv_heads = q->kv_heads;
    out.head_dim = q->head_dim;
    out.attention_scale = q->attention_scale;
    out.scalar = static_cast<kvmem::NativeQueryScalar>(q->scalar);
    out.row_stride = q->row_stride;
    out.head_stride = q->head_stride;
    out.dimension_stride = q->dimension_stride;
    return out;
}

hk_native_current_layer_query native_query_to(const kvmem::NativeCurrentLayerQuery & t) {
    hk_native_current_layer_query out;
    out.data = t.data;
    out.bytes = t.bytes;
    out.layer = t.layer;
    out.rows = t.rows;
    out.query_heads = t.query_heads;
    out.kv_heads = t.kv_heads;
    out.head_dim = t.head_dim;
    out.attention_scale = t.attention_scale;
    out.scalar = static_cast<uint32_t>(t.scalar);
    out.row_stride = t.row_stride;
    out.head_stride = t.head_stride;
    out.dimension_stride = t.dimension_stride;
    return out;
}

hk_query_score_frame score_frame_to(const kvmem::QueryScoreFrame & f) {
    hk_query_score_frame out;
    out.starts = f.starts.data();
    out.start_count = f.starts.size();
    out.counts = f.counts.data();
    out.count_count = f.counts.size();
    out.score_mask = f.score_mask.data();
    out.score_mask_count = f.score_mask.size();
    out.transport_mask = f.transport_mask.data();
    out.transport_mask_count = f.transport_mask.size();
    out.accepted_through = f.accepted_through;
    return out;
}

} // namespace

struct hk_layerwise_request {
    std::unique_ptr<kvmem::LayerwiseQueryRequest> impl;
    const hk_raw_store * keys; // borrowed; same store admitted, passed to the native scorer
};

extern "C" hk_layerwise_request * hk_layerwise_request_create(
    const hk_layerwise_identity * identity, const hk_memory_catalog * catalog,
    const hk_kv_store * policy, const hk_raw_store * keys, const hk_layer_request_spec * specs,
    size_t n_specs, uint64_t native_history_tokens, hk_source_revision_fn revision,
    void * revision_user) {
    clear_error();
    if (!identity || !catalog || !catalog->impl || !policy || !policy->impl || !keys ||
        !keys->impl || (!specs && n_specs)) {
        set_error("hk_layerwise_request_create: invalid arguments");
        return nullptr;
    }
    try {
        std::vector<kvmem::LayerwiseRequestLayer> layers;
        for (size_t i = 0; i < n_specs; ++i) layers.push_back(spec_from(specs[i]));
        kvmem::LayerwiseQueryRequest::SourceRevision rev;
        if (revision) rev = [revision, revision_user]() { return stamp_from(revision(revision_user)); };
        auto req = std::make_unique<kvmem::LayerwiseQueryRequest>(
            identity_from(identity), *catalog->impl, *policy->impl, *keys->impl, std::move(layers),
            native_history_tokens, std::move(rev));
        return new hk_layerwise_request{std::move(req), keys};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_layerwise_request_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_layerwise_request_destroy(hk_layerwise_request * r) { delete r; }

extern "C" void hk_layerwise_request_abort(hk_layerwise_request * r) {
    if (r && r->impl) r->impl->abort();
}

extern "C" uint32_t hk_layerwise_request_phase(const hk_layerwise_request * r) {
    return (r && r->impl) ? static_cast<uint32_t>(r->impl->phase()) : HK_LW_ABORTED;
}

extern "C" int hk_layerwise_request_work(const hk_layerwise_request * r, hk_layerwise_work * out) {
    if (!r || !r->impl || !out) {
        set_error("hk_layerwise_request_work: invalid arguments");
        return HK_ERR;
    }
    const auto w = r->impl->work();
    out->projected_rows = w.projected_rows;
    out->attended_rows = w.attended_rows;
    out->completed_layers = w.completed_layers;
    return HK_OK;
}

extern "C" int hk_layerwise_request_begin_layer(hk_layerwise_request * r,
                                                const hk_layerwise_identity * id,
                                                uint32_t model_layer) {
    if (!r || !r->impl) {
        set_error("hk_layerwise_request_begin_layer: null request");
        return HK_ERR;
    }
    return guarded([&] { r->impl->begin_layer(identity_from(id), model_layer); },
                   "hk_layerwise_request_begin_layer");
}

extern "C" int hk_layerwise_request_projected(hk_layerwise_request * r,
                                              const hk_layerwise_identity * id, uint64_t begin,
                                              const hk_current_layer_query * q) {
    if (!r || !r->impl) {
        set_error("hk_layerwise_request_projected: null request");
        return HK_ERR;
    }
    return guarded([&] { r->impl->projected(identity_from(id), begin, current_query_from(q)); },
                   "hk_layerwise_request_projected");
}

extern "C" int hk_layerwise_request_projected_native(hk_layerwise_request * r,
                                                     const hk_layerwise_identity * id, uint64_t begin,
                                                     const hk_native_current_layer_query * q,
                                                     hk_native_scorer_fn scorer, void * scorer_user) {
    if (!r || !r->impl) {
        set_error("hk_layerwise_request_projected_native: null request");
        return HK_ERR;
    }
    if (!q || !scorer) {
        set_error("hk_layerwise_request_projected_native: null query or scorer");
        return HK_ERR;
    }
    const hk_raw_store * keys = r->keys;
    return guarded([&] {
        // Wrap the owner's C scorer into the core NativeScorer. The core keys
        // reference is the same borrowed store as `keys`; the callback receives
        // the handle so a C owner can read statistics without capturing it.
        kvmem::LayerwiseQuerySelection::NativeScorer fn =
            [scorer, scorer_user, keys](const kvmem::RawKvStore &,
                                        kvmem::NativeCurrentLayerQuery tile,
                                        const kvmem::QueryScoreFrame & frame) -> std::vector<double> {
                const hk_native_current_layer_query ctile = native_query_to(tile);
                const hk_query_score_frame cframe = score_frame_to(frame);
                std::vector<double> out(frame.starts.size(), 0.0);
                const int rc = scorer(scorer_user, keys, &ctile, &cframe, out.data(), out.size());
                if (rc != HK_OK)
                    throw kvmem::MemoryError(kvmem::MemoryErrorCode::InvalidPlan,
                                             "native Query scorer reported failure");
                return out;
            };
        r->impl->projected_native(identity_from(id), begin, native_query_from(q), fn);
    }, "hk_layerwise_request_projected_native");
}

extern "C" int hk_layerwise_request_prepare_restore(hk_layerwise_request * r,
                                                    const hk_layerwise_identity * id,
                                                    hk_layer_plan ** out) {
    clear_error();
    if (!r || !r->impl || !out) {
        set_error("hk_layerwise_request_prepare_restore: invalid arguments");
        return HK_ERR;
    }
    try {
        *out = new hk_layer_plan{r->impl->prepare_restore(identity_from(id))};
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_layerwise_request_prepare_restore: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" int hk_layerwise_request_restored(hk_layerwise_request * r,
                                             const hk_layerwise_identity * id,
                                             const hk_execution_view * view) {
    if (!r || !r->impl) {
        set_error("hk_layerwise_request_restored: null request");
        return HK_ERR;
    }
    return guarded([&] { r->impl->restored(identity_from(id), execution_view_from(view)); },
                   "hk_layerwise_request_restored");
}

extern "C" int hk_layerwise_request_attended(hk_layerwise_request * r,
                                             const hk_layerwise_identity * id, uint64_t begin,
                                             uint64_t count) {
    if (!r || !r->impl) {
        set_error("hk_layerwise_request_attended: null request");
        return HK_ERR;
    }
    return guarded([&] { r->impl->attended(identity_from(id), begin, count); },
                   "hk_layerwise_request_attended");
}

extern "C" int hk_layerwise_request_complete_layer(hk_layerwise_request * r,
                                                   const hk_layerwise_identity * id) {
    if (!r || !r->impl) {
        set_error("hk_layerwise_request_complete_layer: null request");
        return HK_ERR;
    }
    return guarded([&] { r->impl->complete_layer(identity_from(id)); },
                   "hk_layerwise_request_complete_layer");
}

extern "C" int hk_layerwise_request_accepted(hk_layerwise_request * r,
                                             const hk_layerwise_identity * id,
                                             const hk_layer_memory_snapshot * snapshots,
                                             size_t n) {
    if (!r || !r->impl || (!snapshots && n)) {
        set_error("hk_layerwise_request_accepted: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] {
        std::vector<kvmem::LayerMemorySnapshot> snaps;
        for (size_t i = 0; i < n; ++i)
            snaps.push_back({snapshots[i].model_layer, snapshot_from(snapshots[i].memory)});
        r->impl->accepted(identity_from(id), std::move(snaps));
    }, "hk_layerwise_request_accepted");
}

extern "C" uint32_t hk_layerwise_request_selection_layer(const hk_layerwise_request * r) {
    if (!r || !r->impl) return 0;
    try {
        return r->impl->selection().layer;
    } catch (...) {
        return 0;
    }
}
extern "C" size_t hk_layerwise_request_selection_score_count(const hk_layerwise_request * r) {
    if (!r || !r->impl) return 0;
    try {
        return r->impl->selection().scores.size();
    } catch (...) {
        return 0;
    }
}
extern "C" int hk_layerwise_request_selection_score(const hk_layerwise_request * r, size_t i,
                                                    double * out) {
    if (!r || !r->impl || !out) {
        set_error("hk_layerwise_request_selection_score: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] {
        const auto & scores = r->impl->selection().scores;
        if (i >= scores.size()) throw std::out_of_range("selection score index");
        *out = scores[i];
    }, "hk_layerwise_request_selection_score");
}
extern "C" size_t hk_layerwise_request_selection_fixed_count(const hk_layerwise_request * r) {
    if (!r || !r->impl) return 0;
    try {
        return r->impl->selection().fixed.size();
    } catch (...) {
        return 0;
    }
}
extern "C" uint32_t hk_layerwise_request_selection_fixed(const hk_layerwise_request * r, size_t i) {
    if (!r || !r->impl) return 0;
    try {
        const auto & fixed = r->impl->selection().fixed;
        if (i >= fixed.size()) throw std::out_of_range("selection fixed index");
        return fixed[i];
    } catch (...) {
        return 0;
    }
}
extern "C" size_t hk_layerwise_request_selection_selected_count(const hk_layerwise_request * r) {
    if (!r || !r->impl) return 0;
    try {
        return r->impl->selection().selected.size();
    } catch (...) {
        return 0;
    }
}
extern "C" uint32_t hk_layerwise_request_selection_selected(const hk_layerwise_request * r,
                                                            size_t i) {
    if (!r || !r->impl) return 0;
    try {
        const auto & selected = r->impl->selection().selected;
        if (i >= selected.size()) throw std::out_of_range("selection selected index");
        return selected[i];
    } catch (...) {
        return 0;
    }
}
extern "C" uint64_t hk_layerwise_request_selection_historical_tokens(const hk_layerwise_request * r) {
    if (!r || !r->impl) return 0;
    try {
        return r->impl->selection().historical_tokens;
    } catch (...) {
        return 0;
    }
}
extern "C" uint64_t hk_layerwise_request_selection_charged_tokens(const hk_layerwise_request * r) {
    if (!r || !r->impl) return 0;
    try {
        return r->impl->selection().charged_tokens;
    } catch (...) {
        return 0;
    }
}

extern "C" size_t hk_layerwise_request_accepted_layer_count(const hk_layerwise_request * r) {
    if (!r || !r->impl) return 0;
    try {
        return r->impl->accepted_layers().size();
    } catch (...) {
        return 0;
    }
}
extern "C" uint32_t hk_layerwise_request_accepted_model_layer(const hk_layerwise_request * r,
                                                              size_t i) {
    if (!r || !r->impl) return 0;
    try {
        const auto & layers = r->impl->accepted_layers();
        if (i >= layers.size()) throw std::out_of_range("accepted layer index");
        return layers[i].model_layer;
    } catch (...) {
        return 0;
    }
}
extern "C" int hk_layerwise_request_accepted_stamp(const hk_layerwise_request * r, size_t i,
                                                   hk_memory_stamp * out) {
    if (!r || !r->impl || !out) {
        set_error("hk_layerwise_request_accepted_stamp: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] {
        const auto & layers = r->impl->accepted_layers();
        if (i >= layers.size()) throw std::out_of_range("accepted layer index");
        stamp_to(layers[i].memory.stamp, out);
    }, "hk_layerwise_request_accepted_stamp");
}
extern "C" size_t hk_layerwise_request_accepted_block_count(const hk_layerwise_request * r,
                                                            size_t i) {
    if (!r || !r->impl) return 0;
    try {
        const auto & layers = r->impl->accepted_layers();
        if (i >= layers.size()) throw std::out_of_range("accepted layer index");
        return layers[i].memory.blocks.size();
    } catch (...) {
        return 0;
    }
}
extern "C" int hk_layerwise_request_accepted_block(const hk_layerwise_request * r, size_t i,
                                                   size_t b, hk_block_copies * out) {
    if (!r || !r->impl || !out) {
        set_error("hk_layerwise_request_accepted_block: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] {
        const auto & layers = r->impl->accepted_layers();
        if (i >= layers.size()) throw std::out_of_range("accepted layer index");
        const auto & blocks = layers[i].memory.blocks;
        if (b >= blocks.size()) throw std::out_of_range("accepted block index");
        block_desc_to(blocks[b].block, &out->block);
        out->host_version = blocks[b].host_version;
    }, "hk_layerwise_request_accepted_block");
}
extern "C" size_t hk_layerwise_request_accepted_resident_count(const hk_layerwise_request * r,
                                                               size_t i) {
    if (!r || !r->impl) return 0;
    try {
        const auto & layers = r->impl->accepted_layers();
        if (i >= layers.size()) throw std::out_of_range("accepted layer index");
        return layers[i].memory.resident.size();
    } catch (...) {
        return 0;
    }
}
extern "C" uint64_t hk_layerwise_request_accepted_resident(const hk_layerwise_request * r, size_t i,
                                                           size_t b) {
    if (!r || !r->impl) return 0;
    try {
        const auto & layers = r->impl->accepted_layers();
        if (i >= layers.size()) throw std::out_of_range("accepted layer index");
        const auto & resident = layers[i].memory.resident;
        if (b >= resident.size()) throw std::out_of_range("accepted resident index");
        return resident[b];
    } catch (...) {
        return 0;
    }
}

// ── layerwise activation accounts + projected-Q history tape ──

struct hk_spill {
    std::unique_ptr<kvmem::SpillFile> impl;
};
struct hk_reservation {
    kvmem::HostKvReservation impl;
};
struct hk_tape {
    std::unique_ptr<kvmem::QueryProjectionTape> impl;
};
struct hk_tape_chunk {
    kvmem::QueryProjectionChunkSource impl;
};

namespace {

kvmem::QueryProjectionLayout query_layout_from(const hk_query_layout * l) {
    if (!l) throw std::invalid_argument("null query projection layout");
    if (!l->planes && l->plane_count) throw std::invalid_argument("null query projection planes");
    kvmem::QueryProjectionLayout out;
    out.format = l->format ? l->format : "";
    out.version = l->version;
    out.planes.reserve(l->plane_count);
    for (size_t i = 0; i < l->plane_count; ++i)
        out.planes.push_back({l->planes[i].id, l->planes[i].row_stride});
    return out;
}

std::vector<kvmem::QueryProjectionPlaneBytes> query_plane_bytes_from(const hk_query_plane_bytes * p,
                                                                   size_t n) {
    if (!p && n) throw std::invalid_argument("null query plane bytes");
    std::vector<kvmem::QueryProjectionPlaneBytes> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back({p[i].id, p[i].data, p[i].bytes});
    return out;
}

} // namespace

extern "C" hk_spill * hk_spill_create(const char * directory, uint64_t capacity_bytes) {
    clear_error();
    if (!directory || !*directory) {
        set_error("hk_spill_create: null directory");
        return nullptr;
    }
    try {
        return new hk_spill{
            std::make_unique<kvmem::SpillFile>(std::filesystem::path(directory), capacity_bytes)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_spill_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_spill_destroy(hk_spill * s) { delete s; }

extern "C" uint64_t hk_spill_charged_bytes(const hk_spill * s) {
    return (s && s->impl) ? s->impl->charged_bytes() : 0;
}

extern "C" uint64_t hk_spill_capacity_bytes(const hk_spill * s) {
    return (s && s->impl) ? s->impl->capacity_bytes() : 0;
}

extern "C" hk_reservation * hk_storage_reserve(hk_storage * s, uint64_t bytes) {
    clear_error();
    if (!s || !s->impl) {
        set_error("hk_storage_reserve: null storage");
        return nullptr;
    }
    try {
        return new hk_reservation{s->impl->reserve(bytes)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_storage_reserve: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_reservation_destroy(hk_reservation * r) { delete r; }

extern "C" uint64_t hk_reservation_remaining(const hk_reservation * r) {
    return r ? r->impl.remaining() : 0;
}

extern "C" hk_host_record * hk_reservation_allocate(hk_reservation * r, const void * bytes,
                                                    size_t len) {
    clear_error();
    if (!r) {
        set_error("hk_reservation_allocate: null reservation");
        return nullptr;
    }
    try {
        return new hk_host_record{r->impl.allocate(len, bytes)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_reservation_allocate: unknown C++ exception");
    }
    return nullptr;
}

extern "C" hk_tape * hk_tape_create_ram(hk_storage * activation, hk_storage * metadata,
                                       uint64_t max_chunks, const hk_layerwise_identity * identity,
                                       uint32_t model_layer, const hk_query_layout * layout) {
    clear_error();
    if (!activation || !activation->impl || !metadata || !metadata->impl || !identity || !layout) {
        set_error("hk_tape_create_ram: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_tape{std::make_unique<kvmem::QueryProjectionTape>(
            *activation->impl, *metadata->impl, max_chunks, identity_from(identity), model_layer,
            query_layout_from(layout))};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_tape_create_ram: unknown C++ exception");
    }
    return nullptr;
}

extern "C" hk_tape * hk_tape_create_disk(hk_spill * activation, hk_storage * metadata,
                                        uint64_t max_chunks, const hk_layerwise_identity * identity,
                                        uint32_t model_layer, const hk_query_layout * layout) {
    clear_error();
    if (!activation || !activation->impl || !metadata || !metadata->impl || !identity || !layout) {
        set_error("hk_tape_create_disk: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_tape{std::make_unique<kvmem::QueryProjectionTape>(
            *activation->impl, *metadata->impl, max_chunks, identity_from(identity), model_layer,
            query_layout_from(layout))};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_tape_create_disk: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_tape_destroy(hk_tape * t) { delete t; }

extern "C" uint64_t hk_tape_metadata_bound(const hk_query_layout * layout, uint64_t max_chunks) {
    clear_error();
    try {
        return kvmem::QueryProjectionTape::metadata_bound(query_layout_from(layout), max_chunks);
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_tape_metadata_bound: unknown C++ exception");
    }
    return 0;
}

extern "C" uint64_t hk_tape_metadata_bytes(const hk_tape * t) {
    return (t && t->impl) ? t->impl->metadata_bytes() : 0;
}
extern "C" uint64_t hk_tape_max_chunks(const hk_tape * t) {
    return (t && t->impl) ? t->impl->max_chunks() : 0;
}
extern "C" uint64_t hk_tape_total_bytes(const hk_tape * t) {
    return (t && t->impl) ? t->impl->total_bytes() : 0;
}
extern "C" uint64_t hk_tape_reserved_bytes(const hk_tape * t) {
    return (t && t->impl) ? t->impl->reserved_bytes() : 0;
}
extern "C" uint64_t hk_tape_covered_end(const hk_tape * t) {
    return (t && t->impl) ? t->impl->covered_end() : 0;
}
extern "C" size_t hk_tape_chunk_count(const hk_tape * t) {
    return (t && t->impl) ? t->impl->chunk_count() : 0;
}
extern "C" uint32_t hk_tape_model_layer(const hk_tape * t) {
    return (t && t->impl) ? t->impl->model_layer() : 0;
}
extern "C" int hk_tape_complete(const hk_tape * t) {
    return (t && t->impl && t->impl->complete()) ? 1 : 0;
}
extern "C" int hk_tape_aborted(const hk_tape * t) {
    return (t && t->impl && t->impl->aborted()) ? 1 : 0;
}

extern "C" hk_tape_chunk * hk_tape_append(hk_tape * t, const hk_layerwise_identity * identity,
                                         uint32_t model_layer, uint64_t begin, uint64_t count,
                                         const hk_query_plane_bytes * planes, size_t plane_count) {
    clear_error();
    if (!t || !t->impl || !identity) {
        set_error("hk_tape_append: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_tape_chunk{t->impl->append(identity_from(identity), model_layer, begin, count,
                                                 query_plane_bytes_from(planes, plane_count))};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_tape_append: unknown C++ exception");
    }
    return nullptr;
}

extern "C" int hk_tape_finish(hk_tape * t, const hk_layerwise_identity * identity,
                             uint32_t model_layer) {
    if (!t || !t->impl || !identity) {
        set_error("hk_tape_finish: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] { t->impl->finish(identity_from(identity), model_layer); }, "hk_tape_finish");
}

extern "C" hk_tape_chunk * hk_tape_chunk_at(const hk_tape * t, const hk_layerwise_identity * identity,
                                           uint32_t model_layer, size_t index) {
    clear_error();
    if (!t || !t->impl || !identity) {
        set_error("hk_tape_chunk_at: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_tape_chunk{t->impl->chunk(identity_from(identity), model_layer, index)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_tape_chunk_at: unknown C++ exception");
    }
    return nullptr;
}

extern "C" int hk_tape_read_rows(const hk_tape * t, const hk_layerwise_identity * identity,
                                 uint32_t model_layer, uint32_t plane, uint64_t begin, uint64_t count,
                                 void * dest, size_t dest_bytes) {
    if (!t || !t->impl || !identity || (!dest && dest_bytes)) {
        set_error("hk_tape_read_rows: invalid arguments");
        return HK_ERR;
    }
    return guarded(
        [&] {
            t->impl->read_rows(identity_from(identity), model_layer, plane, begin, count, dest,
                               dest_bytes);
        },
        "hk_tape_read_rows");
}

extern "C" void hk_tape_abort(hk_tape * t) {
    if (t && t->impl) t->impl->abort();
}

extern "C" void hk_tape_chunk_destroy(hk_tape_chunk * c) { delete c; }
extern "C" uint32_t hk_tape_chunk_model_layer(const hk_tape_chunk * c) {
    return c ? c->impl.model_layer() : 0;
}
extern "C" uint64_t hk_tape_chunk_begin(const hk_tape_chunk * c) {
    return c ? c->impl.rows().begin : 0;
}
extern "C" uint64_t hk_tape_chunk_end(const hk_tape_chunk * c) {
    return c ? c->impl.rows().end : 0;
}
extern "C" size_t hk_tape_chunk_size(const hk_tape_chunk * c) { return c ? c->impl.size() : 0; }

extern "C" int hk_tape_chunk_read_rows(const hk_tape_chunk * c, const hk_layerwise_identity * identity,
                                       uint32_t model_layer, uint32_t plane, uint64_t begin,
                                       uint64_t count, void * dest, size_t dest_bytes) {
    if (!c || !identity || (!dest && dest_bytes)) {
        set_error("hk_tape_chunk_read_rows: invalid arguments");
        return HK_ERR;
    }
    return guarded(
        [&] {
            c->impl.read_rows(identity_from(identity), model_layer, plane, begin, count, dest,
                              dest_bytes);
        },
        "hk_tape_chunk_read_rows");
}

// ── layerwise attention-mass pressure ──

struct hk_prefill_pressure {
    std::unique_ptr<kvmem::LayerwisePrefillPressure> impl;
};
struct hk_prefill_snapshot {
    kvmem::LayerwisePrefillSnapshot impl;
};
struct hk_output_pressure {
    std::unique_ptr<kvmem::LayerwiseOutputPressure> impl;
};
struct hk_output_snapshot {
    kvmem::LayerwiseOutputSnapshot impl;
};
struct hk_pressure_choices {
    std::vector<kvmem::LayerwisePressureChoice> impl;
};

namespace {

std::vector<uint32_t> copy_u32(const uint32_t * p, size_t n) {
    if (!p || !n) return {};
    return std::vector<uint32_t>(p, p + n);
}
std::vector<double> copy_f64(const double * p, size_t n) {
    if (!p || !n) return {};
    return std::vector<double>(p, p + n);
}
void session_to(const kvmem::SessionIdentity & s, hk_session_identity * out) {
    out->id = s.id;
    out->incarnation = s.incarnation;
}

kvmem::LayerwisePrefillOptions prefill_options_from(const hk_prefill_options * o) {
    if (!o) throw std::invalid_argument("null prefill options");
    kvmem::LayerwisePrefillOptions out;
    out.block_tokens = o->block_tokens;
    out.select_tokens = o->select_tokens;
    out.reserve_tokens = o->reserve_tokens;
    out.interval = o->interval;
    out.stride = o->stride;
    out.half_life = o->half_life;
    return out;
}
void prefill_options_to(const kvmem::LayerwisePrefillOptions & o, hk_prefill_options * out) {
    out->block_tokens = o.block_tokens;
    out->select_tokens = o.select_tokens;
    out->reserve_tokens = o.reserve_tokens;
    out->interval = o.interval;
    out->stride = o.stride;
    out->half_life = o.half_life;
}

kvmem::LayerwisePrefillMass prefill_mass_from(const hk_prefill_mass & m) {
    if (!m.starts && m.start_count) throw std::invalid_argument("null prefill mass starts");
    if (!m.history && m.history_count) throw std::invalid_argument("null prefill mass history");
    if (!m.masses && m.mass_count) throw std::invalid_argument("null prefill mass values");
    kvmem::LayerwisePrefillMass out;
    out.model_layer = m.model_layer;
    out.starts = copy_u64(m.starts, m.start_count);
    out.history = copy_u64(m.history, m.history_count);
    out.masses = copy_f64(m.masses, m.mass_count);
    return out;
}
kvmem::LayerwisePrefillAcceptance prefill_acceptance_from(const hk_prefill_acceptance * a) {
    if (!a) throw std::invalid_argument("null prefill acceptance");
    if (!a->rows && a->row_count) throw std::invalid_argument("null prefill acceptance rows");
    if (!a->layers && a->layer_count) throw std::invalid_argument("null prefill acceptance layers");
    kvmem::LayerwisePrefillAcceptance out;
    out.history = stamp_from(a->history);
    out.count = a->count;
    out.accepted = a->accepted;
    out.local_begin = a->local_begin;
    out.rows = copy_u32(a->rows, a->row_count);
    out.layers.reserve(a->layer_count);
    for (size_t i = 0; i < a->layer_count; ++i) out.layers.push_back(prefill_mass_from(a->layers[i]));
    return out;
}

kvmem::LayerwiseOutputOptions output_options_from(const hk_output_options * o) {
    kvmem::LayerwiseOutputOptions out; // null selects the C++ defaults
    if (o) {
        out.scope = o->scope == HK_OUTPUT_NON_THINKING ? kvmem::LayerwiseOutputScope::NonThinking
                                                       : kvmem::LayerwiseOutputScope::ToolCall;
        out.interval = o->interval;
        out.decay = o->decay;
        out.commit_max_bytes = o->commit_max_bytes;
    }
    return out;
}
kvmem::LayerwiseOutputMass output_mass_from(const hk_output_mass & m) {
    if (!m.history && m.history_count) throw std::invalid_argument("null output mass history");
    if (!m.masses && m.mass_count) throw std::invalid_argument("null output mass values");
    kvmem::LayerwiseOutputMass out;
    out.model_layer = m.model_layer;
    out.history = copy_u64(m.history, m.history_count);
    out.masses = copy_f64(m.masses, m.mass_count);
    return out;
}
kvmem::LayerwiseOutputAcceptance output_acceptance_from(const hk_output_acceptance * a) {
    if (!a) throw std::invalid_argument("null output acceptance");
    if (!a->rows && a->row_count) throw std::invalid_argument("null output acceptance rows");
    if (!a->layers && a->layer_count) throw std::invalid_argument("null output acceptance layers");
    kvmem::LayerwiseOutputAcceptance out;
    out.history = stamp_from(a->history);
    out.request = a->request;
    out.accepted = a->accepted;
    out.rows = copy_u32(a->rows, a->row_count);
    out.layers.reserve(a->layer_count);
    for (size_t i = 0; i < a->layer_count; ++i) out.layers.push_back(output_mass_from(a->layers[i]));
    return out;
}

void pressure_choice_to(const kvmem::LayerwisePressureChoice & c, hk_pressure_choice * out) {
    out->model_layer = c.model_layer;
    out->candidates = c.candidates.data(); out->candidate_count = c.candidates.size();
    out->fixed = c.fixed.data(); out->fixed_count = c.fixed.size();
    out->selected = c.selected.data(); out->selected_count = c.selected.size();
    out->scores = c.scores.data(); out->score_count = c.scores.size();
    out->accepted_score_end = c.accepted_score_end;
    out->samples = c.samples;
    out->chunks = c.chunks;
    out->selected_tokens = c.selected_tokens;
    out->charged_tokens = c.charged_tokens;
    out->source = c.source == kvmem::LayerwisePressureSource::Output ? HK_PRESSURE_OUTPUT
                                                                     : HK_PRESSURE_PREFILL;
    out->output_samples = c.output_samples;
    out->output_rounds = c.output_rounds;
}

} // namespace

extern "C" hk_prefill_pressure * hk_prefill_pressure_create(hk_session_identity session,
                                                            const uint32_t * model_layers,
                                                            size_t n_layers,
                                                            const hk_prefill_options * options) {
    clear_error();
    if ((!model_layers && n_layers) || !options) {
        set_error("hk_prefill_pressure_create: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_prefill_pressure{std::make_unique<kvmem::LayerwisePrefillPressure>(
            kvmem::SessionIdentity{session.id, session.incarnation},
            copy_u32(model_layers, n_layers), prefill_options_from(options))};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_prefill_pressure_create: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_prefill_pressure_destroy(hk_prefill_pressure * p) { delete p; }

extern "C" uint32_t hk_prefill_pressure_rows(const hk_prefill_pressure * p, uint64_t begin,
                                             uint32_t count, uint64_t local_begin, int replay,
                                             uint32_t * out, uint32_t capacity) {
    clear_error();
    if (!p || !p->impl) {
        set_error("hk_prefill_pressure_rows: invalid arguments");
        return UINT32_MAX;
    }
    try {
        const auto rows = p->impl->rows(begin, count, local_begin, replay != 0);
        if (out) {
            const size_t n = rows.size() < capacity ? rows.size() : capacity;
            for (size_t i = 0; i < n; ++i) out[i] = rows[i];
        }
        return static_cast<uint32_t>(rows.size());
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_prefill_pressure_rows: unknown C++ exception");
    }
    return UINT32_MAX;
}

extern "C" int hk_prefill_pressure_accept(hk_prefill_pressure * p, const hk_prefill_acceptance * a) {
    clear_error();
    if (!p || !p->impl || !a) {
        set_error("hk_prefill_pressure_accept: invalid arguments");
        return HK_ERR;
    }
    try {
        return p->impl->accept(prefill_acceptance_from(a)) ? 1 : 0;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_prefill_pressure_accept: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" int hk_prefill_pressure_options(const hk_prefill_pressure * p, hk_prefill_options * out) {
    if (!p || !p->impl || !out) {
        set_error("hk_prefill_pressure_options: invalid arguments");
        return HK_ERR;
    }
    prefill_options_to(p->impl->options(), out);
    return HK_OK;
}

extern "C" hk_prefill_snapshot * hk_prefill_pressure_snapshot(const hk_prefill_pressure * p) {
    clear_error();
    if (!p || !p->impl) {
        set_error("hk_prefill_pressure_snapshot: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_prefill_snapshot{p->impl->snapshot()};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_prefill_pressure_snapshot: unknown C++ exception");
    }
    return nullptr;
}

extern "C" int hk_prefill_pressure_restore(hk_prefill_pressure * p, const hk_prefill_snapshot * s,
                                           uint64_t position) {
    if (!p || !p->impl || !s) {
        set_error("hk_prefill_pressure_restore: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] { p->impl->restore(s->impl, position); }, "hk_prefill_pressure_restore");
}

extern "C" hk_pressure_choices * hk_prefill_pressure_select(const hk_prefill_pressure * p,
                                                            const hk_memory_catalog * catalog,
                                                            const uint64_t * mandatory,
                                                            size_t n_mandatory, uint64_t budget_tokens,
                                                            uint32_t sink_blocks) {
    clear_error();
    if (!p || !p->impl || !catalog || !catalog->impl || (!mandatory && n_mandatory)) {
        set_error("hk_prefill_pressure_select: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_pressure_choices{p->impl->select(*catalog->impl,
            copy_u64(mandatory, n_mandatory), budget_tokens, sink_blocks)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_prefill_pressure_select: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_prefill_snapshot_destroy(hk_prefill_snapshot * s) { delete s; }

extern "C" int hk_prefill_snapshot_session(const hk_prefill_snapshot * s, hk_session_identity * out) {
    if (!s || !out) {
        set_error("hk_prefill_snapshot_session: invalid arguments");
        return HK_ERR;
    }
    session_to(s->impl.session, out);
    return HK_OK;
}

extern "C" size_t hk_prefill_snapshot_layer_count(const hk_prefill_snapshot * s) {
    return s ? s->impl.layers.size() : 0;
}

extern "C" int hk_prefill_snapshot_layer(const hk_prefill_snapshot * s, size_t i, uint32_t * model_layer,
                                         uint64_t * end, uint64_t * samples, uint64_t * chunks) {
    if (!s || i >= s->impl.layers.size()) {
        set_error("hk_prefill_snapshot_layer: invalid arguments");
        return HK_ERR;
    }
    const auto & layer = s->impl.layers[i];
    if (model_layer) *model_layer = layer.model_layer;
    if (end) *end = layer.end;
    if (samples) *samples = layer.samples;
    if (chunks) *chunks = layer.chunks;
    return HK_OK;
}

extern "C" size_t hk_prefill_snapshot_value_count(const hk_prefill_snapshot * s, size_t i) {
    if (!s || i >= s->impl.layers.size()) return 0;
    return s->impl.layers[i].values.size();
}

extern "C" int hk_prefill_snapshot_value(const hk_prefill_snapshot * s, size_t i, size_t v,
                                         uint64_t * original_start, double * mass, double * exposure) {
    if (!s || i >= s->impl.layers.size() || v >= s->impl.layers[i].values.size()) {
        set_error("hk_prefill_snapshot_value: invalid arguments");
        return HK_ERR;
    }
    const auto & value = s->impl.layers[i].values[v];
    if (original_start) *original_start = value.original_start;
    if (mass) *mass = value.mass;
    if (exposure) *exposure = value.exposure;
    return HK_OK;
}

extern "C" void hk_pressure_choices_destroy(hk_pressure_choices * c) { delete c; }

extern "C" size_t hk_pressure_choices_count(const hk_pressure_choices * c) {
    return c ? c->impl.size() : 0;
}

extern "C" int hk_pressure_choices_get(const hk_pressure_choices * c, size_t i, hk_pressure_choice * out) {
    if (!c || !out || i >= c->impl.size()) {
        set_error("hk_pressure_choices_get: invalid arguments");
        return HK_ERR;
    }
    pressure_choice_to(c->impl[i], out);
    return HK_OK;
}

extern "C" hk_output_pressure * hk_output_pressure_create_markers(hk_session_identity session,
                                                                  const uint32_t * model_layers,
                                                                  size_t n_layers,
                                                                  const hk_prefill_options * prefill,
                                                                  const int32_t * markers5,
                                                                  const hk_output_options * options) {
    clear_error();
    if ((!model_layers && n_layers) || !prefill || !markers5) {
        set_error("hk_output_pressure_create_markers: invalid arguments");
        return nullptr;
    }
    try {
        std::array<int32_t, 5> markers{};
        for (size_t i = 0; i < 5; ++i) markers[i] = markers5[i];
        return new hk_output_pressure{std::make_unique<kvmem::LayerwiseOutputPressure>(
            kvmem::SessionIdentity{session.id, session.incarnation},
            copy_u32(model_layers, n_layers), prefill_options_from(prefill), markers,
            output_options_from(options))};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_output_pressure_create_markers: unknown C++ exception");
    }
    return nullptr;
}

extern "C" hk_output_pressure * hk_output_pressure_create_external(hk_session_identity session,
                                                                   const uint32_t * model_layers,
                                                                   size_t n_layers,
                                                                   const hk_prefill_options * prefill,
                                                                   const hk_output_options * options) {
    clear_error();
    if ((!model_layers && n_layers) || !prefill) {
        set_error("hk_output_pressure_create_external: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_output_pressure{std::make_unique<kvmem::LayerwiseOutputPressure>(
            kvmem::SessionIdentity{session.id, session.incarnation},
            copy_u32(model_layers, n_layers), prefill_options_from(prefill),
            kvmem::LayerwiseExternalOutputScope{}, output_options_from(options))};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_output_pressure_create_external: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_output_pressure_destroy(hk_output_pressure * p) { delete p; }

extern "C" int hk_output_pressure_begin(hk_output_pressure * p, const hk_memory_stamp * accepted_history,
                                        uint64_t request, const uint32_t * prompt, size_t prompt_count) {
    if (!p || !p->impl || !accepted_history || (!prompt && prompt_count)) {
        set_error("hk_output_pressure_begin: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] {
        p->impl->begin(stamp_from(*accepted_history), request, copy_u32(prompt, prompt_count));
    }, "hk_output_pressure_begin");
}

extern "C" int hk_output_pressure_begin_scope(hk_output_pressure * p,
                                              const hk_memory_stamp * accepted_history,
                                              uint64_t request, hk_output_scope_state scope) {
    if (!p || !p->impl || !accepted_history) {
        set_error("hk_output_pressure_begin_scope: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] {
        p->impl->begin_scope(stamp_from(*accepted_history), request,
                             kvmem::LayerwiseOutputScopeState{scope.thinking != 0, scope.tool_call != 0});
    }, "hk_output_pressure_begin_scope");
}

extern "C" int hk_output_pressure_accept_prefill(hk_output_pressure * p,
                                                 const hk_memory_stamp * accepted_history,
                                                 uint64_t request, uint64_t begin) {
    if (!p || !p->impl || !accepted_history) {
        set_error("hk_output_pressure_accept_prefill: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] {
        p->impl->accept_prefill(stamp_from(*accepted_history), request, begin);
    }, "hk_output_pressure_accept_prefill");
}

extern "C" int hk_output_pressure_prepare(hk_output_pressure * p, uint64_t request, uint64_t begin,
                                          const uint32_t * consumed_tokens, size_t count,
                                          uint64_t * out_mask) {
    clear_error();
    if (!p || !p->impl || !out_mask || (!consumed_tokens && count)) {
        set_error("hk_output_pressure_prepare: invalid arguments");
        return HK_ERR;
    }
    try {
        *out_mask = p->impl->prepare(request, begin, copy_u32(consumed_tokens, count));
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_output_pressure_prepare: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" int hk_output_pressure_prepare_scope(hk_output_pressure * p, uint64_t request, uint64_t begin,
                                                const hk_output_scope_row * rows, size_t count,
                                                uint64_t * out_mask) {
    clear_error();
    if (!p || !p->impl || !out_mask || (!rows && count)) {
        set_error("hk_output_pressure_prepare_scope: invalid arguments");
        return HK_ERR;
    }
    try {
        std::vector<kvmem::LayerwiseOutputScopeRow> scope_rows;
        scope_rows.reserve(count);
        for (size_t i = 0; i < count; ++i)
            scope_rows.push_back({rows[i].thinking != 0, rows[i].tool_call != 0, rows[i].body != 0});
        *out_mask = p->impl->prepare_scope(request, begin, scope_rows);
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_output_pressure_prepare_scope: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" int hk_output_pressure_accept(hk_output_pressure * p, const hk_output_acceptance * a,
                                         uint64_t * out_mask) {
    clear_error();
    if (!p || !p->impl || !a || !out_mask) {
        set_error("hk_output_pressure_accept: invalid arguments");
        return HK_ERR;
    }
    try {
        *out_mask = p->impl->accept(output_acceptance_from(a));
        return HK_OK;
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_output_pressure_accept: unknown C++ exception");
    }
    return HK_ERR;
}

extern "C" int hk_output_pressure_finish(hk_output_pressure * p, uint64_t request,
                                         const hk_memory_catalog * accepted_history) {
    if (!p || !p->impl || !accepted_history || !accepted_history->impl) {
        set_error("hk_output_pressure_finish: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] { p->impl->finish(request, *accepted_history->impl); },
                   "hk_output_pressure_finish");
}

extern "C" hk_output_snapshot * hk_output_pressure_snapshot(const hk_output_pressure * p) {
    clear_error();
    if (!p || !p->impl) {
        set_error("hk_output_pressure_snapshot: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_output_snapshot{p->impl->snapshot()};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_output_pressure_snapshot: unknown C++ exception");
    }
    return nullptr;
}

extern "C" int hk_output_pressure_restore(hk_output_pressure * p, const hk_output_snapshot * s,
                                          uint64_t position) {
    if (!p || !p->impl || !s) {
        set_error("hk_output_pressure_restore: invalid arguments");
        return HK_ERR;
    }
    return guarded([&] { p->impl->restore(s->impl, position); }, "hk_output_pressure_restore");
}

extern "C" hk_pressure_choices * hk_output_pressure_select(const hk_output_pressure * p,
                                                           const hk_memory_catalog * catalog,
                                                           const hk_prefill_pressure * prefill,
                                                           const uint64_t * mandatory, size_t n_mandatory,
                                                           uint64_t budget_tokens, uint32_t sink_blocks) {
    clear_error();
    if (!p || !p->impl || !catalog || !catalog->impl || !prefill || !prefill->impl ||
        (!mandatory && n_mandatory)) {
        set_error("hk_output_pressure_select: invalid arguments");
        return nullptr;
    }
    try {
        return new hk_pressure_choices{p->impl->select(*catalog->impl, *prefill->impl,
            copy_u64(mandatory, n_mandatory), budget_tokens, sink_blocks)};
    } catch (const std::exception & e) {
        set_error(e.what());
    } catch (...) {
        set_error("hk_output_pressure_select: unknown C++ exception");
    }
    return nullptr;
}

extern "C" void hk_output_snapshot_destroy(hk_output_snapshot * s) { delete s; }

extern "C" int hk_output_snapshot_session(const hk_output_snapshot * s, hk_session_identity * out) {
    if (!s || !out) {
        set_error("hk_output_snapshot_session: invalid arguments");
        return HK_ERR;
    }
    session_to(s->impl.session, out);
    return HK_OK;
}

extern "C" int hk_output_snapshot_meta(const hk_output_snapshot * s, uint64_t * request,
                                       uint64_t * accepted_end, int * active, int * thinking,
                                       int * action, uint64_t * parser_tokens) {
    if (!s) {
        set_error("hk_output_snapshot_meta: invalid arguments");
        return HK_ERR;
    }
    if (request) *request = s->impl.request;
    if (accepted_end) *accepted_end = s->impl.accepted_end;
    if (active) *active = s->impl.active ? 1 : 0;
    if (thinking) *thinking = s->impl.parser.thinking ? 1 : 0;
    if (action) *action = s->impl.parser.action ? 1 : 0;
    if (parser_tokens) *parser_tokens = s->impl.parser.tokens;
    return HK_OK;
}

extern "C" size_t hk_output_snapshot_layer_count(const hk_output_snapshot * s) {
    return s ? s->impl.layers.size() : 0;
}

extern "C" int hk_output_snapshot_layer(const hk_output_snapshot * s, size_t i, uint32_t * model_layer,
                                        uint64_t * samples, uint64_t * rounds) {
    if (!s || i >= s->impl.layers.size()) {
        set_error("hk_output_snapshot_layer: invalid arguments");
        return HK_ERR;
    }
    const auto & layer = s->impl.layers[i];
    if (model_layer) *model_layer = layer.model_layer;
    if (samples) *samples = layer.samples;
    if (rounds) *rounds = layer.rounds;
    return HK_OK;
}

extern "C" size_t hk_output_snapshot_score_count(const hk_output_snapshot * s, size_t i) {
    if (!s || i >= s->impl.layers.size()) return 0;
    return s->impl.layers[i].scores.size();
}

extern "C" int hk_output_snapshot_score(const hk_output_snapshot * s, size_t i, size_t j,
                                        uint64_t * start, double * score) {
    if (!s || i >= s->impl.layers.size() || j >= s->impl.layers[i].scores.size()) {
        set_error("hk_output_snapshot_score: invalid arguments");
        return HK_ERR;
    }
    const auto & entry = s->impl.layers[i].scores[j];
    if (start) *start = entry.first;
    if (score) *score = entry.second;
    return HK_OK;
}

extern "C" size_t hk_output_snapshot_pending_count(const hk_output_snapshot * s, size_t i) {
    if (!s || i >= s->impl.layers.size()) return 0;
    return s->impl.layers[i].pending.size();
}

extern "C" int hk_output_snapshot_pending(const hk_output_snapshot * s, size_t i, size_t j,
                                          uint64_t * start, double * score) {
    if (!s || i >= s->impl.layers.size() || j >= s->impl.layers[i].pending.size()) {
        set_error("hk_output_snapshot_pending: invalid arguments");
        return HK_ERR;
    }
    const auto & entry = s->impl.layers[i].pending[j];
    if (start) *start = entry.first;
    if (score) *score = entry.second;
    return HK_OK;
}
