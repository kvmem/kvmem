#include "llama-memory-kvmem-iswa.h"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>

// Model layers owned by the base (global/full-attention) half of the iSWA cache.
static std::vector<uint32_t> iswa_base_layers(const llama_model & model) {
    std::vector<uint32_t> layers;
    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        if (!model.hparams.is_swa(il) && !model.hparams.is_recr(il) && model.hparams.has_kv(il)) {
            layers.push_back(il);
        }
    }
    return layers;
}

// base GPU 池 = max(KVMem pool, SWA 窗口下限)。
// 原生 iswa: size_swa = GGML_PAD(min(size_base, n_swa*(unified?n_seq_max:1)+n_ubatch), 256)。
// size_base 小于窗口需求会被 min 截断 -> 滑窗损坏，故 base 必须先满足窗口下限。
static uint32_t kvmem_iswa_base_cells(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) {
    const std::vector<uint32_t> layers = iswa_base_layers(model);
    const uint32_t pool = llama_kvmem_pool_cells(
            model, params, cparams, layers.empty() ? nullptr : &layers);
    const uint32_t window = GGML_PAD(model.hparams.n_swa + cparams.n_ubatch, 256);
    return std::max(pool, window);
}

// Target-side trunk layers only; same filter the native gemma4 iswa gets.
static llama_memory_i::layer_filter_cb iswa_target_filter(const llama_model & model) {
    if (model.hparams.n_layer_nextn > 0 && model.hparams.n_layer() > 0 && model.hparams.router_layer < 0) {
        return [&model](int32_t il) { return il < static_cast<int32_t>(model.hparams.n_layer()); };
    }
    return nullptr;
}

llama_memory_kvmem_iswa::llama_memory_kvmem_iswa(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) :
    llama_kv_cache_iswa(
            model,
            params.no_alloc,
            params.type_k,
            params.type_v,
            !cparams.flash_attn,
            cparams.offload_kqv,
            // Never size the SWA half to the whole pool: the borrowed base cache
            // already holds full history, so a window-sized SWA cache is what
            // "swa=native window" means. This fork's server builds its context
            // params with llama_context_default_params(), which leaves swa_full
            // true, and at -c 262144 that alone is a ~21 GiB allocation.
            /* swa_full  */ false,
            /* unified   */ true,
            /* kv_size   */ kvmem_iswa_base_cells(model, params, cparams),
            /* n_seq_max */ 1,
            cparams.n_ubatch,
            /* mem_other */ nullptr,
            /* filter    */ iswa_target_filter(model),
            // gemma4 KV sharing never crosses the base/swa boundary
            /* reuse     */ [&](int32_t il) -> int32_t {
                return model.hparams.has_kv(il) ? -1 : (int32_t) model.hparams.kv_source(il);
            },
            /* share     */ nullptr) {
    attn_kvmem_ = std::make_unique<llama_memory_kvmem>(model, params, cparams, get_base());
    LLAMA_LOG_INFO("%s: KVMem iSWA (base=slot-pool offload, swa=native window)\n", __func__);
}

llama_memory_context_ptr llama_memory_kvmem_iswa::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    // The KVMem pool decides where tokens land, so keep its unified split.
    // A retry would register the append twice; fail instead.
    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    while (true) {
        llama_ubatch ubatch = embd_all ? balloc.split_seq(n_ubatch) : balloc.split_simple(n_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    llama_kv_cache::slot_info_vec_t sinfos_base;
    if (!attn_kvmem_->prepare_ubatches(ubatches, balloc.get_n_tokens(), sinfos_base)) {
        LLAMA_LOG_ERROR("%s: failed to prepare KVMem base ubatches\n", __func__);
        return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    auto sinfos_swa = get_swa()->prepare(ubatches);
    if (sinfos_swa.empty() || sinfos_swa.size() != sinfos_base.size()) {
        LLAMA_LOG_ERROR("%s: failed to prepare SWA ubatches\n", __func__);
        return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_kv_cache_iswa_context>(
            this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
}

void llama_memory_kvmem_iswa::clear(bool data) {
    // Drain capture/packed writers before clearing their source KV.
    attn_kvmem_->clear(data);
    get_swa()->clear(data);
}

bool llama_memory_kvmem_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const bool res = attn_kvmem_->seq_rm(seq_id, p0, p1);
    return get_swa()->seq_rm(seq_id, p0, p1) && res;
}

void llama_memory_kvmem_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    attn_kvmem_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    get_swa()->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_kvmem_iswa::seq_keep(llama_seq_id seq_id) {
    attn_kvmem_->seq_keep(seq_id);
    get_swa()->seq_keep(seq_id);
}

// The KVMem base owns host history; the resident SWA window starts later.
llama_pos llama_memory_kvmem_iswa::seq_pos_min(llama_seq_id seq_id) const {
    return attn_kvmem_->seq_pos_min(seq_id);
}

llama_pos llama_memory_kvmem_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return attn_kvmem_->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_kvmem_iswa::memory_breakdown() const {
    auto mb = attn_kvmem_->memory_breakdown();
    for (const auto & kv : get_swa()->memory_breakdown()) {
        mb[kv.first] += kv.second;
    }
    return mb;
}

void llama_memory_kvmem_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    attn_kvmem_->state_write(io, seq_id, flags);
    get_swa()->state_write(io, seq_id, flags);
}

void llama_memory_kvmem_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    attn_kvmem_->state_read(io, seq_id, flags);
    get_swa()->state_read(io, seq_id, flags);
}
