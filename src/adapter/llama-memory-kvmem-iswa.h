#pragma once

#include "llama-kv-cache-iswa.h"
#include "llama-memory-kvmem.h"

#include <memory>

// llama_kv_cache_iswa whose base (global/full-attention) half is a KVMem
// slot pool offloaded to host RAM; the SWA half stays a native bounded window.
//
// Graph still static_cast's to llama_kv_cache_iswa_context.
class llama_memory_kvmem_iswa : public llama_kv_cache_iswa {
public:
    llama_memory_kvmem_iswa(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams);

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_memory_kvmem * attn_kvmem() { return attn_kvmem_.get(); }

private:
    std::unique_ptr<llama_memory_kvmem> attn_kvmem_;
};
