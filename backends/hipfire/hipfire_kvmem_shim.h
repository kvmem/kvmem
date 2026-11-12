// SPDX-License-Identifier: Apache-2.0
// hipfire KVMem adapter — C ABI over the portable KVMem host library.
//
// hipfire is a Rust engine. This shim is the kvmem-side half of the adapter
// (the Rust half lives in hipfire's `hipfire-kvmem` crate): it exposes the
// block-statistics store (RawKvStore), the block/selection bookkeeping
// (KvMemStore) and the Mean-K retrieval scorer through a stable C ABI, compiled
// host-only (no CUDA/HIP, KVMEM_ROCM stays OFF).
//
// Conventions:
//   * opaque handles are owned by the caller and destroyed explicitly;
//   * every function is noexcept at the ABI boundary — failures return a
//     NULL/false/0/negative value and publish a message via hk_last_error();
//   * byte buffers are HOST pointers. Device transfers stay in hipfire;
//     callers hand this shim the packed Q8 rows after D2H, and memcpy the
//     returned rows back H2D on restore.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hk_storage hk_storage;
typedef struct hk_raw_store hk_raw_store;
typedef struct hk_kv_store hk_kv_store;

typedef enum hk_status {
    HK_OK = 0,
    HK_ERR = -1,
    HK_ERR_TRUNCATED = -2, // output capacity too small; required count returned
} hk_status;

typedef struct hk_raw_store_config {
    uint32_t n_layer;        // FullAttention statistics layers (kv_layer ordinals)
    uint32_t n_embd_k;       // per-token K width in F32 elements
    uint32_t n_embd_v;       // per-token V width in F32 elements (0: unused)
    uint32_t block_tokens;   // logical block size (must match the engine page)
    uint64_t k_row_bytes;    // 0 = FP16 raw rows (n_embd_k * 2)
    uint64_t k_gpu_row_bytes; // packed Q8 K row bytes per token; 0 = disabled
    uint64_t v_gpu_row_bytes; // packed Q8 V row bytes per token; 0 = disabled
} hk_raw_store_config;

typedef struct hk_kv_store_config {
    uint32_t block_tokens;
    uint32_t select_budget;   // semantic selection budget, tokens
    uint32_t prefill_budget;  // 0 => select_budget
    uint32_t gen_budget;
    uint32_t sink_blocks;
    uint32_t recent_blocks;
    uint32_t select_method;    // 0 Retrieval, 1 H2O, 2 Recency
    uint32_t retrieval_method; // 0 MeanK
    uint32_t update_mode;      // 0 Interval, 1 Step
    uint32_t lrfu;             // 0/1
    double   lrfu_lambda;
    double   lrfu_top_fraction;
    uint32_t estimated_gpu_block_capacity;
    uint64_t gpu_resident_block_bytes;
    uint64_t estimated_block_bytes;
    uint64_t cpu_tier_bytes;
    double   gpu_memory_ratio;
    double   gpu_high_watermark;
    double   gpu_low_watermark;
} hk_kv_store_config;

// ── diagnostics ──
const char * hk_last_error(void);
void hk_clear_error(void);
const char * hk_version(void);

// ── host payload storage (RawKvStore packed rows; optional RAM/disk tiers) ──
// `spill_dir` NULL/empty keeps the storage RAM-only; capacity_bytes 0 means
// unlimited. Disk tier sizes are only useful for the packed KV payload.
hk_storage * hk_storage_create(uint64_t capacity_bytes, const char * spill_dir,
                               uint64_t spill_capacity_bytes);
void hk_storage_destroy(hk_storage * s);
uint64_t hk_storage_charged_bytes(const hk_storage * s);
uint64_t hk_storage_disk_bytes(const hk_storage * s);
// Idle reclamation of sealed cold payloads down to resident_target bytes.
void hk_storage_trim(hk_storage * s, uint64_t resident_target);

// ── block statistics / packed KV store ──
hk_raw_store * hk_raw_store_create(const hk_raw_store_config * cfg, hk_storage * storage);
void hk_raw_store_destroy(hk_raw_store * st);
void hk_raw_store_ensure_blocks(hk_raw_store * st, uint32_t block_count);
// Token-major F32 K rows (post-RMSNorm, pre-RoPE), n_embd_k floats per token.
void hk_raw_store_write_layer_mean_k(hk_raw_store * st, uint32_t pos0, uint32_t n,
                                     uint32_t il, const float * k);
// Packed Q8 rows exactly as laid out in the engine arena (rope applied).
void hk_raw_store_write_layer_k_gpu(hk_raw_store * st, uint32_t pos0, uint32_t n,
                                    uint32_t il, const uint8_t * k);
void hk_raw_store_write_layer_v_gpu(hk_raw_store * st, uint32_t pos0, uint32_t n,
                                    uint32_t il, const uint8_t * v);
int hk_raw_store_copy_k_gpu(const hk_raw_store * st, uint32_t block, uint32_t il,
                            uint8_t * out, uint32_t n);
int hk_raw_store_copy_v_gpu(const hk_raw_store * st, uint32_t block, uint32_t il,
                            uint8_t * out, uint32_t n);
void hk_raw_store_mean_k(const hk_raw_store * st, uint32_t block, uint32_t il, float * out);
uint32_t hk_raw_store_mean_k_tokens(const hk_raw_store * st, uint32_t block, uint32_t il);
uint32_t hk_raw_store_n_tokens(const hk_raw_store * st, uint32_t block);
int hk_raw_store_has_k_gpu(const hk_raw_store * st, uint32_t block, uint32_t il, uint32_t n);
int hk_raw_store_has_v_gpu(const hk_raw_store * st, uint32_t block, uint32_t il, uint32_t n);
// Drop everything at/after token_pos, including a partial trailing block.
void hk_raw_store_truncate_to(hk_raw_store * st, uint32_t token_pos);
void hk_raw_store_clear(hk_raw_store * st);
uint64_t hk_raw_store_allocated_bytes(const hk_raw_store * st);
uint64_t hk_raw_store_payload_bytes(const hk_raw_store * st);
// Partial-tail mean sum checkpoint (n_layer * (1 + n_embd_k) floats).
// Returns the element count; writes nothing when out is NULL or capacity is short.
size_t hk_raw_store_mean_checkpoint(const hk_raw_store * st, uint32_t token_pos,
                                    float * out, size_t capacity);
void hk_raw_store_restore_mean_checkpoint(hk_raw_store * st, uint32_t token_pos,
                                          const float * state, size_t len);

// ── block repository / selection ──
hk_kv_store * hk_kv_store_create(const hk_kv_store_config * cfg);
void hk_kv_store_destroy(hk_kv_store * ks);
void hk_kv_store_register_append(hk_kv_store * ks, uint32_t n_new_tokens);
// Rewind the block table; tier slots are not tracked by hipfire, so the
// dropped-block return of the C++ API is intentionally discarded.
void hk_kv_store_truncate_to(hk_kv_store * ks, uint32_t token_pos);
uint32_t hk_kv_store_block_count(const hk_kv_store * ks);
uint32_t hk_kv_store_total_tokens(const hk_kv_store * ks);
uint32_t hk_kv_store_select_budget_tokens(const hk_kv_store * ks);
void hk_kv_store_set_runtime_select_budget(hk_kv_store * ks, uint32_t tokens);
// Selected block IDs in ascending window order. Pass out=NULL/capacity=0 to
// query the required count. Returns the count, or (uint32_t)-1 on error.
uint32_t hk_kv_store_pick_topk_blocks(const hk_kv_store * ks,
                                      const uint32_t * mandatory, uint32_t n_mandatory,
                                      uint32_t * out, uint32_t capacity);
uint32_t hk_kv_store_pick_prefill_pressure_blocks(const hk_kv_store * ks,
                                                  const uint32_t * mandatory,
                                                  uint32_t n_mandatory,
                                                  uint32_t * out, uint32_t capacity);
void hk_kv_store_set_retrieval_scores(hk_kv_store * ks, const double * scores, uint32_t n);
int hk_kv_store_prefill_needs_offload(const hk_kv_store * ks, uint32_t resident_tokens,
                                      uint32_t incoming_tokens, uint32_t pool_tokens);
uint32_t hk_kv_store_gpu_high_watermark_tokens(const hk_kv_store * ks, uint32_t pool_tokens);
uint32_t hk_kv_store_gpu_low_watermark_tokens(const hk_kv_store * ks, uint32_t pool_tokens);
uint64_t hk_kv_store_allocated_bytes(const hk_kv_store * ks);

// ── query-conditioned Mean-K retrieval ──
// `sums` carries `layers * query_heads * head_dim` F32 post-RMSNorm/pre-RoPE Q
// sums; `token_counts` one count per layer. Scores are written per block.
// Returns HK_OK, or HK_ERR when the store/query shape is inconsistent.
int hk_mean_key_retrieval_scores(const hk_raw_store * keys, uint32_t blocks,
                                 const float * sums, const uint32_t * token_counts,
                                 uint32_t layers, uint32_t head_dim, uint32_t query_heads,
                                 uint32_t kv_heads, uint32_t aggregation, double * out);

#ifdef __cplusplus
} // extern "C"
#endif
