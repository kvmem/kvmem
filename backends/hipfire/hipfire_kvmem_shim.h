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
// LRFU round lifecycle. When lrfu is enabled the retrieval scorer must run inside
// an explicit round: begin -> set_retrieval_scores (preview) -> pick_topk_blocks
// -> commit (rewards the picked blocks). Both are no-ops when lrfu is disabled,
// so the default Retrieval path is byte-neutral.
void hk_kv_store_begin_retrieval_round(hk_kv_store * ks);
// Returns 1 when a round was committed (lrfu on), 0 otherwise.
int hk_kv_store_commit_retrieval_round(hk_kv_store * ks);
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

// ── query-span page-softmax retrieval ──
// Multi-row Q scoring: `q_data` carries `layers * rows * query_heads * head_dim`
// F32 post-RMSNorm/pre-RoPE Q rows (layer/token/head/channel order);
// `layer_rows[l]` is the number of present rows for layer l (<= rows). The
// accepted index is frozen into a QueryScoreFrame from `accepted_through` /
// `block_tokens` / `budget_blocks` / `sink_blocks` / `recent_blocks`, then each
// block's softmax mass is averaged over the query rows (FP16-rounded operands,
// scale 1/sqrt(head_dim) when `attention_scale` is 0). `blocks` must equal the
// frame's block count. Scores are written per block; HK_OK or HK_ERR.
int hk_page_softmax_scores(const hk_raw_store * keys, uint32_t blocks,
                           uint64_t accepted_through, uint32_t block_tokens,
                           uint32_t budget_blocks, uint32_t sink_blocks,
                           uint32_t recent_blocks, const float * q_data,
                           const uint32_t * layer_rows, uint32_t layers, uint32_t rows,
                           uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
                           float attention_scale, uint32_t aggregation, double * out);

// ── layerwise native payload foundation (host-only, additive) ──
// These entries expose the portable KVMem layerwise building blocks that the
// hipfire packed-page driver orchestrates: the backend-owned PayloadLayout /
// MemoryDescriptor, per-layer NativeKvLayerRegions (compact-extent reads from a
// sealed host record), the NativeKvArchive write-through store and the raw-KV
// export/import bridge. They are NOT consumed by the legacy residency path;
// nothing here mutates a RawKvStore's statistics or the engine's resident view.
// All handles are caller-owned and destroyed explicitly; byte buffers are HOST
// pointers (device transfers stay in hipfire). Every entry is noexcept at the
// ABI boundary: failures return NULL/HK_ERR and publish hk_last_error().

// Mirrors kvmem::PayloadPlaneKind ordinals (append-only).
typedef enum hk_payload_plane_kind {
    HK_PLANE_KEY = 0,
    HK_PLANE_VALUE = 1,
    HK_PLANE_KEY_SCALE = 2,
    HK_PLANE_VALUE_SCALE = 3,
    HK_PLANE_PACKED_KV = 4
} hk_payload_plane_kind;

typedef struct hk_session_identity {
    uint64_t id;
    uint64_t incarnation;
} hk_session_identity;

// Borrowed NUL-terminated strings; copied into the C++ handle at create time.
typedef struct hk_backend_identity {
    const char * backend;
    const char * revision;
    const char * model;
    const char * state_schema;
} hk_backend_identity;

typedef struct hk_block_descriptor {
    hk_session_identity session;
    uint64_t id;
    uint64_t original_start;
    uint32_t valid_tokens;
    uint64_t content_version;
    uint64_t statistics_version;
} hk_block_descriptor;

typedef struct hk_payload_plane {
    uint32_t layer;          // model layer identity
    uint32_t kind;           // hk_payload_plane_kind
    uint64_t offset;
    uint64_t bytes;
    uint64_t alignment;
    uint64_t token_stride;   // 0 = opaque page; nonzero = contiguous token rows
    uint32_t rank;
    uint32_t head_begin;
    uint32_t head_count;
} hk_payload_plane;

typedef struct hk_memory_capabilities {
    uint32_t page_tokens;
    uint64_t max_position;
    uint32_t native_payload;
    uint32_t sparse_view;
    uint32_t original_rope_positions;
    uint32_t pre_rope_statistics;
    uint32_t state_checkpoint;
    uint32_t asynchronous_transfer;
} hk_memory_capabilities;

typedef struct hk_payload_layout hk_payload_layout;
typedef struct hk_memory_descriptor hk_memory_descriptor;
typedef struct hk_kv_regions hk_kv_regions;
typedef struct hk_host_record hk_host_record;
typedef struct hk_host_source hk_host_source;
typedef struct hk_native_payload hk_native_payload;
typedef struct hk_native_archive hk_native_archive;

// PayloadLayout: build from a packed RawKvStore config (the hipfire Q8 adapter:
// one {Key,Value} token-row plane per attention layer, scales inline), or from
// an explicit plane list. Both validate the layout before returning a handle.
hk_payload_layout * hk_payload_layout_from_raw(const hk_raw_store_config * cfg,
                                               const char * format,
                                               const uint32_t * layers, uint32_t n_layers);
hk_payload_layout * hk_payload_layout_create(const char * format, uint32_t version,
                                             uint32_t block_tokens, uint64_t record_bytes,
                                             uint64_t alignment, const hk_payload_plane * planes,
                                             uint32_t n_planes);
void hk_payload_layout_destroy(hk_payload_layout * l);
uint64_t hk_payload_layout_record_bytes(const hk_payload_layout * l);
uint32_t hk_payload_layout_block_tokens(const hk_payload_layout * l);
uint32_t hk_payload_layout_plane_count(const hk_payload_layout * l);
int hk_payload_layout_plane(const hk_payload_layout * l, uint32_t i, hk_payload_plane * out);

// MemoryDescriptor: backend identity + layout + capabilities, validated.
hk_memory_descriptor * hk_memory_descriptor_create(const hk_backend_identity * identity,
                                                   const hk_payload_layout * layout,
                                                   const hk_memory_capabilities * capabilities);
void hk_memory_descriptor_destroy(hk_memory_descriptor * d);

// Sealed host record / deferred source: the restore source for region reads.
// allocate copies `len` bytes into the storage account; seal makes it immutable
// (and eligible for tiering). source() borrows a deferred read handle.
hk_host_record * hk_host_record_allocate(hk_storage * s, const void * bytes, size_t len);
void hk_host_record_destroy(hk_host_record * r);
void hk_host_record_seal(hk_host_record * r);
size_t hk_host_record_size(const hk_host_record * r);
int hk_host_record_read(const hk_host_record * r, size_t offset, void * dest, size_t bytes);
hk_host_source * hk_host_record_source(const hk_host_record * r);
void hk_host_source_destroy(hk_host_source * s);
size_t hk_host_source_size(const hk_host_source * s);

// NativeKvLayerRegions: one model layer's planes packed into a compact extent.
// read() copies a bounded slice of the packed layer out of a sealed host source
// into caller-owned staging (zero-fills interplane padding). has_rank=0 selects
// every rank of the layer; otherwise only planes with plane.rank==rank.
hk_kv_regions * hk_kv_regions_create(const hk_payload_layout * layout, uint32_t layer,
                                     int has_rank, uint32_t rank);
void hk_kv_regions_destroy(hk_kv_regions * r);
uint64_t hk_kv_regions_packed_bytes(const hk_kv_regions * r);
uint64_t hk_kv_regions_plane_bytes(const hk_kv_regions * r);
uint64_t hk_kv_regions_record_bytes(const hk_kv_regions * r);
uint32_t hk_kv_regions_region_count(const hk_kv_regions * r);
int hk_kv_regions_region(const hk_kv_regions * r, uint32_t i, hk_payload_plane * out_plane,
                         uint64_t * out_packed_offset);
int hk_kv_regions_read(const hk_kv_regions * r, const hk_host_source * source,
                       uint64_t packed_offset, void * dest, size_t bytes, size_t max_read_bytes);

// NativeKvPayload: one canonical block's opaque packed record (all declared
// planes). export gathers a RawKvStore's packed K/V; import writes them back.
// Neither touches mean-K statistics or the resident view.
hk_native_payload * hk_native_payload_export(const hk_raw_store * raw,
                                             const hk_backend_identity * backend,
                                             const hk_payload_layout * layout,
                                             const hk_block_descriptor * block);
hk_native_payload * hk_native_payload_create(const hk_backend_identity * backend,
                                             const hk_payload_layout * layout,
                                             const hk_block_descriptor * block,
                                             const void * bytes, size_t len);
void hk_native_payload_destroy(hk_native_payload * p);
const uint8_t * hk_native_payload_bytes(const hk_native_payload * p);
size_t hk_native_payload_size(const hk_native_payload * p);
int hk_native_payload_block(const hk_native_payload * p, hk_block_descriptor * out);
int hk_native_payload_import(hk_raw_store * raw, const hk_backend_identity * backend,
                             const hk_payload_layout * layout, const hk_native_payload * p);

// NativeKvArchive: bounded write-through store of canonical packed records.
// put copies the payload (the caller keeps its handle); get returns a NEW
// caller-owned handle. Stale/foreign writes are rejected, never silently merged.
hk_native_archive * hk_native_archive_create(const hk_backend_identity * backend,
                                             hk_session_identity session,
                                             const hk_payload_layout * layout,
                                             uint64_t capacity_bytes);
void hk_native_archive_destroy(hk_native_archive * a);
int hk_native_archive_put(hk_native_archive * a, const hk_native_payload * p);
hk_native_payload * hk_native_archive_get(const hk_native_archive * a,
                                          const hk_block_descriptor * expected);
void hk_native_archive_erase(hk_native_archive * a, uint64_t block_id);
uint64_t hk_native_archive_bytes(const hk_native_archive * a);

// ── layerwise catalog + planning (host-only, additive) ──
// Pure planning over one canonical accepted history, mirroring
// kvmem::LayerwiseMemoryCatalog / LayerWorkingSetPlan / LayerwiseWorkingSetPlan.
// No GPU and no mutation of statistics or residency: the packed-page driver
// builds a catalog from its per-layer snapshots, then plans each layer's
// restore/spill/evict at a Query boundary. Input arrays are borrowed and copied
// at call time; output arrays are read through indexed accessors. Every entry is
// noexcept at the boundary (NULL/0/HK_ERR + hk_last_error() on failure).

typedef struct hk_memory_stamp {
    uint32_t valid;
    hk_session_identity session;
    uint64_t history_revision;
    uint64_t view_version;
    uint64_t frontier;
    uint64_t execution_history;
} hk_memory_stamp;

typedef struct hk_block_copies {
    hk_block_descriptor block;
    uint64_t host_version;   // 0: no host copy; smaller than content_version: stale
} hk_block_copies;

// Complete logical catalog (chronological) plus the current resident ID set.
typedef struct hk_memory_snapshot {
    hk_memory_stamp stamp;
    const hk_block_copies * blocks;
    size_t block_count;
    const uint64_t * resident;
    size_t resident_count;
} hk_memory_snapshot;

typedef struct hk_layer_memory_snapshot {
    uint32_t model_layer;
    hk_memory_snapshot memory;
} hk_layer_memory_snapshot;

typedef struct hk_memory_budget {
    uint64_t selected_tokens;
    uint64_t reserve_tokens;
    uint64_t device_tokens;
    uint64_t host_payload_bytes; // one shared canonical archive limit per entry
} hk_memory_budget;

typedef struct hk_view_block {
    hk_block_descriptor block;
    uint64_t compact_start;    // attention address only; never a RoPE rewrite
} hk_view_block;

// One layer's selection input: borrowed logical block-ID arrays plus the
// layer's current native view stamp and its runtime budget.
typedef struct hk_layer_selection {
    uint32_t model_layer;
    hk_memory_stamp expected;
    const uint64_t * selected;
    size_t selected_count;
    const uint64_t * mandatory;
    size_t mandatory_count;
    hk_memory_budget budget;
} hk_layer_selection;

typedef struct hk_memory_catalog hk_memory_catalog;
typedef struct hk_layer_plan hk_layer_plan;
typedef struct hk_layerwise_plan hk_layerwise_plan;

// LayerwiseMemoryCatalog: validates that every target layer shares one accepted
// history/catalog and a consistent resident extent, then plans per layer.
hk_memory_catalog * hk_memory_catalog_create(const hk_memory_descriptor * canonical,
                                             const hk_layer_memory_snapshot * layers,
                                             size_t n_layers);
void hk_memory_catalog_destroy(hk_memory_catalog * c);
size_t hk_memory_catalog_layer_count(const hk_memory_catalog * c);
uint32_t hk_memory_catalog_model_layer(const hk_memory_catalog * c, size_t i);
int hk_memory_catalog_snapshot_stamp(const hk_memory_catalog * c, uint32_t model_layer,
                                     hk_memory_stamp * out);
size_t hk_memory_catalog_snapshot_block_count(const hk_memory_catalog * c, uint32_t model_layer);
int hk_memory_catalog_snapshot_block(const hk_memory_catalog * c, uint32_t model_layer,
                                     size_t i, hk_block_copies * out);
size_t hk_memory_catalog_snapshot_resident_count(const hk_memory_catalog * c, uint32_t model_layer);
uint64_t hk_memory_catalog_snapshot_resident(const hk_memory_catalog * c, uint32_t model_layer,
                                             size_t i);

// plan_layer: pure per-layer plan after that layer's Q exists.
int hk_memory_catalog_plan_layer(const hk_memory_catalog * c, const hk_layer_selection * sel,
                                 hk_layer_plan ** out);
// plan: bulk validation of already-known selections (one entry per target layer).
int hk_memory_catalog_plan(const hk_memory_catalog * c, const hk_layer_selection * sels,
                           size_t n_sels, hk_layerwise_plan ** out);

// LayerWorkingSetPlan reduced view. The per-layer transfer regions are obtained
// separately via hk_layer_plan_payload (a NativeKvLayerRegions handle).
void hk_layer_plan_destroy(hk_layer_plan * p);
uint32_t hk_layer_plan_model_layer(const hk_layer_plan * p);
uint64_t hk_layer_plan_restore_bytes(const hk_layer_plan * p);
uint64_t hk_layer_plan_spill_bytes(const hk_layer_plan * p);
uint64_t hk_layer_plan_device_tokens_required(const hk_layer_plan * p);
uint64_t hk_layer_plan_host_payload_bytes_required(const hk_layer_plan * p);
int hk_layer_plan_next_stamp(const hk_layer_plan * p, hk_memory_stamp * out);
uint64_t hk_layer_plan_next_visible_tokens(const hk_layer_plan * p);
size_t hk_layer_plan_next_block_count(const hk_layer_plan * p);
int hk_layer_plan_next_block(const hk_layer_plan * p, size_t i, hk_view_block * out);
size_t hk_layer_plan_restore_count(const hk_layer_plan * p);
int hk_layer_plan_restore_block(const hk_layer_plan * p, size_t i, hk_block_descriptor * out);
size_t hk_layer_plan_spill_count(const hk_layer_plan * p);
int hk_layer_plan_spill_block(const hk_layer_plan * p, size_t i, hk_block_descriptor * out);
size_t hk_layer_plan_evict_count(const hk_layer_plan * p);
int hk_layer_plan_evict_block(const hk_layer_plan * p, size_t i, hk_block_descriptor * out);
size_t hk_layer_plan_mandatory_count(const hk_layer_plan * p);
uint64_t hk_layer_plan_mandatory(const hk_layer_plan * p, size_t i);
int hk_layer_plan_payload(const hk_layer_plan * p, hk_kv_regions ** out);

// LayerwiseWorkingSetPlan: bulk result, one reduced plan per target layer.
void hk_layerwise_plan_destroy(hk_layerwise_plan * p);
size_t hk_layerwise_plan_layer_count(const hk_layerwise_plan * p);
int hk_layerwise_plan_history(const hk_layerwise_plan * p, hk_memory_stamp * out);
uint64_t hk_layerwise_plan_host_payload_bytes_required(const hk_layerwise_plan * p);
int hk_layerwise_plan_layer(const hk_layerwise_plan * p, size_t i, hk_layer_plan ** out);

// ── layerwise Query request state machine (host-only, additive) ──
// An engine-safe-boundary transaction mirroring kvmem::LayerwiseQueryRequest. It
// connects real projection reports to the qualified scorer/picker and the native
// layer plan; it does NOT execute tensors, copy GPU bytes, fence streams or
// publish state (those stay with the hipfire driver and must precede reports).
// The policy/keys/catalog handles are borrowed and must outlive the request. The
// source-revision callback proves the frozen index stamp at each native boundary.

typedef struct hk_query_range {
    uint64_t begin;
    uint64_t end;
} hk_query_range;

typedef struct hk_layerwise_identity {
    hk_memory_stamp history;
    uint64_t request;
    hk_query_range evaluation; // frozen prefix -> end of public input
    hk_query_range query;      // proven user rows within evaluation
} hk_layerwise_identity;

typedef struct hk_layer_query_geometry {
    uint32_t layer;          // RawKvStore statistics-layer ordinal
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float attention_scale;   // 0 selects 1/sqrt(head_dim)
} hk_layer_query_geometry;

// One target layer's admission spec. mandatory is a borrowed array of canonical
// chronological policy ordinals.
typedef struct hk_layer_request_spec {
    uint32_t model_layer;
    hk_layer_query_geometry query;
    hk_memory_budget budget;
    const uint32_t * mandatory;
    size_t mandatory_count;
} hk_layer_request_spec;

// Borrowed current-forward post-normalization/pre-RoPE rows from one target
// layer: contiguous [rows, query_heads, head_dim] F32.
typedef struct hk_current_layer_query {
    const float * data;
    uint32_t layer;
    uint32_t rows;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float attention_scale;
} hk_current_layer_query;

// Borrowed post-fence native view reported to restored().
typedef struct hk_execution_view {
    hk_memory_stamp stamp;
    const hk_view_block * blocks;
    size_t block_count;
    uint64_t visible_tokens;
} hk_execution_view;

typedef struct hk_layerwise_work {
    uint64_t projected_rows;
    uint64_t attended_rows;
    uint32_t completed_layers;
} hk_layerwise_work;

// Mirrors kvmem::LayerwiseRequestPhase ordinals.
typedef enum hk_layerwise_phase {
    HK_LW_BETWEEN_LAYERS = 0,
    HK_LW_PROJECTING = 1,
    HK_LW_RESTORING = 2,
    HK_LW_ATTENDING = 3,
    HK_LW_AWAITING_ACCEPTANCE = 4,
    HK_LW_ACCEPTED = 5,
    HK_LW_ABORTED = 6
} hk_layerwise_phase;

// Returns the owner's current frozen index stamp; called at each native boundary.
typedef hk_memory_stamp (*hk_source_revision_fn)(void * user);

typedef struct hk_layerwise_request hk_layerwise_request;

hk_layerwise_request * hk_layerwise_request_create(const hk_layerwise_identity * identity,
                                                   const hk_memory_catalog * catalog,
                                                   const hk_kv_store * policy,
                                                   const hk_raw_store * keys,
                                                   const hk_layer_request_spec * specs,
                                                   size_t n_specs,
                                                   uint64_t native_history_tokens,
                                                   hk_source_revision_fn revision,
                                                   void * revision_user);
void hk_layerwise_request_destroy(hk_layerwise_request * r);
void hk_layerwise_request_abort(hk_layerwise_request * r);
uint32_t hk_layerwise_request_phase(const hk_layerwise_request * r);
int hk_layerwise_request_work(const hk_layerwise_request * r, hk_layerwise_work * out);

int hk_layerwise_request_begin_layer(hk_layerwise_request * r, const hk_layerwise_identity * id,
                                     uint32_t model_layer);
int hk_layerwise_request_projected(hk_layerwise_request * r, const hk_layerwise_identity * id,
                                   uint64_t begin, const hk_current_layer_query * q);
int hk_layerwise_request_prepare_restore(hk_layerwise_request * r, const hk_layerwise_identity * id,
                                         hk_layer_plan ** out);
int hk_layerwise_request_restored(hk_layerwise_request * r, const hk_layerwise_identity * id,
                                  const hk_execution_view * view);
int hk_layerwise_request_attended(hk_layerwise_request * r, const hk_layerwise_identity * id,
                                  uint64_t begin, uint64_t count);
int hk_layerwise_request_complete_layer(hk_layerwise_request * r, const hk_layerwise_identity * id);
int hk_layerwise_request_accepted(hk_layerwise_request * r, const hk_layerwise_identity * id,
                                  const hk_layer_memory_snapshot * snapshots, size_t n);

// Frozen selection (available after prepare_restore, through attending). The
// score frame is internal; the driver reads scores/fixed/selected ordinals.
uint32_t hk_layerwise_request_selection_layer(const hk_layerwise_request * r);
size_t hk_layerwise_request_selection_score_count(const hk_layerwise_request * r);
int hk_layerwise_request_selection_score(const hk_layerwise_request * r, size_t i, double * out);
size_t hk_layerwise_request_selection_fixed_count(const hk_layerwise_request * r);
uint32_t hk_layerwise_request_selection_fixed(const hk_layerwise_request * r, size_t i);
size_t hk_layerwise_request_selection_selected_count(const hk_layerwise_request * r);
uint32_t hk_layerwise_request_selection_selected(const hk_layerwise_request * r, size_t i);
uint64_t hk_layerwise_request_selection_historical_tokens(const hk_layerwise_request * r);
uint64_t hk_layerwise_request_selection_charged_tokens(const hk_layerwise_request * r);

// Accepted per-layer snapshots (available only in the Accepted phase).
size_t hk_layerwise_request_accepted_layer_count(const hk_layerwise_request * r);
uint32_t hk_layerwise_request_accepted_model_layer(const hk_layerwise_request * r, size_t i);
int hk_layerwise_request_accepted_stamp(const hk_layerwise_request * r, size_t i,
                                        hk_memory_stamp * out);
size_t hk_layerwise_request_accepted_block_count(const hk_layerwise_request * r, size_t i);
int hk_layerwise_request_accepted_block(const hk_layerwise_request * r, size_t i, size_t b,
                                        hk_block_copies * out);
size_t hk_layerwise_request_accepted_resident_count(const hk_layerwise_request * r, size_t i);
uint64_t hk_layerwise_request_accepted_resident(const hk_layerwise_request * r, size_t i, size_t b);

// ── native (device-ready) Query projection + scorer callback (additive) ──
// projected_native reports a borrowed native Q tensor without a CPU copy or
// layout conversion: strides/extent are BYTES and the token axis need not be
// outermost. The owner's scorer computes per-block page mass for exactly the
// frozen QueryScoreFrame denominator; native stream completion must precede its
// return. The request proves index identity, not the tensor pointer. The keys
// handle passed to the scorer is the same borrowed store the request admitted.

// Mirrors kvmem::NativeQueryScalar ordinals.
typedef enum hk_native_query_scalar {
    HK_QUERY_F32 = 0,
    HK_QUERY_F16 = 1,
    HK_QUERY_BF16 = 2
} hk_native_query_scalar;

typedef struct hk_native_current_layer_query {
    const void * data;
    size_t bytes;
    uint32_t layer;          // RawKvStore statistics-layer ordinal
    uint32_t rows;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float attention_scale;   // 0 selects 1/sqrt(head_dim)
    uint32_t scalar;         // hk_native_query_scalar
    size_t row_stride;       // bytes between consecutive token rows
    size_t head_stride;      // bytes between consecutive query heads
    size_t dimension_stride; // bytes between consecutive channels
} hk_native_current_layer_query;

// Borrowed frozen score denominator (mirrors kvmem::QueryScoreFrame). All arrays
// have one entry per canonical block; score_mask marks scorable blocks (a masked
// block MUST receive score 0), transport_mask marks blocks needing transfer.
typedef struct hk_query_score_frame {
    const uint64_t * starts;
    size_t start_count;
    const uint32_t * counts;
    size_t count_count;
    const uint8_t * score_mask;
    size_t score_mask_count;
    const uint8_t * transport_mask;
    size_t transport_mask_count;
    uint64_t accepted_through;
} hk_query_score_frame;

// Writes frame.start_count per-block scores into out_scores (capacity out_cap).
// Returns HK_OK, or HK_ERR on failure (the request then aborts fail-closed).
typedef int (*hk_native_scorer_fn)(void * user, const hk_raw_store * keys,
                                   const hk_native_current_layer_query * tile,
                                   const hk_query_score_frame * frame,
                                   double * out_scores, size_t out_cap);

int hk_layerwise_request_projected_native(hk_layerwise_request * r,
                                          const hk_layerwise_identity * id, uint64_t begin,
                                          const hk_native_current_layer_query * q,
                                          hk_native_scorer_fn scorer, void * scorer_user);

#ifdef __cplusplus
} // extern "C"
#endif
