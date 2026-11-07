#pragma once

// Public-ish KVMem controls. Compiled into libllama when LLAMA_KVMEM=ON.
// llama.cpp itself only sees llama_memory_kvmem_maybe_create via the factory
// hook header in the submodule patch.

#include "llama.h"

#include <stdint.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_context;
struct ggml_cgraph;
struct ggml_backend_sched;

#ifdef __cplusplus
extern "C" {
#endif

struct llama_kvmem_params {
    bool enabled;
    uint32_t block_tokens;  // 0 → 32
    uint32_t budget;        // tokens; 0 → use n_ctx (identity / no sparse)
    uint32_t gen_reserve;   // tokens; 0 → 256
    uint32_t sink_tokens;   // 0 → one block
    uint32_t recent_tokens; // 0 → pressure uses newest tail only
    int32_t  method;        // 0 recency, 1 retrieval (CLI default 1)
    bool     lrfu = false;
    double   lrfu_lambda = 0.5;
    double   lrfu_top_fraction = 0.9;
    int32_t  score_agg = 0; // mean or max within a GQA group
    bool     query_meanq_cosine = false;
    double   graph_alpha = 0.2;
    int32_t  query_begin;   // original token pos, -1 = unset
    int32_t  query_end;     // exclusive, -1 = end of prompt
    int32_t  force_pos;     // include block containing this orig pos; -1 = none
    float    gpu_memory_ratio;     // 0 → 0.50; cap pool vs device VRAM
    float    gpu_high_watermark;   // 0 → 0.95; trigger prefill offload
    float    gpu_low_watermark;    // 0 → 0.85; documented target after offload
    uint64_t cpu_bytes;            // CPU spill arena; 0 = disabled
    bool     harvest_v;            // prefill D2H V with K (default off)
    uint64_t payload_host_bytes;   // shared native K/V RAM quota; 0 = unlimited RAM
    uint64_t payload_disk_bytes;   // shared active/idle SSD quota; 0 = disabled
    const char * payload_disk_dir; // process-local spill directory, UTF-8
    int32_t  mtp_state;            // 0 snapshots, 1 auto, 2 replay
};

// Call before llama_init_from_model. A null pointer resets to defaults
// (disabled).
LLAMA_API void llama_kvmem_set_params(const struct llama_kvmem_params * params);
LLAMA_API const struct llama_kvmem_params * llama_kvmem_get_params(void);

// Isolated hook, retrieval, conversation and transfer state for one target
// context plus its MTP follower. Activate before creating/using/freeing those
// contexts, on every calling thread. A state must never be active concurrently
// on two threads. Free the contexts before freeing their execution state.
// Exchanging nullptr restores the legacy, single-domain behavior.
struct llama_kvmem_execution_state;
LLAMA_API struct llama_kvmem_execution_state * llama_kvmem_execution_create(void);
LLAMA_API void llama_kvmem_execution_free(struct llama_kvmem_execution_state * state);
LLAMA_API struct llama_kvmem_execution_state * llama_kvmem_execution_exchange(
        struct llama_kvmem_execution_state * state);
struct llama_kvmem_storage_stats {
    uint64_t host_bytes, host_capacity, disk_bytes, disk_capacity;
    uint64_t read_bytes, written_bytes, reads, writes, errors;
};
// Read-only accounting; the execution state must remain alive during the call.
LLAMA_API struct llama_kvmem_storage_stats llama_kvmem_execution_storage_stats(
        const struct llama_kvmem_execution_state * state);

// Legacy eval-callback entry. Always returns false so ggml does not split the
// graph. Capture harvest runs after the full ubatch compute instead.
LLAMA_API bool llama_kvmem_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

// Graph-build / post-compute capture (LLAMA_KVMEM process_ubatch hook).
LLAMA_API void llama_kvmem_register_capture(struct ggml_tensor * t, int il, char which);
LLAMA_API void llama_kvmem_capture_attention(struct ggml_context*,struct ggml_cgraph*,
    struct ggml_tensor* q,struct ggml_tensor* k,struct ggml_tensor* mask,float scale,
    float max_bias,float softcap,int layer);
LLAMA_API void llama_kvmem_attention_begin(const llama_token* prompt,uint32_t count);
LLAMA_API void llama_kvmem_attention_finish(bool commit);
// is_mtp: 1 when the llama_context building this graph is the MTP draft.
// Trunk and MTP must not clear each other's capture pending.
LLAMA_API void llama_kvmem_capture_on_new_graph(int is_mtp);
LLAMA_API void llama_kvmem_harvest_ubatch(struct ggml_backend_sched * sched, int is_mtp);
// True when this ubatch overlaps [query_begin, query_end) and Q nodes should
// be present. n_pos is the per-token position stride (1 for 1-D RoPE).
LLAMA_API bool llama_kvmem_ubatch_needs_q_capture(uint32_t n_tokens, uint32_t n_pos,
                                                  const llama_pos * pos);
// False if the last built graph's Q-capture topology would not match this ubatch.
// MTP graphs do not capture Q; is_mtp=1 always allows reuse from KVMem's side.
LLAMA_API bool llama_kvmem_capture_can_reuse(uint32_t n_tokens, uint32_t n_pos,
                                             const llama_pos * pos, int is_mtp);
// Topology stamp of the target graph last built. 0 if KVMem is not bound.
// Decode-width slots reuse a graph only when this still matches the stamp
// saved at that width's build.
LLAMA_API uint64_t llama_kvmem_capture_stamp(void);
// True while prefill/query-span harvest is still allowed (retrieval, not pinned).
// MTP verify is n>1 after pin; callers must not harvest those ubatches into raw-K.
LLAMA_API bool llama_kvmem_want_prefill_capture(void);
// First-pass Q capture for retrieval scoring; Query replay freezes this seed.
LLAMA_API bool llama_kvmem_want_q_capture(uint32_t n_tokens, uint32_t n_pos,
                                          const llama_pos * pos);
LLAMA_API void llama_kvmem_reset_query(void);
// True after retrieval pin: capture pre-RoPE K for decode mean-K (n=1 and MTP verify).
LLAMA_API bool llama_kvmem_want_decode_mean(void);
// Close prefill harvest before decode / speculative verify.
LLAMA_API void llama_kvmem_end_prefill_capture(void);
// Start a follow-up turn that keeps the GPU prefix. Call seq_rm(n_past,-1)
// on trunk + MTP after this; then prefill only the suffix.
LLAMA_API void llama_kvmem_begin_cached_turn(void);
// Keep a proved resident view while appending; capacity Pressure stays enabled.
LLAMA_API void llama_kvmem_keep_selected(void);
// Pin after skip prefill so decode mean-K uses gen_reserve slots.
LLAMA_API void llama_kvmem_pin_working_set(void);
LLAMA_API uint32_t llama_kvmem_free_slots(void);
LLAMA_API void llama_kvmem_truncate_cached(uint32_t n_past);
// Restore a cold partial tail before append, then validate the resumed view.
LLAMA_API bool llama_kvmem_restore_checkpoint_tail();
// Row-based removal, with native recurrent positions preserved.
LLAMA_API bool llama_kvmem_remove_logical(struct llama_context * ctx, llama_pos begin, llama_pos end);
LLAMA_API llama_pos llama_kvmem_model_pos(uint32_t logical_pos);
LLAMA_API void llama_kvmem_set_media_ranges(const uint32_t * starts, const uint32_t * ends, size_t count);
LLAMA_API uint32_t llama_kvmem_store_n_tokens(void);

// N host KV stores, one GPU working set, time-multiplexed. The server owns
// conversation identity, the caps and the LRU; the adapter owns only the
// mechanism. A process that never calls llama_kvmem_store_create() keeps
// exactly one store and never reaches any of this, so the default path is
// unchanged. Every call below must happen under the server's inference lock
// with no decode of the current request in flight. llama_kvmem_store_switch
// carries the one hard ordering requirement: it must run before
// llama_kvmem_set_request_span and before every staging call of the request it
// maps.
//
// False when a swap can never be safe in this configuration: without flash
// attention V is not mirrored to host.
LLAMA_API bool     llama_kvmem_store_swap_supported(void);
// New empty host store, or -1 when unavailable. Allocates only the bundle.
LLAMA_API int32_t  llama_kvmem_store_create(void);
// Quiesce, drain the GPU working set to host, then rebind the host store and
// the MTP follower mirror in lockstep. The bool is one claim on one store:
// true when the incoming store is attached and holds rows that are live on the
// GPU. False means it holds no rows the caller may decode against, whether it
// was already empty, its working set could not be rebuilt from host RAM and it
// was reset, or the swap itself was refused. That is the same claim
// llama_kvmem_store_n_tokens() > 0 makes, which is what switching to the
// already-active store reports. A refused switch leaves the previous store
// active; otherwise the active store is store_id whenever this returns at all.
// Every refusal logs but one: the guard against an unarmed or foreign memory
// object returns false with no log line, which a caller holding a handle from
// llama_kvmem_store_create() on this context cannot reach. Read
// llama_kvmem_store_current() rather than this bool to learn which store is
// active. The bool describes the INCOMING store only: use
// llama_kvmem_store_rows() to check what a parked store still holds.
// The recurrent half is NOT switched: that state is the server's byte
// snapshot, and a match is only usable when a recurrent checkpoint exists at
// or before it.
LLAMA_API bool     llama_kvmem_store_switch(int32_t store_id);
// Handle of the active store; 0 is the store built with the memory object.
LLAMA_API int32_t  llama_kvmem_store_current(void);
// Free a detached store. Refused for the active one.
LLAMA_API bool     llama_kvmem_store_destroy(int32_t store_id);
// Rows held by one store (llama_kvmem_store_n_tokens is the active one).
LLAMA_API uint32_t llama_kvmem_store_rows(int32_t store_id);
// Accounted host bytes of one store. Walks blocks times layers under the store
// mutex: call it once per request or per status report, never per token.
LLAMA_API uint64_t llama_kvmem_store_bytes(int32_t store_id);
LLAMA_API llama_pos llama_kvmem_recr_pos_max(void);
// After MTP verify: keep the first n_keep batch tokens in the running mean (0 = discard).
LLAMA_API void llama_kvmem_begin_speculative_evaluation();
// One accepted prefix: validate capture, fold GDN when recording, then commit
// target KV and mean-K. False never reports a partially committed batch as usable.
LLAMA_API bool llama_kvmem_commit_speculative(struct llama_context * ctx, uint32_t n_keep);
LLAMA_API void llama_kvmem_decode_mean_commit(uint32_t n_keep);
LLAMA_API void llama_kvmem_decode_mean_discard(void);
// Write any partial-block running mean to the host store (end of turn / evict).
LLAMA_API void llama_kvmem_decode_mean_flush(void);

// Score + reselect + stage-in raw-K/V for retrieval. No-op if method is recency.
LLAMA_API void llama_kvmem_apply_retrieval(struct llama_context * ctx);
LLAMA_API void llama_kvmem_begin_retrieval_round(void);
LLAMA_API void llama_kvmem_accept_retrieval_round(void);
// True when sink + [query_begin, prompt_end) fits the GPU selection budget
// (qw3 kvmem_replay_capacity). False → skip query seq_rm/replay.
LLAMA_API bool llama_kvmem_query_replay_fits(uint32_t query_begin, uint32_t prompt_end);
LLAMA_API void llama_kvmem_set_replay(bool replay);
LLAMA_API void llama_kvmem_trace_cells(struct llama_context * ctx, const char * tag);
// True when the active KVMem memory is hybrid (attn slot-pool + stock GDN).
LLAMA_API bool llama_kvmem_has_recurrent(void);
LLAMA_API bool llama_kvmem_gdn_replay_enabled(void);
LLAMA_API bool llama_kvmem_gdn_replay_begin(llama_pos start, uint32_t width);
LLAMA_API bool llama_kvmem_gdn_replay_commit(struct llama_context * ctx, uint32_t n_keep);
// Update query span / force_pos on the live memory (after llama_init_from_model).
LLAMA_API void llama_kvmem_set_request_span(int32_t query_begin, int32_t query_end, int32_t force_pos);
// Compare never-evicted GPU KV vs host-rebuild from raw-K. block_id -1 = force_pos block.
LLAMA_API void llama_kvmem_dump_kv_compare(struct llama_context * ctx, int32_t block_id);
LLAMA_API void llama_kvmem_dump_kv_writeback(struct llama_context * ctx, int32_t block_id);

#ifdef __cplusplus
}

#include <vector>
#include <string>
#include <memory>
#include "kvmem/lrfu.hpp"
#include "kvmem/graph_recall.hpp"
namespace kvmem { class ContentPrefix; class KvSnapshot; }

namespace kvmem { class SnapshotWriter; class SnapshotReader; struct SnapshotBuffer; }
// Portable host bundle. The active bundle is borrowed by a lane; swapping
// exchanges it with a parked bundle from an identically configured context.
// Startup-only: both execution domains must be quiescent.
LLAMA_API bool llama_kvmem_store_compatible_with_execution(const llama_kvmem_execution_state * other);
struct llama_kvmem_store_bundle;
// Immutable accepted attention image. Capture and restore run on the execution
// thread after producers drain; restoration builds a private native candidate.
struct llama_kvmem_checkpoint_snapshot;
LLAMA_API std::shared_ptr<const llama_kvmem_checkpoint_snapshot> llama_kvmem_checkpoint_capture(uint32_t row);
LLAMA_API bool llama_kvmem_checkpoint_restore(const llama_kvmem_checkpoint_snapshot & snapshot);
LLAMA_API uint64_t llama_kvmem_checkpoint_metadata_bytes(const llama_kvmem_checkpoint_snapshot & snapshot);
LLAMA_API uint64_t llama_kvmem_checkpoint_payload_bytes(const llama_kvmem_checkpoint_snapshot & snapshot);
LLAMA_API const kvmem::KvSnapshot * llama_kvmem_checkpoint_image(const llama_kvmem_checkpoint_snapshot & snapshot);
LLAMA_API llama_kvmem_store_bundle * llama_kvmem_store_bundle_create();
LLAMA_API llama_kvmem_store_bundle * llama_kvmem_store_bundle_fork(llama_kvmem_store_bundle * source = nullptr);
LLAMA_API void llama_kvmem_store_bundle_free(llama_kvmem_store_bundle * bundle);
LLAMA_API bool llama_kvmem_store_bundle_swap(llama_kvmem_store_bundle * bundle);
LLAMA_API void llama_kvmem_store_bundle_reset(llama_kvmem_store_bundle * bundle);
LLAMA_API uint64_t llama_kvmem_store_active_bytes();
LLAMA_API uint32_t llama_kvmem_store_bundle_rows(const llama_kvmem_store_bundle * bundle);
LLAMA_API uint64_t llama_kvmem_store_bundle_bytes(const llama_kvmem_store_bundle * bundle);
LLAMA_API uint64_t llama_kvmem_store_bundle_payload_bytes(const llama_kvmem_store_bundle * bundle);
LLAMA_API void llama_kvmem_store_bundle_freeze(llama_kvmem_store_bundle * bundle, std::vector<kvmem::SnapshotBuffer> & buffers);
LLAMA_API void llama_kvmem_store_bundle_thaw(llama_kvmem_store_bundle * bundle);
// Process-local disk-cache hooks. Park leaves an empty, valid execution store
// attached so the outgoing RAM can be released before a cold store is read.
LLAMA_API bool llama_kvmem_store_park();
LLAMA_API void llama_kvmem_store_freeze(int32_t id, std::vector<kvmem::SnapshotBuffer> & buffers);
LLAMA_API void llama_kvmem_store_thaw(int32_t id);
LLAMA_API void llama_kvmem_store_snapshot_write(int32_t id, kvmem::SnapshotWriter & out);
LLAMA_API void llama_kvmem_store_snapshot_read(int32_t id, kvmem::SnapshotReader & in);
LLAMA_API void llama_kvmem_store_release_payload(int32_t id);
LLAMA_API uint64_t llama_kvmem_store_capacity(uint32_t tokens);
LLAMA_API uint64_t llama_kvmem_store_payload_capacity(uint32_t tokens);
LLAMA_API uint64_t llama_kvmem_payload_budget(void);
LLAMA_API uint64_t llama_kvmem_store_payload_bytes(int32_t store_id, bool disk_only);

struct llama_kvmem_row_range {
    int32_t begin = 0;
    int32_t end = 0;
};
struct llama_kvmem_transfer_stats {
    bool enabled = false;
    uint64_t bytes[3] = {}; // H2D, D2H, D2D; adapter operations, including copy kernels
    uint64_t calls[3] = {};
};
LLAMA_API llama_kvmem_transfer_stats llama_kvmem_get_transfer_stats();
void kvmem_record_transfer(int cuda_kind, uint64_t bytes);
struct llama_kvmem_turn_spans {
    std::vector<llama_kvmem_row_range> query;
    std::vector<llama_kvmem_row_range> mandatory;
    int32_t replay_begin = 0;
};
LLAMA_API void llama_kvmem_set_graph_messages(const std::vector<kvmem::GraphMessageSpan>&);
struct llama_kvmem_query_state {
    // Post-normalization/pre-RoPE token-major Q, one vector per model layer.
    std::vector<std::vector<float>> rows;
    std::vector<uint32_t> count;
    std::shared_ptr<const kvmem::ContentPrefix> prefix;
    std::vector<llama_kvmem_row_range> ranges;
};
struct llama_kvmem_tail_mean_state {
    kvmem::LrfuState history;
    std::shared_ptr<const kvmem::ContentPrefix> prefix;
    std::vector<float> values;
    uint32_t row = 0;
    kvmem::GraphSnapshot graph;
    kvmem::AttentionFeedbackSnapshot feedback;
    kvmem::OutputSampleState scope;
    uint64_t output_begin = UINT64_MAX;
    int32_t think_open = -1, think_close = -1;
};
struct llama_kvmem_attention_view {
    uint64_t epoch = 0;
    uint32_t rows = 0;
    bool valid = false;
    std::vector<uint32_t> blocks;
};
struct llama_kvmem_selection {
    uint64_t epoch = 0;
    uint32_t rows = 0;
    std::vector<uint32_t> blocks;
};
LLAMA_API void llama_kvmem_set_turn_spans(const llama_kvmem_turn_spans & spans);
LLAMA_API llama_kvmem_attention_view llama_kvmem_get_attention_view();
LLAMA_API bool llama_kvmem_can_append(uint32_t end, uint32_t generation_rows, bool all_history, std::string & reason);
LLAMA_API llama_kvmem_selection llama_kvmem_preview_retrieval(const llama_kvmem_tail_mean_state* prefix = nullptr);
LLAMA_API bool llama_kvmem_selection_fits(const llama_kvmem_selection & selection, uint32_t end, uint32_t native_workspace);
LLAMA_API bool llama_kvmem_commit_unchanged(const llama_kvmem_attention_view & view, const llama_kvmem_selection & selection);
LLAMA_API void llama_kvmem_apply_selection(const llama_kvmem_selection & selection);
LLAMA_API bool llama_kvmem_commit_resident(bool canonical = true);
LLAMA_API bool llama_kvmem_get_query(llama_kvmem_query_state & state);
LLAMA_API bool llama_kvmem_set_query(const llama_kvmem_query_state & state);
LLAMA_API void llama_kvmem_freeze_query(bool frozen);
LLAMA_API bool llama_kvmem_get_tail_mean(uint32_t row, llama_kvmem_tail_mean_state & state);
// Validate before restoring recurrent/draft bytes. Input/media prefix matching
// and checkpoint selection remain the driver's responsibility.
LLAMA_API bool llama_kvmem_tail_mean_valid(uint32_t row, const llama_kvmem_tail_mean_state & state);
LLAMA_API bool llama_kvmem_set_tail_mean(uint32_t row, const llama_kvmem_tail_mean_state & state);
LLAMA_API void llama_kvmem_restore_history(const kvmem::LrfuState & state);
LLAMA_API kvmem::LrfuState llama_kvmem_get_history();
#endif
