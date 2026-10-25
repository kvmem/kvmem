#pragma once

// Shared draft wiring for llama-kvmem-cli / llama-kvmem-server.
// Does not reimplement llama.cpp's speculative state machine.

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "kvmem/speculative.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

struct kvmem_spec_opts {
    kvmem::SpeculativeAlgorithm algorithm = kvmem::SpeculativeAlgorithm::Mtp;
    int32_t n_max = 2;
    int32_t n_min = 0;
    float   p_min = 0.0f;
    int32_t n_gpu_layers = 99;
    int32_t n_ctx = 0;
    int32_t n_batch = 512;
    int32_t n_ubatch = 512;
    int32_t n_threads = -1;
    int32_t n_threads_batch = -1;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool    kvmem_enabled = false;
    std::string draft_model; // optional sidecar GGUF; empty = welded nextn
    ggml_type type_k = GGML_TYPE_Q8_0;
    ggml_type type_v = GGML_TYPE_Q8_0;
    ggml_type draft_type = GGML_TYPE_COUNT; // inherit target K/V types unless overridden
};

// Supported cache types: f16, f32, q8_0, q5_0, q4_0.
// q5_0 requires GGML_CUDA_FA_QUANTS=all (enabled by this project's build).
ggml_type kvmem_parse_cache_type(const char * s, bool * ok);
// Quantized K/V may independently use q8_0, q5_0 or q4_0; no float/quantized mixing.
bool kvmem_cache_types_ok(ggml_type type_k, ggml_type type_v);

struct kvmem_spec_session {
    kvmem::SpeculativeConfig config;
    std::shared_ptr<kvmem::SpeculativeTelemetry> telemetry = std::make_shared<kvmem::SpeculativeTelemetry>();
    common_params spec_params;
    common_speculative_init_result_ptr init;
    common_speculative * spec = nullptr;
    llama_context * ctx_dft = nullptr;
    bool use_ckpt_tgt = false;
    bool use_ckpt_dft = false;
    bool use_gdn_replay = false;
    uint32_t n_rs_tgt = 0;
    bool ok = false;
    bool independent_draft() const { return config.algorithm == kvmem::SpeculativeAlgorithm::DFlash2 ||
                                             config.algorithm == kvmem::SpeculativeAlgorithm::DSpark; }
};

bool kvmem_spec_start(kvmem_spec_session & sess,
                      llama_model * model_tgt,
                      llama_context * ctx_tgt,
                      const kvmem_spec_opts & opts,
                      const std::shared_ptr<llama_model> & model_dft = {});

void kvmem_spec_stop(kvmem_spec_session & sess);

// Bridge existing adapter batches to upstream's state-aware batch interface.
bool kvmem_spec_process(common_speculative * spec, llama_context * ctx, const llama_batch & batch);

// decode_span / spec_decode_span: client disconnected mid-prefill.
constexpr int KVMEM_DECODE_ABORT = -100;

// Decode [pos0, pos1) with explicit pos/seq_id so process() can catch up draft KV.
// `abort` if set: return KVMEM_DECODE_ABORT when it returns true (checked between batches).
int kvmem_spec_decode_span(llama_context * ctx,
                           common_speculative * spec,
                           const llama_token * toks,
                           int pos0, int pos1, int n_batch,
                           const char * what,
                           const std::function<bool()> & abort = {});

using kvmem_spec_on_token =
        std::function<void(llama_token id, const std::string & piece, bool from_draft)>;

struct kvmem_spec_gen_stats {
    int n_past = 0;
    int n_gen = 0;
    int n_drafted = 0;
    int n_accept = 0;
    int n_restore = 0;
    kvmem::SpeculativeAlgorithm algorithm = kvmem::SpeculativeAlgorithm::Mtp;
    // Native draft counters exclude the bonus token and internal replay proposals.
    uint64_t draft_tokens = 0;
    uint64_t accepted_tokens = 0;
    uint64_t verification_steps = 0;
    uint64_t generation_rounds = 0;
    uint64_t fallback_rounds = 0;
    std::vector<uint64_t> accepted_per_position;
    double wall_ms = 0.0;
    bool failed = false;
    nlohmann::json metrics() const {
        return {{"backend", algorithm == kvmem::SpeculativeAlgorithm::DFlash2 ? "draft-dflash"
                             : algorithm == kvmem::SpeculativeAlgorithm::DSpark ? "draft-dspark"
                             : "draft-mtp"},
            {"timing_scope", "generation_loop"}, {"drafted_tokens", draft_tokens},
            {"accepted_tokens", accepted_tokens}, {"verification_steps", verification_steps},
            {"generation_rounds", generation_rounds}, {"fallback_rounds", fallback_rounds},
            {"restore_rounds", n_restore},
            {"uncommitted_rounds", generation_rounds - verification_steps - fallback_rounds - (uint64_t)n_restore},
            {"accepted_per_position", accepted_per_position},
            {"output_tokens", n_gen}, {"wall_ms", wall_ms},
            {"acceptance_rate", draft_tokens ? nlohmann::json((double) accepted_tokens / draft_tokens) : nlohmann::json(nullptr)},
            {"tokens_per_round", generation_rounds ? nlohmann::json((double) n_gen / generation_rounds) : nlohmann::json(nullptr)},
            {"ms_per_round", generation_rounds ? nlohmann::json(wall_ms / generation_rounds) : nlohmann::json(nullptr)}};
    }
};

// `prompt` is the full prompt; the last token is id_last and must not already
// be in the target KV (same contract as llama.cpp speculative-simple).
kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        float temp,
        kvmem_spec_on_token on_token,
        uint32_t seed = LLAMA_DEFAULT_SEED);

// Same as above, but uses a fully-specified sampler (grammar / lazy triggers).
// `abort` if set: stop the generate loop when it returns true (not a failure).
kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        common_params_sampling sparams,
        kvmem_spec_on_token on_token,
        const std::function<bool()> & abort = {},
        llama_pos position_offset = 0);
