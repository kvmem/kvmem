#include "kvmem/query_policy.hpp"
#include "llama.h"
#include "../backends/llamacpp/src/llama-ext.h"
#include "llama-kvmem-hooks.h"
#include "kvmem/retrieval_options.hpp"
#include "kvmem/cli_options.h"
#include "kvmem-spec.h"
#include "kvmem-server-devices.h"
#include "kvmem-execution-scope.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf [options] [prompt]\n"
            "\n"
            "  -m, --model PATH           GGUF path\n"
            "  -f, --file PATH            read prompt from file (instead of trailing args)\n"
            "  -n, --n-predict N          tokens to generate (default 32)\n"
            "  -c, --ctx-size N           context size (default prompt + n_predict)\n"
            "  -b, --batch-size N         logical batch (default 512)\n"
            "  -ub, --ubatch-size N       physical ubatch (default 512)\n"
            "  -ngl, --n-gpu-layers N     GPU layers; all required for multi-GPU (default 99)\n"
            "  --list-devices            list available ggml devices\n"
            "  --device NAMES            CUDA devices, e.g. CUDA0,CUDA1\n"
            "  --split-mode MODE         none | layer | tensor\n"
            "  --tensor-split N,...      proportions, one per selected GPU\n"
            "  --temp T                   temperature; 0 = greedy (default 0)\n"
            "  --tokens-only              print generated token ids, one per line\n"
            "  --no-prompt                do not echo the prompt (generation only)\n"
            "  --kvmem                    enable KVMem slot-pool memory\n"
            "  --kvmem-block-tokens N     block size (default 32)\n"
            "  --kvmem-budget N           GPU working-set tokens; 0 = n_ctx (identity)\n"
            "  --kvmem-gen-reserve N      extra GPU tokens for decode (default 256)\n"
            "  --kvmem-sink-tokens N      always-kept prefix; 0 = one block\n"
            "  --kvmem-recent-tokens N    always-kept suffix blocks (default 0)\n"
            "  --kvmem-method NAME        recency | retrieval (default retrieval)\n"
            "  --kvmem-query-scorer NAME    page-softmax (default) | meanq-cosine\n"
            "  --kvmem-score-agg MODE       mean | max (default: page-softmax mean, cosine max)\n"
            "  --kvmem-lrfu / --no-kvmem-lrfu  Query history (default on only for cosine)\n"
            "  --kvmem-lrfu-lambda X        history decay in [0,1] (default 0.5)\n"
            "  --kvmem-lrfu-top-frac X      history hit fraction in (0,1] (default 0.9)\n"
            "  --kvmem-graph-alpha X        Query propagation in [0,1] (default: page-softmax 0.2; cosine 0)\n"
            "  --kvmem-query-max-tokens N cap raw Query capture (default 512)\n"
            "  --kvmem-query-replay-max-tokens N bound full replay suffix (default 2048)\n"
            "  --kvmem-query-last N       last N prompt tokens are the retrieval query (default 64)\n"
            "  --kvmem-force-substr S     force-select the block containing substring S\n"
            "  --kvmem-gpu-ratio R        cap slot pool at this fraction of GPU VRAM (default 0.50)\n"
            "  --kvmem-gpu-high R         prefill offload high watermark (default 0.95)\n"
            "  --kvmem-gpu-low R          prefill offload low watermark (default 0.85)\n"
            "  --kvmem-host-mib MIB      shared native KV RAM budget (default 16384 MiB)\n"
            "  --kvmem-disk-mib MIB      shared native KV SSD budget; 0 = off\n"
            "  --kvmem-disk-path PATH    process-local SSD directory; pair with disk-mib\n"
            "  --kvmem-cpu-gb GB          CPU spill arena in GiB (0 = off)\n"
            "  --kvmem-harvest-v          optional prefill D2H V capture (default off)\n"
            "  --kvmem-dump-kv            after prefill, compare raw-rebuild KV vs GPU KV\n"
            "  --kv-dtype NAME            GPU KV cache type for K and V: f16 | f32 | q8_0 | q5_0 | q4_0 (default q8_0)\n"
            "  -ctk, --cache-type-k TYPE  GPU K cache type (llama.cpp name; default q8_0)\n"
            "  -ctv, --cache-type-v TYPE  GPU V cache type (quantized: independently q8_0 | q5_0 | q4_0)\n"
            "  --seed N                   RNG seed; -1=random (ordinary default 0, MTP random)\n"
            "  --spec-type TYPE           none | draft-mtp | draft-dflash | draft-dspark (default none)\n"
            "  --spec-kv-dtype TYPE       draft K/V type (default: inherit target K/V types)\n"
            "  --spec-draft-n-max N       draft tokens (default 2)\n"
            "  --spec-draft-p-min P       min draft probability (default 0)\n"
            "  --kvmem-mtp-state MODE     snapshots | replay (default snapshots)\n"
            "  --spec-draft-model PATH    DFlash2 draft GGUF; optional sidecar MTP GGUF\n",
            argv0);
}

static bool eq(const char * a, const char * b) {
    return std::strcmp(a, b) == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "Hello my name is";
    std::string prompt_file;
    int n_predict = 32;
    int n_ctx = 0;
    int n_batch = 512;
    int n_ubatch = 512;
    int ngl = 99;
    bool list_devices = false;
    kvmem_server_options device_options;
    kvmem_server_devices device_config;
    float temp = 0.0f;
    bool tokens_only = false;
    bool no_prompt = false;

    std::unique_ptr<llama_kvmem_execution_state, decltype(&llama_kvmem_execution_free)> execution{
        llama_kvmem_execution_create(), llama_kvmem_execution_free};
    kvmem_execution_scope execution_scope(execution.get());
    llama_kvmem_params kparams = {};
    kvmem::CliOptions scoring_options;
    kparams.block_tokens = 32;
    kparams.gen_reserve = 256;
    kparams.method = 1;  // retrieval
    kparams.query_begin = -1;
    kparams.query_end = -1;
    kparams.force_pos = -1;
    int query_last = 64;
    uint32_t query_max_tokens=512, query_replay_max_tokens=2048;
    std::string force_substr;
    kvmem::CliOptions storage_options;
    bool dump_kv = false;
    ggml_type cache_type_k = GGML_TYPE_Q8_0;
    ggml_type cache_type_v = GGML_TYPE_Q8_0;
    ggml_type spec_cache_type = GGML_TYPE_COUNT;
    uint32_t seed = 0;
    bool seed_set = false;
    bool spec_enabled = false;
    auto spec_algorithm = kvmem::SpeculativeAlgorithm::None;
    int spec_n_max = 2;
    float spec_p_min = 0.0f;
    std::string spec_draft_model;

    int i = 1;
    for (; i < argc; ++i) {
        const char * arg = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                print_usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };
        try {
            kvmem::reject_retired_storage_option(arg);
            if (eq(arg, "--kvmem-host-mib") || eq(arg, "--kvmem-disk-mib") || eq(arg, "--kvmem-disk-path")) {
                storage_options.parse(arg, need);
                continue;
            }
        } catch (const std::invalid_argument & error) {
            fprintf(stderr, "%s\n", error.what());
            return 1;
        }
        if (eq(arg, "-h") || eq(arg, "--help")) {
            print_usage(argv[0]);
            return 0;
        } else if (eq(arg, "-m") || eq(arg, "--model")) {
            model_path = need(arg);
        } else if (eq(arg, "-f") || eq(arg, "--file")) {
            prompt_file = need(arg);
        } else if (eq(arg, "-n") || eq(arg, "--n-predict")) {
            n_predict = std::atoi(need(arg));
        } else if (eq(arg, "-c") || eq(arg, "--ctx-size")) {
            n_ctx = std::atoi(need(arg));
        } else if (eq(arg, "-b") || eq(arg, "--batch-size")) {
            n_batch = std::atoi(need(arg));
        } else if (eq(arg, "-ub") || eq(arg, "--ubatch-size")) {
            n_ubatch = std::atoi(need(arg));
        } else if (eq(arg, "-ngl") || eq(arg, "--n-gpu-layers")) {
            const char * value = need(arg);
            ngl = eq(value, "all") ? -2 : std::atoi(value);
        } else if (eq(arg, "--list-devices")) {
            list_devices = true;
        } else if (eq(arg, "--device") || eq(arg, "-dev") ||
                   eq(arg, "--split-mode") || eq(arg, "-sm") ||
                   eq(arg, "--tensor-split") || eq(arg, "-ts") ||
                   eq(arg, "--main-gpu") || eq(arg, "-mg")) {
            try {
                device_options.parse(arg, [&](const char *) { return need(arg); });
            } catch (const std::exception & e) {
                fprintf(stderr, "invalid GPU option: %s\n", e.what());
                return 1;
            }
        } else if (eq(arg, "--temp")) {
            temp = std::atof(need(arg));
        } else if (eq(arg, "--tokens-only")) {
            tokens_only = true;
        } else if (eq(arg, "--no-prompt")) {
            no_prompt = true;
        } else if (eq(arg, "--kvmem")) {
            kparams.enabled = true;
        } else if (eq(arg, "--kvmem-block-tokens")) {
            kparams.block_tokens = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-budget")) {
            kparams.budget = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-gen-reserve")) {
            kparams.gen_reserve = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-sink-tokens")) {
            kparams.sink_tokens = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-recent-tokens")) {
            kparams.recent_tokens = static_cast<uint32_t>(std::atoi(need(arg)));
        } else if (eq(arg, "--kvmem-method")) {
            const char * m = need(arg);
            if (eq(m, "retrieval") || eq(m, "retrieve")) {
                kparams.method = 1;
            } else {
                kparams.method = 0;
            }
        } else if (eq(arg,"--kvmem-query-scorer") || eq(arg,"--kvmem-graph-alpha") ||
                   eq(arg,"--kvmem-lrfu") || eq(arg,"--no-kvmem-lrfu") ||
                   eq(arg,"--kvmem-lrfu-lambda") || eq(arg,"--kvmem-lrfu-top-frac") || eq(arg,"--kvmem-score-agg")) {
            try { scoring_options.parse(arg,need); }
            catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
        } else if (eq(arg,"--kvmem-query-max-tokens")) {
            try { query_max_tokens=kvmem::CliOptions::integer(arg,need(arg),1); }
            catch(const std::invalid_argument& error) {fprintf(stderr,"%s\n",error.what());return 1;}
        } else if (eq(arg,"--kvmem-query-replay-max-tokens")) {
            try { query_replay_max_tokens=kvmem::CliOptions::integer(arg,need(arg),1); }
            catch(const std::invalid_argument& error) {fprintf(stderr,"%s\n",error.what());return 1;}
        } else if (eq(arg, "--kvmem-query-last")) {
            query_last = std::atoi(need(arg));
        } else if (eq(arg, "--kvmem-force-substr")) {
            force_substr = need(arg);
        } else if (eq(arg, "--kvmem-dump-kv")) {
            dump_kv = true;
        } else if (eq(arg, "--kv-dtype") || eq(arg, "-ctk") || eq(arg, "--cache-type-k")
                   || eq(arg, "-ctv") || eq(arg, "--cache-type-v")) {
            bool ok = false;
            const ggml_type t = kvmem_parse_cache_type(need(arg), &ok);
            if (!ok) {
                fprintf(stderr, "unsupported cache type (want f16|f32|q8_0|q5_0|q4_0)\n");
                return 1;
            }
            if (eq(arg, "-ctv") || eq(arg, "--cache-type-v")) {
                cache_type_v = t;
            } else if (eq(arg, "-ctk") || eq(arg, "--cache-type-k")) {
                cache_type_k = t;
            } else {
                cache_type_k = t;
                cache_type_v = t;
            }
        } else if (eq(arg, "--kvmem-gpu-ratio")) {
            kparams.gpu_memory_ratio = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--kvmem-gpu-high")) {
            kparams.gpu_high_watermark = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--kvmem-gpu-low")) {
            kparams.gpu_low_watermark = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--kvmem-cpu-gb")) {
            const double gb = std::atof(need(arg));
            kparams.cpu_bytes = gb <= 0.0 ? 0
                : static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (eq(arg, "--kvmem-harvest-v")) {
            kparams.harvest_v = true;
        } else if (eq(arg, "--spec-kv-dtype")) {
            bool ok = false;
            spec_cache_type = kvmem_parse_cache_type(need(arg), &ok);
            if (!ok) {
                fprintf(stderr, "unsupported MTP cache type (want f16|q8_0|q5_0|q4_0|f32)\n");
                return 1;
            }
        } else if (eq(arg, "--seed") || eq(arg, "-s")) {
            const char * value = need(arg);
            char * end = nullptr;
            const auto parsed = std::strtoll(value, &end, 10);
            if (!*value || *end || parsed < -1 || (parsed >= 0 && (uint64_t)parsed > UINT32_MAX)) {
                fprintf(stderr, "--seed requires -1 or an integer in 0..4294967295\n");
                return 1;
            }
            seed = (uint32_t)parsed;
            seed_set = true;
        } else if (eq(arg, "--spec-type")) {
            const char * t = need(arg);
            if (eq(t, "draft-mtp") || eq(t, "mtp")) {
                spec_algorithm = kvmem::SpeculativeAlgorithm::Mtp;
            } else if (eq(t, "draft-dflash") || eq(t, "dflash2")) {
                spec_algorithm = kvmem::SpeculativeAlgorithm::DFlash2;
            } else if (eq(t, "draft-dspark") || eq(t, "dspark")) {
                spec_algorithm = kvmem::SpeculativeAlgorithm::DSpark;
            } else if (eq(t, "none")) {
                spec_algorithm = kvmem::SpeculativeAlgorithm::None;
            } else {
                fprintf(stderr, "unsupported --spec-type %s (none|draft-mtp|draft-dflash|draft-dspark)\n", t);
                return 1;
            }
            spec_enabled = spec_algorithm != kvmem::SpeculativeAlgorithm::None;
        } else if (eq(arg, "--spec-draft-n-max")) {
            spec_n_max = std::atoi(need(arg));
        } else if (eq(arg, "--spec-draft-p-min")) {
            spec_p_min = std::strtof(need(arg), nullptr);
        } else if (eq(arg, "--kvmem-mtp-state")) {
            const char * mode = need(arg);
            if (eq(mode, "snapshots")) kparams.mtp_state = 0;
            else if (eq(mode, "replay")) kparams.mtp_state = 2;
            else {
                fprintf(stderr, "unsupported --kvmem-mtp-state (want snapshots|replay)\n");
                return 1;
            }
        } else if (eq(arg, "--spec-draft-model") || eq(arg, "-md")) {
            spec_draft_model = need(arg);
        } else if (arg[0] == '-') {
            fprintf(stderr, "unknown flag: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        } else {
            break;
        }
    }
    if (storage_options.payload_disk_dir.empty() != (storage_options.payload_disk_bytes == 0)) {
        fprintf(stderr, "SSD requires --kvmem-disk-path and positive --kvmem-disk-mib together; omit the path when disabled\n");
        return 1;
    }
    if ((storage_options.payload_host_bytes || storage_options.payload_disk_bytes) && !kparams.enabled) {
        fprintf(stderr, "RAM/SSD storage options require --kvmem\n");
        return 1;
    }
    if (storage_options.payload_disk_bytes && kparams.cpu_bytes) {
        fprintf(stderr, "shared KV SSD cannot be combined with the legacy CPU spill arena\n");
        return 1;
    }
    kparams.payload_host_bytes = kparams.enabled ? (storage_options.payload_host_bytes ?
        storage_options.payload_host_bytes : kvmem::CliOptions::default_host_bytes) : 0;
    kparams.payload_disk_bytes = storage_options.payload_disk_bytes;
    kparams.payload_disk_dir = storage_options.payload_disk_dir.empty() ? nullptr : storage_options.payload_disk_dir.c_str();
    // Validate before backend initialization and loading a potentially large model.
    if (!kvmem_cache_types_ok(cache_type_k, cache_type_v)) {
        fprintf(stderr, "incompatible KV cache types: K=%s, V=%s; quantized K/V must both use q8_0, q5_0 or q4_0; "
                "set both -ctk and -ctv, or use --kv-dtype TYPE to set both\n",
                ggml_type_name(cache_type_k), ggml_type_name(cache_type_v));
        return 1;
    }
    if (list_devices) {
        ggml_backend_load_all();
        for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
            auto * dev = ggml_backend_dev_get(d);
            fprintf(stdout, "%s: %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        }
        return 0;
    }
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    try {
        if (!kparams.enabled && (scoring_options.auxiliary_set || scoring_options.lrfu))
            throw std::invalid_argument("KVMem scoring settings require --kvmem");
        scoring_options.resolve_query_scoring();
        kparams.query_meanq_cosine = scoring_options.query_scorer == kvmem::QueryScorer::MeanQCosine;
        kparams.graph_alpha = scoring_options.graph_alpha;
        kparams.score_agg = scoring_options.score_agg;
        kparams.lrfu = scoring_options.lrfu;
        kparams.lrfu_lambda = scoring_options.lrfu_lambda;
        kparams.lrfu_top_fraction = scoring_options.lrfu_top_fraction;
    } catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
    if (!prompt_file.empty()) {
        std::ifstream in(prompt_file);
        if (!in) {
            fprintf(stderr, "failed to read prompt file: %s\n", prompt_file.c_str());
            return 1;
        }
        prompt.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    } else if (i < argc) {
        prompt = argv[i++];
        for (; i < argc; ++i) {
            prompt += " ";
            prompt += argv[i];
        }
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    model_params.load_mtp = spec_algorithm == kvmem::SpeculativeAlgorithm::Mtp;
    try {
        device_config.apply(device_options, model_params);
    } catch (const std::exception & e) {
        fprintf(stderr, "invalid GPU configuration: %s\n", e.what());
        return 1;
    }
    if ((spec_algorithm == kvmem::SpeculativeAlgorithm::DFlash2 ||
         spec_algorithm == kvmem::SpeculativeAlgorithm::DSpark) &&
            (spec_draft_model.empty() || device_config.devices.size() > 2 || model_params.split_mode == LLAMA_SPLIT_MODE_TENSOR)) {
        fprintf(stderr, "DFlash2/DSpark requires --spec-draft-model and one GPU\n");
        return 1;
    }
    if (model_params.split_mode == LLAMA_SPLIT_MODE_TENSOR && spec_enabled && kparams.enabled && kparams.mtp_state != 0) {
        fprintf(stderr, "tensor KVMem currently supports MTP snapshots only; use --kvmem-mtp-state snapshots\n");
        return 1;
    }
    if (device_config.devices.size() > 2 && spec_enabled && !kparams.enabled &&
        model_params.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
        fprintf(stderr, "multi-GPU MTP requires --kvmem\n");
        return 1;
    }
    if (device_config.devices.size() > 2 && spec_enabled && !spec_draft_model.empty()) {
        fprintf(stderr, "multi-GPU MTP currently requires an embedded nextn draft layer\n");
        return 1;
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }
    if ((spec_algorithm == kvmem::SpeculativeAlgorithm::DFlash2 ||
         spec_algorithm == kvmem::SpeculativeAlgorithm::DSpark) && llama_model_n_devices(model) > 1) {
        fprintf(stderr, "DFlash2/DSpark requires a single target GPU; select one device\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }
    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }

    if (n_ctx <= 0) {
        n_ctx = n_prompt + n_predict;
    }
    if (n_batch <= 0) {
        n_batch = 512;
    }
    if (n_ubatch <= 0) {
        n_ubatch = n_batch;
    }

    if (kparams.enabled && kparams.budget > 0) {
        const uint32_t bt = kparams.block_tokens ? kparams.block_tokens : 32u;
        const uint32_t sink = kparams.sink_tokens == 0 ? bt : kparams.sink_tokens;
        const uint32_t room = kparams.budget > sink ? kparams.budget - sink : bt;
        const uint32_t cap = std::min(kparams.gen_reserve ? kparams.gen_reserve : 256u, room);
        if (static_cast<uint32_t>(n_batch) > cap) {
            fprintf(stderr, "llama-kvmem-cli: clamping -b %d → %u so each prefill chunk fits the slot pool\n",
                    n_batch, cap);
            n_batch = static_cast<int>(cap);
        }
        if (n_ubatch > n_batch) {
            n_ubatch = n_batch;
        }
    }

    if (kparams.enabled) {
        if (!spec_enabled) kparams.mtp_state = 0;
        if (dump_kv) {
#ifdef _WIN32
            _putenv_s("KVMEM_DUMP_CAPTURE", "1");
#else
            setenv("KVMEM_DUMP_CAPTURE", "1", 1);
#endif
        }
        if (query_last > 0 && n_prompt > 0) {
            const int last = std::min({query_last, n_prompt, int(query_max_tokens)});
            kparams.query_begin = n_prompt - last;
            kparams.query_end = n_prompt;
        }
        if (!force_substr.empty()) {
            std::string acc;
            for (int ti = 0; ti < n_prompt; ++ti) {
                char buf[256];
                const int n = llama_token_to_piece(vocab, prompt_tokens[ti], buf, sizeof(buf), 0, true);
                if (n > 0) {
                    acc.append(buf, static_cast<size_t>(n));
                }
                if (acc.find(force_substr) != std::string::npos) {
                    kparams.force_pos = ti;
                    break;
                }
            }
            fprintf(stderr, "KVMEM_TRACE n_prompt=%d force_substr='%s' force_pos=%d block~=%d\n",
                    n_prompt, force_substr.c_str(), kparams.force_pos,
                    kparams.force_pos >= 0 && kparams.block_tokens
                        ? kparams.force_pos / static_cast<int>(kparams.block_tokens)
                        : -1);
        } else {
            fprintf(stderr, "KVMEM_TRACE n_prompt=%d\n", n_prompt);
        }
        llama_kvmem_set_params(&kparams);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
    ctx_params.n_batch = static_cast<uint32_t>(n_batch);
    ctx_params.n_ubatch = static_cast<uint32_t>(n_ubatch);
    ctx_params.n_seq_max = 1;
    ctx_params.no_perf = false;
    ctx_params.type_k = cache_type_k;
    ctx_params.type_v = cache_type_v;
    if (spec_enabled) {
        const uint32_t n_out = (uint32_t) (1 + std::max(0, spec_n_max));
        ctx_params.n_outputs_max = n_out;
        ctx_params.n_outputs_max_per_seq = n_out;
        // GPU GDN snapshot planes for MTP verify reject (qw3-style). Host
        // PARTIAL_ONLY is only used if the arch clamps this to 0.
        ctx_params.n_rs_seq = (uint32_t) std::max(0, spec_n_max);
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "failed to create llama_context\n");
        return 1;
    }

    kvmem_spec_session spec_sess;
    if (spec_enabled) {
        kvmem_spec_opts sopts;
        sopts.n_max = spec_n_max;
        sopts.p_min = spec_p_min;
        sopts.n_gpu_layers = ngl;
        sopts.n_ctx = n_ctx;
        sopts.n_batch = n_batch;
        sopts.n_ubatch = n_ubatch;
        sopts.kvmem_enabled = kparams.enabled;
        sopts.draft_model = spec_draft_model;
        sopts.algorithm = spec_algorithm;
        sopts.type_k = cache_type_k;
        sopts.type_v = cache_type_v;
        sopts.draft_type = spec_cache_type;
        if (!kvmem_spec_start(spec_sess, model, ctx, sopts)) {
            return 1;
        }
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }

    if (!tokens_only && !no_prompt) {
        for (auto id : prompt_tokens) {
            char buf[256];
            const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "token_to_piece failed\n");
                return 1;
            }
            fwrite(buf, 1, static_cast<size_t>(n), stdout);
        }
        fflush(stdout);
    }

    auto decode_span = [&](int pos0, int pos1, const char * what) -> int {
        if (spec_sess.ok) {
            return kvmem_spec_decode_span(ctx, spec_sess.spec, prompt_tokens.data(),
                                          pos0, pos1, n_batch, what);
        }
        int n_pos = pos0;
        while (n_pos < pos1) {
            const int n = std::min(n_batch, pos1 - n_pos);
            llama_batch batch = llama_batch_get_one(prompt_tokens.data() + n_pos, n);
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                fprintf(stderr, "llama_decode(%s) failed rc=%d at pos=%d n=%d\n",
                        what, rc, n_pos, n);
                return rc;
            }
            n_pos += n;
        }
        return 0;
    };

    // Speculative-simple leaves the last prompt token as id_last (not in KV).
    const int eval_end = spec_sess.ok ? n_prompt - 1 : n_prompt;
    if (spec_sess.ok && n_prompt < 1) {
        fprintf(stderr, "speculative decode needs a non-empty prompt\n");
        return 1;
    }

    const bool do_retr = kparams.enabled && kparams.method == 1 && kparams.query_begin > 0;
    const bool recr_ckpt = do_retr && llama_kvmem_has_recurrent();
    const int prefix_end = do_retr ? kparams.query_begin : eval_end;
    llama_kvmem_tail_mean_state query_prefix;
    llama_kvmem_attention_view probe_view;
    std::vector<uint8_t> query_draft;
    if(do_retr) {
        llama_kvmem_turn_spans spans;
        spans.query={{prefix_end,std::min(kparams.query_end,eval_end)}};
        spans.mandatory={{prefix_end,eval_end}};
        spans.replay_begin=prefix_end;
        llama_kvmem_set_turn_spans(spans);
    }

    if (kparams.lrfu) llama_kvmem_set_request_span(kparams.query_begin,
        std::min(kparams.query_end > 0 ? kparams.query_end : eval_end, eval_end), kparams.force_pos);
    llama_kvmem_begin_retrieval_round();
    if (decode_span(0, prefix_end, "prefill") != 0) {
        return 1;
    }

    std::vector<uint8_t> gdn_ckpt;
    if (recr_ckpt) {
        llama_synchronize(ctx);
        llama_kvmem_trace_cells(ctx, "before_gdn_ckpt");
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, fl);
        if (sz == 0) {
            fprintf(stderr, "llama_state_seq_get_size_ext(PARTIAL_ONLY) returned 0\n");
            return 1;
        }
        gdn_ckpt.resize(sz);
        const size_t ncopy = llama_state_seq_get_data_ext(ctx, gdn_ckpt.data(), sz, 0, fl);
        if (ncopy != sz) {
            fprintf(stderr, "GDN checkpoint copy failed (%zu != %zu)\n", ncopy, sz);
            return 1;
        }
        fprintf(stderr, "KVMEM_TRACE gdn_ckpt pos_end=%d bytes=%zu query_begin=%d\n",
                prefix_end, sz, kparams.query_begin);
    }
    if(do_retr) {
        if(!llama_kvmem_get_tail_mean(prefix_end,query_prefix)) return 1;
        probe_view=llama_kvmem_get_attention_view();
        if(spec_sess.ok) common_speculative_get_state(spec_sess.spec,0,query_draft);
        if(decode_span(prefix_end,eval_end,"prefill-query")!=0) return 1;
    }
    llama_synchronize(ctx);

    const llama_perf_context_data perf_prefill = llama_perf_context(ctx);
    double retrieval_ms = 0.0;
    double replay_wall_ms = 0.0;
    int replay_n = 0;
    double replay_eval_ms = 0.0;

    if (kparams.enabled && dump_kv) {
        fprintf(stderr, "KVMEM_KV --- after prefill ---\n");
        llama_kvmem_dump_kv_compare(ctx, 0);
        llama_kvmem_dump_kv_compare(ctx, -1);
        fprintf(stderr, "KVMEM_KV --- overwrite needle block from raw, compare to snapshot ---\n");
        llama_kvmem_dump_kv_writeback(ctx, -1);
    }

    if(do_retr) {
        const uint32_t block=kparams.block_tokens;
        const uint32_t budget=kparams.budget ? kparams.budget : uint32_t(n_ctx);
        const auto plan=kvmem::query_replay_plan({uint64_t(prefix_end),uint64_t(eval_end)},
            prefix_end,eval_end,query_replay_max_tokens,block,
            std::min<uint64_t>((uint64_t(prefix_end)+block-1)/block,budget/block),
            (uint64_t(budget)+kparams.gen_reserve)/block,spec_sess.ok ? spec_n_max+1 : 0);
        bool replay=false;
        const auto t0=std::chrono::steady_clock::now();
        if(plan.admitted) {
            const auto selection=llama_kvmem_preview_retrieval(&query_prefix);
            if(!llama_kvmem_commit_unchanged(probe_view,selection) &&
                llama_kvmem_selection_fits(selection,eval_end,spec_sess.ok ? spec_n_max+1 : 0)) {
                llama_kvmem_apply_selection(selection);
                replay=true;
            }
        }
        retrieval_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        if(replay) {
            if(recr_ckpt) {
                const auto flags=LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
                if(llama_state_seq_set_data_ext(ctx,gdn_ckpt.data(),gdn_ckpt.size(),0,flags)!=gdn_ckpt.size()) return 1;
            }
            if(!llama_kvmem_set_tail_mean(prefix_end,query_prefix)) return 1;
            llama_kvmem_set_replay(true);
            if(!llama_kvmem_remove_logical(ctx,prefix_end,-1)) return 1;
            if(spec_sess.ok) {
                if(!llama_kvmem_remove_logical(spec_sess.ctx_dft,prefix_end,-1)) return 1;
                common_speculative_set_state(spec_sess.spec,0,query_draft);
            }
            const auto before=llama_perf_context(ctx);
            const auto started=std::chrono::steady_clock::now();
            const int rc=decode_span(prefix_end,eval_end,"query replay");
            llama_synchronize(ctx);
            replay_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            const auto after=llama_perf_context(ctx);
            replay_n=after.n_p_eval-before.n_p_eval;
            replay_eval_ms=after.t_p_eval_ms-before.t_p_eval_ms;
            llama_kvmem_set_replay(false);
            if(rc) return 1;
        }
        llama_kvmem_pin_working_set();
        fprintf(stderr,"KVMEM_PREFILL_DECISION path=%s reason=%s replay_rows=%d\n",
            replay?"query_replay":plan.admitted?"first_pass":"query_replay_skipped",
            plan.reason,replay?eval_end-prefix_end:0);
    }

    std::vector<llama_token> gen;
    gen.reserve(static_cast<size_t>(n_predict));
    llama_kvmem_accept_retrieval_round();
    llama_kvmem_end_prefill_capture();
    llama_kvmem_attention_begin(prompt_tokens.data(),uint32_t(prompt_tokens.size()));
    llama_synchronize(ctx);
    const auto t_gen0 = std::chrono::steady_clock::now();
    if (spec_sess.ok) {
        const kvmem_spec_gen_stats gst = kvmem_spec_generate(
                ctx, model, spec_sess, prompt_tokens, n_predict, temp,
                [&](llama_token id, const std::string & piece, bool /*from_draft*/) {
                    gen.push_back(id);
                    if (tokens_only) {
                        printf("%d\n", id);
                    } else {
                        fwrite(piece.data(), 1, piece.size(), stdout);
                        fflush(stdout);
                    }
                }, seed_set ? seed : LLAMA_DEFAULT_SEED);
        fprintf(stderr, "KVMEM_SPEC_METRICS %s\n", gst.metrics().dump().c_str());
        if (gst.failed) {
            kvmem_spec_stop(spec_sess);
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    } else {
        for (int n_gen = 0; n_gen < n_predict; ++n_gen) {
            const llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }
            gen.push_back(new_token);
            if (tokens_only) {
                printf("%d\n", new_token);
            } else {
                char buf[256];
                const int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    fprintf(stderr, "token_to_piece failed\n");
                    return 1;
                }
                fwrite(buf, 1, static_cast<size_t>(n), stdout);
                fflush(stdout);
            }
            llama_batch batch = llama_batch_get_one(&gen.back(), 1);
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                fprintf(stderr, "llama_decode(gen) failed rc=%d\n", rc);
                return 1;
            }
        }
        llama_kvmem_decode_mean_flush();
    }
    if (!tokens_only) {
        printf("\n");
    }

    llama_synchronize(ctx);
    const char * hash_gdn = std::getenv("KVMEM_GDN_HASH");
    if (hash_gdn && hash_gdn[0] == '1' && llama_kvmem_has_recurrent()) {
        const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t size = llama_state_seq_get_size_ext(ctx, 0, flags);
        std::vector<uint8_t> state(size);
        if (!size || llama_state_seq_get_data_ext(ctx, state.data(), size, 0, flags) != size) {
            fprintf(stderr, "KVMEM_GDN_HASH failed to read recurrent state\n");
            return 1;
        }
        uint64_t hash = 14695981039346656037ull;
        for (uint8_t byte : state) hash = (hash ^ byte) * 1099511628211ull;
        fprintf(stderr, "KVMEM_GDN_HASH bytes=%zu fnv64=%016llx\n", size, (unsigned long long) hash);
    }
    const double gen_wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_gen0).count();
    llama_perf_context_print(ctx);
    {
        const llama_perf_context_data p = llama_perf_context(ctx);
        const double prompt_tps =
            p.t_p_eval_ms > 0.0 ? 1000.0 * static_cast<double>(p.n_p_eval) / p.t_p_eval_ms : 0.0;
        const double gen_tps =
            p.t_eval_ms > 0.0 ? 1000.0 * static_cast<double>(p.n_eval) / p.t_eval_ms : 0.0;
        fprintf(stderr,
                "KVMEM_PERF load_ms=%.2f prompt_n=%d prompt_ms=%.2f prompt_toks=%.2f gen_n=%d gen_ms=%.2f gen_toks=%.2f\n",
                p.t_load_ms, p.n_p_eval, p.t_p_eval_ms, prompt_tps, p.n_eval, p.t_eval_ms, gen_tps);
        const double prefill_tps = perf_prefill.t_p_eval_ms > 0.0
            ? 1000.0 * static_cast<double>(perf_prefill.n_p_eval) / perf_prefill.t_p_eval_ms
            : 0.0;
        fprintf(stderr,
                "KVMEM_STAGE prefill_n=%d prefill_ms=%.2f prefill_toks=%.2f "
                "retrieval_ms=%.2f replay_n=%d replay_eval_ms=%.2f replay_wall_ms=%.2f\n",
                perf_prefill.n_p_eval, perf_prefill.t_p_eval_ms, prefill_tps,
                retrieval_ms, replay_n, replay_eval_ms, replay_wall_ms);
        const int n_out = (int) gen.size();
        const double gen_wall_tps =
            gen_wall_ms > 0.0 ? 1000.0 * static_cast<double>(n_out) / gen_wall_ms : 0.0;
        fprintf(stderr, "KVMEM_GEN_WALL n=%d ms=%.2f toks=%.2f\n",
                n_out, gen_wall_ms, gen_wall_tps);
    }

    if (kparams.enabled) {
        const auto storage = llama_kvmem_execution_storage_stats(execution.get());
        fprintf(stderr, "KVMEM_STORAGE host_bytes=%llu host_budget=%llu disk_bytes=%llu disk_budget=%llu "
                "read_bytes=%llu written_bytes=%llu errors=%llu\n",
                (unsigned long long)storage.host_bytes, (unsigned long long)storage.host_capacity,
                (unsigned long long)storage.disk_bytes, (unsigned long long)storage.disk_capacity,
                (unsigned long long)storage.read_bytes, (unsigned long long)storage.written_bytes,
                (unsigned long long)storage.errors);
    }
    llama_kvmem_attention_finish(true);
    kvmem_spec_stop(spec_sess);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
