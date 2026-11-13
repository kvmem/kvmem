#include "batched_observer.hpp"
#include "engine.h"
#include "hf_model.h"
#include "radiance_capabilities.hpp"
#include <charconv>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    if (argc==2 && std::string_view(argv[1])=="--kvmem-capabilities") {
        std::cout << kvmem_radiance_capabilities << '\n'; return 0;
    }
    kvmem::radiance::SelectionOptions options;
    options.max_draft_tokens=kvmem_radiance_max_draft_tokens;
    bool aggregation_set = false, alpha_set = false, lrfu_set = false;
    std::vector<char*> engine_args{argv[0]};
    try {
    for (int i = 1; i < argc; ++i) {
        std::string_view flag(argv[i]);
#ifdef KVMEM_RADIANCE_ARGV_PARSE_TEST
        // config_parse gives help precedence over errors. The parser test removes its trailing
        // help marker so every preceding option is validated, then exits before model startup.
        if (flag=="--help") {
            if(i!=argc-1)throw std::invalid_argument("parser test requires --help at the end");
            continue;
        }
#endif
        if (flag=="--kvmem") continue;
        if (flag=="--kvmem-drafter") {
            if (++i>=argc) throw std::invalid_argument("missing native drafter mode");
            options.drafter_mode=argv[i];
            if (options.drafter_mode!="off" && options.drafter_mode!="mtp" && options.drafter_mode!="dflash2")
                throw std::invalid_argument("native drafter mode must be off, mtp or dflash2");
            continue;
        }
        if (flag=="--kvmem-gen-reserve") flag="--kvmem-replay-tokens";
        if (flag=="--kvmem-sessions") flag="--kvmem-cache-sessions";
        if (flag=="--kvmem-query-max-tokens") flag="--kvmem-query-tokens";
        if (flag=="--kvmem-query-replay-max-tokens") flag="--kvmem-query-replay-tokens";
        if (flag=="--kvmem-lrfu-top-frac") flag="--kvmem-lrfu-top-fraction";
        if(flag=="--kvmem-disk-path") {
            if(++i>=argc || !*argv[i])throw std::invalid_argument("missing SSD cache directory");
            options.disk_path=argv[i];continue;
        }
        if (flag == "--kvmem-rolling-generation") { options.rolling_generation = true; continue; }
        if (flag == "--kvmem-lrfu" || flag == "--no-kvmem-lrfu") {
            options.lrfu = flag == "--kvmem-lrfu"; lrfu_set = true; continue;
        }
        if (flag == "--kvmem-query-scorer" || flag == "--kvmem-score-agg" || flag == "--kvmem-graph-alpha" ||
            flag == "--kvmem-lrfu-lambda" || flag == "--kvmem-lrfu-top-fraction") {
            if (++i >= argc) throw std::invalid_argument("missing recall option value");
            if (flag == "--kvmem-query-scorer") options.scorer = kvmem::parse_query_scorer(argv[i]);
            else if (flag == "--kvmem-score-agg") { options.aggregation = kvmem::parse_score_aggregation(argv[i]); aggregation_set = true; }
            else if (flag == "--kvmem-graph-alpha") { options.graph_alpha = kvmem::parse_graph_alpha(argv[i]); alpha_set = true; }
            else if (flag == "--kvmem-lrfu-lambda") options.lrfu_lambda = kvmem::parse_lrfu_real(argv[i],true);
            else options.lrfu_top_fraction = kvmem::parse_lrfu_real(argv[i],false);
            continue;
        }
        if (flag == "--kvmem-sink-tokens" || flag == "--kvmem-recent-tokens") {
            if (++i >= argc) { std::cerr << "missing " << flag << " value\n"; return 2; }
            int64_t value = 0;
            const std::string_view text(argv[i]);
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < -1 || value > INT32_MAX) {
                std::cerr << "invalid " << flag << " value\n"; return 2;
            }
            if (flag == "--kvmem-sink-tokens") options.sink_tokens = value;
            else options.recent_tokens = value;
            continue;
        }
        if(flag=="--kvmem-cache-min-tokens") {
            if(++i>=argc)throw std::invalid_argument("missing cache minimum token count");
            uint32_t value=0;const std::string_view text(argv[i]);
            const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
            if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || value>INT32_MAX)
                throw std::invalid_argument("cache minimum must be between 0 and 2147483647");
            options.cache_min_tokens=value;continue;
        }
        if (flag == "--kvmem-cache-sessions" || flag == "--kvmem-cache-records") {
            if (++i>=argc) { std::cerr << "missing cache group/record count\n"; return 2; }
            uint32_t value=0; const std::string_view text(argv[i]);
            const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
            if (parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || value>1024) {
                std::cerr << "cache group/record count must be between 0 and 1024\n"; return 2;
            }
            if(flag == "--kvmem-cache-records")options.cache_records=value;
            else options.cache_sessions=value;
            continue;
        }
        if (flag == "--kvmem-budget" || flag == "--kvmem-block-tokens" || flag == "--kvmem-host-mib" || flag=="--kvmem-disk-mib" ||
            flag == "--kvmem-replay-tokens" || flag == "--kvmem-query-tokens" || flag == "--kvmem-query-replay-tokens") {
            if (++i >= argc) { std::cerr << "missing " << flag << " value\n"; return 2; }
            uint64_t value = 0;
            const std::string_view text(argv[i]);
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !value || value > INT32_MAX) {
                std::cerr << "invalid " << flag << " value\n"; return 2;
            }
            if (flag == "--kvmem-budget") options.budget_tokens = uint32_t(value);
            else if (flag == "--kvmem-block-tokens") options.block_tokens = uint32_t(value);
            else if (flag == "--kvmem-replay-tokens") options.replay_tokens = uint32_t(value);
            else if (flag == "--kvmem-query-tokens") options.query_tokens = uint32_t(value);
            else if (flag == "--kvmem-query-replay-tokens") options.query_replay_tokens = uint32_t(value);
            else if (flag == "--kvmem-disk-mib") options.disk_bytes=value*1024*1024;
            else options.host_bytes = value * 1024 * 1024;
        } else engine_args.push_back(argv[i]);
    }
    if (options.scorer == kvmem::QueryScorer::MeanQCosine) {
        if (!aggregation_set) options.aggregation = 1;
        if (!alpha_set) options.graph_alpha = 0;
        if (!lrfu_set) options.lrfu = true;
    }
    if(options.cache_min_tokens && !options.budget_tokens)
        throw std::invalid_argument("cache minimum requires a positive KVMem budget");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
    engine_args.push_back(nullptr);
    rad::Config cfg;
    const int status = rad::config_parse(int(engine_args.size()) - 1, engine_args.data(), &cfg);
    if (status == 1) {
        rad::config_usage(argv[0]);
        std::cout << "KVMem qualification: --kvmem-budget TOKENS --kvmem-block-tokens TOKENS --kvmem-host-mib MiB --kvmem-replay-tokens TOKENS\n";
        std::cout << "Always-kept bands: --kvmem-sink-tokens N --kvmem-recent-tokens N (-1: common KVMem auto allocation)\n";
        std::cout << "Query: --kvmem-query-scorer page-softmax|meanq-cosine --kvmem-score-agg mean|max --kvmem-graph-alpha [0,1]\n";
        std::cout << "Bounds: --kvmem-query-tokens 512 --kvmem-query-replay-tokens 2048; generation reserve: --kvmem-replay-tokens\n";
        std::cout << "History: --kvmem-lrfu|--no-kvmem-lrfu --kvmem-lrfu-lambda [0,1] --kvmem-lrfu-top-fraction (0,1]\n";
        std::cout << "Retained content groups: --kvmem-cache-sessions N (default 8; 0 disables), sharing --kvmem-host-mib\n";
        std::cout << "Minimum restorable history: --kvmem-cache-min-tokens N (default 0: disabled)\n";
        std::cout << "Retained records: --kvmem-cache-records N (default 0: follow group limit; each record has up to 3 restore points)\n";
        std::cout << "Optional SSD: --kvmem-disk-path DIR --kvmem-disk-mib MiB (shared native KV and retained state)\n";
        return 0;
    }
    if (status < 0) { rad::config_usage(argv[0]); return 2; }
    // The fused GDN path holds the deployment window (drafts + one target
    // token), not just the current adaptive step. Enforce the same qualified
    // cap as the launcher before model resolution or GPU initialization.
    if (cfg.n_spec > kvmem_radiance_max_draft_tokens) {
        std::cerr << "Radiance KVMem supports at most " << kvmem_radiance_max_draft_tokens
                  << " speculative draft tokens; larger GDN deployment windows are not supported\n";
        return 2;
    }
#ifdef KVMEM_RADIANCE_ARGV_PARSE_TEST
    std::cout << "KVMEM_NATIVE_ARGV_VALID\n";
    return 0;
#else
    if (!rad::resolve_hf_model(cfg)) return 1;
    rad::Engine engine(std::move(cfg), std::make_unique<kvmem::radiance::BatchedStatisticsObserver>(options));
    if (engine.start() < 0) return 1;
    if (engine.config().debug_graph) return engine.dump();
    return engine.serve() < 0 ? 1 : 0;
#endif
}
