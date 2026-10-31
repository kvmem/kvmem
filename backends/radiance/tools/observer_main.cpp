#include "statistics_observer.hpp"
#include "engine.h"
#include "hf_model.h"
#include <charconv>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    kvmem::radiance::SelectionOptions options;
    std::vector<char*> engine_args{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--kvmem-rolling-generation") { options.rolling_generation = true; continue; }
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
        if (flag == "--kvmem-budget" || flag == "--kvmem-block-tokens" || flag == "--kvmem-host-mib" || flag == "--kvmem-replay-tokens") {
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
            else options.host_bytes = value * 1024 * 1024;
        } else engine_args.push_back(argv[i]);
    }
    engine_args.push_back(nullptr);
    rad::Config cfg;
    const int status = rad::config_parse(int(engine_args.size()) - 1, engine_args.data(), &cfg);
    if (status == 1) {
        rad::config_usage(argv[0]);
        std::cout << "KVMem qualification: --kvmem-budget TOKENS --kvmem-block-tokens TOKENS --kvmem-host-mib MiB --kvmem-replay-tokens TOKENS\n";
        std::cout << "Always-kept bands: --kvmem-sink-tokens N --kvmem-recent-tokens N (-1: common KVMem auto allocation)\n";
        std::cout << "Experimental reusable generation workspace: --kvmem-rolling-generation\n";
        return 0;
    }
    if (status < 0) { rad::config_usage(argv[0]); return 2; }
    if (!rad::resolve_hf_model(cfg)) return 1;
    rad::Engine engine(std::move(cfg), std::make_unique<kvmem::radiance::StatisticsObserver>(options));
    if (engine.start() < 0) return 1;
    if (engine.config().debug_graph) return engine.dump();
    return engine.serve() < 0 ? 1 : 0;
}
