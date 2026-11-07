#pragma once

#include "kvmem/retrieval_options.hpp"
#include "kvmem/layerwise_options.hpp"
#include "kvmem/layerwise_mtp.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace kvmem {

inline void reject_retired_storage_option(const std::string & arg) {
    const auto name = arg.substr(0, arg.find('='));
    if (name == "--kvmem-nvme-gb" || name == "--kvmem-nvme-dir" ||
        name == "--kvmem-raw-k-nvme" || name == "--kvmem-session-nvme-gb" ||
        name == "--kvmem-session-cache-dir")
        throw std::invalid_argument(name + " was retired; use --kvmem-host-mib H with "
            "--kvmem-disk-path PATH --kvmem-disk-mib D (one shared RAM/SSD budget; no automatic conversion)");
}

// Version 1 of the public KVMem CLI contract. Backend-native flags do not
// belong here. Adapters validate their capabilities after resolving this object.
struct CliOptions {
    static constexpr uint32_t default_budget = 32768;
    static constexpr uint32_t default_reserve = 1024;
    static constexpr uint64_t default_host_bytes = uint64_t(16384) << 20;
    bool enabled = false, lrfu = false;
    int enabled_override = -1;
    uint32_t budget = 0, gen_reserve = 0, sink_tokens = 0, recent_tokens = 64, sessions = 4;
    uint64_t payload_host_bytes = 0, payload_disk_bytes = 0;
    std::string payload_disk_dir;
    double lrfu_lambda = 0.5, lrfu_top_fraction = 0.9;
    int score_agg = 0;
    QueryScorer query_scorer = QueryScorer::PageSoftmax;
    RecallStrategy recall_strategy = RecallStrategy::Legacy;
    LayerwiseOptions layerwise;
    bool layerwise_limits_set = false;
    bool sink_tokens_set = false, recent_tokens_set = false;
    double graph_alpha = 0.2;
    bool score_agg_set = false, lrfu_set = false, graph_alpha_set = false, lrfu_tuning_set = false;

    void resolve_query_scoring() {
        const bool cosine = query_scorer == QueryScorer::MeanQCosine;
        if (!score_agg_set) score_agg = cosine ? 1 : 0;
        if (!lrfu_set) lrfu = cosine;
        if (!graph_alpha_set) graph_alpha = cosine ? 0.0 : 0.2;
        validate_query_scoring(query_scorer, score_agg, graph_alpha);
        if (lrfu_tuning_set && !lrfu)
            throw std::invalid_argument("LRFU tuning requires enabled LRFU");
    }
    bool budget_set = false, sessions_set = false, auxiliary_set = false;
    LayerwiseMtpHistory layerwise_mtp_history = LayerwiseMtpHistory::Unspecified;

    static uint32_t integer(const std::string & name, const char * value, uint32_t minimum) {
        const std::string text(value);
        uint64_t number = 0;
        if (text.empty()) throw std::invalid_argument(name + " requires an integer");
        for (const char c : text) {
            if (c < '0' || c > '9') throw std::invalid_argument(name + " requires an unsigned decimal integer");
            number = number * 10 + uint32_t(c - '0');
            if (number > uint64_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument(name + " exceeds 2147483647");
        }
        if (number < minimum) throw std::invalid_argument(name + " must be positive");
        return uint32_t(number);
    }

    template<class Need> bool parse(const std::string & arg, const Need & need) {
        reject_retired_storage_option(arg);
        if (arg == "--kvmem" || arg == "--no-kvmem") {
            enabled_override = arg == "--kvmem" ? 1 : 0;
            return true;
        }
        if (arg == "--kvmem-budget") {
            budget = integer(arg, need(arg.c_str()), 1);
            budget_set = true;
            return true;
        }
        if (arg == "--kvmem-recall-strategy") recall_strategy = parse_recall_strategy(need(arg.c_str()));
        else if (arg == "--kvmem-layerwise-mtp-history") layerwise_mtp_history = parse_layerwise_mtp_history(need(arg.c_str()));
        else if (arg == "--kvmem-layerwise-activation-ram-mib" || arg == "--kvmem-layerwise-activation-disk-mib" ||
                 arg == "--kvmem-layerwise-staging-mib" || arg == "--kvmem-layerwise-metadata-mib" ||
                 arg == "--kvmem-layerwise-activation-metadata-mib") {
            const auto bytes=uint64_t(integer(arg,need(arg.c_str()),
                arg=="--kvmem-layerwise-activation-ram-mib" || arg=="--kvmem-layerwise-activation-disk-mib" ? 0:1))<<20;
            if (arg=="--kvmem-layerwise-activation-ram-mib") layerwise.activation_ram_bytes=bytes;
            else if (arg=="--kvmem-layerwise-activation-disk-mib") layerwise.activation_disk_bytes=bytes;
            else if (arg=="--kvmem-layerwise-staging-mib") layerwise.staging_bytes=bytes;
            else if (arg=="--kvmem-layerwise-metadata-mib") layerwise.metadata_bytes=bytes;
            else layerwise.activation_metadata_bytes=bytes;
            layerwise_limits_set=true;
        }
        else if (arg == "--kvmem-gen-reserve") gen_reserve = integer(arg, need(arg.c_str()), 1);
        else if (arg == "--kvmem-sink-tokens") {sink_tokens = integer(arg, need(arg.c_str()), 0);sink_tokens_set=true;}
        else if (arg == "--kvmem-recent-tokens") {recent_tokens = integer(arg, need(arg.c_str()), 0);recent_tokens_set=true;}
        else if (arg == "--kvmem-sessions") {
            sessions = integer(arg, need(arg.c_str()), 1);
            sessions_set = true;
        } else if (arg == "--kvmem-host-mib") payload_host_bytes = uint64_t(integer(arg, need(arg.c_str()), 1)) << 20;
        else if (arg == "--kvmem-disk-mib") {
            payload_disk_bytes = uint64_t(integer(arg, need(arg.c_str()), 0)) << 20;
            return true;
        }
        else if (arg == "--kvmem-disk-path") {
            payload_disk_dir = need(arg.c_str());
            if (payload_disk_dir.empty()) throw std::invalid_argument(arg + " must not be empty");
        } else if (arg == "--kvmem-lrfu" || arg == "--no-kvmem-lrfu") {
            lrfu = arg == "--kvmem-lrfu";
            lrfu_set = true;
            return true;
        }
        else if (arg == "--kvmem-query-scorer") query_scorer = parse_query_scorer(need(arg.c_str()));
        else if (arg == "--kvmem-graph-alpha") { graph_alpha = parse_graph_alpha(need(arg.c_str())); graph_alpha_set = true; }
        else if (arg == "--kvmem-lrfu-lambda") { lrfu_lambda = parse_lrfu_real(need(arg.c_str()), true); lrfu_tuning_set = true; }
        else if (arg == "--kvmem-lrfu-top-frac") { lrfu_top_fraction = parse_lrfu_real(need(arg.c_str()), false); lrfu_tuning_set = true; }
        else if (arg == "--kvmem-score-agg") { score_agg = parse_score_aggregation(need(arg.c_str())); score_agg_set = true; }
        else return false;
        auxiliary_set = true;
        return true;
    }

    void resolve() {
        // Positive B keeps existing explicit-budget launch commands working.
        // Omission is disabled on both EXEs; --kvmem selects shared defaults.
        enabled = enabled_override < 0 ? budget_set : enabled_override != 0;
        if (!enabled && (budget_set || auxiliary_set || payload_disk_bytes || lrfu))
            throw std::invalid_argument("KVMem settings require --kvmem (or a positive --kvmem-budget); remove --no-kvmem");
        const bool uses_disk = payload_disk_bytes ||
            (recall_strategy==RecallStrategy::Layerwise && layerwise.activation_disk_bytes);
        if (payload_disk_dir.empty() == uses_disk)
            throw std::invalid_argument("KVMem SSD requires --kvmem-disk-path with a positive canonical or layerwise activation SSD quota; omit the path when both are disabled");
        if (!enabled) return;
        if (layerwise_limits_set && recall_strategy!=RecallStrategy::Layerwise)
            throw std::invalid_argument("layerwise limits require --kvmem-recall-strategy layerwise");
        if (layerwise_mtp_history!=LayerwiseMtpHistory::Unspecified && recall_strategy!=RecallStrategy::Layerwise)
            throw std::invalid_argument("MTP history policy requires --kvmem-recall-strategy layerwise");
        if (recall_strategy==RecallStrategy::Layerwise) validate_layerwise_options(layerwise);
        resolve_query_scoring();
        if (!budget) budget = default_budget;
        if (!gen_reserve) gen_reserve = default_reserve;
        if (!payload_host_bytes) payload_host_bytes = default_host_bytes;
    }
};

} // namespace kvmem
