#pragma once
#include <string>

namespace kvmem {
// Model-free CLI contract used by the local console before launching a worker.
inline std::string launch_capabilities(const char* backend, bool inkling2 = false, bool inkling2_fp8 = false,
                                       bool inkling2_dspark = false, bool inkling2_multi_session = false,
                                       bool inkling2_dspark_batch = false) {
    const auto query_limits = std::string(backend) == "ninfer"
        ? "\"legacy_query_replay_revision\":1,\"query_limits\":{\"query_max_tokens\":0,\"query_replay_max_tokens\":0}}"
        : "\"query_limits\":{\"query_max_tokens\":512,\"query_replay_max_tokens\":2048}}";
    const auto blocks = std::string(backend) == "polystrata"
        ? "\"block_tokens\":{\"default\":32,\"supported\":[32,64],\"startup_fixed\":true}," : "";
    std::string models;
    if (std::string(backend) == "polystrata") {
        models = "\"model_architectures\":[\"qwen35moe\",\"gemma4\"";
        if (inkling2) models += ",\"inkling2_loop\"";
        models += "],\"model_kv_formats\":{\"qwen35moe\":[\"f16\",\"q8_0\"],\"gemma4\":[\"f16\",\"int8\",\"int8k\"]";
        if (inkling2) models += inkling2_fp8 ? ",\"inkling2_loop\":[\"f16\",\"fp8\"]"
                                            : ",\"inkling2_loop\":[\"f16\"]";
        models += "},";
        if (inkling2 && inkling2_dspark) models += "\"inkling2_dspark\":true,";
        if (inkling2 && inkling2_multi_session) models += "\"inkling2_multi_session\":true,";
        if (inkling2 && inkling2_dspark_batch) models += "\"inkling2_dspark_batch\":true,";
    }
    return std::string("{\"schema_version\":1,\"backend\":\"") + backend +
        "\"," + blocks + models + "\"cache_admission_revision\":1,\"query_scoring_revision\":2,"
        "\"query_scorers\":[\"page-softmax\",\"meanq-cosine\"],"
        "\"score_aggregations\":[\"mean\",\"max\"],"
        "\"independent_graph_lrfu\":true,\"signed_lrfu_hits\":true,"
        + query_limits;
}
} // namespace kvmem
