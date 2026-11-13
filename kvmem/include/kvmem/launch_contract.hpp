#pragma once
#include <string>

namespace kvmem {
// Model-free CLI contract used by the local console before launching a worker.
inline std::string launch_capabilities(const char* backend) {
    const auto blocks = std::string(backend) == "polystrata"
        ? "\"block_tokens\":{\"default\":32,\"supported\":[32,64],\"startup_fixed\":true}," : "";
    const auto models = std::string(backend) == "polystrata"
        ? "\"model_architectures\":[\"qwen35moe\",\"gemma4\"],"
          "\"model_kv_formats\":{\"qwen35moe\":[\"f16\",\"q8_0\"],\"gemma4\":[\"f16\",\"int8\",\"int8k\"]}," : "";
    return std::string("{\"schema_version\":1,\"backend\":\"") + backend +
        "\"," + blocks + models + "\"cache_admission_revision\":1,\"query_scoring_revision\":2,"
        "\"query_scorers\":[\"page-softmax\",\"meanq-cosine\"],"
        "\"score_aggregations\":[\"mean\",\"max\"],"
        "\"independent_graph_lrfu\":true,\"signed_lrfu_hits\":true,"
        "\"query_limits\":{\"query_max_tokens\":512,\"query_replay_max_tokens\":2048}}";
}
} // namespace kvmem
