#pragma once

// Public configuration and telemetry only. Native engines own proposal,
// verification, sampling, KV and recurrent-state execution.
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <utility>

namespace kvmem {
enum class SpeculativeAlgorithm { None, Mtp, DFlash, DFlash2, DSpark };
inline const char * speculative_name(SpeculativeAlgorithm value) noexcept {
    switch (value) {
        case SpeculativeAlgorithm::Mtp: return "mtp";
        case SpeculativeAlgorithm::DFlash: return "dflash";
        case SpeculativeAlgorithm::DFlash2: return "dflash2";
        case SpeculativeAlgorithm::DSpark: return "dspark";
        default: return "none";
    }
}
struct SpeculativeConfig {
    SpeculativeAlgorithm algorithm = SpeculativeAlgorithm::None;
    uint32_t draft_tokens = 0; // effective maximum, after native model limits
    bool adaptive = false;
    uint32_t ngram_draft_tokens = 0;
};
struct SpeculativeCapability {
    SpeculativeAlgorithm algorithm = SpeculativeAlgorithm::None;
    uint32_t max_draft_tokens = 0;
    bool adaptive = false;
    bool ngram = false;
    uint32_t max_concurrency = 1;
    uint32_t max_devices = 1;
    std::string draft_asset; // embedded, optional_gguf, required_gguf
};
struct SpeculativeStatistics {
    uint64_t requests = 0;
    uint64_t rounds = 0;
    uint64_t drafted_tokens = 0; // native proposal candidates, excluding internal replay proposals
    uint64_t accepted_tokens = 0; // accepted candidates; excludes correction/bonus tokens
    uint64_t generated_tokens = 0; // committed output, including correction/bonus tokens
    uint64_t fallback_steps = 0;
    void add(const SpeculativeStatistics & other) noexcept {
        requests += other.requests; rounds += other.rounds;
        drafted_tokens += other.drafted_tokens; accepted_tokens += other.accepted_tokens;
        generated_tokens += other.generated_tokens; fallback_steps += other.fallback_steps;
    }
};
class SpeculativeTelemetry {
public:
    void add(const SpeculativeStatistics & value) {
        std::lock_guard<std::mutex> guard(mutex_); value_.add(value);
    }
    SpeculativeStatistics snapshot() const {
        std::lock_guard<std::mutex> guard(mutex_); return value_;
    }
private:
    mutable std::mutex mutex_;
    SpeculativeStatistics value_;
};

// Json is supplied by the host. The public library has no JSON dependency.
template<class Json> Json speculative_json(const SpeculativeConfig & config,
        const std::vector<SpeculativeCapability> & capabilities,
        const SpeculativeStatistics & stats = {}) {
    Json result = Json::object();
    result["schema_version"] = 1;
    result["enabled"] = config.algorithm != SpeculativeAlgorithm::None;
    result["algorithm"] = speculative_name(config.algorithm);
    result["draft_tokens"] = config.draft_tokens;
    result["adaptive"] = config.adaptive;
    result["ngram_draft_tokens"] = config.ngram_draft_tokens;
    result["capabilities"] = Json::array();
    for (const auto & cap : capabilities) {
        Json item = Json::object();
        item["algorithm"] = speculative_name(cap.algorithm);
        item["max_draft_tokens"] = cap.max_draft_tokens;
        item["adaptive"] = cap.adaptive; item["ngram"] = cap.ngram;
        item["max_concurrency"] = cap.max_concurrency;
        item["max_devices"] = cap.max_devices; item["draft_asset"] = cap.draft_asset;
        result["capabilities"].push_back(std::move(item));
    }
    Json counters = Json::object();
    counters["scope"] = "completed_requests";
    counters["requests"] = stats.requests; counters["rounds"] = stats.rounds;
    counters["drafted_tokens"] = stats.drafted_tokens;
    counters["accepted_tokens"] = stats.accepted_tokens;
    counters["generated_tokens"] = stats.generated_tokens;
    counters["fallback_steps"] = stats.fallback_steps;
    counters["acceptance_rate"] = stats.drafted_tokens ? Json(double(stats.accepted_tokens) / double(stats.drafted_tokens)) : Json(nullptr);
    result["statistics"] = std::move(counters);
    return result;
}
} // namespace kvmem
