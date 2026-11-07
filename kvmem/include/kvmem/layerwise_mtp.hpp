#pragma once

#include "layerwise_options.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kvmem {

// User-approved alternatives. Backends resolve an omitted policy before
// validation; NInfer defaults to FirstLayer. Both leave target fresh-Q selection
// unchanged and restore draft-owned K/V for the selected original token IDs.
// UpstreamGlobal requires the pinned upstream private probe/global reduction;
// averaging the real forward's fresh layer scores is a different algorithm.
enum class LayerwiseMtpHistory : uint8_t {
    Unspecified = 0,
    UpstreamGlobal = 1,
    FirstLayer = 2,
};

inline const char * layerwise_mtp_history_name(LayerwiseMtpHistory value) {
    switch (value) {
        case LayerwiseMtpHistory::Unspecified: return "unspecified";
        case LayerwiseMtpHistory::UpstreamGlobal: return "upstream-global";
        case LayerwiseMtpHistory::FirstLayer: return "first-layer";
    }
    throw std::invalid_argument("invalid layerwise MTP history policy");
}

inline LayerwiseMtpHistory parse_layerwise_mtp_history(const char * text) {
    const std::string value(text ? text : "");
    if (value=="upstream-global") return LayerwiseMtpHistory::UpstreamGlobal;
    if (value=="first-layer") return LayerwiseMtpHistory::FirstLayer;
    throw std::invalid_argument("layerwise MTP history must be upstream-global or first-layer");
}

inline void validate_layerwise_mtp_history(RecallStrategy strategy,bool mtp_enabled,LayerwiseMtpHistory value) {
    (void)layerwise_mtp_history_name(value);
    if (value!=LayerwiseMtpHistory::Unspecified && (strategy!=RecallStrategy::Layerwise || !mtp_enabled))
        throw std::invalid_argument("--kvmem-layerwise-mtp-history requires layerwise recall and native MTP");
    if (strategy==RecallStrategy::Layerwise && mtp_enabled && value==LayerwiseMtpHistory::Unspecified)
        throw std::invalid_argument("layerwise MTP policy was not resolved; select --kvmem-layerwise-mtp-history upstream-global|first-layer or use a backend-declared default");
}

} // namespace kvmem
