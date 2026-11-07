#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <optional>

namespace kvmem {

// The execution strategy is independent of the legacy Query scorer. Layerwise
// always uses its reference current-layer-Q page-softmax selection/Pressure.
enum class RecallStrategy : uint8_t { Legacy = 0, Layerwise = 1 };

inline const char * recall_strategy_name(RecallStrategy value) {
    switch (value) {
        case RecallStrategy::Legacy: return "legacy";
        case RecallStrategy::Layerwise: return "layerwise";
    }
    throw std::invalid_argument("invalid KVMem recall strategy");
}

inline RecallStrategy parse_recall_strategy(const char * text) {
    const std::string value(text ? text : "");
    if (value=="legacy") return RecallStrategy::Legacy;
    if (value=="layerwise") return RecallStrategy::Layerwise;
    throw std::invalid_argument("KVMem recall strategy must be legacy or layerwise");
}

struct LayerwiseOptions {
    uint64_t activation_ram_bytes = 256u<<20;
    uint64_t activation_disk_bytes = 0;
    uint64_t staging_bytes = 16u<<20;
    uint64_t metadata_bytes = 256u<<20;
    // Native graph/tape descriptors are admitted independently of retained
    // source/index/checkpoint metadata, whose account is metadata_bytes.
    uint64_t activation_metadata_bytes = 128u<<20;
};

inline void validate_layerwise_options(const LayerwiseOptions & options) {
    if ((!options.activation_ram_bytes && !options.activation_disk_bytes) || !options.staging_bytes ||
        !options.metadata_bytes || !options.activation_metadata_bytes ||
        options.activation_ram_bytes==UINT64_MAX || options.activation_disk_bytes==UINT64_MAX ||
        options.staging_bytes==UINT64_MAX || options.metadata_bytes==UINT64_MAX || options.activation_metadata_bytes==UINT64_MAX)
        throw std::invalid_argument("layerwise recall requires finite activation, staging and metadata limits");
}

// Resolve once at the startup historical budget K=C-G. None uses the pinned
// native percentage/clamp allocation; explicit zero is literal. Request-final
// B changes do not resolve the percentages again.
struct LayerwiseKeepBands {
    uint32_t sink_blocks = 0;
    uint32_t recent_blocks = 0;
};
LayerwiseKeepBands resolve_layerwise_keep_bands(uint32_t block_tokens,uint32_t startup_tokens,
    std::optional<uint32_t> sink_tokens = {},std::optional<uint32_t> recent_tokens = {});

// Selector precondition only. Owners first apply the reference H>=B gate;
// H<B executes its accepted view without constructing a Query selector or
// applying this constraint. Pressure reserves sink but has no recent band.
void validate_layerwise_selection_bands(LayerwiseKeepBands,uint32_t block_tokens,
    uint64_t selected_tokens,bool pressure = false);

} // namespace kvmem
