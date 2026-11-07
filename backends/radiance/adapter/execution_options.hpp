#pragma once
#include <cstdlib>
#include <cstring>

namespace kvmem::radiance {
// Process-wide ablation switches. Read once, before any request can run.
inline bool early_statistics_collection() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("KVMEM_RAD_EARLY_COLLECT");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
inline bool borrow_spill_capture() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("KVMEM_RAD_BORROW_SPILL");
        return !value || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}
} // namespace kvmem::radiance
