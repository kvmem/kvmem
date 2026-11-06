#pragma once
#include "native_pages.hpp"
#include "kvmem/kv_snapshot.hpp"

namespace kvmem::radiance {
// Produced only by NativeMemory at a checked accepted boundary. Spans describe
// complete immutable records, in the exact NativePages archive representation.
struct PrefixCaptureBlock {
    std::shared_ptr<SealedKvBlock> origin;
    std::vector<CopySpan> spans;
};
// Ephemeral capture plan: the caller holds the drained scheduler boundary
// until PrefixLease has retained its pages. The engine owner prevents ABA reuse
// of the manager address while any plan remains alive.
struct PrefixCapture {
    std::shared_ptr<void> engine_lifetime;
    MemoryStamp stamp;
    const void* manager=nullptr;
    uint64_t sequence=0;
    std::vector<PrefixCaptureBlock> blocks;
};
} // namespace kvmem::radiance
