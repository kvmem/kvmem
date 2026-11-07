#pragma once
#include "native_pages.hpp"
#include "rad_device.h"
#include <memory>
#include <span>

namespace kvmem::radiance {
struct GroupPool {
    int32_t group;
    void* base;
    uint64_t bytes;
};

// Bounded transfer primitive; page ownership and view publication belong to the
// MemoryBackend transaction. The caller holds pools/pages and the compute stream
// alive until drain succeeds. Uses Radiance streams, NOT casted HIP streams.
class PageTransfer {
public:
    // A borrowed stream must belong to device and outlive this transfer. The
    // native transaction uses the engine's drained compute lane: no per-page
    // queue creation, and no ownership of the borrowed lane is transferred.
    PageTransfer(int device, uint64_t record_bytes, RadStream borrowed = nullptr);
    ~PageTransfer();
    PageTransfer(const PageTransfer&) = delete;
    PageTransfer& operator=(const PageTransfer&) = delete;
    void spill(const std::vector<CopySpan>&, const std::vector<GroupPool>&, RadStream compute);
    void restore(const std::vector<CopySpan>&, const std::vector<GroupPool>&,
                 RadStream compute, std::span<const uint8_t> payload);
    // A completed successful restore may reuse its allocation/events for the
    // next chunk (up to the original capacity). Spill handles stay single-use.
    TransferPoll poll();
    std::vector<uint8_t> payload(); // completed spill only
    // Immutable CPU view, after the completion event. Valid until this transfer
    // is destroyed. An asynchronous reader must finish before its owner releases
    // the transfer; the view itself performs no device calls on the I/O worker.
    std::span<const uint8_t> payload_view();
    // Quiesce ALL copies, even if event creation/record failed after a partial
    // enqueue. False indicates a fatal device error: invalidate the engine.
    bool drain() noexcept;
private:
    struct State;
    std::unique_ptr<State> state_;
    void begin(bool out, const std::vector<CopySpan>&, const std::vector<GroupPool>&,
               RadStream compute, std::span<const uint8_t> payload);
};
} // namespace kvmem::radiance
