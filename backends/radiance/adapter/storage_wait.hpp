#pragma once
#include <exception>

namespace kvmem::radiance {
// Optional Radiance-only seam. The portable MemoryTransfer ABI is unchanged.
// Cancellation never releases a started worker's leases, buffers or native pages.
class StorageWaiter {
public:
    virtual ~StorageWaiter() = default;
    virtual bool awaiting_storage() const noexcept = 0;
    virtual bool cpu_io_pending() const noexcept = 0;
    virtual bool storage_quiescent() = 0;
    virtual void cancel_queued_storage() noexcept = 0;
    virtual std::exception_ptr storage_error() const noexcept { return {}; }
};
} // namespace kvmem::radiance
