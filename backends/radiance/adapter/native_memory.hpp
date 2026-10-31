#pragma once
#include "kvmem/memory_contract.hpp"
#include "rad_device.h"
#include <memory>

namespace rad { class KVManager; struct KVGroupInfo; }
namespace kvmem::radiance {

// First engine boundary: one rank, FULL attention. Incomplete accepted tail
// blocks stay resident and unarchived until they close.
// Recurrent/conv slots and native drafter windows stay engine-owned while full KV moves;
// this is state retention, not a checkpoint/replay implementation.
// Caller holds the scheduler boundary for the WHOLE MemorySession transaction.
// engine_lifetime must own the KV manager, pools and compute stream. No decode,
// preemption or accepted-frontier update may race a transaction.
struct MemoryBinding {
    std::shared_ptr<void> engine_lifetime;
    rad::KVManager* manager = nullptr;
    RadStream compute = nullptr;
    int device = -1;
    uint32_t ranks = 1;
};

// Actual native page backend, not a policy implementation. Initial catalog must
// describe a canonical, evaluated and accepted sequence.
// Query-statistic capture and incremental/pending tails are a separate bridge;
// this backend does not advertise them before that bridge has been qualified.
class NativeMemory final : public MemoryBackend {
public:
    struct ReuseStats {
        uint64_t transactions = 0, recycled_pages = 0, journal_bytes_peak = 0, rollbacks = 0;
    };
    NativeMemory(MemoryBinding, const std::vector<rad::KVGroupInfo>&,
                 BackendIdentity, MemoryStamp, const std::vector<BlockDescriptor>&,
                 uint32_t block_tokens, uint64_t max_position, uint64_t host_capacity);
    ~NativeMemory() override;
    MemoryDescriptor descriptor() const override;
    MemorySnapshot snapshot() const override;
    MemoryStamp stamp() const noexcept override;
    ReuseStats reuse_stats() const noexcept;
    std::unique_ptr<MemoryTransfer> prepare(const WorkingSetPlan&) override;
    // Engine has evaluated/accepted a contiguous append. Existing compact pages
    // must be an unchanged prefix, and no transfer may be pending. Descriptors
    // cover the formerly partial tail (if any, with a newer content version)
    // followed by the new blocks, ending at the exact accepted frontier.
    // Failure invalidates the now-divergent snapshot; never reuse its old stamp.
    void accept_append(uint64_t frontier, const std::vector<BlockDescriptor>& added);
    // Drop a complete-block suffix after draining the probe. Preserves archived
    // earlier history and surviving native pages. State checkpoint is separate.
    void rewind(uint64_t frontier);
    // Save metadata matching a separately owned PrefixLease. Later bounded
    // prefill may change the live view; no native page/host payload is duplicated.
    void retain_view();
    // Reattach an EXACT retained view after its prefix/state lease was restored
    // into a fresh engine sequence. Logical session/archive identity is unchanged.
    // Caller proves token/query identity and restores recurrent state first.
    void rebind_sequence(uint64_t sequence, bool relocated_prefix = false);
    uint64_t archive_bytes() const noexcept;
    bool needs_unleased_pages(const std::vector<uint64_t>& selected,
                              const std::vector<uint64_t>& mandatory, const MemoryBudget&) const;
    void invalidate() noexcept override;
private:
    struct State;
    class Transaction;
    std::shared_ptr<State> state_;
};
} // namespace kvmem::radiance
