#pragma once
#include "page_transfer.hpp"
#include "mem/kv.h"
#include "state_materialization.hpp"
#include "tiered_buffer.hpp"
#include <stdexcept>

namespace kvmem::radiance {
// One immutable, adapter-owned recurrent/conv snapshot. It neither borrows nor
// replaces the engine's prefix-cache checkpoint arena. Caller holds the drained
// scheduler boundary and the engine lifetime throughout save/restore.
class StateCheckpoint {
    struct Slot { int32_t group, slot; void* base; int64_t bytes; };
    std::vector<Slot> slots_;
    std::vector<GroupPool> pools_;
    std::vector<CopySpan> spans_;
    std::vector<uint8_t> bytes_;
    std::unique_ptr<TieredBuffer> parked_;
    std::shared_ptr<StorageIoQueue> io_;
    uint64_t sequence_ = 0;
    const rad::KVManager* owner_ = nullptr;
    bool valid_ = false;
public:
    StateCheckpoint fork() const {
        if(!valid_)throw std::runtime_error("cannot fork an incomplete recurrent checkpoint");
        StateCheckpoint next;
        next.slots_=slots_;next.pools_=pools_;next.spans_=spans_;next.bytes_=bytes_;
        if(parked_)next.parked_=parked_->fork();
        next.io_=io_;next.sequence_=sequence_;next.owner_=owner_;next.valid_=true;
        return next;
    }
    void save(rad::KVManager& kv, uint64_t sequence, int device, RadStream stream,
              const AcceptedState& accepted = {}) {
        if (valid_ || !kv.has_sequence(sequence)) throw std::runtime_error("invalid checkpoint save");
        uint64_t offset = 0;
        for (const auto& p : kv.plans()) if (p.stateful()) {
            const int32_t slot = kv.state_slot(sequence, p.index);
            // Prefill checkpoints read copy0 only. Speculative replay factors
            // in the other copies are engine scratch, not committed history.
            if (p.state_copies < 1 || slot < 0 || !p.base || p.layer_bytes_per_state() <= 0)
                throw std::runtime_error("checkpoint needs a committed recurrent slot");
            slots_.push_back({p.index, slot, p.base, p.pool_bytes});
            pools_.push_back({p.index, p.base, uint64_t(p.pool_bytes)});
            for (int32_t layer = 0; layer < p.n_layers; ++layer) {
                auto* state = static_cast<char*>(kv.state_ptr(p.index, layer, slot));
                if (!state) throw std::runtime_error("checkpoint state address missing");
                const uint64_t count = uint64_t(p.layer_bytes_per_state());
                if (offset > UINT64_MAX - count) throw std::runtime_error("checkpoint size overflow");
                spans_.push_back({p.index, uint64_t(state - static_cast<char*>(p.base)), offset, count});
                offset += count;
            }
        }
        if (offset) {
            PageTransfer copy(device, offset, stream);
            copy.spill(spans_, pools_, stream);
            if (!copy.drain()) throw std::runtime_error("checkpoint save failed");
            bytes_ = copy.payload();
        }
        if (accepted.rows) {
            size_t at = 0;
            for (const auto& p : kv.plans()) if (p.stateful())
                for (int32_t layer = 0; layer < p.n_layers; ++layer) {
                    const auto count = size_t(p.layer_bytes_per_state());
                    materialize_state(kv, sequence, p, layer, device, stream, accepted,
                                      bytes_.data() + at, count);
                    at += count;
                }
        }
        sequence_ = sequence; owner_ = &kv; valid_ = true;
    }
    void restore(rad::KVManager& kv, uint64_t sequence, int device, RadStream stream) const {
        if (!valid_ || owner_ != &kv || sequence != sequence_ || !kv.has_sequence(sequence))
            throw std::runtime_error("checkpoint sequence mismatch");
        // Validate EVERY group before enqueuing the first write.
        for (const auto& saved : slots_) {
            const auto* p = kv.plan(saved.group);
            if (!p || p->base != saved.base || p->pool_bytes != saved.bytes ||
                p->state_copies < 1 || kv.state_slot(sequence, saved.group) != saved.slot)
                throw std::runtime_error("checkpoint state ownership changed");
        }
        if (bytes()) {
            const auto loaded=parked_ ? read_buffer(*parked_,io_) : std::vector<uint8_t>{};
            const auto& payload=parked_ ? loaded : bytes_;
            PageTransfer copy(device, payload.size(), stream);
            copy.restore(spans_, pools_, stream, payload);
            if (!copy.drain()) throw std::runtime_error("checkpoint restore failed");
        }
    }
    // Session-cache building block: restore into a newly owned sequence in the
    // SAME engine. Caller must prove token-prefix/checkpoint identity and drain
    // pending state clears before using it. This method neither selects a hit
    // nor advances the scheduler; unrelated requests cannot reuse it implicitly.
    std::vector<uint8_t> load_cpu() const {
        if(!valid_)throw std::runtime_error("cannot load an incomplete checkpoint");
        return parked_ ? parked_->read() : bytes_;
    }
    void restore_rebound(rad::KVManager& kv, uint64_t destination, int device, RadStream stream,
                         const std::vector<uint8_t>* prepared=nullptr) const {
        if (!valid_ || owner_ != &kv || !kv.has_sequence(destination))
            throw std::runtime_error("checkpoint destination ownership mismatch");
        std::vector<CopySpan> destination_spans;
        uint64_t offset = 0;
        for (const auto& saved : slots_) {
            const auto* p = kv.plan(saved.group);
            const int32_t slot = kv.state_slot(destination, saved.group);
            if (!p || p->base != saved.base || p->pool_bytes != saved.bytes ||
                p->state_copies < 1 || slot < 0 || p->layer_bytes_per_state() <= 0)
                throw std::runtime_error("checkpoint destination geometry mismatch");
            for (int32_t layer = 0; layer < p->n_layers; ++layer) {
                auto* state = static_cast<char*>(kv.state_ptr(p->index, layer, slot));
                const uint64_t count = uint64_t(p->layer_bytes_per_state());
                if (!state || offset > bytes() || count > bytes() - offset)
                    throw std::runtime_error("checkpoint destination range mismatch");
                destination_spans.push_back({p->index, uint64_t(state - static_cast<char*>(p->base)), offset, count});
                offset += count;
            }
        }
        if (offset != bytes() || (prepared && prepared->size()!=offset)) throw std::runtime_error("checkpoint destination size mismatch");
        // All groups validated before the first write, just as strict restore.
        if (offset) {
            const auto loaded=!prepared && parked_ ? read_buffer(*parked_,io_) : std::vector<uint8_t>{};
            const auto& payload=prepared ? *prepared : parked_ ? loaded : bytes_;
            PageTransfer copy(device, offset, stream);
            copy.restore(destination_spans, pools_, stream, payload);
            if (!copy.drain()) throw std::runtime_error("checkpoint rebound restore failed");
        }
    }
    void park(const std::shared_ptr<HostKvStorage>& storage,const std::shared_ptr<StorageIoQueue>& io) {
        if(!valid_)throw std::runtime_error("cannot park an incomplete recurrent checkpoint");
        park_buffer(bytes_,parked_,storage,io);io_=io;
    }
    void account(SharedPayloadAccounting& out) const {if(parked_)parked_->account(out);}
    size_t bytes() const noexcept { return parked_ ? parked_->size() : bytes_.size(); }
};
} // namespace kvmem::radiance
