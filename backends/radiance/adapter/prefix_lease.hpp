#pragma once
#include "state_checkpoint.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace kvmem::radiance {
// An immutable native execution-view lease for serial session continuation.
// It retains physical pages (including a shared partial tail, protected by COW)
// and a matching recurrent snapshot. The portable archive/mean-K/query identity
// must be retained alongside it by the session controller. This is NOT an
// alternative cache policy and does not by itself report a scheduler cache hit.
// Each cache entry owns its lease; ordinary engine prefix caching is disabled.
// Capture, restore and destruction all require the drained scheduler boundary.
class PrefixLease {
    std::shared_ptr<void> lifetime_;
    rad::KVManager* owner_;
    std::vector<rad::KVManager::FullTable> tables_;
    std::vector<std::pair<int32_t, int32_t>> refs_;
    std::vector<int32_t> tokens_;
    StateCheckpoint state_;
    int64_t visible_ = -1;
    std::vector<uint8_t> host_;
    std::unique_ptr<TieredBuffer> parked_host_;
    std::shared_ptr<StorageIoQueue> io_;
    std::shared_ptr<const int> payload_identity_=std::make_shared<const int>(0);
    struct ForkTag {};
    PrefixLease(const PrefixLease& source,ForkTag)
        : lifetime_(source.lifetime_),owner_(source.owner_),tables_(source.tables_),tokens_(source.tokens_),
          state_(source.state_.fork()),visible_(source.visible_),host_(source.host_),io_(source.io_),payload_identity_(source.payload_identity_) {
        if(!source.refs_.empty() || !source.host_bytes())throw std::runtime_error("fork requires a parked prefix");
        if(source.parked_host_)parked_host_=source.parked_host_->fork();
    }
    struct Copies { std::vector<GroupPool> pools; std::vector<CopySpan> spans; uint64_t bytes = 0; };
    Copies copies(const std::vector<rad::KVManager::FullTable>& tables) const {
        Copies out;
        for (const auto& t : tables) {
            const auto* p = owner_->plan(t.group);
            if (!p || p->kind != RAD_KV_FULL || !p->base || p->pool_bytes <= 0 || p->layer_bytes_per_block() <= 0)
                throw std::runtime_error("prefix copy geometry changed");
            out.pools.push_back({t.group, p->base, uint64_t(p->pool_bytes)});
            for (int32_t layer = 0; layer < p->n_layers; ++layer) for (auto page : t.blocks) {
                const uint64_t bytes = uint64_t(p->layer_bytes_per_block());
                if (page < 0 || page >= owner_->live_blocks(t.group) ||
                    owner_->block_refcount(t.group, page) < 1 || out.bytes > uint64_t(INT64_MAX) - bytes)
                    throw std::runtime_error("prefix copy page is not owned/backed");
                out.spans.push_back({t.group, uint64_t(layer) * p->layer_stride() + uint64_t(page) * bytes, out.bytes, bytes});
                out.bytes += bytes;
            }
        }
        return out;
    }
    void release() noexcept {
        for (const auto& [group, block] : refs_) owner_->release_block(group, block);
        refs_.clear();
    }
public:
    void account(SharedPayloadAccounting& out) const {
        if(parked_host_)parked_host_->account(out);state_.account(out);
    }
    class Payload {
        friend class PrefixLease;
        std::shared_ptr<const int> identity_;
    public:
        std::vector<uint8_t> kv,state;
    };
    // CPU-worker operation on an immutable parked fork. No device calls.
    Payload load_cpu() const {
        if(!refs_.empty() || !host_bytes())throw std::runtime_error("CPU prefix load requires parked pages");
        Payload out;out.identity_=payload_identity_;
        out.kv=parked_host_ ? parked_host_->read() : host_;out.state=state_.load_cpu();return out;
    }
    PrefixLease(std::shared_ptr<void> lifetime, rad::KVManager& kv, uint64_t source,
                std::vector<int32_t> evaluated_prefix, int device, RadStream stream,
                const AcceptedState& accepted = {})
        : lifetime_(std::move(lifetime)), owner_(&kv), tokens_(std::move(evaluated_prefix)) {
        if (!lifetime_ || tokens_.empty() || kv.seq_tokens(source) != int64_t(tokens_.size()))
            throw std::runtime_error("prefix lease requires the exact evaluated frontier");
        size_t count = 0;
        for (const auto& p : kv.plans()) {
            if (p.kind < 0 || p.stateful() || p.kind == RAD_KV_WINDOW) continue;
            if (p.kind != RAD_KV_FULL || p.zero) throw std::runtime_error("unsupported prefix view geometry");
            const auto visible = int64_t(tokens_.size()) - kv.execution_offset(source, p.index);
            if (visible < 0 || (visible_ >= 0 && visible != visible_))
                throw std::runtime_error("prefix view frontiers differ across groups");
            visible_ = visible;
            tables_.push_back({p.index, kv.block_table(source, p.index)});
            for (auto page : tables_.back().blocks)
                if (kv.block_refcount(p.index, page) < 1)
                    throw std::runtime_error("prefix source does not own its pages");
            if (kv.reserve_release_space(p.index, tables_.back().blocks.size()) < 0)
                throw std::runtime_error("prefix release bookkeeping reservation failed");
            count += tables_.back().blocks.size();
        }
        if (tables_.empty()) throw std::runtime_error("prefix lease has no attention view");
        refs_.reserve(count);
        try {
            state_.save(kv, source, device, stream, accepted);
            for (const auto& t : tables_) for (const auto block : t.blocks) {
                if (kv.retain_block(t.group, block) < 0) throw std::runtime_error("prefix page retention failed");
                refs_.emplace_back(t.group, block);
            }
        } catch (...) { release(); throw; }
    }
    PrefixLease(const PrefixLease&) = delete;
    PrefixLease& operator=(const PrefixLease&) = delete;
    ~PrefixLease() { release(); }
    // Each branch restores its own device pages/state. Only sealed host/disk
    // bytes are shared, and their physical quota remains charged exactly once.
    std::unique_ptr<PrefixLease> fork() const { return std::unique_ptr<PrefixLease>(new PrefixLease(*this,ForkTag{})); }
    bool matches(const std::vector<int32_t>& prompt) const {
        // At least one unevaluated token: neither a state nor a KV lease restores
        // logits. Comparing the whole prefix prevents shared chat headers alone
        // from being mistaken for a conversation match.
        return prompt.size() > tokens_.size() && std::equal(tokens_.begin(), tokens_.end(), prompt.begin());
    }
    size_t frontier() const noexcept { return tokens_.size(); }
    size_t host_bytes() const noexcept { return parked_host_ ? parked_host_->size() : host_.size(); }
    size_t state_bytes() const noexcept { return state_.bytes(); }
    uint64_t page_bytes() const { return host_bytes() ? host_bytes() : copies(tables_).bytes; }
    // Only called at a drained boundary when the live selection needs leased
    // pages. No new page allocator; save opaque native bytes before releasing
    // refs. Copy failure leaves the original lease intact. The caller shares
    // its host budget with the archive; transient pinned staging is additional.
    void demote(int device, RadStream stream, uint64_t max_bytes) {
        if (host_bytes()) return;
        const auto batch = copies(tables_);
        if (!batch.bytes || batch.bytes > max_bytes)
            throw MemoryError(MemoryErrorCode::BudgetExceeded, "host allowance cannot hold retained prefix");
        PageTransfer copy(device, batch.bytes, stream);
        copy.spill(batch.spans, batch.pools, stream);
        if (!copy.drain()) throw std::runtime_error("prefix demotion copy failed");
        auto payload = copy.payload();
        host_.swap(payload);
        release();
    }
    void park(const std::shared_ptr<HostKvStorage>& storage,const std::shared_ptr<StorageIoQueue>& io) {
        if(!refs_.empty() || !host_bytes())throw std::runtime_error("prefix must be demoted before tiering");
        state_.park(storage,io);park_buffer(host_,parked_host_,storage,io);io_=io;
    }
    void restore(rad::KVManager& kv, uint64_t destination, const std::vector<int32_t>& prompt,
                 int device, RadStream stream,const Payload* prepared=nullptr) const {
        if (&kv != owner_ || !matches(prompt)) throw std::runtime_error("prefix identity mismatch");
        if(prepared && (prepared->identity_!=payload_identity_ || prepared->kv.size()!=host_bytes() || prepared->state.size()!=state_bytes()))
            throw std::runtime_error("prepared prefix payload size mismatch");
        const auto expected = kv.seq_tokens(destination);
        auto tables = tables_;
        std::vector<std::pair<int32_t, int32_t>> reserved;
        auto release_reserved = [&] {
            for (const auto& [group, page] : reserved) kv.release_block(group, page);
            reserved.clear();
        };
        bool published = false;
        try {
            if (host_bytes()) {
                // File reads finish before reserving/writing destination pages.
                const auto loaded=!prepared && parked_host_ ? read_buffer(*parked_host_,io_) : std::vector<uint8_t>{};
                const auto& payload=prepared ? prepared->kv : parked_host_ ? loaded : host_;
                size_t count = 0;
                for (const auto& t : tables) {
                    if (kv.reserve_release_space(t.group, t.blocks.size() + kv.block_table(destination, t.group).size()) < 0)
                        throw std::runtime_error("prefix restore release bookkeeping failed");
                    count += t.blocks.size();
                }
                reserved.reserve(count);
                for (auto& t : tables) for (auto& page : t.blocks) {
                    int32_t allocated = -1;
                    if (kv.reserve_block(t.group, &allocated) < 0)
                        throw MemoryError(MemoryErrorCode::BudgetExceeded, "prefix restore page allocation failed");
                    reserved.emplace_back(t.group, allocated); page = allocated;
                }
                const auto batch = copies(tables);
                if (batch.bytes != payload.size()) throw std::runtime_error("prefix restore payload geometry changed");
                PageTransfer copy(device, payload.size(), stream);
                try {
                    copy.restore(batch.spans, batch.pools, stream, payload);
                    if (!copy.drain()) throw std::runtime_error("prefix restore copy failed");
                } catch (...) {
                    if (!copy.drain()) reserved.clear(); // fatal runtime: never recycle in-flight IDs
                    throw;
                }
            }
            if (kv.restore_full_tables(destination, expected, int64_t(tokens_.size()), visible_, tables) < 0)
                throw std::runtime_error("prefix page publication failed");
            published = true;
            state_.restore_rebound(kv, destination, device, stream,prepared ? &prepared->state : nullptr);
            release_reserved(); // publication retained its own native references
        }
        catch (...) {
            // Never leave a half-restored destination eligible for inference.
            if (published) kv.free_sequence(destination);
            release_reserved();
            throw;
        }
    }
};
} // namespace kvmem::radiance
