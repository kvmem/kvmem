#pragma once
#include "kvmem/native_kv_payload.hpp"
#include "shared_payload_accounting.hpp"
#include "kvmem/host_kv_storage.hpp"
#include <cstring>
#include <map>
#include <memory>

namespace kvmem::radiance {
// Native format identity stays in the adapter catalog. Shared storage owns only
// sealed opaque bytes; a source survives catalog deletion without pinning RAM.
// Catalog operations require the engine boundary. CPU workers may read sources.
class TieredArchive {
    struct Entry { BlockDescriptor block; HostKvRecord record; };
    BackendIdentity backend_;
    SessionIdentity session_;
    PayloadLayout layout_;
    uint64_t capacity_, bytes_=0;
    std::shared_ptr<HostKvStorage> storage_;
    std::map<uint64_t, Entry> entries_;
    [[noreturn]] static void fail(MemoryErrorCode code,const char* text) { throw MemoryError(code,text); }
    const Entry& find(const BlockDescriptor& expected) const {
        const auto it=entries_.find(expected.id);
        if(it==entries_.end())fail(MemoryErrorCode::Stale,"native archive record is missing");
        const auto& actual=it->second.block;
        if(!(expected.session==actual.session) || expected.original_start!=actual.original_start ||
            expected.valid_tokens!=actual.valid_tokens || expected.content_version!=actual.content_version)
            fail(MemoryErrorCode::Stale,"native archive identity/version/frontier differs");
        return it->second;
    }
public:
    class Prepared {
        friend class TieredArchive;
        BackendIdentity backend;
        PayloadLayout layout;
        BlockDescriptor block;
        HostKvRecord record;
    public:
        Prepared(Prepared&&)=default;
        Prepared& operator=(Prepared&&)=default;
    private:
        Prepared()=default;
    };
    // CPU-worker operation: no catalog/GPU access and no publication. Failed
    // writes destroy this private record; existing sources remain authoritative.
    static Prepared stage(NativeKvPayload payload,const std::shared_ptr<HostKvStorage>& storage,
                          HostKvRecord previous={}) {
        if(!storage)fail(MemoryErrorCode::InvalidPlan,"missing native payload storage");
        Prepared next;
        next.backend=payload.backend();next.layout=payload.layout();next.block=payload.block();
        if(previous) {
            if(previous.size()!=payload.bytes().size())fail(MemoryErrorCode::Stale,"previous archive record size differs");
            std::vector<uint8_t> bytes(previous.size());previous.read(0,bytes.data(),bytes.size());
            if(bytes!=payload.bytes())fail(MemoryErrorCode::Stale,"changed bytes under unchanged native version");
            next.record=std::move(previous);
        } else next.record=storage->allocate(payload.bytes().size(),payload.bytes().data());
        next.record.seal();return next;
    }
    void publish(Prepared next) {
        if(!(next.backend==backend_) || !(next.layout==layout_) || !(next.block.session==session_))
            fail(MemoryErrorCode::InvalidPlan,"foreign prepared native archive payload");
        auto it=entries_.find(next.block.id);
        if(it!=entries_.end()) {
            const auto& old=it->second.block;
            if(next.block.original_start!=old.original_start || next.block.content_version<old.content_version)
                fail(MemoryErrorCode::Stale,"prepared archive replacement is not a newer version");
            if(next.block.content_version==old.content_version) {
                if(next.block.valid_tokens!=old.valid_tokens ||
                    next.record.storage_identity()!=it->second.record.storage_identity())
                    fail(MemoryErrorCode::Stale,"unverified equal-version archive replacement");
                return; // worker verified bytes against this exact immutable record
            }
            it->second=Entry{next.block,std::move(next.record)};
        } else {
            if(layout_.record_bytes>capacity_-bytes_)
                fail(MemoryErrorCode::BudgetExceeded,"native archive logical capacity exhausted");
            entries_.emplace(next.block.id,Entry{next.block,std::move(next.record)});
            bytes_+=layout_.record_bytes;
        }
    }
    std::shared_ptr<HostKvStorage> storage() const noexcept { return storage_; }
    // Native-proven capture in this exact layout: no copy or extra charge.
    void import(const BlockDescriptor& block,const HostKvRecord& record) {
        if(record.size()!=layout_.record_bytes)
            fail(MemoryErrorCode::InvalidPlan,"captured archive layout size differs");
        Prepared next;next.backend=backend_;next.layout=layout_;next.block=block;
        next.record=record.fork();publish(std::move(next));
    }
    TieredArchive fork(SessionIdentity branch,const MemorySnapshot& snapshot) const {
        if(branch==session_)fail(MemoryErrorCode::InvalidPlan,"archive branch needs a distinct session identity");
        TieredArchive next(backend_,branch,layout_,capacity_,storage_);
        for(const auto& item:snapshot.blocks)if(item.host_version) {
            if(item.host_version!=item.block.content_version)
                fail(MemoryErrorCode::Stale,"retained archive version is not current");
            const auto& source=find(item.block);
            auto block=item.block;block.session=branch;
            next.entries_.emplace(block.id,Entry{block,source.record.fork()});next.bytes_+=layout_.record_bytes;
        }
        return next;
    }
    HostKvRecord previous(const BlockDescriptor& block) const {
        const auto it=entries_.find(block.id);
        if(it==entries_.end())return {};
        const auto& old=it->second.block;
        if(!(block.session==old.session) || block.original_start!=old.original_start || block.content_version<old.content_version)
            fail(MemoryErrorCode::Stale,"stale native archive staging request");
        if(block.content_version==old.content_version) {
            if(block.valid_tokens!=old.valid_tokens)fail(MemoryErrorCode::Stale,"stale native archive staging frontier");
            return it->second.record.fork();
        }
        return {};
    }
    TieredArchive(BackendIdentity backend,SessionIdentity session,PayloadLayout layout,uint64_t capacity,
                  std::shared_ptr<HostKvStorage> storage={})
        : backend_(std::move(backend)),session_(session),layout_(std::move(layout)),capacity_(capacity),
          storage_(storage ? std::move(storage) : std::make_shared<HostKvStorage>(capacity)) {
        layout_.validate();
        if(backend_.backend.empty() || backend_.revision.empty() || backend_.model.empty() ||
            backend_.state_schema.empty() || !session.id || !session.incarnation)
            fail(MemoryErrorCode::InvalidPlan,"native archive requires complete backend/session identity");
    }
    void put(NativeKvPayload payload) {
        if(!(payload.backend()==backend_) || !(payload.layout()==layout_) || !(payload.block().session==session_))
            fail(MemoryErrorCode::InvalidPlan,"foreign native archive payload");
        const auto& block=payload.block();
        auto it=entries_.find(block.id);
        if(it!=entries_.end()) {
            const auto& old=it->second.block;
            if(block.original_start!=old.original_start || block.content_version<old.content_version)
                fail(MemoryErrorCode::Stale,"stale native archive replacement");
            if(block.content_version==old.content_version) {
                if(block.valid_tokens!=old.valid_tokens)
                    fail(MemoryErrorCode::Stale,"changed native frontier under unchanged version");
                // Equal versions must also preserve all scale planes and padding.
                std::vector<uint8_t> previous(payload.bytes().size());
                it->second.record.read(0,previous.data(),previous.size());
                if(previous!=payload.bytes())fail(MemoryErrorCode::Stale,"changed bytes under unchanged native version");
                return;
            }
        } else if(layout_.record_bytes>capacity_-bytes_)
            fail(MemoryErrorCode::BudgetExceeded,"native archive logical capacity exhausted");
        // Allocate/write/seal privately. Failed allocation or SSD I/O cannot
        // replace a still-valid catalog record. Its old source remains readable.
        publish(stage(std::move(payload),storage_));
    }
    HostKvSource source(const BlockDescriptor& expected) const { return find(expected).record.source(); }
    void read(const BlockDescriptor& expected,void* destination,size_t bytes) const {
        if(bytes!=layout_.record_bytes)fail(MemoryErrorCode::InvalidPlan,"native archive read size differs");
        find(expected).record.read(0,destination,bytes);
    }
    void erase(uint64_t id) { if(entries_.erase(id))bytes_-=layout_.record_bytes; }
    uint64_t bytes() const noexcept { return bytes_; }
    void account(SharedPayloadAccounting& out) const {
        for(const auto& [id,entry]:entries_)out.add(entry.record.storage_identity(),entry.record.size());
    }
    // Only identical sealed allocations are aliases. Equal logical block IDs
    // or equal byte counts do not prove shared storage after a branch changes.
    uint64_t shared_bytes(const TieredArchive& other) const noexcept {
        uint64_t shared = 0;
        for (const auto& [id, entry] : entries_) {
            const auto it = other.entries_.find(id);
            if (it != other.entries_.end() && entry.record.storage_identity() == it->second.record.storage_identity())
                shared += entry.record.size();
        }
        return shared;
    }
    const PayloadLayout& layout() const noexcept { return layout_; }
};
} // namespace kvmem::radiance
