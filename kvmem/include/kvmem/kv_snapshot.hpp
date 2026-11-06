#pragma once
#include "kvmem/host_kv_storage.hpp"
#include "kvmem/memory_contract.hpp"
#include <algorithm>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace kvmem {
// One native-proven immutable block origin. Copies of its shared_ptr preserve
// that proof; an equal token/position/version tuple is NOT a substitute for it.
// The adapter alone creates origins, after draining accepted device producers.
// A worker may complete the origin once; no device calls occur under this lock.
class SealedKvBlock {
    const size_t bytes_;
    mutable std::mutex mutex_;
    HostKvRecord record_;
public:
    explicit SealedKvBlock(size_t bytes) : bytes_(bytes) {
        if(!bytes)throw MemoryError(MemoryErrorCode::InvalidPlan,"empty snapshot block");
    }
    explicit SealedKvBlock(const HostKvRecord& record) : SealedKvBlock(record.size()) {
        record_=record.fork();
    }
    size_t size() const noexcept {return bytes_;}
    HostKvRecord record() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return record_ ? record_.fork() : HostKvRecord{};
    }
    bool ready() const {std::lock_guard<std::mutex> lock(mutex_);return bool(record_);}
    // No storage allocation/I/O while holding the publication mutex. Two
    // producers of this SAME proven origin may race; only one allocation lives.
    bool seal(HostKvStorage& storage,const void* bytes,size_t size) {
        if(!bytes || size!=bytes_)throw MemoryError(MemoryErrorCode::InvalidPlan,"snapshot block size mismatch");
        if(ready())return false;
        auto next=storage.allocate(size,bytes);next.seal();
        std::lock_guard<std::mutex> lock(mutex_);
        if(record_)return false;
        record_=std::move(next);return true;
    }
};

// Portable layout of an immutable native execution image. Native adapters
// supply exact span mappings; the public layer owns the shared RAM/SSD bytes.
// Blocks can use different sizes/layouts and need not match GPU page geometry.
class KvSnapshot {
public:
    struct Span {size_t block=0, source=0, destination=0, bytes=0;};
private:
    std::vector<std::shared_ptr<SealedKvBlock>> blocks_;
    std::vector<Span> spans_;
    std::vector<std::vector<Span>> by_block_;
    size_t bytes_=0;
public:
    KvSnapshot(std::vector<std::shared_ptr<SealedKvBlock>> blocks,std::vector<Span> spans,size_t bytes)
        :blocks_(std::move(blocks)),spans_(std::move(spans)),bytes_(bytes) {
        if(!bytes_ || blocks_.empty())throw MemoryError(MemoryErrorCode::InvalidPlan,"empty snapshot image");
        by_block_.resize(blocks_.size());
        std::unordered_set<const SealedKvBlock*> origins;
        for(const auto& block:blocks_) {
            if(!block)throw MemoryError(MemoryErrorCode::InvalidPlan,"missing snapshot origin");
            if(!origins.insert(block.get()).second)
                throw MemoryError(MemoryErrorCode::InvalidPlan,"duplicate snapshot origin");
        }
        std::sort(spans_.begin(),spans_.end(),[](const Span& a,const Span& b){return a.destination<b.destination;});
        size_t offset=0;
        for(const auto& span:spans_) {
            if(span.block>=blocks_.size() || !span.bytes || span.destination!=offset || offset>bytes_ ||
               span.bytes>bytes_-offset || span.source>blocks_[span.block]->size() ||
               span.bytes>blocks_[span.block]->size()-span.source)
                throw MemoryError(MemoryErrorCode::InvalidPlan,"invalid snapshot mapping");
            offset+=span.bytes;by_block_[span.block].push_back(span);
        }
        if(offset!=bytes_)throw MemoryError(MemoryErrorCode::InvalidPlan,"incomplete snapshot image");
        // Full records are referenced exactly once, enabling exact alias credit.
        for(size_t i=0;i<blocks_.size();++i) {
            auto& ranges=by_block_[i];
            std::sort(ranges.begin(),ranges.end(),[](const Span&a,const Span&b){return a.source<b.source;});
            size_t end=0;
            for(const auto& span:ranges) {
                if(span.source!=end)throw MemoryError(MemoryErrorCode::InvalidPlan,"overlapping snapshot block ranges");
                end+=span.bytes;
            }
            if(end!=blocks_[i]->size())throw MemoryError(MemoryErrorCode::InvalidPlan,"incomplete snapshot block");
        }
    }
    // Owned object/vector capacities; allocator/control-block overhead remains
    // process overhead, just as for the existing host-storage metadata.
    size_t manifest_bytes() const noexcept {
        size_t n=sizeof(*this)+blocks_.capacity()*sizeof(blocks_[0])+spans_.capacity()*sizeof(Span)+
            by_block_.capacity()*sizeof(by_block_[0]);
        for(const auto& spans:by_block_)n+=spans.capacity()*sizeof(Span);
        return n;
    }
    size_t metadata_bytes() const noexcept {return manifest_bytes()+blocks_.size()*sizeof(SealedKvBlock);}
    template<class Visitor> void account_metadata(Visitor&& visitor) const {
        visitor(this,manifest_bytes());
        for(const auto& block:blocks_)visitor(block.get(),sizeof(SealedKvBlock));
    }
    size_t size() const noexcept {return bytes_;}
    const auto& blocks() const noexcept {return blocks_;}
    const auto& spans() const noexcept {return spans_;}
    bool ready() const {return std::all_of(blocks_.begin(),blocks_.end(),[](const auto& b){return b->ready();});}
    size_t missing_bytes() const {
        size_t n=0;for(const auto& b:blocks_)if(!b->ready())n+=b->size();return n;
    }
    template<class Visitor> void account(Visitor&& visitor) const {
        for(const auto& block:blocks_)if(auto record=block->record())visitor(record.storage_identity(),record.size());
    }
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> out(bytes_);
        for(size_t i=0;i<blocks_.size();++i) {
            auto record=blocks_[i]->record();
            if(!record)throw MemoryError(MemoryErrorCode::Pending,"snapshot block capture is incomplete");
            const auto lease=record.lease();
            for(const auto& span:by_block_[i])
                lease.read(span.source,out.data()+span.destination,span.bytes);
        }
        return out;
    }
};
} // namespace kvmem
