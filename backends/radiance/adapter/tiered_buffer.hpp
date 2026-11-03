#pragma once
#include "kvmem/host_kv_storage.hpp"
#include "shared_payload_accounting.hpp"
#include "storage_work.hpp"
#include <algorithm>
#include <vector>

namespace kvmem::radiance {
// Immutable opaque prefix/state bytes, chunked so one retained recurrent state
// may exceed H. Chunks and native KV use the same storage account and disk file.
class TieredBuffer {
    std::vector<HostKvRecord> records_;
    size_t bytes_=0;
    TieredBuffer()=default;
public:
    TieredBuffer(const std::vector<uint8_t>& bytes,const std::shared_ptr<HostKvStorage>& storage) : bytes_(bytes.size()) {
        if(!storage || !storage->capacity_bytes())throw MemoryError(MemoryErrorCode::BudgetExceeded,"missing retained-state RAM capacity");
        const size_t chunk=size_t(std::min<uint64_t>(1u<<20,storage->capacity_bytes()));
        records_.reserve(bytes_/chunk+(bytes_%chunk!=0));
        for(size_t offset=0;offset<bytes_;offset+=chunk) {
            auto record=storage->allocate(std::min(chunk,bytes_-offset),bytes.data()+offset);
            record.seal();records_.push_back(std::move(record));
        }
    }
    size_t size() const noexcept { return bytes_; }
    void account(SharedPayloadAccounting& out) const {
        for(const auto& record:records_)out.add(record.storage_identity(),record.size());
    }
    std::unique_ptr<TieredBuffer> fork() const {
        auto next=std::unique_ptr<TieredBuffer>(new TieredBuffer);
        next->bytes_=bytes_;next->records_.reserve(records_.size());
        for(const auto& record:records_)next->records_.push_back(record.fork());
        return next;
    }
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> bytes(bytes_);size_t offset=0;
        for(const auto& record:records_) { record.read(0,bytes.data()+offset,record.size());offset+=record.size(); }
        return bytes;
    }
};
inline void park_buffer(std::vector<uint8_t>& bytes,std::unique_ptr<TieredBuffer>& parked,
                        const std::shared_ptr<HostKvStorage>& storage,const std::shared_ptr<StorageIoQueue>& io) {
    if(parked || bytes.empty())return;
    std::unique_ptr<TieredBuffer> next;
    complete_storage(io,bytes.size(),[&]{next=std::make_unique<TieredBuffer>(bytes,storage);});
    parked=std::move(next);std::vector<uint8_t>().swap(bytes);
}
inline std::vector<uint8_t> read_buffer(const TieredBuffer& parked,const std::shared_ptr<StorageIoQueue>& io) {
    std::vector<uint8_t> bytes;
    complete_storage(io,parked.size(),[&]{bytes=parked.read();});return bytes;
}
} // namespace kvmem::radiance
