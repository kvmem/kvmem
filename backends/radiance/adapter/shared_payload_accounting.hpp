#pragma once
#include <cstdint>
#include <map>
#include <stdexcept>

namespace kvmem::radiance {
// Count only proven aliases of sealed allocations. Metadata, mutable buffers
// and independently captured equal bytes receive no sharing credit.
class SharedPayloadAccounting {
    std::map<const void*, uint64_t> records_;
    uint64_t credit_=0;
public:
    void add(const void* identity,uint64_t bytes) {
        if(!identity || !bytes)throw std::runtime_error("invalid shared payload identity");
        const auto [it,inserted]=records_.emplace(identity,bytes);
        if(!inserted) {
            if(it->second!=bytes || bytes>UINT64_MAX-credit_)
                throw std::runtime_error("shared payload accounting mismatch");
            credit_+=bytes;
        }
    }
    uint64_t credit() const noexcept {return credit_;}
};
} // namespace kvmem::radiance
