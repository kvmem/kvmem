#pragma once
#include "prefix_lease.hpp"

namespace kvmem::radiance {
// A cache entry is usable only as a complete set of rank leases. Sealed CPU
// payloads remain independently owned by each rank and share the global quota.
class RankPrefix {
    struct Item {
        int device;
        RadStream stream;
        rad::KVManager* kv;
        std::unique_ptr<PrefixLease> lease;
    };
    std::vector<Item> ranks_;
    static uint64_t add(uint64_t a,uint64_t b) {
        if(b>UINT64_MAX-a)throw std::runtime_error("rank prefix size overflow");return a+b;
    }
public:
    void account(SharedPayloadAccounting& out) const {for(const auto& r:ranks_)r.lease->account(out);}
    struct Payload {std::vector<PrefixLease::Payload> ranks;};
    void add_rank(int device,RadStream stream,rad::KVManager& kv,std::unique_ptr<PrefixLease> lease) {
        if(!lease || (!ranks_.empty() && lease->frontier()!=frontier()))throw std::runtime_error("rank prefix frontiers differ");
        ranks_.push_back({device,stream,&kv,std::move(lease)});
    }
    std::unique_ptr<RankPrefix> fork() const {
        auto out=std::make_unique<RankPrefix>();
        for(const auto& r:ranks_)out->add_rank(r.device,r.stream,*r.kv,r.lease->fork());
        return out;
    }
    size_t frontier() const {if(ranks_.empty())throw std::runtime_error("empty rank prefix");return ranks_[0].lease->frontier();}
    uint64_t host_bytes() const {uint64_t n=0;for(const auto& r:ranks_)n=add(n,r.lease->host_bytes());return n;}
    uint64_t state_bytes() const {uint64_t n=0;for(const auto& r:ranks_)n=add(n,r.lease->state_bytes());return n;}
    uint64_t page_bytes() const {uint64_t n=0;for(const auto& r:ranks_)n=add(n,r.lease->page_bytes());return n;}
    Payload load_cpu() const {
        Payload out;for(const auto& r:ranks_)out.ranks.push_back(r.lease->load_cpu());return out;
    }
    void demote(uint64_t max_bytes) {
        if(page_bytes()>max_bytes)throw MemoryError(MemoryErrorCode::BudgetExceeded,"host allowance cannot hold all prefix ranks");
        for(auto& r:ranks_)r.lease->demote(r.device,r.stream,r.lease->page_bytes());
    }
    void park(const std::shared_ptr<HostKvStorage>& storage,const std::shared_ptr<StorageIoQueue>& io) {
        for(auto& r:ranks_)r.lease->park(storage,io);
    }
    void restore(uint64_t sequence,const std::vector<int32_t>& prompt,const Payload* prepared=nullptr) const {
        if(ranks_.empty() || (prepared && prepared->ranks.size()!=ranks_.size()))throw std::runtime_error("rank prefix payload count mismatch");
        for(const auto& r:ranks_)if(!r.lease->matches(prompt))throw std::runtime_error("rank prefix token identity mismatch");
        // A partial failure is fatal to this request. Engine rejection/retire
        // owns sequence cleanup; no scheduler frontier is advanced on failure.
        for(size_t i=0;i<ranks_.size();++i) {
            const auto& r=ranks_[i];
            r.lease->restore(*r.kv,sequence,prompt,r.device,r.stream,prepared ? &prepared->ranks[i] : nullptr);
        }
    }
};
class RankCheckpoint {
    struct Item {int device;RadStream stream;rad::KVManager* kv;StateCheckpoint state;WindowCheckpoint window;};
    std::vector<Item> ranks_;
public:
    void save_rank(rad::KVManager& kv,uint64_t sequence,int device,RadStream stream,int64_t frontier) {
        Item item{device,stream,&kv,{},{}};
        item.state.save(kv,sequence,device,stream);
        item.window.save(kv,sequence,frontier,device,stream);
        ranks_.push_back(std::move(item));
    }
    void restore(uint64_t sequence) const {
        for(const auto& r:ranks_) {
            r.state.restore(*r.kv,sequence,r.device,r.stream);
            r.window.restore(*r.kv,sequence,r.device,r.stream);
        }
    }
    uint64_t bytes() const {
        uint64_t total=0;for(const auto& r:ranks_) {
            const auto bytes=r.state.bytes()+r.window.bytes();if(bytes>UINT64_MAX-total)throw std::runtime_error("rank checkpoint size overflow");total+=bytes;
        }return total;
    }
};
} // namespace kvmem::radiance
