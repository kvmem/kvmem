#pragma once
#include "rank_memory.hpp"
#include "native_memory.hpp"

namespace kvmem::radiance {
class RankNativeMemory final : public RankMemory {
    std::vector<std::shared_ptr<NativeMemory>> native_;
    static std::vector<std::shared_ptr<MemoryBackend>> bases(const std::vector<std::shared_ptr<NativeMemory>>& v) {
        return {v.begin(),v.end()};
    }
    template<class F> void update(F operation) {
        try {for(auto& rank:native_)operation(*rank);(void)snapshot();}
        catch(...){invalidate();throw;}
    }
public:
    RankNativeMemory(std::vector<std::shared_ptr<NativeMemory>> ranks,BackendIdentity identity)
        :RankMemory(bases(ranks),std::move(identity)),native_(std::move(ranks)) {}
    void accept_append(uint64_t frontier,const std::vector<BlockDescriptor>& added) {
        update([&](auto& rank){rank.accept_append(frontier,added);});
    }
    void rewind(uint64_t frontier) {update([&](auto& rank){rank.rewind(frontier);});}
    void retain_view() {update([](auto& rank){rank.retain_view();});}
    PrefixCapture prefix_capture(size_t rank) {return native_.at(rank)->prefix_capture();}
    void refresh_captured() {update([](auto& rank){rank.refresh_captured();});}
    void rebind_sequence(uint64_t sequence,bool relocated=false) {
        update([&](auto& rank){rank.rebind_sequence(sequence,relocated);});
    }
    std::shared_ptr<RankNativeMemory> fork_retained(SessionIdentity session) const {
        std::vector<std::shared_ptr<NativeMemory>> branch;
        for(const auto& rank:native_)branch.push_back(rank->fork_retained(session));
        return std::make_shared<RankNativeMemory>(std::move(branch),descriptor().identity);
    }
    void account_archive(SharedPayloadAccounting& out) const {for(const auto& rank:native_)rank->account_archive(out);}
    uint64_t archive_bytes() const noexcept {
        uint64_t total=0;for(const auto& rank:native_) {
            const auto bytes=rank->archive_bytes();if(bytes>UINT64_MAX-total)return UINT64_MAX;total+=bytes;
        }return total;
    }
    uint64_t shared_archive_bytes(const RankNativeMemory& other) const noexcept {
        if(native_.size()!=other.native_.size())return 0;
        uint64_t total=0;
        for(size_t i=0;i<native_.size();++i) {
            const auto bytes=native_[i]->shared_archive_bytes(*other.native_[i]);
            if(bytes>UINT64_MAX-total)return UINT64_MAX;
            total+=bytes;
        }
        return total;
    }
    NativeMemory::ReuseStats reuse_stats() const noexcept {
        NativeMemory::ReuseStats out;
        for(const auto& rank:native_) {
            const auto stats=rank->reuse_stats();
            out.transactions+=stats.transactions;out.recycled_pages+=stats.recycled_pages;
            out.journal_bytes_peak+=stats.journal_bytes_peak;out.rollbacks+=stats.rollbacks;
            out.spill_copy_bytes_avoided+=stats.spill_copy_bytes_avoided;
            out.spill_staging_bytes_avoided_peak+=stats.spill_staging_bytes_avoided_peak;
        }return out;
    }
};
} // namespace kvmem::radiance
