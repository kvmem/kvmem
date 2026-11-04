#pragma once
#include "kvmem/host_kv_storage.hpp"
#include "kvmem/memory_contract.hpp"
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cstdio>

namespace kvmem::radiance {
// Only immutable token/media/position metadata. A checkpoint may be earlier,
// but this object is created only after the complete canonical input is accepted.
template<class Media> struct NativeCanonicalInput {
    std::shared_ptr<HostKvReservation> charge; // Destroy last, after vectors.
    std::vector<int32_t> tokens;
    std::vector<Media> images;
    uint64_t bytes() const {return sizeof(*this)+128+tokens.capacity()*sizeof(int32_t)+images.capacity()*sizeof(Media);}
    static bool media_prefix(const std::vector<Media>& a,const std::vector<Media>& b,size_t end) {
        size_t i=0,j=0;
        while(i<a.size() && uint64_t(a[i].begin)<end) {
            if(uint64_t(a[i].end)>end || j>=b.size() || !(a[i]==b[j]))return false;
            ++i;++j;
        }
        return j>=b.size() || uint64_t(b[j].begin)>=end;
    }
    bool comparable(const NativeCanonicalInput& other) const {
        const auto n=std::min(tokens.size(),other.tokens.size());
        return n && std::equal(tokens.begin(),tokens.begin()+n,other.tokens.begin()) &&
            media_prefix(images,other.images,n) && media_prefix(other.images,images,n);
    }
};
template<class Media,class Reserve>
std::shared_ptr<const NativeCanonicalInput<Media>> capture_canonical_input_reserved(const std::vector<int32_t>& tokens,
        uint64_t end,uint64_t accepted,const std::vector<Media>& images,Reserve reserve) {
    if(!end || end>tokens.size() || accepted<end)return {};
    size_t count=0;
    for(const auto& image:images)if(uint64_t(image.begin)<end) {
        if(uint64_t(image.end)>end)return {}; ++count;
    }
    using Input=NativeCanonicalInput<Media>;
    try {
        const auto bytes=sizeof(Input)+128+end*sizeof(int32_t)+count*sizeof(Media);
        auto charge=reserve(bytes);
        auto input=std::make_shared<Input>();input->charge=std::move(charge);
        input->tokens.assign(tokens.begin(),tokens.begin()+std::ptrdiff_t(end));
        input->images.assign(images.begin(),images.begin()+std::ptrdiff_t(count));
        if(input->bytes()>input->charge->remaining())throw std::logic_error("canonical metadata capacity exceeds preflight");
        return input;
    } catch(const MemoryError& error) {
        std::fprintf(stderr,"KVMEM_CANONICAL_SKIP reason=optional_storage code=%d error=%s\n",int(error.code()),error.what());
    } catch(const std::bad_alloc&) {std::fprintf(stderr,"KVMEM_CANONICAL_SKIP reason=allocation\n");}
    return {}; // Unknown fairness identity never weakens native restore proof.
}
template<class Media>
std::shared_ptr<const NativeCanonicalInput<Media>> capture_canonical_input(const std::vector<int32_t>& tokens,
        uint64_t end,uint64_t accepted,const std::vector<Media>& images,HostKvStorage* storage) {
    if(!storage)return {};
    return capture_canonical_input_reserved(tokens,end,accepted,images,[&](uint64_t bytes) {
        return std::make_shared<HostKvReservation>(storage->reserve(bytes));
    });
}
template<class Media> struct CanonicalGroupMember {
    uint64_t group;
    std::shared_ptr<const NativeCanonicalInput<Media>> input;
};
template<class Media>
std::optional<uint64_t> classify_canonical_group(const std::shared_ptr<const NativeCanonicalInput<Media>>& input,
        const std::vector<CanonicalGroupMember<Media>>& members) {
    if(!input || input->tokens.empty())return {};
    struct Group {bool valid=true,exact=false;size_t depth=0;};std::map<uint64_t,Group> groups;
    for(const auto& member:members)if(member.group) {
        auto& group=groups[member.group];const auto& prior=member.input;
        if(!prior || !input->comparable(*prior)){group.valid=false;continue;}
        group.exact|=input->tokens.size()==prior->tokens.size();
        group.depth=std::max(group.depth,std::min(input->tokens.size(),prior->tokens.size()));
    }
    std::optional<uint64_t> chosen;Group best;
    for(const auto& item:groups)if(item.second.valid && (!chosen || (item.second.exact && !best.exact) ||
            (item.second.exact==best.exact && item.second.depth>best.depth))){chosen=item.first;best=item.second;}
    return chosen;
}
} // namespace kvmem::radiance
