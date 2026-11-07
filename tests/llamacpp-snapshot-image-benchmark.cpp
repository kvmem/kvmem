// CPU-only attribution fixture; never used as a record format or restore path.
// Link with the same release kvmem host library as the backend being measured.
#include "kvmem/checkpoint_image.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
    constexpr size_t n = 52691548; // pinned hybrid's PARTIAL_ONLY payload
    std::vector<uint8_t> payload(n), destination(n);
    for (size_t i=0; i<n; ++i) payload[i]=static_cast<uint8_t>((i*31u)^(i>>12));
    kvmem::HostKvStorage storage;
    const auto bytes=kvmem::checkpoint_image_size(n,2);
    auto image=storage.allocate(bytes), plain=storage.allocate(bytes);
    const uint64_t id=42;
    const std::vector<kvmem::CheckpointWriteSpan> input{{payload.data(),n},{nullptr,0}};
    const std::vector<kvmem::CheckpointReadSpan> output{{destination.data(),n},{nullptr,0}};
    kvmem::write_checkpoint_image(image,id,input);
    // The read arm uses a distinct, sealed hot image and a private destination.
    auto sealed=storage.allocate(bytes);
    kvmem::write_checkpoint_image(sealed,id,input);
    sealed.seal();
    auto lease=sealed.lease();
    constexpr size_t offset=4*sizeof(uint64_t); // first payload after frame header/length
    for (size_t i=0;i<n;i+=kvmem::snapshot_chunk)
        plain.write(offset+i,payload.data()+i,std::min(n-i,kvmem::snapshot_chunk));
    auto simple=sealed.lease();
    volatile uint64_t observed=0;
    const char * names[]={"framed_write","copy_write","framed_read","copy_read","hash_only"};
    for(int round=-2;round<12;++round) {
        // Rotate and reverse the five operations, with independent warmups.
        for(int j=0;j<5;++j) {
            const int mode=(round+7+((round%2)==0?j:4-j))%5;
            auto start=std::chrono::steady_clock::now();
            if(mode==0) kvmem::write_checkpoint_image(image,id,input);
            if(mode==1) for(size_t i=0;i<n;i+=kvmem::snapshot_chunk)
                plain.write(offset+i,payload.data()+i,std::min(n-i,kvmem::snapshot_chunk));
            if(mode==2) kvmem::read_checkpoint_image(lease,id,output);
            if(mode==3) for(size_t i=0;i<n;i+=kvmem::snapshot_chunk)
                simple.read(offset+i,destination.data()+i,std::min(n-i,kvmem::snapshot_chunk));
            if(mode==4) {
                kvmem::SnapshotWriter writer([](const void *,size_t){});
                writer.write(payload.data(),n);
                observed=writer.hash();
            }
            auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(round>=0) std::printf("%d,%s,%.6f\n",round,names[mode],ms);
        }
        if(destination!=payload || observed==0) return 1;
    }
    return 0;
}
