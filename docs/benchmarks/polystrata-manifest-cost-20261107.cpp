#include "kvmem/kv_snapshot.hpp"
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

int main() {
    using namespace kvmem;
    using Clock = std::chrono::steady_clock;
    for (const size_t count : {100u,512u,4096u,4097u}) {
        std::vector<std::shared_ptr<SealedKvBlock>> origins;
        std::vector<uint64_t> versions(count,1);
        for (size_t i=0;i<count;++i) origins.push_back(std::make_shared<SealedKvBlock>(4096));
        for (int reserve : {0,1,1,0}) {
            double micros=0, version_us=0;
            size_t manifest_bytes=0,version_bytes=0;
            std::vector<std::shared_ptr<KvSnapshot>> live(3);
            std::vector<std::shared_ptr<const std::vector<uint64_t>>> live_versions(3);
            for (int rep=0;rep<201;++rep) {
                auto begin=Clock::now();
                std::vector<KvSnapshot::Span> spans;
                if (reserve) spans.reserve(count);
                for(size_t i=0;i<count;++i) spans.push_back({i,0,i*4096,4096});
                auto snapshot=std::make_shared<KvSnapshot>(origins,std::move(spans),count*4096);
                auto mid=Clock::now();
                auto v=std::make_shared<const std::vector<uint64_t>>(versions);
                auto end=Clock::now();
                if(rep) {
                    micros+=std::chrono::duration<double,std::micro>(mid-begin).count();
                    version_us+=std::chrono::duration<double,std::micro>(end-mid).count();
                }
                manifest_bytes=snapshot->manifest_bytes();
                version_bytes=v->capacity()*sizeof(uint64_t);
                live[rep%3]=std::move(snapshot);
                live_versions[rep%3]=std::move(v);
            }
            std::printf("%zu,%d,%zu,%zu,%.3f,%.3f\n",count,reserve,manifest_bytes,version_bytes,micros/200,version_us/200);
        }
    }
}
