#include "rank_statistics.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
using namespace kvmem::radiance;
int main() {
    using Clock=std::chrono::steady_clock;
    for(unsigned rows:{8u,256u}) {
        AcceptedStatistics seed;seed.session={1,1};seed.rows=rows;seed.key_width=512;seed.query_width=4096;
        for(int l=0;l<16;++l){seed.layers.push_back(l);seed.keys.emplace_back(rows*512,.25f);seed.query.tokens.emplace_back(8*4096,.5f);seed.query.token_counts.push_back(8);}
        seed.query.sums.resize(16*4096,4.f);
        double times[2]={};double checksum=0;
        for(int repeat=0;repeat<400;++repeat)for(int order=0;order<2;++order) {
            int mode=order^(repeat&1);std::vector<AcceptedStatistics> input{seed};
            const auto start=Clock::now();
            auto result=mode ? merge_rank_statistics(std::move(input)) : merge_rank_statistics(input);
            const auto end=Clock::now();
            if(repeat>=40)times[mode]+=std::chrono::duration<double,std::micro>(end-start).count();
            checksum+=result.keys[0][0]+result.query.sums[0];
        }
        std::cout<<"{\"rows\":"<<rows<<",\"samples_per_mode\":360,\"copy_us\":"<<times[0]/360
                 <<",\"move_us\":"<<times[1]/360<<",\"checksum\":"<<checksum<<"}\n";
    }
}
