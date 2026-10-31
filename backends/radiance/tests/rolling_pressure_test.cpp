#include "rolling_pressure.hpp"
#include <iostream>
#include <numeric>
using namespace kvmem::radiance::rolling;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while(false)
template<class F> void rejects(F f) { bool threw=false; try { f(); } catch(const std::invalid_argument&) { threw=true; } REQUIRE(threw); }
int main() {
    REQUIRE(!retainable_prefix_boundary(32843,32843,128));
    REQUIRE(!retainable_prefix_boundary(64,64,128));
    REQUIRE(!retainable_prefix_boundary(32768,32843,128));
    REQUIRE(!retainable_prefix_boundary(0,0,128));
    REQUIRE(retainable_prefix_boundary(32768,32768,128));
    REQUIRE(retainable_prefix_boundary(128,128,128));
    std::vector<uint32_t> required{629,0,628,629};
    canonicalize_mandatory(required);
    REQUIRE(required==std::vector<uint32_t>({0,628,629}));
    REQUIRE(pressure_selection({0,628,629},{1,1,1},required,3)==required);
    REQUIRE(query_checkpoint_due(64512,320,64842,128));
    REQUIRE(!query_checkpoint_due(64000,512,64842,128));
    REQUIRE(query_checkpoint_due(64512,398,64884,128));
    REQUIRE(!query_checkpoint_due(0,128,128,128));
    REQUIRE(query_checkpoint_due(128,64,128,128));
    // Actual capacity contract: unchanged B, R and batch, and three reusable windows.
    constexpr uint32_t B=36864,R=16384,batch=512,block=128;
    REQUIRE(!needs_pressure(B+R-batch-block,B,R,batch,block));
    REQUIRE(needs_pressure(B+R-batch-block+1,B,R,batch,block));
    rejects([]{needs_pressure(1,36864,640,512,128);});
    std::vector<uint32_t> resident(B/block);
    std::iota(resident.begin(),resident.end(),0);
    uint32_t next=B/block;
    for(int cycle=0;cycle<3;++cycle) {
        while(!needs_pressure(resident.size()*block,B,R,batch,block)) resident.push_back(next++);
        const auto tail=resident.back();
        std::vector<double> score(resident.size(),1);
        auto selected=pressure_selection(resident,score,{0,tail},B/block);
        REQUIRE(selected.size()==B/block);
        REQUIRE(selected.front()==0 && selected.back()==tail);
        for(auto id:selected) REQUIRE(std::find(resident.begin(),resident.end(),id)!=resident.end());
        resident=selected;
    }
    REQUIRE(next*block>B+2*R); // original positions keep growing, pool does not.
    REQUIRE(pressure_selection({1,3,9},{.2,.8,.7},{1},2)==std::vector<uint32_t>({1,3}));
    REQUIRE(pressure_selection({1,3,9},{1,1,1},{},2)==std::vector<uint32_t>({3,9}));
    rejects([]{pressure_selection({1,1},{1,1},{},2);});
    rejects([]{pressure_selection({1},{-1},{},1);});
    rejects([]{pressure_selection({1},{1},{1,2},1);});
    std::cout<<"rolling pressure: three windows, fixed capacity, resident-only and safety checks PASS\n";
}
