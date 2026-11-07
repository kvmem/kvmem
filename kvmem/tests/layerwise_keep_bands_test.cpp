#include "kvmem/layerwise_options.hpp"
#include "kvmem/cli_options.h"
#include "kvmem/kvmem_store.hpp"
#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
namespace upstream_keep_oracle {
kvmem::KvMemKeepAllocation resolve_kvmem_keep_allocation(uint32_t,uint32_t,int64_t,int64_t,int64_t,int64_t);
std::vector<uint32_t> fixed_selection(uint32_t,uint32_t,uint32_t,const std::vector<uint32_t>&,uint32_t,bool);
}
namespace {
void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class F>bool rejects(F f){try{f();return false;}catch(const std::exception&){return true;}}
}
int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--oracle") {
        uint32_t block,budget;int64_t sb,rb,st,rt;
        while(std::cin>>block>>budget>>sb>>rb>>st>>rt) {
            try {const auto out=upstream_keep_oracle::resolve_kvmem_keep_allocation(block,budget,sb,rb,st,rt);
                std::cout<<"ok "<<out.sink_blocks<<' '<<out.recent_blocks<<'\n'<<std::flush;
            }catch(const std::exception&){std::cout<<"error\n"<<std::flush;}
        }
        return 0;
    }
    try {
        uint64_t compared=0;
        for(uint32_t block:{1u,4u,32u,64u,128u})for(uint32_t budget:{128u,256u,4096u,5120u,8192u,32768u,131072u,262144u}) {
            if(budget%block)continue;
            const std::vector<std::optional<uint32_t>> values{{},0u,block,33u,1024u,4096u,UINT32_MAX};
            for(auto sink:values)for(auto recent:values) {
                kvmem::LayerwiseKeepBands actual; kvmem::KvMemKeepAllocation expected;
                const bool a=rejects([&]{actual=kvmem::resolve_layerwise_keep_bands(block,budget,sink,recent);});
                const bool b=rejects([&]{expected=upstream_keep_oracle::resolve_kvmem_keep_allocation(block,budget,-1,-1,
                    sink?int64_t(*sink):-1,recent?int64_t(*recent):-1);});
                check(a==b,"startup keep-band rejection differs from pinned native function");
                if(!a)check(actual.sink_blocks==expected.sink_blocks && actual.recent_blocks==expected.recent_blocks,
                    "startup keep-band allocation differs from pinned native function");
                ++compared;
            }
        }
        const auto bands=kvmem::resolve_layerwise_keep_bands(32,131072);
        check(bands.sink_blocks==41 && bands.recent_blocks==328,"reference auto percentages/clamps changed");
        kvmem::validate_layerwise_selection_bands(bands,32,32768);
        check(rejects([&]{kvmem::validate_layerwise_selection_bands(bands,32,1024);}),"Query fixed bands silently shrank to B");
        kvmem::validate_layerwise_selection_bands(bands,32,1312,true);
        check(rejects([&]{kvmem::validate_layerwise_selection_bands(bands,32,1280,true);}),"Pressure lost its configured sink bound");
        const auto zero=kvmem::resolve_layerwise_keep_bands(32,128,0,0);
        check(!zero.sink_blocks && !zero.recent_blocks,"explicit0 became auto");
        check(rejects([]{kvmem::resolve_layerwise_keep_bands(32,128);}),"invalid tiny auto budget was clamped to legacy bands");
        // Hard pins consume recent room even when the configured S+R fits B.
        // Compare the actual native picker to the exact upstream method body.
        kvmem::KvMemStoreConfig fixed_cfg;fixed_cfg.block_tokens=32;fixed_cfg.select_budget=128;
        fixed_cfg.sink_blocks=1;fixed_cfg.recent_blocks=2;
        kvmem::KvMemStore policy(fixed_cfg);policy.register_append(256);
        const auto expected=upstream_keep_oracle::fixed_selection(8,1,2,{2,3},4,false);
        check(expected==std::vector<uint32_t>({0,2,3,7}),"pinned hard-pin/recent priority changed");
        check(policy.query_fixed_blocks({2,3})==expected,"native hard pins were unioned with the entire recent band");
        uint64_t fixed_cases=0;
        for(uint32_t n=1;n<=9;++n)for(uint32_t budget=1;budget<=n+1;++budget)
          for(uint32_t sink=0;sink<=budget;++sink)for(uint32_t recent=0;recent<=budget-sink;++recent) {
            fixed_cfg.select_budget=budget*32;fixed_cfg.sink_blocks=sink;fixed_cfg.recent_blocks=recent;
            kvmem::KvMemStore candidate(fixed_cfg);candidate.register_append(n*32);
            for(const auto pins:{std::vector<uint32_t>{},std::vector<uint32_t>{n/2},std::vector<uint32_t>{0,n-1}}) {
                std::vector<uint32_t> actual,want;
                const bool a=rejects([&]{actual=candidate.query_fixed_blocks(pins);});
                const bool b=rejects([&]{want=upstream_keep_oracle::fixed_selection(n,sink,recent,pins,budget,false);});
                check(a==b && (a || actual==want),"fixed-band picker differs from pinned canonical method");
                ++fixed_cases;
            }
          }
        kvmem::CliOptions cli;
        check(!cli.sink_tokens_set && !cli.recent_tokens_set,"absent CLI bands became explicit");
        cli.parse("--kvmem-sink-tokens",[](const char*){return "0";});
        cli.parse("--kvmem-recent-tokens",[](const char*){return "0";});
        check(cli.sink_tokens_set && cli.recent_tokens_set && !cli.sink_tokens && !cli.recent_tokens,
            "CLI explicit zero lost its provenance");
        std::cout<<"PASS keep-band native oracle cases="<<compared<<", fixed priority cases="<<fixed_cases
            <<", explicit0/None, finalB/Pressure preconditions, CLI provenance\n";
        return 0;
    } catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}
}
