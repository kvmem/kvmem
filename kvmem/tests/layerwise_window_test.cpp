#include "kvmem/layerwise_window.hpp"
#include "layerwise_window_upstream_prefix.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value,const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> int64_t result(F function) {
    try { return int64_t(function()); }
    catch (const kvmem::MemoryError & e) { return e.code()==kvmem::MemoryErrorCode::BudgetExceeded ? -2:-1; }
}
void fixtures(const char * path) {
    std::ifstream file(path); require(bool(file),"missing independently generated pinned RequestPlan fixtures");
    std::string line; size_t cases=0;
    while (std::getline(file,line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream row(line); std::vector<int64_t> v; int64_t value;
        while (row>>value) v.push_back(value);
        require(v.size()==25,"invalid RequestPlan fixture shape");
        kvmem::LayerwiseRequestEnvelope e;
        e.block_tokens=uint32_t(v[1]); e.prompt_end=uint64_t(v[2]); e.generation_tokens=uint64_t(v[3]);
        e.output_tokens=uint64_t(v[4]); e.rolling_active_window=v[5]!=0;
        e.prefill_window=kvmem::LayerwiseWindowPolicy(v[6]); e.reserve_query_output=v[7]!=0;
        e.probe_tokens=uint64_t(v[8]); e.pressure_probe_tokens=uint64_t(v[9]); e.has_query_mandatory=v[10]!=0;
        if (v[11]>=0) e.query=kvmem::QueryRange{uint64_t(v[11]),uint64_t(v[12])};
        if (v[13]>=0) e.replay=kvmem::QueryRange{uint64_t(v[13]),uint64_t(v[14])};
        kvmem::LayerwiseWindowPlan plan(e,uint64_t(v[15]));
        const kvmem::LayerwiseWindowState state{uint64_t(v[16]),uint64_t(v[17])};
        const int64_t actual[]{result([&]{return int(plan.action(state));}),
            result([&]{return plan.admit(state,uint64_t(v[18]));}),
            result([&]{return plan.selection_budget(state.history);}),
            plan.prefill_checkpoint() ? int64_t(*plan.prefill_checkpoint()):-1,
            int64_t(plan.generation_checkpoint()),plan.prefill_boundary(state.history)};
        for (size_t i=0;i<6;++i) if (actual[i]!=v[19+i])
            throw std::runtime_error("pinned RequestPlan fixture "+std::to_string(v[0])+" field "+std::to_string(i)+
                " actual="+std::to_string(actual[i])+" expected="+std::to_string(v[19+i]));
        ++cases;
    }
    require(cases>=1000,"insufficient independent ordinary request fixtures");
    std::cout<<"pinned RequestPlan AST fixtures passed: "<<cases<<'\n';
}
}

int main(int argc,char ** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--native-reference-stream") {
            unsigned kind;uint64_t a,b,c;
            while(std::cin>>kind>>a>>b>>c)
                std::cout<<(kind==0 ? qw3::detail::kvmem_branch_safe_prompt_checkpoint_split(a,b,uint32_t(c)) :
                    qw3::detail::kvmem_replay_capture_begin(a,b,c))<<'\n';
            return 0;
        }
        if (argc==5 && std::string(argv[1])=="--native-reference") {
            const auto prompt=std::stoull(argv[2]),base=std::stoull(argv[3]);
            const auto block=uint32_t(std::stoul(argv[4]));
            std::cout<<qw3::detail::kvmem_branch_safe_prompt_checkpoint_split(prompt,base,block)<<'\n'; return 0;
        }
        if (argc==5 && std::string(argv[1])=="--native-replay") {
            std::cout<<qw3::detail::kvmem_replay_capture_begin(std::stoull(argv[2]),std::stoull(argv[3]),std::stoull(argv[4]))<<'\n';
            return 0;
        }
        kvmem::LayerwiseRequestEnvelope e; e.prompt_end=512; e.block_tokens=32; e.generation_tokens=128;
        kvmem::LayerwiseWindowPlan plan(e,256);
        require(plan.action({192,192})==kvmem::LayerwiseWindowAction::Pressure,"prefill did not visit C-2b");
        require(plan.action({191,191})==kvmem::LayerwiseWindowAction::None,"unaligned accepted tail selected early");
        require(plan.admit({191,191},64)==1,"admission did not seal the accepted partial tail at boundary");
        e.prefill_window=kvmem::LayerwiseWindowPolicy::KeepSelected;
        require(kvmem::LayerwiseWindowPlan(e,256).action({224,224})==kvmem::LayerwiseWindowAction::Exhausted,
                "keep-selected silently became rolling Pressure");
        e.rolling_active_window=true;
        require(kvmem::LayerwiseWindowPlan(e,256).action({512,224})==kvmem::LayerwiseWindowAction::Pressure,
                "rolling output did not use C-b");
        e.prefill_window=kvmem::LayerwiseWindowPolicy::Pressure; e.query=kvmem::QueryRange{403,431};
        e.replay=kvmem::QueryRange{399,439};
        kvmem::LayerwiseWindowPlan query(e,256);
        require(query.query_evaluation()->begin==384 && query.query_evaluation()->end==439,"Query/header origin was shortened");
        require(query.admit({384,64},54)==0 && query.admit({384,64},55)==55,"atomic Query was split");
        require(result([&]{return query.admit({385,65},64);})==-1,"accepted origin inside complete Query was allowed");
        require(result([&]{return query.admit({384,224},55);})==-2,"atomic Query bypassed native capacity proof");
        require(result([&]{return plan.admit({100,101},32);})==-1,"foreign physical extent was accepted");
        require(result([&]{return plan.admit({UINT64_MAX,0},32);})==-1,"native coordinate overflow was accepted");
#ifdef KVMEM_WINDOW_FIXTURE_PATH
        fixtures(argc==2 ? argv[1]:KVMEM_WINDOW_FIXTURE_PATH);
#else
        require(argc==2,"provide pinned AST fixture file"); fixtures(argv[1]);
#endif
        return 0;
    } catch (const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
