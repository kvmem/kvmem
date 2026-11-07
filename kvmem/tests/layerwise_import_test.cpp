#include "kvmem/layerwise_import.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> int error(F call) {
    try { call(); return 0; }
    catch (const kvmem::MemoryError & e) { return e.code() == kvmem::MemoryErrorCode::BudgetExceeded ? -2 : -1; }
}
void fixtures(const char * path) {
    std::ifstream input(path); check(bool(input), "missing pinned import-start fixtures");
    std::string line; size_t count=0;
    while (std::getline(input,line)) {
        if (line.empty() || line.front()=='#') continue;
        std::istringstream row(line); std::vector<int64_t> v; int64_t x;
        while(row>>x) v.push_back(x);
        check(v.size()==24,"invalid pinned import-start fixture shape");
        kvmem::LayerwiseRequestEnvelope e;
        e.block_tokens=uint32_t(v[1]);e.prompt_end=uint64_t(v[2]);e.generation_tokens=uint64_t(v[3]);
        e.output_tokens=uint64_t(v[4]);e.rolling_active_window=v[5]!=0;
        e.prefill_window=kvmem::LayerwiseWindowPolicy(v[6]);e.reserve_query_output=v[7]!=0;
        e.probe_tokens=uint64_t(v[8]);e.pressure_probe_tokens=uint64_t(v[9]);
        if(v[10]>=0)e.query=kvmem::QueryRange{uint64_t(v[10]),uint64_t(v[11])};
        if(v[12]>=0)e.replay=kvmem::QueryRange{uint64_t(v[12]),uint64_t(v[13])};
        kvmem::LayerwiseWindowPlan plan(e,uint64_t(v[14]));
        kvmem::LayerwiseImportStart actual;
        const auto code=error([&]{actual=kvmem::plan_layerwise_import(plan,{uint64_t(v[15]),uint64_t(v[16])});});
        if(code!=v[17])throw std::runtime_error("import-start fixture error "+std::to_string(v[0]));
        if(!code) {
            const uint64_t values[]{actual.terminal_reserve_overflow,actual.prefill_probe_overflow,
                actual.query_replay_overflow,actual.budget_tokens,actual.restore_tokens,actual.selection_budget};
            for(size_t i=0;i<6;++i)if(values[i]!=uint64_t(v[18+i]))
                throw std::runtime_error("import-start fixture "+std::to_string(v[0])+" field "+std::to_string(i));
        }
        ++count;
    }
    check(count>=10000,"insufficient independent import-start fixtures");
    std::cout<<"pinned SessionLedger AST import fixtures passed: "<<count<<'\n';
}
}

int main(int argc,char ** argv) {
    try {
        kvmem::LayerwiseRequestEnvelope e;
        e.block_tokens=32;e.prompt_end=721;e.generation_tokens=128;e.query=kvmem::QueryRange{641,721};
        kvmem::LayerwiseWindowPlan plan(e,384);
        check(plan.action({635,347})==kvmem::LayerwiseWindowAction::None,"partial import unexpectedly used ordinary dispatch");
        auto p=kvmem::plan_layerwise_import(plan,{635,347});
        check(p.pressure() && p.prefill_probe_overflow && !p.terminal_reserve_overflow && !p.query_replay_overflow &&
            p.budget_tokens==256 && p.restore_tokens==251 && p.selection_budget==256,"partial warm import lost startup K or accepted tail");
        check(p.restore_tokens+5==256,"partial tail reconstruction did not reach K at A640");
        p=kvmem::plan_layerwise_import(plan,{640,288});
        check(p.query_replay_overflow && p.restore_tokens==256,"exact Query-origin import did not reserve its old-view probe");
        p=kvmem::plan_layerwise_import(plan,{640,256});
        check(!p.pressure() && p.restore_tokens==256 && !p.selection_budget,"safe exact Query-origin import selected unnecessarily");
        e.query.reset();e.prompt_end=650;e.output_tokens=64;
        p=kvmem::plan_layerwise_import(kvmem::LayerwiseWindowPlan(e,384),{645,325});
        check(p.terminal_reserve_overflow && p.budget_tokens==256 && p.restore_tokens==229 &&
            p.selection_budget==256,"terminal import failed exact partial accounting");
        e.output_tokens=128;
        p=kvmem::plan_layerwise_import(kvmem::LayerwiseWindowPlan(e,384),{645,325});
        check(p.budget_tokens==192 && p.restore_tokens==165,"terminal output did not lower startup budget");
        e.rolling_active_window=true;
        p=kvmem::plan_layerwise_import(kvmem::LayerwiseWindowPlan(e,384),{645,325});
        check(!p.terminal_reserve_overflow,"rolling mode reserved a full output window");
        check(error([&]{kvmem::plan_layerwise_import(plan,{635,348});})==-1,"foreign partial-tail extent accepted");
        check(error([&]{kvmem::plan_layerwise_import(plan,{722,18});})==-1,"import beyond trusted prompt accepted");
        e.rolling_active_window=false;e.output_tokens=384;
        check(error([&]{kvmem::plan_layerwise_import(kvmem::LayerwiseWindowPlan(e,384),{645,325});})==-2,
              "negative signed import budget wrapped");
#ifdef KVMEM_IMPORT_FIXTURE_PATH
        fixtures(argc==2?argv[1]:KVMEM_IMPORT_FIXTURE_PATH);
#else
        check(argc==2,"provide pinned import-start fixtures");fixtures(argv[1]);
#endif
        std::cout<<"layerwise import-start tests passed\n";
        return 0;
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n';return 1; }
}
