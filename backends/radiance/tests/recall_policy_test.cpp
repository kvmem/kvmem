#include "recall_policy.hpp"
#include "attention_plan.hpp"
#include <cmath>
#include <iostream>
using namespace kvmem;
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(false)
template<class F> void rejects(F f) { bool failed=false; try{f();}catch(const std::exception&){failed=true;} CHECK(failed); }
int main() {
    auto pages=attention_pages(nullptr,0,10,4);
    CHECK(pages.size()==3 && pages.back().count==2 && pages.back().compact==8);
    MemorySnapshot frozen;frozen.stamp.valid=true;frozen.stamp.frontier=10;
    frozen.resident={0,2};
    frozen.blocks={{{{},0,0,4,1,1},0},{{{},1,4,4,1,1},1},{{{},2,8,2,1,1},0}};
    // Native reservation may now be ahead of 10. The accepted view itself stays
    // frozen; extend its partial tail and later appends without reading native KV.
    pages=attention_pages(&frozen,13,4,4);
    CHECK(pages.size()==4 && pages[1].start==8 && pages[1].count==4);
    CHECK(pages[2].start==12 && pages[2].compact==8 && pages[3].count==1);
    rejects([&]{attention_pages(&frozen,9,1,4);});
    frozen.resident={0};rejects([&]{attention_pages(&frozen,10,1,4);});
    // A 775-token fixed replay is admitted by the 5376-token native pool,
    // although its last batch begins above the normal 4096+512 Pressure threshold.
    // Keep all 32 selected pages fixed; sampling must not demand a new selection.
    MemorySnapshot replay;replay.stamp.valid=true;replay.stamp.frontier=11520;
    for(uint32_t id=0;id<90;++id) {
        replay.blocks.push_back({{{},id,uint64_t(id)*128,128,1,1},0});
        if(id<16 || id>=74)replay.resident.push_back(id);
    }
    CHECK(query_replay_plan({11776,12288},11520,12295,2048,128,32,42,256).admitted);
    uint64_t last_sample=11520,last_visible=0;
    unsigned replay_steps=0;
    for(uint32_t begin=11520;begin<12295;) {
        const auto count=std::min<uint32_t>(256,12295-begin);
        const auto view=attention_pages(&replay,begin,count,128);
        const uint64_t visible=view.back().compact+view.back().count-count;
        const auto samples=prefill_attention_samples(begin,count,last_sample,visible,4608,5376,true,128);
        for(auto row:samples)CHECK(row<count);
        if(!samples.empty())last_sample=begin+count;
        last_visible=visible;++replay_steps;begin+=count;
    }
    CHECK(replay_steps==4 && last_visible==4864);
    rejects([&]{prefill_attention_samples(12288,7,last_sample,4864,4608,5376,false,128);});
    rejects([&]{prefill_attention_samples(12288,7,last_sample,5377,4608,5376,true,128);});
    CHECK(prefill_attention_samples(128,128,128,128,4608,5376,false,128)==
          prefill_sample_rows(128,128,128,128,4608,1024,32,128));
    auto global=std::make_shared<GraphStatisticsBudget>(256u<<20);
    RecallPolicy policy({},4,global);
    policy.start_request(true,false);
    policy.accept(0,12,true,{{3,{{0,1}}},{7,{{0,.8},{4,.2}}},{11,{{0,.7},{4,.2},{8,.1}}}}, {0,4,8},{});
    const auto base=policy.checkpoint();
    CHECK(policy.pressure(SelectionReason::PrefillPressure,{0,1,2},{2},2)==std::vector<uint32_t>({0,2}));
    rejects([&]{policy.pressure(SelectionReason::DecodePressure,{1,2},{0},2);});
    rejects([&]{policy.pressure(SelectionReason::Query,{0,1,2},{2},2);});
    policy.accept(12,1,false,{{12,{{0,.1},{4,.8},{8,.1}}}},{0,4,8,12},{false,16});
    CHECK(policy.pressure(SelectionReason::DecodePressure,{0,1,2,3},{3},2)==std::vector<uint32_t>({1,3}));
    // A pending output preview must not commit or decay on repeated Pressure.
    auto first=policy.feedback();
    (void)policy.pressure(SelectionReason::DecodePressure,{0,1,2,3},{3},2);
    CHECK(policy.feedback().output_samples==first.output_samples);
    CHECK(policy.feedback().pending_output==first.pending_output);
    policy.restore(base);
    CHECK(policy.graph().accepted_through==12 && policy.output().eligible_tokens==0);
    CHECK(policy.feedback().output_samples==0);
    CHECK(policy.pressure(SelectionReason::DecodePressure,{1,2},{2},1)==std::vector<uint32_t>({2}));
    // An invalid acceptance must leave the complete old snapshot usable.
    rejects([&]{policy.accept(12,1,false,{{13,{{0,1}}}},{0},{});});
    CHECK(policy.graph().accepted_through==12);
    RawKvStore keys({1,2,0,4});
    const float a[]{8,0},b[]{0,8},c[]{-8,0};
    keys.write_layer_mean_sum(0,4,0,a);keys.write_layer_mean_sum(4,4,0,b);keys.write_layer_mean_sum(8,4,0,c);
    const float q[]{1,0,1,0};const uint32_t rows[]{2};
    const PageQueryView view{q,rows,1,2,1,1,2};
    const auto frame=query_score_frame(12,4,2,0,0,{2});
    CHECK(policy.query(keys,view,frame,{2},2)==std::vector<uint32_t>({0,2}));
    for (auto scorer:{QueryScorer::PageSoftmax,QueryScorer::MeanQCosine}) for(int agg:{0,1}) {
        RecallOptions options;options.scorer=scorer;options.aggregation=agg;options.lrfu=true;options.graph_alpha=0;
        RecallPolicy p(options,4);p.start_request(true,false);p.accept(0,12,true,{}, {},{});
        const auto saved=p.checkpoint();
        CHECK(p.query(keys,view,frame,{2},2)==std::vector<uint32_t>({0,2}));p.commit_query();
        CHECK(p.checkpoint().lrfu.committed);
        p.restore(saved);CHECK(!p.checkpoint().lrfu.committed);
    }
    const auto range=bounded_query(4,1000,1004,512);CHECK(range.begin==492 && range.end==1004);
    CHECK(!query_replay_plan(range,384,3000,2048,128,4,32,512).admitted);
    CHECK(query_replay_plan(range,384,1100,2048,128,4,32,512).admitted);
    std::cout<<"Radiance common recall/Pressure/checkpoint invariants passed\n";
}
