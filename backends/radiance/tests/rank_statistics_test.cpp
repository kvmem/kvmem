#include "rank_statistics.hpp"
#include <iostream>
#include <limits>
using namespace kvmem;using namespace kvmem::radiance;
#define CHECK(x) do{if(!(x))throw std::runtime_error(#x);}while(false)
template<class F> void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}CHECK(caught);}
int main() {
    AcceptedStatistics a{{9,2},64,2,2,4,{3,7},{{1,2,3,4},{5,6,7,8}},
        {{11,12,13,14,21,22,23,24},{1,1},{{11,12,13,14},{21,22,23,24}}}};
    auto b=a;
    for(auto& layer:b.keys)for(auto& x:layer)x+=100;
    for(auto& layer:b.query.tokens)for(auto& x:layer)x+=100;
    for(auto& x:b.query.sums)x+=100;
    const auto merged=merge_rank_statistics({a,b});
    CHECK(merged.keys[0]==std::vector<float>({1,2,101,102,3,4,103,104}));
    CHECK(merged.keys[1]==std::vector<float>({5,6,105,106,7,8,107,108}));
    CHECK(merged.query.tokens[1]==std::vector<float>({21,22,23,24,121,122,123,124}));
    CHECK(merged.query.token_counts==a.query.token_counts && merged.key_width==4 && merged.query_width==8);
    CHECK(merge_rank_statistics({a}).keys==a.keys);
    auto bad=b;bad.first++;rejects([&]{merge_rank_statistics({a,bad});});
    bad=b;bad.session.incarnation++;rejects([&]{merge_rank_statistics({a,bad});});
    bad=b;bad.query.token_counts[1]=0;rejects([&]{merge_rank_statistics({a,bad});});
    bad=b;bad.layers[1]=8;rejects([&]{merge_rank_statistics({a,bad});});
    bad=b;bad.keys[0].pop_back();rejects([&]{merge_rank_statistics({a,bad});});
    bad=b;bad.query.sums[0]=std::numeric_limits<float>::infinity();rejects([&]{merge_rank_statistics({a,bad});});
    std::vector<AcceptedAttentionRow> left{{95,{{0,.8},{64,.2}}}},right{{95,{{0,.2},{64,.8}}}};
    const auto attention=merge_rank_attention({left,right},{1,3});
    CHECK(std::abs(attention[0].mass[0].mass-.35)<1e-12);
    CHECK(std::abs(attention[0].mass[1].mass-.65)<1e-12);
    right[0].source_position++;rejects([&]{merge_rank_attention({left,right},{1,1});});
    right=left;right[0].mass[0].target=1;rejects([&]{merge_rank_attention({left,right},{1,1});});
    right=left;right[0].mass[0].mass=.5;rejects([&]{merge_rank_attention({left,right},{1,1});});
    rejects([&]{merge_rank_attention({left},{0});});
    // Consuming TP1 preserves allocations and all validation. A rejected frame
    // has not been moved from; a const caller still gets an independent copy.
    std::vector<AcceptedStatistics> single{a};
    const auto* keys=single[0].keys[0].data();const auto* tokens=single[0].query.tokens[1].data();
    const auto* sums=single[0].query.sums.data();
    const auto copied=merge_rank_statistics(single);CHECK(copied.keys[0].data()!=keys);
    const auto moved=merge_rank_statistics(std::move(single));
    CHECK(moved.keys==a.keys && moved.query.tokens==a.query.tokens && moved.query.sums==a.query.sums);
    CHECK(moved.keys[0].data()==keys && moved.query.tokens[1].data()==tokens && moved.query.sums.data()==sums);
    single={a};single[0].keys[0][0]=std::numeric_limits<float>::quiet_NaN();keys=single[0].keys[0].data();
    rejects([&]{merge_rank_statistics(std::move(single));});CHECK(single[0].keys[0].data()==keys);
    single={a};single[0].query.tokens[0].pop_back();rejects([&]{merge_rank_statistics(std::move(single));});
    single={a};single[0].query.sums.pop_back();rejects([&]{merge_rank_statistics(std::move(single));});
    std::vector<std::vector<AcceptedAttentionRow>> one{left};const auto* masses=one[0][0].mass.data();
    const auto moved_attention=merge_rank_attention(std::move(one),{1});CHECK(moved_attention[0].mass.data()==masses);
    CHECK(moved_attention[0].mass[0].mass==left[0].mass[0].mass);
    one={left};one[0][0].mass[0].mass=std::numeric_limits<double>::quiet_NaN();masses=one[0][0].mass.data();
    rejects([&]{merge_rank_attention(std::move(one),{1});});CHECK(one[0][0].mass.data()==masses);
    one={left};one[0][0].mass[0].mass=.1;rejects([&]{merge_rank_attention(std::move(one),{1});});
    one={left};rejects([&]{merge_rank_attention(std::move(one),{});});
    one={{{95,{{0,-0.0},{64,1.0}}}}};
    CHECK(!std::signbit(merge_rank_attention(std::move(one),{1})[0].mass[0].mass));
    std::cout<<"rank statistics: row/head order, logical Query counts, acceptance identity, finite masses PASS\n";
}
