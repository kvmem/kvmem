#include "memory_test_support.hpp"
#include "kvmem/layerwise_pressure.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

namespace {
void near(double a,long double b) {CHECK(std::abs(static_cast<long double>(a)-b)<1e-12L*std::max(1.L,std::abs(b)));}
size_t bucket(uint64_t row,uint64_t key,uint32_t block) {
    long double distance=std::max(0.L,static_cast<long double>(row)-key-block);
    const uint64_t limits[]={32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072};
    for (size_t i=0;i<13;++i) if (distance<limits[i]) return i;
    return 13;
}
struct Oracle {
    struct Value {long double mass=0,exposure=0,expected=0;};
    std::map<uint64_t,Value> values;
    std::array<long double,14> masses{},pairs{};
    uint64_t end=0;
    void accept(const LayerwisePrefillAcceptance& packet,const LayerwisePrefillMass& input,uint32_t block,double half_life) {
        std::map<uint64_t,Value> next;
        const long double decay=std::exp2(-static_cast<long double>(packet.history.frontier-end)/half_life);
        for (auto key:input.history) {auto old=values[key];next[key]={old.mass*decay,old.exposure*decay,old.expected*decay};}
        const auto R=packet.rows.size(),B=input.history.size();
        // Deliberately materialize a naive per-row/per-block expectation matrix;
        // no production bucket helper, row histogram or normalized table is used.
        std::vector<std::vector<long double>> expected(R,std::vector<long double>(B));
        for (size_t r=0;r<R;++r) for (size_t b=0;b<B;++b) {
            if (input.starts[b]>packet.local_begin+packet.rows[r]) continue;
            const auto at=packet.history.frontier-packet.count+packet.rows[r];
            const auto g=bucket(at,input.history[b],block);
            masses[g]+=input.masses[r*B+b];pairs[g]+=1;
        }
        for (size_t r=0;r<R;++r) {
            long double norm=0;size_t visible=0;
            for (size_t b=0;b<B;++b) if (input.starts[b]<=packet.local_begin+packet.rows[r]) {
                const auto g=bucket(packet.history.frontier-packet.count+packet.rows[r],input.history[b],block);
                expected[r][b]=pairs[g]>0?masses[g]/pairs[g]:0;
                norm+=expected[r][b];++visible;
            }
            std::map<uint64_t,bool> seen;
            for (size_t b=0;b<B;++b) if (input.starts[b]<=packet.local_begin+packet.rows[r]) {
                auto& v=next[input.history[b]];
                v.mass+=static_cast<long double>(input.masses[r*B+b])/R;
                v.expected+=(norm>0?expected[r][b]/norm:1.L/std::max<size_t>(visible,1))/R;
                if (!seen[input.history[b]]) {v.exposure+=1.L/R;seen[input.history[b]]=true;}
            }
        }
        values=std::move(next);end=packet.history.frontier;
    }
    void compare(const LayerwisePrefillState& state) const {
        CHECK(state.values.size()==values.size());
        for (const auto& v:state.values) {
            const auto& x=values.at(v.original_start);
            near(v.mass,x.mass);near(v.exposure,x.exposure);near(v.expected,x.expected);
            if (v.expected>0) near(v.mass/v.expected,x.mass/x.expected);
        }
        for (size_t b=0;b<14;++b) {near(state.bucket_mass[b],masses[b]);near(state.bucket_pairs[b],pairs[b]);}
    }
};
LayerwisePrefillOptions options(uint32_t block=64) {return {block,4*block,4*block,block,block/2,32768.,LayerwisePrefillScorer::Lift};}
LayerwisePrefillAcceptance packet(uint32_t block,uint64_t begin,bool compact,bool split=false) {
    LayerwisePrefillAcceptance p;
    p.count=p.accepted=4*block;p.history={true,{42,3},begin+1,begin+1,begin+p.count,begin+1};
    std::vector<uint64_t> history;
    if (begin) history={0,1024,65536};
    for (uint64_t i=0;i<4;++i) history.push_back(begin+i*block);
    p.local_begin=compact?(history.size()-4)*block:begin;
    for (uint32_t i=block/2-1;i<p.count;i+=block/2) p.rows.push_back(i);
    std::vector<uint64_t> starts;
    for (size_t i=0;i<history.size();++i) starts.push_back(compact?i*block:history[i]);
    if (split) {history.insert(history.begin()+1,history.front());starts.insert(starts.begin()+1,starts.front()+block/2);}
    for (uint32_t layer:{2u,7u}) {
        LayerwisePrefillMass input;input.model_layer=layer;input.history=history;input.starts=starts;
        for (auto row:p.rows) {
            std::vector<double> weights;
            double sum=0;
            for (size_t b=0;b<history.size();++b) {
                double weight=0;
                if (starts[b]<=p.local_begin+row) {
                    const auto distance=std::max(0.,double(begin+row)-double(history[b]));
                    weight=(1.+((b+layer)%3)*.25)/(1.+distance/block);
                    if (begin && history[b]==0) weight*=layer==2?4.:.5;
                }
                weights.push_back(weight);sum+=weight;
            }
            for (auto v:weights) input.masses.push_back(v/sum);
        }
        p.layers.push_back(std::move(input));
    }
    return p;
}
void same(const LayerwisePrefillSnapshot& a,const LayerwisePrefillSnapshot& b) {
    CHECK(a.session==b.session && a.scorer==b.scorer && a.layers.size()==b.layers.size());
    for (size_t i=0;i<a.layers.size();++i) {
        const auto& x=a.layers[i];const auto& y=b.layers[i];
        CHECK(x.model_layer==y.model_layer && x.end==y.end && x.samples==y.samples && x.chunks==y.chunks);
        CHECK(x.bucket_mass==y.bucket_mass && x.bucket_pairs==y.bucket_pairs && x.values.size()==y.values.size());
        for (size_t k=0;k<x.values.size();++k) {
            CHECK(x.values[k].original_start==y.values[k].original_start);
            CHECK(x.values[k].mass==y.values[k].mass && x.values[k].exposure==y.values[k].exposure && x.values[k].expected==y.values[k].expected);
        }
    }
}
void numerical_and_coordinates() {
    for (uint32_t block:{32u,64u}) {
        LayerwisePrefillPressure compact({42,3},{2,7},options(block)),logical({42,3},{2,7},options(block));
        auto ordinary=options(block);ordinary.scorer=LayerwisePrefillScorer::Attention;
        LayerwisePrefillPressure unchanged({42,3},{2,7},ordinary);
        std::array<Oracle,2> refs;
        for (uint64_t begin:{0ull,131072ull,262144ull}) {
            auto p=packet(block,begin,true);
            CHECK(compact.accept(p));CHECK(logical.accept(packet(block,begin,false)));
            same(compact.snapshot(),logical.snapshot());CHECK(unchanged.accept(p));
            for (size_t l=0;l<2;++l) for (size_t k=0;k<compact.snapshot().layers[l].values.size();++k) {
                const auto& lift=compact.snapshot().layers[l].values[k];
                const auto& attention=unchanged.snapshot().layers[l].values[k];
                CHECK(lift.mass==attention.mass && lift.exposure==attention.exposure && attention.expected==0);
            }
            for (size_t l=0;l<2;++l) {refs[l].accept(p,p.layers[l],block,32768.);refs[l].compare(compact.snapshot().layers[l]);}
        }
        // Duplicate physical segments share logical mass/expected accumulators,
        // while exposure counts each logical key only once per sampled row.
        auto split=packet(block,393216,true,true);CHECK(compact.accept(split));
        for (size_t l=0;l<2;++l) {refs[l].accept(split,split.layers[l],block,32768.);refs[l].compare(compact.snapshot().layers[l]);}
        const auto saved=compact.snapshot();
        auto copy=compact;copy.restore(saved,saved.layers[0].end);same(copy.snapshot(),saved);
        auto next=packet(block,524288,true);CHECK(copy.accept(next));CHECK(compact.accept(next));same(copy.snapshot(),compact.snapshot());
        auto invalid=compact.snapshot();invalid.layers[1].values.front().expected=-1;
        rejects(MemoryErrorCode::InvalidPlan,[&]{copy.restore(invalid,invalid.layers[0].end);});same(copy.snapshot(),compact.snapshot());
        invalid=compact.snapshot();invalid.layers[1].bucket_mass[13]=std::numeric_limits<double>::quiet_NaN();
        rejects(MemoryErrorCode::InvalidPlan,[&]{copy.restore(invalid,invalid.layers[0].end);});same(copy.snapshot(),compact.snapshot());
        auto bad=packet(block,655360,true);bad.layers.back().masses.back()=-1;
        rejects(MemoryErrorCode::InvalidPlan,[&]{copy.accept(bad);});same(copy.snapshot(),compact.snapshot());
        CHECK(!copy.accept(next));same(copy.snapshot(),compact.snapshot());
        copy.restore(saved,saved.layers[0].end-1);
        LayerwisePrefillPressure fresh({42,3},{2,7},options(block));same(copy.snapshot(),fresh.snapshot());
        CHECK(copy.accept(packet(block,0,true)));CHECK(fresh.accept(packet(block,0,true)));same(copy.snapshot(),fresh.snapshot());
        LayerwisePrefillPressure old({42,3},{2,7},ordinary);
        rejects(MemoryErrorCode::InvalidPlan,[&]{old.restore(saved,saved.layers[0].end);});
    }
}
void bucket_boundaries_and_large_coordinates() {
    for (const uint32_t block:{32u,64u}) {
        const uint64_t begin=(UINT64_MAX/1024-4)*1024;
        auto config=options(block);config.stride=1;
        LayerwisePrefillPressure p({42,3},{2},config);
        LayerwisePrefillAcceptance a;a.count=a.accepted=block;
        a.history={true,{42,3},1,1,begin+block,1};
        LayerwisePrefillMass layer;layer.model_layer=2;
        for (uint64_t distance:{0ull,32ull,64ull,128ull,256ull,512ull,1024ull,2048ull,4096ull,8192ull,16384ull,32768ull,65536ull,131072ull,262144ull}) {
            // Align the historical block; stride-one rows test either side of
            // every representable bucket edge near UINT64_MAX.
            layer.history.push_back((begin-distance-block)/block*block);
            layer.starts.push_back(0);
        }
        a.local_begin=layer.history.size()*block;
        for (uint32_t row=0;row<block;++row) {
            a.rows.push_back(row);double total=0;
            for (size_t col=0;col<layer.history.size();++col) total+=1.+((row+col)%5);
            for (size_t col=0;col<layer.history.size();++col) layer.masses.push_back((1.+((row+col)%5))/total);
        }
        a.layers.push_back(layer);CHECK(p.accept(a));
        Oracle reference;reference.accept(a,layer,block,config.half_life);reference.compare(p.snapshot().layers[0]);
        CHECK(p.snapshot().layers[0].bucket_pairs.back()>0);
    }
}
MemoryDescriptor desc() {
    MemoryDescriptor d;
    d.identity={"lift-reference","native-v1","one-layer","state-v1"};
    d.layout={"opaque-native",1,64,8,4,{{2,PayloadPlaneKind::Key,0,4,4,0},{2,PayloadPlaneKind::Value,4,4,4,0}}};
    d.capabilities={64,1024,true,true,true,true,true,false};d.validate();return d;
}
void ranking_and_decode() {
    auto config=options();LayerwisePrefillPressure p({42,3},{2},config);
    auto state=p.snapshot();state.layers[0].values={{0,.4,1,.4},{64,.1,1,.01},{128,0,0,0}};
    state.layers[0].end=192;state.layers[0].samples=2;state.layers[0].chunks=1;
    state.layers[0].bucket_mass[0]=.5;state.layers[0].bucket_pairs[0]=2;p.restore(state,192);
    MemorySnapshot memory;memory.stamp={true,{42,3},9,8,192,11};
    memory.blocks={{{{42,3},900,0,64,1,1},1},{{{42,3},100,64,64,1,1},1},{{{42,3},500,128,64,1,1},1}};
    memory.resident={900,100,500};
    auto catalog=LayerwiseMemoryCatalog(desc(),{{2,memory}});
    CHECK(p.select(catalog,{},64)[0].selected==std::vector<uint64_t>{500});
    CHECK(p.select(catalog,{},128)[0].selected==std::vector<uint64_t>({100,500}));
    CHECK(p.select(catalog,{},128,1)[0].selected==std::vector<uint64_t>({900,500}));
    const std::array<int32_t,5> markers{{10,11,12,13,14}};
    LayerwiseOutputPressure output({42,3},{2},config,markers);
    CHECK(output.select(catalog,p,{},64)[0].selected==std::vector<uint64_t>{900});
    auto os=output.snapshot();os.accepted_end=192;os.layers[0].scores={{0,.1},{64,.9},{128,.2}};
    output.restore(os,192);
    CHECK(output.select(catalog,p,{},64)[0].selected==std::vector<uint64_t>{100});
    memory.resident={900,500};catalog=LayerwiseMemoryCatalog(desc(),{{2,memory}});
    CHECK(p.select(catalog,{},128)[0].selected==std::vector<uint64_t>({900,500}));
    // With no positive lift and no unobserved key, use ordinary scores; the
    // established all-zero pressure contract remains Pending.
    for (auto& value:state.layers[0].values) {value.mass=0;value.expected=1;value.exposure=1;}
    p.restore(state,192);rejects(MemoryErrorCode::Pending,[&]{p.select(catalog,{},64);});
}
}
int main() {
    numerical_and_coordinates();bucket_boundaries_and_large_coordinates();ranking_and_decode();
    std::puts("lift Pressure: independent long-double oracle, 32/64-token geometry, compacted coordinates, duplicate segments, complete restore/rewind, atomic acceptance and unchanged decode ranking passed");
}
