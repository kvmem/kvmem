#include "kvmem/layerwise_probe_selection.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
void check(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
template<class F>void rejects(F f) {
    bool bad=false;try {f();}catch(const std::exception&){bad=true;}check(bad,"invalid probe operation was accepted");
}
kvmem::LayerwiseQueryIdentity identity(uint32_t pages,uint32_t rows) {
    kvmem::LayerwiseQueryIdentity out;
    out.history.valid=true;out.history.session={7,3};out.history.history_revision=8;
    out.history.execution_history=11;out.history.frontier=uint64_t(pages)*4;
    out.request=29;out.evaluation={out.history.frontier,out.history.frontier+rows+6};
    out.query={out.history.frontier+3,out.history.frontier+rows+3};return out;
}
kvmem::KvMemStoreConfig config(uint32_t budget,uint32_t sink=0,uint32_t recent=0) {
    kvmem::KvMemStoreConfig cfg;cfg.block_tokens=4;cfg.select_budget=4*budget;
    cfg.sink_blocks=sink;cfg.recent_blocks=recent;return cfg;
}
std::vector<kvmem::LayerwiseQueryGeometry> geometries(uint32_t heads=1,float scale=1) {
    // The index has a fourth auxiliary/draft layer. It is intentionally absent.
    return {{0,heads,1,1,scale},{2,heads,1,1,scale},{1,heads,1,1,scale}};
}
void populate(kvmem::RawKvStore& keys,uint32_t pages) {
    for(uint32_t layer=0;layer<4;++layer)for(uint32_t page=0;page<pages;++page) {
        const float key=layer==0?(page==0?6.f:0.f):layer==3?float(page*64):(page==1?2.f:0.f);
        const float sum=key*4;keys.write_layer_mean_sum(page*4,4,layer,&sum);
    }
}
std::vector<double> oracle(const kvmem::RawKvStore& keys,const std::vector<kvmem::LayerwiseQueryGeometry>& geometry,
    const std::vector<float>& query,uint32_t rows,const std::vector<uint8_t>& mask) {
    std::vector<double> result(mask.size(),0.);
    // Independent probability definition: logits per target/token/head,
    // same eligible page denominator, sum tokens / mean layers and heads.
    for(const auto& g:geometry)for(uint32_t t=0;t<rows;++t)for(uint32_t h=0;h<g.query_heads;++h) {
        std::vector<double> values(mask.size(),0.);double top=-INFINITY;
        for(size_t p=0;p<mask.size();++p)if(mask[p]) {
            const double key=g.layer==0?(p==0?6.:0.):(p==1?2.:0.);
            values[p]=query[size_t(t)*g.query_heads+h]*key*g.attention_scale;top=std::max(top,values[p]);
        }
        if(!std::isfinite(top))continue;
        double sum=0;for(size_t p=0;p<mask.size();++p)if(mask[p])sum+=std::exp(values[p]-top);
        for(size_t p=0;p<mask.size();++p)if(mask[p])
            result[p]+=std::exp(values[p]-top)/sum/double(geometry.size()*g.query_heads);
    }
    (void)keys;return result;
}
void global_differs_from_first_and_tiles() {
    kvmem::KvMemStore policy(config(1));policy.register_append(12);
    kvmem::RawKvStore keys({4,1,0,4});populate(keys,3);
    const auto id=identity(3,5);auto stamp=id.history;const auto g=geometries();
    kvmem::HostKvStorage metadata(1u<<20);
    const std::vector<float> query(5,1.f);
    std::vector<double> baseline;
    for(uint32_t tile:{1u,2u,3u,5u}) {
        const auto before=metadata.charged_bytes();
        {
            kvmem::LayerwiseProbeSelection selection(policy,keys,id,g,[&]{return stamp;},metadata);
            check(metadata.charged_bytes()==before+selection.metadata_bytes(),"probe retained metadata was not charged");
            rejects([&]{selection.finish(id);});
            for(uint32_t at=0;at<5;at+=tile)for(const auto& target:g) {
                const auto rows=std::min(tile,5-at);
                selection.append(id,id.query.begin+at,{query.data()+at,target.layer,rows,1,1,1,1});
            }
            auto result=selection.finish(id);
            check(result.selected==std::vector<uint32_t>{1},"global probe used first layer or top-k union");
            check(result.fixed.empty() && result.historical_tokens==4 && result.charged_tokens==4,"probe selection budget differs");
            const auto expected=oracle(keys,g,query,5,result.frame.score_mask);
            for(size_t p=0;p<expected.size();++p)check(std::abs(expected[p]-result.scores[p])<1e-7,"global probability formula differs");
            check(std::abs(std::accumulate(result.scores.begin(),result.scores.end(),0.)-5)<1e-7,"query tokens were averaged or layers summed");
            if(baseline.empty())baseline=result.scores;
            else for(size_t p=0;p<baseline.size();++p)check(std::abs(baseline[p]-result.scores[p])<1e-12,"probe scores depend on tile size");
            rejects([&]{selection.finish(id);});
        }
        check(metadata.charged_bytes()==before,"probe metadata outlived its owner");
    }
    const float q=1;
    check(kvmem::select_layerwise_history(policy,keys,{&q,0,1,1,1,1,1}).selected==std::vector<uint32_t>{0},
        "independent fixture failed to distinguish first and global policies");
}
void masks_pins_heads_scale_and_native() {
    kvmem::KvMemStore policy(config(4,1,2));policy.register_append(28);
    policy.set_media_ranges({{12,28}}); // Must never implicitly promote/expand.
    kvmem::RawKvStore keys({4,1,0,4});populate(keys,7);
    const auto id=identity(7,5);const auto g=geometries(2,.5f);auto stamp=id.history;
    const std::vector<float> query{1,-1,2,1,-2,2,1,1,0,-1};
    kvmem::HostKvStorage metadata(1u<<20);
    std::vector<double> baseline;
    for(uint32_t tile:{1u,2u,5u}) {
        kvmem::LayerwiseProbeSelection selection(policy,keys,id,g,[&]{return stamp;},metadata,{2,3});
        kvmem::NativeCurrentLayerQuery masked{query.data(),2*sizeof(float),0,1,2,1,1,.5f,
            kvmem::NativeQueryScalar::Float32,2*sizeof(float),sizeof(float),sizeof(float)};
        rejects([&]{selection.append_native(id,id.query.begin,masked,[](const auto&,auto,const auto&) {
            return std::vector<double>{1,0,0,0,0,0,0};
        });});
        check(selection.scored_rows()==0 && selection.next_layer()==0,"masked native mass advanced probe coverage");
        for(uint32_t at=0;at<5;at+=tile)for(const auto& target:g) {
            const auto rows=std::min(tile,5-at);const auto* ptr=query.data()+at*2;
            kvmem::NativeCurrentLayerQuery native{ptr,size_t(rows)*2*sizeof(float),target.layer,rows,2,1,1,.5f,
                kvmem::NativeQueryScalar::Float32,2*sizeof(float),sizeof(float),sizeof(float)};
            selection.append_native(id,id.query.begin+at,native,[&](const auto& index,auto q,const auto& frame) {
                check(frame.score_mask==std::vector<uint8_t>({0,1,1,1,1,0,0}),"score mask changed with hard pins");
                check(frame.transport_mask==std::vector<uint8_t>({0,1,0,0,1,1,0}),"hard pins did not consume soft recent room");
                return kvmem::layerwise_mean_key_scores(index,{static_cast<const float*>(q.data),q.layer,q.rows,
                    q.query_heads,q.kv_heads,q.head_dim,q.attention_scale},frame);
            });
        }
        const auto result=selection.finish(id);const auto expected=oracle(keys,g,query,5,result.frame.score_mask);
        check(result.fixed==std::vector<uint32_t>({0,2,3,6}),"fixed sink/pin/recent priority differs from upstream");
        check(result.selected==result.fixed,"fixed budget was expanded by legacy media");
        for(size_t p=0;p<expected.size();++p)check(std::abs(expected[p]-result.scores[p])<1e-7,"heads/scale/masked denominator differ");
        if(baseline.empty())baseline=result.scores;
        else for(size_t p=0;p<baseline.size();++p)check(std::abs(baseline[p]-result.scores[p])<1e-12,"native global reduction depends on tile size");
    }
    check(policy.media_ranges()==std::vector<std::pair<uint32_t,uint32_t>>{{12,28}},"probe mutated the legacy owner media ranges");
}
void invalid_operations_are_atomic() {
    kvmem::KvMemStore policy(config(1));policy.register_append(12);
    kvmem::RawKvStore keys({4,1,0,4});populate(keys,3);
    const auto id=identity(3,2);const auto g=geometries();auto stamp=id.history;
    kvmem::HostKvStorage metadata(1u<<20);const float query[]{1,1};
    kvmem::LayerwiseProbeSelection s(policy,keys,id,g,[&]{return stamp;},metadata);
    auto foreign=id;++foreign.request;
    rejects([&]{s.append(foreign,id.query.begin,{query,0,1,1,1,1,1});});
    rejects([&]{s.append(id,id.query.begin+1,{query,0,1,1,1,1,1});});
    rejects([&]{s.append(id,id.query.begin,{query,2,1,1,1,1,1});});
    s.append(id,id.query.begin,{query,0,1,1,1,1,1});
    rejects([&]{s.append(id,id.query.begin,{query,2,2,1,1,1,1});});
    rejects([&]{s.finish(id);});check(s.scored_rows()==0 && s.next_layer()==1,"incomplete tile published row coverage");
    kvmem::NativeCurrentLayerQuery native{query,sizeof(float),2,1,1,1,1,1,
        kvmem::NativeQueryScalar::Float32,sizeof(float),sizeof(float),sizeof(float)};
    for(const auto bad:{std::vector<double>{1,2},std::vector<double>{NAN,0,0},std::vector<double>{-1,0,0}})
        rejects([&]{s.append_native(id,id.query.begin,native,[&](const auto&,auto,const auto&){return bad;});});
    check(s.scored_rows()==0 && s.next_layer()==1,"invalid native mass advanced target coverage");
    for(const auto& target:std::vector<kvmem::LayerwiseQueryGeometry>{g[1],g[2]})
        s.append(id,id.query.begin,{query,target.layer,1,1,1,1,1});
    check(s.scored_rows()==1 && s.next_layer()==0,"complete token-major group lost coverage");
    s.append(id,id.query.begin+1,{query,0,1,1,1,1,1});
    native.layer=2;
    rejects([&]{s.append_native(id,id.query.begin+1,native,[&](const auto&,auto,const auto&) {
        ++stamp.history_revision;return std::vector<double>{0,1,0};});});
    check(s.phase()==kvmem::LayerwiseSelectionPhase::Aborted,"changed native source did not abort probe selection");
    rejects([&]{s.finish(id);});
}
void half_max_after_probe_layer_and_query_reduction() {
    auto cfg=config(1);cfg.layerwise_half_max=true;
    kvmem::KvMemStore policy(cfg);policy.register_append(8);
    kvmem::RawKvStore keys({2,1,0,2});
    const float query=1.f;
    for(uint32_t layer=0;layer<2;++layer)
        for(uint32_t half=0;half<4;++half) keys.write_layer_mean_sum(2*half,2,layer,&query);
    const auto id=identity(2,2);
    const std::vector<kvmem::LayerwiseQueryGeometry> g{{0,1,1,1,1},{1,1,1,1,1}};
    kvmem::HostKvStorage metadata(kvmem::LayerwiseProbeSelection::metadata_bound(policy,2,true));
    kvmem::LayerwiseProbeSelection selection(policy,keys,id,g,[&]{return id.history;},metadata);
    for(uint32_t i=0;i<2;++i) for(uint32_t layer=0;layer<2;++layer) {
        kvmem::NativeCurrentLayerQuery tile{&query,sizeof(query),layer,1,1,1,1,1,
            kvmem::NativeQueryScalar::Float32,sizeof(query),sizeof(query),sizeof(query)};
        selection.append_native(id,id.query.begin+i,tile,[&](const auto&,auto,const auto& frame) {
            check(frame.starts==std::vector<uint64_t>({0,2,4,6}),"incomplete probe half-page frame");
            return ((i+layer)%2)?std::vector<double>{.01,.40,.30,.29}:std::vector<double>{.40,.01,.30,.29};
        });
    }
    auto x=selection.finish(id);
    check(x.selected==std::vector<uint32_t>{1} && x.charged_tokens==4 && x.scores.size()==2 &&
          std::abs(x.scores[0]-.41)<1e-12 && std::abs(x.scores[1]-.60)<1e-12,
          "probe took max before layer or query reduction");
}
void construction_and_quota() {
    kvmem::KvMemStore policy(config(1));policy.register_append(12);
    kvmem::RawKvStore keys({4,1,0,4});populate(keys,3);
    const auto id=identity(3,1);const auto g=geometries();auto stamp=id.history;
    const auto bound=kvmem::LayerwiseProbeSelection::metadata_bound(policy,uint32_t(g.size()));
    kvmem::HostKvStorage tiny(bound-1);
    rejects([&]{kvmem::LayerwiseProbeSelection s(policy,keys,id,g,[&]{return stamp;},tiny);});
    check(tiny.charged_bytes()==0,"failed probe metadata admission leaked quota");
    kvmem::HostKvStorage metadata(bound);
    auto bad=g;bad[1]=bad[0];rejects([&]{kvmem::LayerwiseProbeSelection s(policy,keys,id,bad,[&]{return stamp;},metadata);});
    check(metadata.charged_bytes()==0,"invalid geometry leaked metadata");
    bad=g;bad[1].attention_scale=.5f;rejects([&]{kvmem::LayerwiseProbeSelection s(policy,keys,id,bad,[&]{return stamp;},metadata);});
    kvmem::RawKvStore missing({4,1,0,4});
    rejects([&]{kvmem::LayerwiseProbeSelection s(policy,missing,id,g,[&]{return stamp;},metadata);});
    kvmem::RawKvStore partial({4,1,0,4});
    const float sum=1;for(uint32_t l=0;l<4;++l)for(uint32_t p=0;p<3;++p)
        partial.write_layer_mean_sum(p*4,l==2&&p==2?3:4,l,&sum);
    rejects([&]{kvmem::LayerwiseProbeSelection s(policy,partial,id,g,[&]{return stamp;},metadata);});
    {
        kvmem::LayerwiseProbeSelection s(policy,keys,id,g,[&]{return stamp;},metadata);
        check(metadata.charged_bytes()==bound,"exact probe metadata limit was not honored");
        s.abort();rejects([&]{s.finish(id);});
    }
    check(metadata.charged_bytes()==0,"aborted probe retained metadata quota");
    kvmem::LayerwiseProbeHistorySelection retained;
    {
        kvmem::LayerwiseProbeSelection s(policy,keys,id,g,[&]{return stamp;},metadata);
        const float q=1;
        for(const auto& target:g)s.append(id,id.query.begin,{&q,target.layer,1,1,1,1,1});
        retained=s.finish(id);
    }
    check(metadata.charged_bytes()==bound,"returned probe result lost its retained metadata charge");
    retained={};check(metadata.charged_bytes()==0,"retired probe result did not release metadata");
}
}
int main() {
    try {
        global_differs_from_first_and_tiles();masks_pins_heads_scale_and_native();
        invalid_operations_are_atomic();construction_and_quota();half_max_after_probe_layer_and_query_reduction();
        std::cout<<"PASS pinned private-probe global probability oracle, token-major coverage, mask/pin priority, native mass validation and finite metadata lifecycle\n";
    }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
