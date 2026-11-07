#include "memory_test_support.hpp"
#include "kvmem/layerwise_pressure.hpp"
#include <algorithm>
#include <cmath>
#include <string>

namespace {
LayerwiseOutputGrammarConfig config() {return {100,101,105,106,48,49,"thought",64};}
const std::vector<LayerwiseOutputGrammarToken> stream{
    {105,"<|turn>"},{200,"mo"},{201,"del\n"},{100,"<|channel>"},{202,"tho"},{203,"ught"},
    {204,"\n"},{205,"secret"},{101,"<channel|>"},{206,"visibleA"},{48,"<|tool_call>"},
    {207,"arg1"},{208,"arg2"},{49,"<tool_call|>"},{209,"visibleB"},{105,"<|turn>"},{210,"model"}};
MemoryStamp stamp(uint64_t end) {return {true,{42,3},1,1,end,1};}
LayerwiseMemoryCatalog catalog(uint64_t end) {
    MemoryDescriptor d;d.identity={"scope-test","native-v1","one-layer","state-v1"};
    d.layout={"opaque",1,4,8,4,{{7,PayloadPlaneKind::Key,0,4,4,0},{7,PayloadPlaneKind::Value,4,4,4,0}}};
    d.capabilities={4,1024,true,true,true,true,true,false};d.validate();
    MemorySnapshot memory;memory.stamp=stamp(end);
    for(uint64_t at=0;at<end;at+=4) {
        const auto count=uint32_t(std::min<uint64_t>(4,end-at));
        memory.blocks.push_back({{{42,3},100+at,at,count,1,1},1});memory.resident.push_back(100+at);
    }
    return {d,{{7,memory}}};
}
double score(const LayerwiseOutputSnapshot& s,uint64_t key) {
    const auto& values=s.layers[0].scores;
    const auto found=std::find_if(values.begin(),values.end(),[&](const auto& x){return x.first==key;});
    return found==values.end()?0.:found->second;
}
std::vector<uint32_t> sampled(uint64_t mask,size_t count) {
    std::vector<uint32_t> result;
    for(uint32_t row=0;row<count;++row)if((mask>>row)&1)result.push_back(row);
    return result;
}
LayerwiseOutputAcceptance packet(uint64_t begin,uint32_t accepted,const std::vector<uint32_t>& rows,size_t global) {
    LayerwiseOutputAcceptance p;p.history=stamp(begin+accepted);p.request=9;p.accepted=accepted;p.rows=rows;
    LayerwiseOutputMass layer;layer.model_layer=7;layer.history={0,4};
    for(auto row:rows) {
        const auto at=global+row;
        const auto a=at==9?.25:at==11?.8:.5;
        layer.masses.push_back(a);layer.masses.push_back(1.-a);
    }
    p.layers.push_back(std::move(layer));return p;
}
void chunk_and_clock() {
    for(auto scope:{LayerwiseOutputScope::ToolCall,LayerwiseOutputScope::NonThinking})
    for(size_t tile:{1u,2u,3u,5u,16u,64u}) {
        LayerwiseOutputGrammar grammar(config());
        LayerwiseOutputOptions options;options.scope=scope;options.interval=2;
        LayerwiseOutputPressure output({42,3},{7},{4,8,8,6,2,8.},LayerwiseExternalOutputScope{},options);
        output.begin_scope(stamp(8),9,grammar.snapshot().scope);
        uint64_t global_mask=0;
        for(size_t at=0;at<stream.size();) {
            const auto count=std::min(tile,stream.size()-at);
            std::vector<LayerwiseOutputGrammarToken> tokens(stream.begin()+at,stream.begin()+at+count);
            const auto semantic=grammar.prepare(tokens);const auto mask=output.prepare_scope(9,8+at,semantic);
            global_mask|=mask<<at;
            const auto p=packet(8+at,uint32_t(count),sampled(mask,count),at);
            CHECK(output.accept(p)==mask);grammar.accept(uint32_t(count));
            CHECK(output.accept(p)==0);grammar.accept(uint32_t(count));at+=count;
        }
        const bool all=scope==LayerwiseOutputScope::NonThinking;
        CHECK(global_mask==(all?((uint64_t(1)<<9)|(uint64_t(1)<<12)):(uint64_t(1)<<11)));
        CHECK(output.snapshot().parser.tokens==(all?4u:2u));
        CHECK(output.snapshot().layers[0].samples==(all?2u:1u));
        CHECK(grammar.snapshot().header==LayerwiseOutputHeader::Turn);
        output.finish(9,catalog(8+stream.size()));
        CHECK(std::abs(score(output.snapshot(),0)-(all?.375:.8))<1e-14);
        CHECK(std::abs(score(output.snapshot(),4)-(all?.625:.2))<1e-14);
    }
}
void accepted_prefix_and_prompt() {
    for(uint32_t accepted:{0u,1u,4u,6u,7u,8u,10u,12u,14u,17u}) {
        LayerwiseOutputGrammar grammar(config());
        LayerwiseOutputOptions options;options.scope=LayerwiseOutputScope::NonThinking;options.interval=2;
        LayerwiseOutputPressure output({42,3},{7},{4,8,8,6,2,8.},LayerwiseExternalOutputScope{},options);
        output.begin_scope(stamp(8),9,{});
        const auto rows=grammar.prepare(stream);const auto mask=output.prepare_scope(9,8,rows);
        const auto p=packet(8,accepted,sampled(mask,stream.size()),0);
        output.accept(p);grammar.accept(accepted);
        LayerwiseOutputGrammar serial(config());
        for(uint32_t i=0;i<accepted;++i)serial.prompt(stream[i]);
        const auto x=grammar.snapshot(),y=serial.snapshot();
        CHECK(x.header==y.header && x.matched==y.matched && x.mismatch==y.mismatch);
        CHECK(x.scope.thinking==y.scope.thinking && x.scope.tool_call==y.scope.tool_call);
        CHECK(output.snapshot().parser.tokens==uint64_t(accepted>9)+uint64_t(accepted>11)+
            uint64_t(accepted>12)+uint64_t(accepted>14));
        CHECK(output.snapshot().layers[0].samples==uint64_t(accepted>9)+uint64_t(accepted>12));
        auto invalid=p;invalid.rows.clear();
        const auto before=grammar.snapshot();
        rejects(MemoryErrorCode::InvalidPlan,[&]{grammar.accept(accepted?accepted-1:uint32_t(stream.size()+1));});
        CHECK(grammar.snapshot().header==before.header);
    }
    LayerwiseOutputGrammar grammar(config());
    grammar.prompt({100,"<|channel>"});grammar.prompt({200,"tho"});
    const auto partial=grammar.snapshot();CHECK(partial.header==LayerwiseOutputHeader::Channel);
    grammar.prepare({{201,"ught\n"},{202,"secret"},{101,"<channel|>"}});grammar.accept(0);
    CHECK(grammar.snapshot().matched==3 && !grammar.snapshot().scope.thinking);
    grammar.restore(partial);
    const auto near=grammar.prepare({{201,"ughtful\n"},{202,"ordinary"}});
    CHECK(!near[0].body && !near[0].thinking && near[1].body && !near[1].thinking);
    grammar.accept(2);
}
void bounds_and_modes() {
    auto bad=config();bad.tool_close=bad.tool_open;
    rejects(MemoryErrorCode::InvalidPlan,[&]{LayerwiseOutputGrammar x(bad);});
    LayerwiseOutputGrammar grammar(config());const auto bytes=grammar.metadata_bytes();
    grammar.prompt({105,"<|turn>"});
    for(unsigned i=0;i<10000;++i)grammar.prompt({200,"junk"});
    CHECK(grammar.snapshot().header==LayerwiseOutputHeader::Turn && grammar.metadata_bytes()==bytes);
    rejects(MemoryErrorCode::InvalidPlan,[&]{grammar.prepare(std::vector<LayerwiseOutputGrammarToken>(65,{200,"a"}));});
    const std::string large(65,'a');
    rejects(MemoryErrorCode::InvalidPlan,[&]{grammar.prompt({200,large});});
    auto invalid=grammar.snapshot();invalid.matched=999;
    rejects(MemoryErrorCode::InvalidPlan,[&]{grammar.restore(invalid);});
    LayerwiseOutputPressure external({42,3},{7},{4,8,8,6,2,8.},LayerwiseExternalOutputScope{});
    rejects(MemoryErrorCode::Unsupported,[&]{external.begin(stamp(8),9,{});});
    external.begin_scope(stamp(8),9,{});
    rejects(MemoryErrorCode::Unsupported,[&]{external.prepare(9,8,{1});});
    rejects(MemoryErrorCode::InvalidPlan,[&]{external.prepare_scope(9,8,{{true,false,true}});});
    LayerwiseOutputPressure markers({42,3},{7},{4,8,8,6,2,8.},std::array<int32_t,5>{1,2,3,4,5});
    rejects(MemoryErrorCode::Unsupported,[&]{markers.begin_scope(stamp(8),9,{});});
    rejects(MemoryErrorCode::Unsupported,[&]{markers.prepare_scope(9,8,{});});
}
}
int main() {
    chunk_and_clock();accepted_prefix_and_prompt();bounds_and_modes();
    std::puts("PASS native output grammar, accepted header/body clock, prefix rollback, chunk invariance and exact S/M/N");
}
