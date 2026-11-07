#include "memory_test_support.hpp"
#include "kvmem/layerwise_pressure.hpp"
#include "layerwise_pressure_fixture.hpp"
#include "layerwise_pressure_output_fixture.hpp"
#include "layerwise_pressure_parser_reference.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
LayerwisePrefillOptions options() { return {4,8,8,6,2,8.}; }
LayerwisePrefillPressure policy() { return {{42,3},{2,7},options()}; }
LayerwisePrefillAcceptance packet(unsigned step) {
    LayerwisePrefillAcceptance p;
    p.history={true,{42,3},step+1,step+1,4*(step+1),step+1};
    p.count=p.accepted=4; p.rows={1,3}; p.local_begin=step?4:0;
    if (step==0) p.layers={{2,{0},{0},{1.002,1.002}},{7,{0},{0},{.998,.998}}};
    if (step==1) p.layers={{2,{0,4,6},{0,4,4},{.2,.8,0., .1,.3,.6}},
                          {7,{0,4,6},{0,4,4},{.9,.1,0., .8,.1,.1}}};
    if (step==2) p.layers={{2,{0,4},{4,8},{.9,.101, .4,.6}},
                          {7,{0,4},{0,8},{.1,.9, .2,.8}}};
    return p;
}
void close(double actual,double expected) {
    CHECK(std::abs(actual-expected)<=2e-14*std::max(1.,std::abs(expected)));
}
void same(const LayerwisePrefillSnapshot& a,const LayerwisePrefillSnapshot& b) {
    CHECK(a.session==b.session && a.layers.size()==b.layers.size());
    for (size_t l=0;l<a.layers.size();++l) {
        const auto& x=a.layers[l]; const auto& y=b.layers[l];
        CHECK(x.model_layer==y.model_layer && x.end==y.end && x.samples==y.samples && x.chunks==y.chunks);
        CHECK(x.values.size()==y.values.size());
        for (size_t i=0;i<x.values.size();++i) {
            CHECK(x.values[i].original_start==y.values[i].original_start);
            CHECK(x.values[i].mass==y.values[i].mass && x.values[i].exposure==y.values[i].exposure);
        }
    }
}
void oracle(const LayerwisePrefillSnapshot& state,size_t step) {
    CHECK(state.layers.size()==2);
    for (size_t l=0;l<2;++l) {
        const auto& actual=state.layers[l]; const auto& expected=pressure_fixture::snapshots[step][l];
        CHECK(actual.model_layer==(l?7u:2u) && actual.end==4*(step+1));
        CHECK(actual.samples==2*(step+1) && actual.chunks==step+1 && actual.values.size()==expected.size());
        for (size_t i=0;i<expected.size();++i) {
            CHECK(actual.values[i].original_start==expected[i].start);
            close(actual.values[i].mass,expected[i].mass); close(actual.values[i].exposure,expected[i].exposure);
            if (actual.values[i].exposure>0)
                close(actual.values[i].mass/actual.values[i].exposure,expected[i].score);
        }
    }
}
MemoryDescriptor native_descriptor() {
    MemoryDescriptor d;
    d.identity={"pressure-reference","native-v1","two-target-layers","state-v1"};
    d.layout={"opaque-native",1,4,16,4,{
        {2,PayloadPlaneKind::Key,0,4,4,0},{2,PayloadPlaneKind::Value,4,4,4,0},
        {7,PayloadPlaneKind::Key,8,4,4,0},{7,PayloadPlaneKind::Value,12,4,4,0}}};
    d.capabilities={4,1024,true,true,true,true,true,false}; d.validate(); return d;
}
LayerwiseMemoryCatalog catalog(bool all_resident=false,uint32_t tail=4) {
    MemorySnapshot s;
    s.stamp={true,{42,3},9,8,8+tail,11};
    // IDs deliberately disagree with chronological order and original starts.
    s.blocks={{{{42,3},900,0,4,1,1},1},{{{42,3},100,4,4,1,1},1},
              {{{42,3},500,8,tail,1,1},1}};
    s.resident=all_resident?std::vector<uint64_t>{900,100,500}:std::vector<uint64_t>{100,500};
    std::vector<LayerMemorySnapshot> layers{{2,s}};
    s.resident=all_resident?std::vector<uint64_t>{900,100,500}:std::vector<uint64_t>{900,500};
    ++s.stamp.view_version; layers.push_back({7,s});
    return {native_descriptor(),layers};
}

void pinned_python_oracle_and_candidates() {
    auto p=policy();
    for (unsigned step=0;step<3;++step) {
        auto input=packet(step);
        CHECK(p.rows(input.history.frontier-input.count,4,input.local_begin)==input.rows);
        CHECK(p.accept(input)); oracle(p.snapshot(),step);
        if (step==1) {
            // At the previous accepted frontier, both blocks are resident.
            MemorySnapshot s; s.stamp={true,{42,3},2,2,8,2};
            s.blocks={{{{42,3},900,0,4,1,1},1},{{{42,3},100,4,4,1,1},1}};
            s.resident={900,100};
            const LayerwiseMemoryCatalog c(native_descriptor(),{{2,s},{7,s}});
            const auto choices=p.select(c,{},4);
            CHECK(choices[0].selected==std::vector<uint64_t>{100});
            CHECK(choices[1].selected==std::vector<uint64_t>{900});
        }
    }
    // The dictionary retains only current physical history keys, as upstream.
    CHECK(p.snapshot().layers[0].values[0].original_start==4);
    CHECK(p.snapshot().layers[1].values[0].original_start==0);
    const auto choices=p.select(catalog(),{},4);
    CHECK(choices.size()==2 && choices[0].candidates==std::vector<uint64_t>({100,500}));
    CHECK(choices[1].candidates==std::vector<uint64_t>({900,500}));
    CHECK(choices[0].selected==std::vector<uint64_t>{100});
    CHECK(choices[1].selected==std::vector<uint64_t>{500});
    for (const auto& c:choices) {
        CHECK(c.selected_tokens==4 && c.charged_tokens==4);
        CHECK(c.accepted_score_end==12 && c.samples==6 && c.chunks==3);
    }
    const auto before=p.snapshot();
    const auto fixed=p.select(catalog(true),{500},8,1);
    for (const auto& c:fixed) {
        CHECK(c.fixed==std::vector<uint64_t>({900,500}));
        CHECK(c.selected==std::vector<uint64_t>({900,500}));
    }
    same(before,p.snapshot()); // selection never decays/finishes history
    rejects(MemoryErrorCode::Stale,[&]{p.select(catalog(),{100},8);});
    rejects(MemoryErrorCode::Stale,[&]{p.select(catalog(),{},8,1);});
    rejects(MemoryErrorCode::BudgetExceeded,[&]{p.select(catalog(true),{100,500},4);});
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.select(catalog(),{},12);});
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.select(catalog(),{500,500},8);});

    // A high score for cold native ID900 never joins layer2's candidates.
    auto snapshot=p.snapshot();
    snapshot.layers[0].values.insert(snapshot.layers[0].values.begin(),{0,1000,1});
    p.restore(snapshot,12);
    CHECK(p.select(catalog(),{},4)[0].selected==std::vector<uint64_t>{100});
    snapshot=p.snapshot();
    for (auto& layer:snapshot.layers) for (auto& v:layer.values) v.mass=.5*v.exposure;
    p.restore(snapshot,12);
    const auto ties=p.select(catalog(),{},4);
    CHECK(ties[0].selected==std::vector<uint64_t>{500} && ties[1].selected==std::vector<uint64_t>{500});
    // All-zero independent attention must fail rather than change algorithms.
    for (auto& layer:snapshot.layers) for (auto& v:layer.values) v.mass=0;
    p.restore(snapshot,12);
    rejects(MemoryErrorCode::Pending,[&]{p.select(catalog(),{},4);});
}

void sampling_and_rewind() {
    auto p=policy(); for (unsigned step=0;step<3;++step) CHECK(p.accept(packet(step)));
    CHECK(p.rows(12,1,0).empty());
    CHECK(p.rows(12,4,3).empty()); // remaining nine > threshold eight, interval not reached
    CHECK(p.rows(12,4,4)==std::vector<uint32_t>({1,3})); // equality is near capacity
    CHECK(p.rows(14,4,0)==std::vector<uint32_t>({1,3})); // exact interval boundary
    CHECK(p.rows(12,4,100)==std::vector<uint32_t>({1,3})); // signed capacity arithmetic
    CHECK(p.rows(12,4,4,true).empty());
    CHECK(p.rows(8,4,4).empty() && p.rows(12,0,0).empty());
    auto zero=options(); zero.stride=0;
    LayerwisePrefillPressure last({42,3},{2,7},zero);
    CHECK(last.rows(0,4,0)==std::vector<uint32_t>{3});
    zero.stride=128; LayerwisePrefillPressure sparse({42,3},{2,7},zero);
    CHECK(sparse.rows(0,4,0)==std::vector<uint32_t>{3});
    auto invalid_options=options(); invalid_options.interval=0;
    rejects(MemoryErrorCode::InvalidPlan,[&]{LayerwisePrefillPressure({42,3},{2,7},invalid_options);});
    const auto long_query=p.rows(12,131,0);
    CHECK(long_query.size()==66 && long_query.front()==0 && long_query.back()==130);
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.rows(UINT64_MAX,4,0);});
    const auto snapshot=p.snapshot();
    p.restore(snapshot,11);
    for (const auto& layer:p.snapshot().layers)
        CHECK(layer.values.empty() && layer.end==0 && layer.samples==0 && layer.chunks==0);
    p.restore(snapshot,12); same(p.snapshot(),snapshot);
    auto invalid=snapshot; ++invalid.session.incarnation;
    rejects(MemoryErrorCode::Stale,[&]{p.restore(invalid,12);});
    invalid=snapshot; --invalid.layers[1].end;
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.restore(invalid,12);});
    invalid=snapshot; invalid.layers[1].values[0].exposure=std::numeric_limits<double>::infinity();
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.restore(invalid,12);});
    same(p.snapshot(),snapshot);
}

void atomic_acceptance_and_raw_mass() {
    auto p=policy(); const auto empty=p.snapshot(); auto input=packet(0);
    auto invalid=input; invalid.accepted=3;
    CHECK(!p.accept(invalid)); same(p.snapshot(),empty);
    invalid=input; invalid.rows.clear();
    CHECK(!p.accept(invalid)); same(p.snapshot(),empty);
    invalid=input; invalid.layers.pop_back();
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.accept(invalid);});
    invalid=input; invalid.layers[1].model_layer=99; // includes a draft/foreign layer
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.accept(invalid);});
    invalid=input; invalid.layers[1].model_layer=2;
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.accept(invalid);});
    invalid=input; invalid.layers[1].masses.pop_back();
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.accept(invalid);});
    invalid=input; invalid.layers[1].masses[1]=std::numeric_limits<double>::quiet_NaN();
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.accept(invalid);});
    invalid=input; invalid.layers[1].masses[1]=-.001;
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.accept(invalid);});
    invalid=input; invalid.layers[1].masses[1]=1.00500001;
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.accept(invalid);});
    invalid=input; invalid.rows={0,3};
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.accept(invalid);});
    invalid=input; ++invalid.history.session.incarnation;
    rejects(MemoryErrorCode::Stale,[&]{p.accept(invalid);});
    same(p.snapshot(),empty); // even a late layer failure publishes nothing
    CHECK(p.accept(input)); oracle(p.snapshot(),0);
    CHECK(p.snapshot().layers[0].values[0].mass==1.002); // tolerance is NOT renormalization
    const auto accepted=p.snapshot(); CHECK(!p.accept(input)); same(p.snapshot(),accepted);
    auto threshold=policy(); input=packet(0);
    for (auto& layer:input.layers) layer.masses={1.005,1.005};
    CHECK(threshold.accept(input));
    close(threshold.snapshot().layers[0].values[0].mass,1.005);
}

void unequal_partial_geometry_rejected() {
    auto p=policy(); CHECK(p.accept(packet(0))); CHECK(p.accept(packet(1)));
    auto input=packet(1); input.history.frontier=10; ++input.history.history_revision;
    input.count=input.accepted=2; input.local_begin=8; input.rows={1};
    input.layers={{2,{0,4,8},{0,4,8},{.1,.8,.1}}, {7,{0,4,8},{0,4,8},{0,0,1}}};
    CHECK(p.accept(input));
    // Both native maps contain the same four/four/two rows before Pressure.
    // Layer2 picks a full block; layer7 picks the new partial tail. The pinned
    // experiment refuses this unequal causal length rather than silently pad.
    rejects(MemoryErrorCode::InvalidPlan,[&]{p.select(catalog(true,2),{},4);});
}

const std::array<int32_t,5> markers{{10,11,12,13,14}};
MemoryStamp stamp(uint64_t end) { return {true,{42,3},20,20,end,20}; }
LayerwiseOutputPressure output_policy(uint32_t interval=2,
    LayerwiseOutputScope scope=LayerwiseOutputScope::NonThinking,uint64_t limit=16777216) {
    return {{42,3},{2,7},options(),markers,{scope,interval,.5,limit}};
}
void same_output(const LayerwiseOutputSnapshot& a,const LayerwiseOutputSnapshot& b) {
    CHECK(a.session==b.session && a.request==b.request && a.accepted_end==b.accepted_end && a.active==b.active);
    CHECK(a.parser.thinking==b.parser.thinking && a.parser.action==b.parser.action && a.parser.tokens==b.parser.tokens);
    CHECK(a.layers.size()==b.layers.size());
    for (size_t i=0;i<a.layers.size();++i) {
        const auto& x=a.layers[i]; const auto& y=b.layers[i];
        CHECK(x.model_layer==y.model_layer && x.samples==y.samples && x.rounds==y.rounds);
        CHECK(x.scores==y.scores && x.pending==y.pending);
    }
}
LayerwiseOutputAcceptance output_packet(uint32_t accepted=1) {
    LayerwiseOutputAcceptance p;
    p.history=stamp(8+accepted); p.request=1; p.accepted=accepted; p.rows={0,1,2};
    p.layers={{2,{0,4,4},{.501,.25,.25, 100,0,0, .1,.4,.5}},
              {7,{0,4,4},{.1,.4,.5, 100,0,0, .8,.1,.1}}};
    return p;
}
double value(const std::vector<std::pair<uint64_t,double>>& values,uint64_t key) {
    for (const auto& v:values) if (v.first==key) return v.second;
    return 0;
}
void output_preview(const std::vector<LayerwisePressureChoice>& choices,const double expected[2][3]) {
    for (size_t i=0;i<2;++i) {
        CHECK(choices[i].source==LayerwisePressureSource::Output);
        CHECK(choices[i].candidates==std::vector<uint64_t>({900,100,500}));
        CHECK(choices[i].scores.size()==3);
        for (size_t c=0;c<3;++c) close(choices[i].scores[c],expected[i][c]);
    }
}
void pinned_output_mass_oracle() {
    auto output=output_policy(); auto prefill=policy();
    output.begin(stamp(8),1,{14});
    CHECK(output.prepare(1,8,{42,42,42})==5); // first, then third consumed token
    CHECK(output.accept(output_packet())==1);
    for (size_t l=0;l<2;++l) {
        const auto& layer=output.snapshot().layers[l]; CHECK(layer.samples==1);
        for (size_t c=0;c<3;++c) close(value(layer.pending,4*c),pressure_fixture::output_pending_first[l][c]);
    }
    auto repeated=output_packet(); repeated.rows.clear(); repeated.layers.clear();
    const auto first=output.snapshot(); CHECK(output.accept(repeated)==0); same_output(first,output.snapshot());
    CHECK(output.accept(output_packet(3))==4);
    for (size_t l=0;l<2;++l) {
        const auto& layer=output.snapshot().layers[l]; CHECK(layer.samples==2 && layer.rounds==0);
        for (size_t c=0;c<3;++c) close(value(layer.pending,4*c),pressure_fixture::output_pending_second[l][c]);
    }
    output_preview(output.select(catalog(true,3),prefill,{},8),pressure_fixture::output_before_finish_preview);
    const auto pending=output.snapshot();
    output.finish(1,catalog(true,3));
    for (size_t l=0;l<2;++l) {
        const auto& layer=output.snapshot().layers[l];
        CHECK(layer.pending.empty() && layer.samples==2 && layer.rounds==1);
        for (size_t c=0;c<3;++c) close(value(layer.scores,4*c),pressure_fixture::output_finished[l][c]);
    }
    // Source-defined closed-window interval retains N until actual next begin.
    output_preview(output.select(catalog(true,3),prefill,{},8),pressure_fixture::output_closed_preview);
    const auto closed=output.snapshot(); output.begin(stamp(11),1,{14});
    same_output(closed,output.snapshot()); // same finished ID is still a no-op
    output.begin(stamp(11),2,{14});
    output_preview(output.select(catalog(true,3),prefill,{},8),pressure_fixture::output_next_begin_preview);
    CHECK(output.prepare(2,11,{42})==1);
    LayerwiseOutputAcceptance next{stamp(12),2,1,{0},{{2,{0,4},{1,0}},{7,{0,4},{0,1}}}};
    CHECK(output.accept(next)==1);
    output_preview(output.select(catalog(true),prefill,{},8),pressure_fixture::output_second_window_preview);
    output.finish(2,catalog(true));
    for (size_t l=0;l<2;++l) for (size_t c=0;c<3;++c)
        close(value(output.snapshot().layers[l].scores,4*c),pressure_fixture::output_second_finished[l][c]);
    output.begin(stamp(12),3,{14}); const auto unsampled=output.snapshot();
    output.finish(3,catalog(true));
    for (size_t l=0;l<2;++l) {
        CHECK(output.snapshot().layers[l].samples==0);
        CHECK(output.snapshot().layers[l].rounds==unsampled.layers[l].rounds);
        CHECK(output.snapshot().layers[l].scores==unsampled.layers[l].scores);
    }

    // Accepted checkpoint includes pending mass and the committed parser; a
    // speculative prepared candidate is deliberately discarded on restore.
    output.restore(pending,11); same_output(output.snapshot(),pending);
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(output_packet(3));});
    CHECK(output.prepare(1,11,{42,42})==2); // committed eligible count was three
    auto foreign=pending; ++foreign.session.incarnation;
    rejects(MemoryErrorCode::Stale,[&]{output.restore(foreign,11);});
    rejects(MemoryErrorCode::Stale,[&]{output.restore(pending,10);});
    auto bad=pending; ++bad.layers[1].samples;
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.restore(bad,11);});
    same_output(output.snapshot(),pending);
}

void native_parser_oracle() {
    // This test compiles the UNMODIFIED pinned upstream parser header and runs
    // it directly. Expected masks are not a second implementation in the test.
    using Native=qw3::detail::ActionAttentionParser;
    const std::vector<uint32_t> tokens{12,42,13,10,42,42,42,11,42,14,42,10,42,12,42,13,42,11,42,42,42,42};
    for (auto scope:{LayerwiseOutputScope::ToolCall,LayerwiseOutputScope::NonThinking})
        for (uint32_t interval:{1u,3u,16u}) for (uint32_t chunk:{1u,3u,8u,64u}) {
            auto output=output_policy(interval,scope); output.begin(stamp(8),1,{12,14});
            Native native; native.scope=scope==LayerwiseOutputScope::ToolCall?Native::Scope::ToolCall:Native::Scope::NonThinking;
            native.open=10;native.close=11;native.think_open=12;native.think_close=13;native.interval=interval;
            uint32_t first=0;
            while (first<tokens.size()) {
                const uint32_t n=std::min<uint32_t>(chunk,uint32_t(tokens.size())-first);
                std::vector<uint32_t> batch(tokens.begin()+first,tokens.begin()+first+n);
                native.prepare(batch.data(),n);
                CHECK(output.prepare(1,8+first,batch)==native.mask);
                LayerwiseOutputAcceptance input{stamp(8+first+n),1,n,{},{}};
                for (uint32_t row=0;row<n;++row) input.rows.push_back(row);
                input.layers={{2,{0},std::vector<double>(n,1.)},{7,{0},std::vector<double>(n,1.)}};
                const auto expected=native.commit(n);
                CHECK(output.accept(input)==expected);
                const auto& state=output.snapshot().parser;
                CHECK(state.thinking==native.committed.thinking && state.action==native.committed.action && state.tokens==native.committed.tokens);
                first+=n;
            }
        }
    auto max=output_policy(1); max.begin(stamp(0),1,{14});
    CHECK(max.prepare(1,0,std::vector<uint32_t>(64,42))==UINT64_MAX);
    LayerwiseOutputAcceptance full{stamp(64),1,64,{},{}};
    for (uint32_t r=0;r<64;++r) full.rows.push_back(r);
    full.layers={{2,{0},std::vector<double>(64,1.)},{7,{0},std::vector<double>(64,1.)}};
    CHECK(max.accept(full)==UINT64_MAX && max.snapshot().layers[0].samples==64);
    const auto before=max.snapshot();
    rejects(MemoryErrorCode::InvalidPlan,[&]{max.prepare(1,64,std::vector<uint32_t>(65,42));});
    same_output(before,max.snapshot());

    // Rejected candidate think-open must not leak into the committed scope.
    auto partial=output_policy(2); partial.begin(stamp(8),1,{14});
    CHECK(partial.prepare(1,8,{42,12,42})==1);
    LayerwiseOutputAcceptance one{stamp(9),1,1,{0},{{2,{0},{1}},{7,{0},{1}}}};
    CHECK(partial.accept(one)==1 && !partial.snapshot().parser.thinking);
    CHECK(partial.prepare(1,9,{42,42})==2);
    // The actual prompt's final marker controls initial state; message reset
    // and explicit think-open are caller-supplied schema, never token guesses.
    auto thinking=output_policy(); thinking.begin(stamp(8),1,{14,12});
    CHECK(thinking.prepare(1,8,{42,13,42})==4);
}

void output_atomic_failures_and_fallback() {
    auto output=output_policy(); output.begin(stamp(8),1,{14}); output.prepare(1,8,{42,42,42});
    const auto before=output.snapshot(); auto input=output_packet();
    auto bad=input; bad.layers.pop_back();
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(bad);});
    bad=input; bad.layers[1].model_layer=99;
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(bad);});
    bad=input; bad.layers[1].masses[0]=std::numeric_limits<double>::quiet_NaN();
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(bad);});
    bad=input; bad.layers[1].masses[0]=.106; // accepted sum exceeds .005 tolerance
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(bad);});
    bad=input; bad.rows={2};
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(bad);});
    bad=input; bad.rows={0,0,2};
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(bad);});
    bad=input; ++bad.history.frontier;
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(bad);});
    bad=input; ++bad.request;
    rejects(MemoryErrorCode::Stale,[&]{output.accept(bad);});
    same_output(before,output.snapshot());
    // Native validation reads only newly accepted sampled rows. Both the
    // unsampled middle row and rejected sampled suffix may contain junk.
    input.layers[0].masses[3]=std::numeric_limits<double>::quiet_NaN();
    input.layers[1].masses[6]=std::numeric_limits<double>::quiet_NaN();
    CHECK(output.accept(input)==1);
    const auto first=output.snapshot(); input.accepted=3; input.history=stamp(11);
    rejects(MemoryErrorCode::InvalidPlan,[&]{output.accept(input);});
    same_output(first,output.snapshot());
    CHECK(output.accept(output_packet(3))==4); // complete retry after late-layer validation failure
    auto small=output_policy(2,LayerwiseOutputScope::NonThinking,5055);
    small.begin(stamp(8),1,{14}); small.prepare(1,8,{42,42,42});
    rejects(MemoryErrorCode::BudgetExceeded,[&]{small.accept(output_packet());});
    CHECK(small.snapshot().parser.tokens==0 && small.snapshot().layers[0].samples==0);
    auto missing=markers; missing[0]=-1;
    rejects(MemoryErrorCode::Unsupported,[&]{LayerwiseOutputPressure({42,3},{2,7},options(),missing);});

    auto prefill=policy(); for (unsigned i=0;i<3;++i) CHECK(prefill.accept(packet(i)));
    auto fallback=output_policy();
    CHECK(fallback.select(catalog(),prefill,{},4)[0].source==LayerwisePressureSource::Prefill);
    auto snapshot=fallback.snapshot(); snapshot.accepted_end=12;
    // Output mass on a cold block is not evidence for this layer's seed choice.
    snapshot.layers[0].scores={{0,1000}}; snapshot.layers[1].scores={{4,1000}};
    fallback.restore(snapshot,12);
    const auto cold=fallback.select(catalog(),prefill,{},4);
    CHECK(cold[0].source==LayerwisePressureSource::Prefill && cold[0].selected==std::vector<uint64_t>{100});
    CHECK(cold[1].source==LayerwisePressureSource::Prefill && cold[1].selected==std::vector<uint64_t>{500});
    // A positive fixed resident sink establishes the output seed BEFORE the
    // fixed mask: zero remaining output scores use resident recency, not prefill.
    for (auto& layer:snapshot.layers) layer.scores={{0,1}};
    fallback.restore(snapshot,12);
    const auto fixed=fallback.select(catalog(true),prefill,{},8,1);
    for (const auto& c:fixed) {
        CHECK(c.source==LayerwisePressureSource::Output);
        CHECK(c.selected==std::vector<uint64_t>({900,500}));
    }
}

void output_request_prefill_boundary() {
    auto output=output_policy();
    output.begin(stamp(8),1,{14});
    auto expected=output.snapshot();expected.accepted_end=11;
    output.accept_prefill(stamp(11),1,8);
    same_output(expected,output.snapshot());
    rejects(MemoryErrorCode::Stale,[&]{output.accept_prefill(stamp(12),1,8);});
    rejects(MemoryErrorCode::Stale,[&]{output.accept_prefill(stamp(12),2,11);});
    auto foreign=stamp(12);++foreign.session.incarnation;
    rejects(MemoryErrorCode::Stale,[&]{output.accept_prefill(foreign,1,11);});
    output.prepare(1,11,{42});
    rejects(MemoryErrorCode::Stale,[&]{output.accept_prefill(stamp(12),1,11);});
    same_output(expected,output.snapshot());
}
} // namespace

int main() {
    pinned_python_oracle_and_candidates(); sampling_and_rewind();
    atomic_acceptance_and_raw_mass(); unequal_partial_geometry_rejected();
    pinned_output_mass_oracle(); native_parser_oracle(); output_atomic_failures_and_fallback();
    output_request_prefill_boundary();
    std::puts("layerwise Pressure: pinned prefill/output fixtures, native scope parser oracle, atomic accept and resident candidates passed");
}
