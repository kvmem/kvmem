#include "kvmem/graph_recall.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/kvmem_store.hpp"
#include "kvmem/cli_options.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void close(double actual, double expected, double tolerance = 1e-12) {
    require(std::abs(actual-expected) <= tolerance, "independent reference mismatch");
}
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid transaction was accepted");
}
void frames_and_scores() {
    auto frame = kvmem::query_score_frame(12, 4, 2, 0, 0, {2});
    require(frame.score_mask == std::vector<uint8_t>({1,1,1}), "mandatory changed softmax denominator");
    require(frame.transport_mask == std::vector<uint8_t>({1,1,0}), "mandatory transported graph mass");
    auto small = kvmem::query_score_frame(11,4,4,1,1,{0,2});
    require(small.score_mask == std::vector<uint8_t>({1,1,1}) &&
            small.transport_mask == std::vector<uint8_t>({0,0,0}) && small.counts.back() == 3,
            "small-history or partial-tail domain mismatch");
    auto constrained = kvmem::query_score_frame(12,4,2,1,1,{0,2});
    require(constrained.score_mask == std::vector<uint8_t>({0,1,0}), "keep-band score mask mismatch");
    kvmem::RawKvStore keys({1, 2, 0, 4});
    const float a[]{8,0}, b[]{0,4}, c[]{40,0};
    keys.write_layer_mean_sum(0,4,0,a);
    keys.write_layer_mean_sum(4,4,0,b);
    keys.write_layer_mean_sum(8,4,0,c);
    const float q[]{2,0, 0,2}; const uint32_t rows[]{2};
    const auto result = kvmem::page_softmax_scores(keys,{q,rows,1,2,1,1,2},frame);
    // Naive long-double reference, independent softmax and exact FP16 operands.
    long double expected[3]{};
    const long double logits[2][3]{{4/std::sqrt(2.L),0,20/std::sqrt(2.L)},
                                  {0,2/std::sqrt(2.L),0}};
    for (const auto& line : logits) {
        long double denominator = 0;
        for (auto value : line) denominator += std::exp(value);
        for (size_t i=0;i<3;++i) expected[i] += std::exp(line[i])/denominator;
    }
    for (size_t i=0;i<3;++i) close(result[i],double(expected[i]));
    require(result[1] > result[0], "mandatory-domain ranking regression");
    frame.score_mask[2] = 0;
    const auto wrong_domain = kvmem::page_softmax_scores(keys,{q,rows,1,2,1,1,2},frame);
    require(wrong_domain[0] > wrong_domain[1], "fixture does not discriminate the score domain");
    close(std::accumulate(result.begin(),result.end(),0.),2);
    frame.score_mask.assign(3,0);
    const auto empty = kvmem::page_softmax_scores(keys,{q,rows,1,2,1,1,2},frame);
    close(std::accumulate(empty.begin(),empty.end(),0.),0);
    close(kvmem::recall_fp16(1.00048828125f),1);
    close(kvmem::recall_fp16(1.00146484375f),1.001953125);
    close(kvmem::recall_fp16(std::ldexp(1.f,-24)),std::ldexp(1.,-24));
    close(kvmem::recall_fp16(std::ldexp(1.f,-25)),0);
    rejects([] { kvmem::recall_fp16(65536); });
}
void gqa_and_large_domain() {
    kvmem::RawKvStore keys({2,1,0,1});
    for (uint32_t l=0;l<2;++l) {
        const float a[]{1}, b[]{-1};
        keys.write_layer_mean_sum(0,1,l,a); keys.write_layer_mean_sum(1,1,l,b);
    }
    auto frame = kvmem::query_score_frame(2,1,1,0,0,{});
    const float q[]{1,-1, 2,-2}; const uint32_t rows[]{1,1};
    const auto result=kvmem::page_softmax_scores(keys,{q,rows,2,1,2,1,1},frame);
    close(result[0],.5); close(result[1],.5);
    // Boundary sizes must all normalize over the complete history.
    for (const uint32_t n : {8192u,8193u,32768u,32769u}) {
        kvmem::RawKvStore large({1,1,0,1});
        const float value[]{1};
        for (uint32_t b=0;b<n;++b) large.write_layer_mean_sum(b,1,0,value);
        auto f=kvmem::query_score_frame(n,1,1,0,0,{});
        const float query[]{1}; const uint32_t count[]{1};
        auto scores=kvmem::page_softmax_scores(large,{query,count,1,1,1,1,1},f);
        close(scores.front(),1./n); close(scores.back(),1./n);
        close(std::accumulate(scores.begin(),scores.end(),0.),1,2e-12);
    }
}
void scorer_variants() {
    kvmem::RawKvStore keys({1,2,0,1});
    const float k0[]{1,0},k1[]{0,1},k2[]{-1,-1};
    keys.write_layer_mean_sum(0,1,0,k0); keys.write_layer_mean_sum(1,1,0,k1); keys.write_layer_mean_sum(2,1,0,k2);
    const float q[]{2,0,0,2, 2,0,0,2}; const uint32_t counts[]{2};
    kvmem::PageQueryView view{q,counts,1,2,2,1,2};
    const auto frame=kvmem::query_score_frame(3,1,2,0,0,{});
    // Independent normalized-max oracle: two heads have mirrored distributions.
    const double a=std::exp(std::sqrt(2.)),b=1,c=std::exp(-std::sqrt(2.));
    const double maximum=a/(a+b+c),negative=c/(a+b+c),mass=2*maximum+negative;
    auto scores=kvmem::page_softmax_scores(keys,view,frame,kvmem::GqaScoreAggregation::Max);
    close(scores[0],2*maximum/mass);close(scores[1],2*maximum/mass);close(scores[2],2*negative/mass);
    scores=kvmem::mean_query_cosine_scores(keys,view,3,kvmem::GqaScoreAggregation::Max);
    close(scores[0],1,1e-6);close(scores[1],1,1e-6);close(scores[2],-1/std::sqrt(2.),1e-6);
    const auto mean=kvmem::mean_query_cosine_scores(keys,view,3,kvmem::GqaScoreAggregation::Mean);
    close(mean[0],1/std::sqrt(2.),1e-6);close(mean[2],-1,1e-6);
    kvmem::KvMemStoreConfig cfg;cfg.block_tokens=1;cfg.select_budget=2;cfg.sink_blocks=0;cfg.recent_blocks=0;
    kvmem::KvMemStore store(cfg);store.register_append(3);
    require(store.pick_scored_blocks({0,1,2},{-.8,-.2,-.5},{},2,true)==std::vector<uint32_t>({1,2}),
        "signed Query order lost");
    rejects([&] {store.pick_scored_blocks({0,1,2},{-.8,-.2,-.5},{},2);});
}
void graph_lifecycle() {
    constexpr size_t limit=1u<<20;
    kvmem::GraphSnapshot graph; graph.block_tokens=64;
    graph=kvmem::prepare_graph_accept(graph,0,64,{},limit);
    graph=kvmem::prepare_graph_accept(graph,64,65,{{64,{{0,.8},{64,.2}}}},limit);
    require(graph.chunks.empty() && graph.partial.samples==1, "partial source sealed early");
    const auto partial_checkpoint=graph;
    graph=kvmem::prepare_graph_accept(graph,65,128,{},limit);
    require(graph.chunks.size()==1 && graph.partial.samples==0 && graph.completed_through==128,
            "unsampled acceptance did not seal the source");
    close(graph.chunks[0]->at(0).edges[0].mass,.8);
    graph=kvmem::prepare_graph_accept(graph,128,192,{{128,{{64,1}}}},limit);
    auto frame=kvmem::query_score_frame(192,64,2,0,0,{});
    auto result=kvmem::propagate_query_scores({1,0,1},frame,graph);
    close(result[0],.95); close(result[1],.2); close(result[2],.85);
    close(std::accumulate(result.begin(),result.end(),0.),2);
    require(partial_checkpoint.chunks.empty() && partial_checkpoint.partial.samples==1,
            "later acceptance mutated an immutable checkpoint");
    frame.transport_mask[0]=0;
    result=kvmem::propagate_query_scores({8,0,1},frame,graph);
    close(result[0],8); close(result[1],.15); close(result[2],.85);
    rejects([&] { kvmem::prepare_graph_accept(graph,192,193,{{192,{{256,1}}}},limit); });
    rejects([&] { kvmem::prepare_graph_accept(graph,192,193,{{192,{{0,.2}}}},limit); });
    rejects([&] { kvmem::prepare_graph_accept(graph,192,256,{},1); });
    require(graph.accepted_through==192, "failed graph preparation changed its input");
    // Aggregate before top-k: an initially runner-up target can win later.
    kvmem::GraphSnapshot top; top.block_tokens=4;
    top=kvmem::prepare_graph_accept(top,0,12,{},limit);
    top=kvmem::prepare_graph_accept(top,12,14,
        {{12,{{0,.6},{4,.4}}},{13,{{4,.4},{8,.6}}}},limit,1);
    top=kvmem::prepare_graph_accept(top,14,16,{},limit,1);
    require(top.chunks[0]->at(0).edges[0].target==4,"top-k happened before source aggregation");
    close(top.chunks[0]->at(0).edges[0].mass,.4);
}
void feedback_and_pressure() {
    kvmem::AttentionFeedback feedback;
    feedback.observe_prefill(0,64,64,{{63,{{0,.2},{32,.8}}}},{0,32});
    auto initial=feedback.pressure_scores({0,32}); close(initial[0],.2); close(initial[1],.8);
    feedback.observe_output({{0,.75},{32,.25}});
    const auto checkpoint=feedback.snapshot();
    const auto preview=feedback.pressure_scores({0,32});
    close(preview[0],.75); close(preview[1],.25);
    require(feedback.snapshot().output.empty() && feedback.snapshot().output_samples==1,
            "pressure preview committed or decayed state");
    feedback.finish_output(); feedback.finish_output();
    close(feedback.pressure_scores({0,32})[0],.75);
    feedback.observe_output({{0,.25},{32,.75}});
    close(feedback.pressure_scores({0,32})[0],.625);
    feedback.restore(checkpoint);
    close(feedback.pressure_scores({0,32})[0],.75);
    // Fallback is evaluated before fixed/mandatory nodes are filtered.
    close(feedback.pressure_scores({0})[0],.75);
    feedback.observe_prefill(64,128,32,{{127,{{0,1}}}},{0});
    require(feedback.snapshot().prefill_end==64,"partially accepted prefill changed feedback");
    const auto picked=kvmem::pressure_selection({0,2,3},{.1,.8,0},{0},2);
    require(picked==std::vector<uint32_t>({0,2}),"pressure reopened cold history or lost a mandatory page");
    const auto recency=kvmem::pressure_selection({0,2,3},{0,0,0},{0},2);
    require(recency==std::vector<uint32_t>({0,3}),"pressure recency fill mismatch");
    rejects([] { kvmem::pressure_selection({0},{1},{0,1},1); });
}
void orthogonal_query_scoring() {
    kvmem::GraphSnapshot graph; graph.block_tokens=1;
    // Three adjacent pages; the final page also has an accepted attention edge to page 0.
    graph=kvmem::prepare_graph_accept(graph,0,3,{{2,{{0,1}}}},1u<<20);
    const auto frame=kvmem::query_score_frame(3,1,2,0,0,{});
    const std::vector<double> raw{.8,-.2,.4};
    auto actual=kvmem::propagate_query_scores(raw,frame,graph,.4);
    // Explicit independent transport matrix (columns = sources):
    // [ .9 .1 .2; .1 .8 .1; 0 .1 .7 ] applied to [.8,0,.4], plus [0,-.2,0].
    const std::vector<double> expected{.8,-.08,.28};
    for (size_t i=0;i<3;++i) close(actual[i],expected[i]);
    close(std::accumulate(actual.begin(),actual.end(),0.),1.);
    require(kvmem::propagate_query_scores(raw,frame,graph,0)==raw,"alpha zero changed signed scores");
    const std::vector<double> negative{-.8,-.2,-.4};
    require(kvmem::propagate_query_scores(negative,frame,graph,1)==negative,
        "negative scores generated transport evidence");
    auto masked=frame; masked.transport_mask[2]=0;
    actual=kvmem::propagate_query_scores(raw,masked,graph,.4);
    close(actual[0],.72);close(actual[1],-.12);close(actual[2],.4);
    rejects([&]{kvmem::propagate_query_scores({0,std::numeric_limits<double>::quiet_NaN(),0},frame,graph,0);});
    rejects([&]{kvmem::propagate_query_scores({0,std::numeric_limits<double>::infinity(),0},frame,graph,.4);});

    kvmem::RawKvStore keys({1,1,0,1});
    const float a[]{1},b[]{-1},c[]{.5},q[]{1};const uint32_t count[]{1};
    keys.write_layer_mean_sum(0,1,0,a);keys.write_layer_mean_sum(1,1,0,b);keys.write_layer_mean_sum(2,1,0,c);
    const kvmem::PageQueryView view{q,count,1,1,1,1,1};
    for (const auto* scorer : {"page-softmax","meanq-cosine"})
        for (const auto* agg : {"mean","max"}) for (bool history : {false,true})
            for (const auto* alpha : {"0","0.4","1"}) {
                kvmem::CliOptions options;
                auto parse=[&](const char* flag,const char* value="") {
                    require(options.parse(flag,[&](const char*){return value;}),"unknown Query option");
                };
                // Explicit overrides before scorer selection must survive resolution.
                parse("--kvmem");parse(history ? "--kvmem-lrfu" : "--no-kvmem-lrfu");
                parse("--kvmem-graph-alpha",alpha);parse("--kvmem-score-agg",agg);
                parse("--kvmem-query-scorer",scorer);options.resolve();
                require(options.lrfu==history && options.graph_alpha==std::stod(alpha),"Query controls coupled");
                const auto aggregation=options.score_agg ? kvmem::GqaScoreAggregation::Max : kvmem::GqaScoreAggregation::Mean;
                const auto scores=options.query_scorer==kvmem::QueryScorer::MeanQCosine ?
                    kvmem::mean_query_cosine_scores(keys,view,3,aggregation) :
                    kvmem::page_softmax_scores(keys,view,frame,aggregation);
                const auto propagated=kvmem::propagate_query_scores(scores,frame,graph,options.graph_alpha);
                kvmem::KvMemStoreConfig cfg;cfg.block_tokens=1;cfg.select_budget=2;cfg.sink_blocks=0;cfg.recent_blocks=0;
                cfg.lrfu=history;cfg.lrfu_lambda=.5;cfg.lrfu_top_fraction=1;
                kvmem::KvMemStore store(cfg);store.register_append(3);store.begin_retrieval_round();
                store.set_retrieval_scores(propagated);
                for(size_t i=0;i<3;++i) close(store.selection_score(uint32_t(i)),propagated[i]);
                if(!history) continue;
                require(store.commit_retrieval_round(),"accepted Query failed to commit");
                const auto checkpoint=store.retrieval_checkpoint();
                store.begin_retrieval_round();
                // The next query has no positive evidence: old positive history must not diffuse again.
                store.set_retrieval_scores(negative);store.set_retrieval_scores(negative);
                for(size_t i=0;i<3;++i) close(store.selection_score(uint32_t(i)),.5*propagated[i]+negative[i]);
                require(store.commit_retrieval_round() && !store.commit_retrieval_round(),"Query rewarded twice");
                store.restore_retrieval(checkpoint);
                for(size_t i=0;i<3;++i) close(store.selection_score(uint32_t(i)),propagated[i]);
            }
}
void accepted_sampling() {
    std::vector<int32_t> tokens(35,42);
    tokens[0]=10; tokens[9]=11; tokens[26]=10; tokens[30]=11;
    auto plan=kvmem::accepted_output_samples({},tokens.data(),35,10,11);
    require(plan.rows==std::vector<uint32_t>({25}) && plan.state.eligible_tokens==20 && !plan.state.thinking,
        "thinking markers or rejected suffix polluted output sampling");
    auto prefix=kvmem::accepted_output_samples({},tokens.data(),26,10,11);
    auto suffix=kvmem::accepted_output_samples(prefix.state,tokens.data()+26,9,10,11);
    require(prefix.rows==plan.rows && suffix.rows.empty() && suffix.state.eligible_tokens==plan.state.eligible_tokens,
        "output sampling depends on GPU batch boundaries");
    require(kvmem::prefill_sample_rows(0,128,0,0,1024)==std::vector<uint32_t>({31,63,95,127}),
        "prefill sample stride differs from reference");
    require(kvmem::prefill_sample_rows(128,128,128,128,4096).empty(),"prefill interval ignored");
    require(!kvmem::prefill_sample_rows(128,128,128,720,1024).empty(),"near-capacity sample missed page slack");
}
void fixed_picker() {
    kvmem::KvMemStoreConfig config;config.block_tokens=4;config.select_budget=16;
    config.sink_blocks=1;config.recent_blocks=1;
    kvmem::KvMemStore store(config);store.register_append(24);
    store.set_media_ranges({{4,9},{9,13}});
    require(store.query_fixed_blocks({3})==std::vector<uint32_t>({0,1,2,3}),
        "overlapping image boundary groups did not close backwards");
    store.set_retrieval_scores({0,0,0,0,9,8});
    require(store.pick_graph_blocks({3})==store.query_fixed_blocks({3}),
        "actual picker differs from the graph transport fixed set");
    rejects([&]{store.pick_scored_blocks({0,3,4,5},{0,1,9,8},{0,3},4);});
}
void statistics_budget() {
    auto global=std::make_shared<kvmem::GraphStatisticsBudget>(40000);
    auto first=std::make_shared<kvmem::GraphStatisticsBudget>(35000,global);
    auto second=std::make_shared<kvmem::GraphStatisticsBudget>(35000,global);
    {
        kvmem::GraphSnapshot a,b;a.set_budget(first);b.set_budget(second);
        std::vector<kvmem::GraphMessageSpan> spans(1000);
        for(size_t i=0;i<spans.size();++i) spans[i]={i*8,i*8+8,true};
        a.set_messages(spans);
        const auto before=global->used();
        rejects([&]{b.set_messages(spans);});
        require(global->used()==before && !b.messages,"failed global admission changed its snapshot");
        auto copy = a;
        require(copy.messages == a.messages && global->used() == before + sizeof(copy),
            "a fork charged its shared messages twice");
        copy.include_feedback(128);
        require(global->used() == before + sizeof(copy) + 128, "private feedback lost its charge");
        auto tight = std::make_shared<kvmem::GraphStatisticsBudget>(1024, global);
        const auto charged = global->used();
        rejects([&]{copy.set_budget(tight);});
        require(!tight->used() && global->used() == charged && copy.budget() == first,
            "failed rebind leaked an ancestor charge or changed the published owner");
        copy.set_budget(second);
        b.set_messages(a.messages);
        require(global->used() == before + sizeof(copy) + 128, "sibling budgets duplicated the parent charge");
        a.set_budget({});
        require(first->used() == 0 && second->used() == global->used(),
            "rebind retained the old budget or released the last shared charge");
        const auto held = global->used();
        rejects([&]{copy.include_feedback(40000);});
        require(global->used() == held && copy.messages == b.messages, "failed private growth changed shared state");
    }
    require(!global->used() && !first->used() && !second->used(),"retired graph snapshots retained budget charges");
}
void forked_graph_budget() {
    auto budget = std::make_shared<kvmem::GraphStatisticsBudget>(32u << 20);
    {
        kvmem::GraphSnapshot source;
        source.set_budget(budget);
        source.block_tokens = 1;
        source.set_messages(std::vector<kvmem::GraphMessageSpan>(262144));
        source = kvmem::prepare_graph_accept(source, 0, 2, {{1, {{0, 1.0}}}}, 32u << 20);
        const auto before = budget->used();
        std::vector<kvmem::GraphSnapshot> branches(8, source);
        require(budget->used() - before < 8192, "forks duplicated the shared history charge");
        auto& branch = branches.front();
        require(branch.chunks.front() == source.chunks.front(), "fork copied an immutable chunk");
        auto held = budget->used();
        branch = kvmem::prepare_graph_accept(branch, 2, 3, {{2, {{0, 1.0}}}}, 32u << 20);
        require(source.chunks.front()->size() == 1 && branch.chunks.front()->size() == 2 &&
            branch.chunks.front() != source.chunks.front() && budget->used() > held,
            "copy-on-write tail was not charged independently");
        rejects([&]{branch.set_messages(std::vector<kvmem::GraphMessageSpan>(1200000));});
        require(branch.messages == source.messages && source.accepted_through == 2,
            "rejected allocation damaged a published branch");
    }
    require(budget->used() == 0, "fork retirement leaked graph budget");
}
void message_locality() {
    kvmem::GraphSnapshot graph;graph.block_tokens=64;
    graph=kvmem::prepare_graph_accept(graph,0,192,{},1u<<20);
    auto frame=kvmem::query_score_frame(192,64,2,0,0,{});
    graph.set_messages(std::vector<kvmem::GraphMessageSpan>{{0,64,true},{64,192,true}});
    const auto crossed=kvmem::propagate_query_scores({1,0,0},frame,graph);
    close(crossed[0],.9875);close(crossed[1],.0125);
    graph.set_messages(std::vector<kvmem::GraphMessageSpan>{{0,192,true}});
    const auto same=kvmem::propagate_query_scores({1,0,0},frame,graph);
    close(same[0],.95);close(same[1],.05);
}
} // namespace
int main() {
    try {
        frames_and_scores(); gqa_and_large_domain(); scorer_variants(); graph_lifecycle(); orthogonal_query_scoring(); feedback_and_pressure(); accepted_sampling(); fixed_picker(); statistics_budget(); forked_graph_budget(); message_locality();
        std::cout << "graph recall numerical and accepted-state checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
