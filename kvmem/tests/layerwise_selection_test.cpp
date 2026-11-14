#include "kvmem/layerwise_selection.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace upstream_keep_oracle {
std::vector<uint32_t> fixed_selection(uint32_t,uint32_t,uint32_t,const std::vector<uint32_t>&,uint32_t,bool);
}

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected,"invalid layerwise selection was accepted");
}
kvmem::KvMemStoreConfig config(uint32_t budget, uint32_t sink=0, uint32_t recent=0) {
    kvmem::KvMemStoreConfig cfg;
    cfg.block_tokens=4; cfg.select_budget=budget*4;
    cfg.sink_blocks=sink; cfg.recent_blocks=recent;
    return cfg;
}
void populate(kvmem::RawKvStore& keys, uint32_t blocks) {
    for (uint32_t id=0; id<blocks; ++id) {
        const float sum[]{float(id*4)};
        keys.write_layer_mean_sum(id*4,4,1,sum);
    }
}
void fixed_masks_and_current_query() {
    kvmem::KvMemStore policy(config(4,1,1)); policy.register_append(24);
    policy.set_retrieval_scores({100,80,60,40,20,0});
    std::vector<double> previous;
    for (const auto& b : policy.blocks()) previous.push_back(b.attn_score);
    kvmem::RawKvStore keys({2,1,0,4}); populate(keys,6);
    const float positive[]{1};
    const auto result=kvmem::select_layerwise_history(policy,keys,{positive,1,1,1,1,1},{2});
    require(result.fixed==std::vector<uint32_t>({0,2,5}),"fixed Query reservations differ");
    require(result.frame.score_mask==std::vector<uint8_t>({0,1,1,1,1,0}),
            "mandatory identity removed a softmax logit");
    require(result.frame.transport_mask==std::vector<uint8_t>({0,1,0,1,1,0}),
            "fixed picker set was conflated with the score domain");
    require(result.selected==std::vector<uint32_t>({0,2,4,5}),
            "fixed and highest current-layer score were not selected");
    require(result.layer==1 && result.historical_tokens==16 && result.charged_tokens==16,
            "layer identity or budget accounting differs");
    for (size_t i=0;i<previous.size();++i)
        require(policy.blocks()[i].attn_score==previous[i],"pure selection mutated shared legacy scores");
    const float negative[]{-1};
    const auto changed=kvmem::select_layerwise_history(policy,keys,{negative,1,1,1,1,1},{2});
    require(changed.selected==std::vector<uint32_t>({0,1,2,5}),
            "selection ignored this invocation's fresh layer Query");
}
void half_page_softmax_preserves_short_evidence() {
    kvmem::KvMemStore policy(config(2,1));
    policy.register_append(12);
    // First physical page is a fixed sink. Page 1 has a strong and a weak
    // half. Page 2 is uniformly moderate and wins if each page is averaged first.
    const float values[]{100,100,8,-8,2,2};
    kvmem::RawKvStore fine({1,1,0,2}), coarse({1,1,0,4});
    for (uint32_t i=0;i<6;++i) {
        const float sum=2*values[i];
        fine.write_layer_mean_sum(2*i,2,0,&sum);
        coarse.write_layer_mean_sum(2*i,2,0,&sum);
    }
    const float q[]{1};
    const auto small=kvmem::select_layerwise_history(policy,fine,{q,0,1,1,1,1});
    const auto large=kvmem::select_layerwise_history(policy,coarse,{q,0,1,1,1,1});
    const double strong=std::exp(8.)+std::exp(-8.);
    const double moderate=2*std::exp(2.);
    require(small.scores.size()==3 && small.scores[0]==0 &&
            std::abs(small.scores[1]-strong/(strong+moderate))<1e-5 &&
            std::abs(small.scores[2]-moderate/(strong+moderate))<1e-5,
            "half-page masses differ from the independent softmax");
    require(small.selected==std::vector<uint32_t>({0,1}) &&
            large.selected==std::vector<uint32_t>({0,2}) &&
            small.charged_tokens==8 && small.historical_tokens==8,
            "fine scores must restore and charge whole pages");
    kvmem::RawKvStore incomplete({1,1,0,2});
    for (uint32_t i=0;i<5;++i) incomplete.write_layer_mean_sum(2*i,2,0,&values[i]);
    rejects([&]{kvmem::select_layerwise_history(policy,incomplete,{q,0,1,1,1,1});});
}
void strongest_half_is_taken_after_the_complete_query() {
    // Two physical 64-token pages. Two query dimensions independently encode
    // logits [1,-4,.75,.75] and [-4,1,.75,.75], all exactly representable
    // through the scorer's FP16 input boundary. Query rows alternate the
    // winning half. A premature max picks page 0; the full Query picks page 1.
    const float left[]{1,-4,.75f,.75f}, right[]{-4,1,.75f,.75f};
    const double z=std::exp(1.)+std::exp(-4.)+2*std::exp(.75);
    const double expected0=(std::exp(1.)+std::exp(-4.))/z, expected1=2*std::exp(.75)/z;
    const float q[]{1,0,0,1};
    kvmem::RawKvStore keys({1,2,0,32});
    for (uint32_t i=0;i<4;++i) {
        const float sums[]{32*left[i],32*right[i]};
        keys.write_layer_mean_sum(32*i,32,0,sums);
    }
    auto cfg=config(1);cfg.block_tokens=64;cfg.select_budget=64;
    kvmem::KvMemStore sum(cfg);sum.register_append(128);
    cfg.layerwise_half_max=true;
    kvmem::KvMemStore max(cfg);max.register_append(128);
    const auto baseline=kvmem::select_layerwise_history(sum,keys,{q,0,1,1,1,2,1});
    const auto peaked=kvmem::select_layerwise_history(max,keys,{q,0,1,1,1,2,1});
    require(baseline.selected==std::vector<uint32_t>{1} && peaked.selected==std::vector<uint32_t>{0},
            "sum and strongest-half should rank these pages differently");
    auto check=[&](const kvmem::LayerwiseHistorySelection& x) {
        require(x.selected==std::vector<uint32_t>{1} && x.charged_tokens==64 && x.historical_tokens==64 &&
            x.scores.size()==2 && std::abs(x.scores[0]-expected0)<1e-6 && std::abs(x.scores[1]-expected1)<1e-6,
            "strongest half differs from the independent complete-query oracle");
    };
    check(kvmem::select_layerwise_history(max,keys,{q,0,2,1,1,2,1}));
    const kvmem::LayerwiseQueryIdentity id{{true,{11,2},3,4,128,5},7,{128,130},{128,130}};
    for (uint32_t chunk : {1u,2u}) {
        kvmem::LayerwiseQuerySelection query(max,keys,id,{0,1,1,2,1},[&]{return id.history;});
        for (uint32_t i=0;i<2;i+=chunk)
            query.append(id,128+i,{q+2*i,0,chunk,1,1,2,1});
        check(query.finish(id));
    }
    // Production append_native: its scorer emits one value per half. Deliberately
    // alternate the peak between Query tiles, just as the host fixture does.
    kvmem::LayerwiseQuerySelection native(max,keys,id,{0,1,1,2,1},[&]{return id.history;});
    for (uint32_t i=0;i<2;++i) {
        kvmem::NativeCurrentLayerQuery tile{q+2*i,2*sizeof(float),0,1,1,1,2,1,
            kvmem::NativeQueryScalar::Float32,2*sizeof(float),2*sizeof(float),sizeof(float)};
        native.append_native(id,128+i,tile,[&](const auto&,auto,const auto& frame){
            require(frame.starts==std::vector<uint64_t>({0,32,64,96}),"native denominator did not cover four halves");
            const float* values=i?right:left;
            std::vector<double> probabilities;
            for (int h=0;h<4;++h) probabilities.push_back(std::exp(double(values[h]))/z);
            return probabilities;
        });
    }
    check(native.finish(id));
}
void upstream_newer_first_ties() {
    kvmem::KvMemStore policy(config(2)); policy.register_append(24);
    kvmem::RawKvStore keys({2,1,0,4}); populate(keys,6);
    const float zero[]{0};
    const auto result=kvmem::select_layerwise_history(policy,keys,{zero,1,1,1,1,1});
    // Upstream72a81cc take_top/final topk uses earlier(b,a). In the canonical
    // directory historical order is logical ID order, so equal mass picks 5,4.
    require(result.selected==std::vector<uint32_t>({4,5}),"upstream chronological tie rule changed");
    policy.set_retrieval_scores(std::vector<double>(6,1.));
    require(policy.pick_topk_blocks()==result.selected,"native TopK and scored picker ties differ");
}
void explicit_pins_without_legacy_media_promotion() {
    kvmem::KvMemStore policy(config(1));policy.register_append(32);
    policy.set_media_ranges({{24,32}}); // Two legacy grouped pages exceed B=1.
    kvmem::RawKvStore keys({2,1,0,4});populate(keys,8);
    const float q[]{0};
    const auto result=kvmem::select_layerwise_history(policy,keys,{q,1,1,1,1,1});
    const auto fixed=upstream_keep_oracle::fixed_selection(8,0,0,{},1,false);
    require(result.fixed==fixed && fixed.empty(),"legacy newest media was promoted into fixed pins");
    require(result.selected==std::vector<uint32_t>{7},"legacy media grouping changed raw page selection");
    const auto pinned=kvmem::select_layerwise_history(policy,keys,{q,1,1,1,1,1},{2});
    require(pinned.fixed==upstream_keep_oracle::fixed_selection(8,0,0,{2},1,false) &&
        pinned.selected==std::vector<uint32_t>{2},"explicit request pins were lost with legacy media metadata");
    require(policy.media_ranges()==std::vector<std::pair<uint32_t,uint32_t>>{{24,32}},
        "layerwise wrapper mutated the legacy owner policy");
    rejects([&]{kvmem::select_layerwise_history(policy,keys,{q,1,1,1,1,1},{6,7});});
    kvmem::LayerwiseQueryIdentity identity;
    identity.history.valid=true;identity.history.session={1,1};identity.history.history_revision=1;
    identity.history.execution_history=1;identity.history.frontier=32;
    identity.request=1;identity.evaluation={32,33};identity.query={32,33};
    kvmem::LayerwiseQuerySelection tiled(policy,keys,identity,{1,1,1,1},[&]{return identity.history;});
    tiled.append(identity,32,{q,1,1,1,1,1});
    const auto completed=tiled.finish(identity);
    require(completed.fixed==fixed && completed.selected==result.selected,
        "tiled wrapper retained a legacy media promotion/expansion");
}
void fitting_and_runtime_budget() {
    kvmem::KvMemStore policy(config(8,1,1)); policy.register_append(24);
    kvmem::RawKvStore keys({2,1,0,4}); populate(keys,6);
    const float q[]{1};
    const auto full=kvmem::select_layerwise_history(policy,keys,{q,1,1,1,1,1},{2});
    require(full.selected==std::vector<uint32_t>({0,1,2,3,4,5}),"fitting history did not retain all blocks");
    require(full.frame.score_mask==std::vector<uint8_t>(6,1) &&
            full.frame.transport_mask==std::vector<uint8_t>(6,0),"fitting score-frame semantics changed");
    policy.set_runtime_select_budget(12);
    const auto reduced=kvmem::select_layerwise_history(policy,keys,{q,1,1,1,1,1});
    require(reduced.selected==std::vector<uint32_t>({0,4,5}) && reduced.charged_tokens==12,
            "runtime selection budget was not used consistently");
    require(policy.runtime_select_budget_tokens()==12,"selection changed the frozen runtime budget");
}
void native_picker_equivalence() {
    kvmem::RawKvStore keys({2,1,0,4}); populate(keys,7);
    const float q[]{1};
    for(uint32_t budget=1;budget<=8;++budget) {
        for(uint32_t sink=0;sink<=std::min(2u,budget);++sink) {
            for(uint32_t recent=0;recent<=std::min(2u,budget-sink);++recent) {
                kvmem::KvMemStore policy(config(budget,sink,recent)); policy.register_append(28);
                const std::vector<uint32_t> mandatory=budget>sink ? std::vector<uint32_t>{3} : std::vector<uint32_t>{};
                const auto result=kvmem::select_layerwise_history(policy,keys,{q,1,1,1,1,1},mandatory);
                policy.set_retrieval_scores(result.scores);
                require(policy.pick_topk_blocks(mandatory)==result.selected,
                        "canonical native TopK differs from unfused layer selection");
            }
        }
    }
}
void invalid_and_capacity() {
    const float q[]{1};
    kvmem::RawKvStore keys({2,1,0,4}); populate(keys,6);
    auto run=[&](kvmem::KvMemStore& p,const std::vector<uint32_t>& pins=std::vector<uint32_t>{}) {
        return kvmem::select_layerwise_history(p,keys,{q,1,1,1,1,1},pins);
    };
    kvmem::KvMemStore partial(config(2)); partial.register_append(23);
    rejects([&]{run(partial);});
    kvmem::KvMemStore policy(config(2,1)); policy.register_append(24);
    rejects([&]{run(policy,{1,2});});
    rejects([&]{run(policy,{6});});
    // A request-final B preserves startup-resolved bands. The pinned outer
    // selection_config rejects S+R>B before the private best-effort picker.
    auto crowded_cfg=config(2,1,2);
    kvmem::KvMemStore crowded(crowded_cfg);crowded.register_append(24);
    rejects([&]{run(crowded);});
    for(int variant=0;variant<4;++variant) {
        auto cfg=config(2);
        if(variant==0) cfg.lrfu=true;
        if(variant==1) cfg.select_policy=kvmem::KvMemSelectPolicy::Quota;
        if(variant==2) cfg.select_method=kvmem::KvMemMethod::Recency;
        if(variant==3) cfg.n_subblocks=2;
        kvmem::KvMemStore unsupported(cfg); unsupported.register_append(24);
        rejects([&]{run(unsupported);});
    }
    kvmem::RawKvStore missing({2,1,0,4});
    rejects([&]{kvmem::select_layerwise_history(policy,missing,{q,1,1,1,1,1});});
}

kvmem::LayerwiseQueryIdentity tiled_identity(uint64_t rows) {
    return {{true,{11,2},3,4,24,5},7,{24,25+rows},{25,25+rows}};
}
void complete_query_tile_invariance() {
    const float q[]{2,-1,1,3,0,-2,1};
    kvmem::KvMemStore policy(config(3,1,1)); policy.register_append(24);
    kvmem::RawKvStore keys({2,1,0,4}); populate(keys,6);
    const auto identity=tiled_identity(7);
    const auto expected=kvmem::select_layerwise_history(policy,keys,{q,1,7,1,1,1});
    for(uint32_t chunk : {1u,2u,3u,7u,256u}) {
        kvmem::LayerwiseQuerySelection selection(policy,keys,identity,{1,1,1,1},
                                                [&]{return identity.history;});
        rejects([&]{selection.finish(identity);});
        for(uint32_t offset=0;offset<7;) {
            const auto count=std::min(chunk,7-offset);
            selection.append(identity,identity.query.begin+offset,{q+offset,1,count,1,1,1});
            offset+=count;
            require(selection.scored_rows()==offset,"Query tile row accounting changed");
            if(offset<7) rejects([&]{selection.finish(identity);});
        }
        const auto actual=selection.finish(identity);
        require(actual.selected==expected.selected && actual.fixed==expected.fixed &&
                actual.frame.score_mask==expected.frame.score_mask &&
                actual.frame.transport_mask==expected.frame.transport_mask,
                "tiled Query selected using a different domain, fixed set or partial scores");
        for(size_t i=0;i<actual.scores.size();++i)
            require(std::abs(actual.scores[i]-expected.scores[i])<2e-12,
                    "raw Query token mass was not accumulated before selection");
        require(selection.phase()==kvmem::LayerwiseSelectionPhase::Complete,
                "complete Query did not close the accumulator");
        rejects([&]{selection.finish(identity);});
        rejects([&]{selection.append(identity,identity.query.begin,{q,1,1,1,1,1});});
    }
}

void frozen_policy_and_source_revision() {
    const float q[]{1,2};
    kvmem::KvMemStore policy(config(3,1,1)); policy.register_append(24);
    kvmem::RawKvStore keys({2,1,0,4}); populate(keys,6);
    const auto identity=tiled_identity(2);
    auto revision=identity.history;
    kvmem::LayerwiseQuerySelection frozen(policy,keys,identity,{1,1,1,1},[&]{return revision;});
    frozen.append(identity,25,{q,1,1,1,1,1});
    // An unrelated mutable policy update cannot alter this Query's already
    // frozen candidate domain, fixed set or current runtime budget.
    policy.set_runtime_select_budget(8);
    policy.set_retrieval_scores(std::vector<double>(6,100));
    frozen.append(identity,26,{q+1,1,1,1,1,1});
    const auto actual=frozen.finish(identity);
    require(actual.selected==std::vector<uint32_t>({0,4,5}) && actual.charged_tokens==12,
            "mutable caller policy changed a frozen full-Query decision");
    kvmem::LayerwiseQuerySelection stale(policy,keys,identity,{1,1,1,1},[&]{return revision;});
    stale.append(identity,25,{q,1,1,1,1,1});
    const float replacement[]{400}; keys.write_layer_mean_sum(0,4,1,replacement);
    ++revision.history_revision;
    rejects([&]{stale.append(identity,26,{q+1,1,1,1,1,1});});
    require(stale.phase()==kvmem::LayerwiseSelectionPhase::Aborted && stale.scored_rows()==1,
            "changed Mean-K revision was combined with old Query mass");
    revision=identity.history;
    rejects([&]{stale.finish(identity);});
    ++revision.history_revision;
    auto fresh_identity=identity; fresh_identity.history=revision;
    kvmem::LayerwiseQuerySelection finish_stale(policy,keys,fresh_identity,{1,1,1,1},[&]{return revision;});
    finish_stale.append(fresh_identity,25,{q,1,2,1,1,1});
    ++revision.history_revision;
    rejects([&]{finish_stale.finish(fresh_identity);});
    require(finish_stale.phase()==kvmem::LayerwiseSelectionPhase::Aborted,
            "final picker published a decision after its index revision changed");
}

void tiled_bounds_geometry_and_owner() {
    const float q[]{1,2};
    kvmem::KvMemStore policy(config(3)); policy.register_append(24);
    kvmem::RawKvStore keys({2,1,0,4}); populate(keys,6);
    const auto identity=tiled_identity(2);
    kvmem::LayerwiseQuerySelection selection(policy,keys,identity,{1,1,1,1},
                                             [&]{return identity.history;});
    auto foreign=identity; ++foreign.request;
    rejects([&]{selection.append(foreign,25,{q,1,1,1,1,1});});
    rejects([&]{selection.finish(foreign);});
    rejects([&]{selection.append(identity,24,{q,1,1,1,1,1});});
    rejects([&]{selection.append(identity,25,{q,1,3,1,1,1});});
    rejects([&]{selection.append(identity,25,{q,0,1,1,1,1});});
    rejects([&]{selection.append(identity,25,{q,1,1,2,1,1});});
    rejects([&]{selection.append(identity,25,{q,1,1,1,1,1,.5f});});
    require(selection.scored_rows()==0,"rejected tile advanced Query progress");
    selection.append(identity,25,{q,1,1,1,1,1});
    rejects([&]{selection.append(identity,25,{q,1,1,1,1,1});});
    rejects([&]{selection.append(identity,27,{q+1,1,1,1,1,1});});
    selection.abort();
    rejects([&]{selection.append(identity,26,{q+1,1,1,1,1,1});});
    rejects([&]{selection.finish(identity);});
    rejects([&]{kvmem::LayerwiseQuerySelection missing(policy,keys,identity,{1,1,1,1},{});});
}
}

int main() {
    try {
        fixed_masks_and_current_query();
        upstream_newer_first_ties();
        half_page_softmax_preserves_short_evidence();
        strongest_half_is_taken_after_the_complete_query();
        explicit_pins_without_legacy_media_promotion();
        fitting_and_runtime_budget();
        native_picker_equivalence();
        invalid_and_capacity();
        complete_query_tile_invariance();
        frozen_policy_and_source_revision();
        tiled_bounds_geometry_and_owner();
        std::cout << "layerwise_selection_test: ok\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
