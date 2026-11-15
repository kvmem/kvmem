#include "kvmem/historical_kv_semantic.hpp"
#include "kvmem/historical_kv_page_gc.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace kvmem;
static void check_impl(bool ok, int line) {
    if (!ok) throw std::runtime_error("step redundancy assertion at line " + std::to_string(line));
}
#define check(...) check_impl((__VA_ARGS__), __LINE__)
struct Fixture {
    std::shared_ptr<HistoricalKvPageFeatures> features=std::make_shared<HistoricalKvPageFeatures>(16,2);
    std::shared_ptr<HistoricalKvStepEvidence> evidence;
    HistoricalKvSnapshot snapshot{1,7,{}};
    size_t mark() { return evidence->commit_selection_boundary(std::vector<uint8_t>(16)); }
    Fixture(uint32_t sources=4,uint32_t witnesses=1,bool charge=true,uint32_t max_steps=64,
            std::vector<uint8_t> witness_masks={}, bool observe=false, bool relaxed=false) {
        HistoricalKvStepEvidence::Config cfg;
        cfg.feature_gate=true;cfg.min_observations=1;cfg.min_cold_observations=0;cfg.max_importance=1;
        cfg.charge_result_pages=charge;
        cfg.observe_only=observe;
        cfg.relaxed_attention_redundancy=relaxed;
        if (relaxed) cfg.similarity=.85;
        cfg.max_steps=max_steps;
        evidence=std::make_shared<HistoricalKvStepEvidence>(16,64,cfg);
        evidence->publish({{"owned","thinking",0,sources*64,8*64,(8+witnesses)*64,1,1,false,true,
            std::move(witness_masks)}},1024);
        for (uint32_t p=0;p<16;++p) {
            HistoricalKvBlockMeta b;
            b.handle={7,p,1};b.logical_id=p;b.original_begin=p*64;b.valid_tokens=64;
            b.life=HistoricalKvLife::Live;b.sealed=true;b.content_version=1;
            snapshot.blocks.push_back(b);evidence->bind(p,b.handle);
            features->publish_page(p,{1,0});
        }
        features->publish_question_scores(std::vector<double>(16,.5));
    }
};
static void relaxed_attention_contract() {
    Fixture f(4,1,false,64,{0,0,0,0},false,true);
    f.features->truncate_to(512,64);
    f.features->publish_page(8,{.86f,float(std::sqrt(1-.86*.86))});
    auto query=std::vector<double>(16,.01);query[0]=.99;
    f.features->publish_question_scores(query);
    f.evidence->evaluate(f.features,f.snapshot);
    check(f.mark()==0); // Query similarity is not a substitute for attention.
    const auto revision=f.features->observation_revision();
    for (uint32_t id=0;id<4;++id) f.features->publish_output_attention(id,id==1?.8:.2);
    f.features->publish_output_attention(8,.3);
    check(f.features->observation_revision()>revision);
    f.evidence->evaluate(f.features,f.snapshot); // Newly observed attention retries this scope.
    auto pins=std::vector<uint8_t>(16);pins[2]=2;
    check(f.evidence->commit_selection_boundary(pins)==2);
    check(f.evidence->selection_excluded(0) && f.evidence->selection_excluded(3));
    check(!f.evidence->selection_excluded(1)); // More important source remains.
    check(!f.evidence->selection_excluded(2)); // Reader/safety protection still wins.
    double importance=0;float key[2];
    check(f.features->read_observed(f.snapshot.blocks[0],key,importance,true) && importance==.2);
    check(f.features->read_observed(f.snapshot.blocks[0],key,importance) && importance==.99);

    for (const auto similarity : {.84,.86}) {
        Fixture g(4,1,false,64,{0,0,0,0},false,true);
        g.features->truncate_to(512,64);
        g.features->publish_page(8,{float(similarity),float(std::sqrt(1-similarity*similarity))});
        for (uint32_t id=0;id<4;++id) g.features->publish_output_attention(id,0);
        g.features->publish_output_attention(8,0); // Measured zero differs from missing.
        g.evidence->evaluate(g.features,g.snapshot);
        check(g.mark()==(similarity>.85?4:0));
    }
    Fixture stale(4,1,false,64,{0,0,0,0},false,true);
    for (uint32_t id=0;id<9;++id) stale.features->publish_output_attention(id,.3);
    stale.evidence->evaluate(stale.features,stale.snapshot);
    stale.features->clear_question();
    check(stale.mark()==0);
}
static void literal_witness_contract() {
    // Digits 11 and 12 form one run across the source page seam. The first
    // result repeats the value with another label. The second repeats context.
    const std::vector<int32_t> source{0,1,2,3,4,96,10,11,12,13,23,5,6,7,8,9,0,0,0,0,0,0,0,0};
    std::vector<uint8_t> digits(source.size());
    digits[7]=digits[8]=1;
    const std::vector<int32_t> results{99,10,11,12,13,23,0,0,96,10,11,12,13,23,0,0};
    check(historical_kv_literal_witnesses(source,digits,results,8)==std::vector<uint8_t>{2,2,3});
    const std::vector<int32_t> split_result{0,0,0,0,96,10,11,12,13,23,0,0,0,0,0,0};
    check(historical_kv_literal_witnesses(source,digits,split_result,8)==std::vector<uint8_t>{0,0,3});
    digits.front()=1;
    check(historical_kv_literal_witnesses(source,digits,results,8)==std::vector<uint8_t>{0,2,3});

    Fixture f(4,2,false,64,{0,2,1,3});
    f.evidence->evaluate(f.features,f.snapshot);
    check(f.mark()==3);
    check(!f.evidence->selection_excluded(0) && f.evidence->capacity_reclaimable(f.snapshot.blocks[0]));
    const auto witness=f.evidence->retired(1,f.snapshot.blocks[1].handle);
    check(witness.page==9 && witness.result_end==640);
    bool rejected=false;
    try { Fixture invalid(4,2,false,64,{0,4,1,3}); }
    catch (const std::invalid_argument&) { rejected=true; }
    check(rejected);
}
static void duplicate_feedback_contract() {
    HistoricalKvStepEvidence::Config config;
    config.feature_gate=true;
    config.min_observations=1;
    config.min_cold_observations=0;
    config.max_importance=1;
    config.charge_result_pages=false;
    HistoricalKvStepEvidence evidence(16,64,config);
    HistoricalKvStepEvidence::Update update{"repeated-output","duplicate_tool_feedback",
        128,256,512,704,1,1,false,true,{},530};
    bool rejected=false;
    auto truncated=update;
    truncated.result_end=640;
    try { evidence.publish({truncated},1024); }
    catch (const std::invalid_argument&) { rejected=true; }
    check(rejected);
    evidence.publish({update},1024);
    auto features=std::make_shared<HistoricalKvPageFeatures>(16,2);
    HistoricalKvSnapshot snapshot{1,7,{}};
    for (uint32_t page=0;page<16;++page) {
        HistoricalKvBlockMeta block;
        block.handle={7,page,1};
        block.logical_id=page;
        block.valid_tokens=64;
        block.life=HistoricalKvLife::Live;
        block.sealed=true;
        snapshot.blocks.push_back(block);
        evidence.bind(page,block.handle);
    }
    std::vector<uint8_t> protections(16);
    snapshot.blocks[10].sealed=false;
    evidence.evaluate(features,snapshot);
    check(evidence.commit_selection_boundary(protections, 0, 1)==0); // One source would add two pins.
    protections[9] = 4;
    check(evidence.commit_selection_boundary(protections, 0, 2)==1);
    check(evidence.eligible(2) && !evidence.eligible(3));
    snapshot.blocks[10].sealed=true;
    evidence.evaluate(features,snapshot);
    features->clear_question(); // Exact feedback coverage is independent of Query scores.
    check(evidence.commit_selection_boundary(protections, 0, 2)==1);
    const auto pending_first=evidence.selection_witness(2);
    const auto pending_second=evidence.selection_witness(3);
    check(pending_first.page==8 && pending_first.page_count==2 && pending_first.result_end==704);
    check(pending_second.page==9 && pending_second.page_count==2 && pending_second.result_end==704);
    const auto first=evidence.retired(2,snapshot.blocks[2].handle);
    const auto second=evidence.retired(3,snapshot.blocks[3].handle);
    check(first.page==8 && first.page_count==2 && first.result_end==704);
    check(second.page==9 && second.page_count==2 && second.result_end==704);
}
static void selection_witness_lifetime() {
    Fixture f;
    check(f.evidence->selection_witness(0).result_end==0);
    f.evidence->evaluate(f.features,f.snapshot);
    check(f.evidence->selection_witness(0).result_end==0); // Worker proposals are not marks.
    check(f.mark()==4);
    const auto witness=f.evidence->selection_witness(0);
    check(witness.page==8 && witness.page_count==1 && witness.result_end==576);
    check(f.evidence->selection_witness(8).result_end==0);
    check(f.evidence->selection_witness(16).result_end==0);
    check(f.evidence->retired(0,f.snapshot.blocks[0].handle).result_end==576);
    check(f.evidence->selection_witness(0).result_end==0); // Retirement owns its separate tombstone.
    f.evidence->reset_bindings();
    check(f.evidence->selection_witness(1).result_end==576); // Native owner movement preserves the mark.
    f.evidence->truncate_to(576);
    check(f.evidence->selection_witness(1).result_end==576);
    f.evidence->truncate_to(575);
    check(f.evidence->selection_witness(1).result_end==0); // Rewritten witness revokes the projection.
    Fixture observed(4,1,true,64,{},true);
    observed.evidence->evaluate(observed.features,observed.snapshot);
    check(observed.mark()==0 && observed.evidence->selection_witness(0).result_end==0);
    check(observed.evidence->status().front().state=="observed_no_delete");
}
static void pressure_policy_contract() {
    Fixture f(4,1,false);
    HistoricalKvStatisticalPolicy policy(f.evidence,f.features,16);
    HistoricalKvDecisionBuffer out(16);
    // Evaluated scopes remain advisory under pressure until the owner commits.
    // No pressure means no cold deletions.
    policy.step(f.snapshot,{12,2,6},{16,16},out);
    check(out.values().empty());
    ++f.snapshot.round_id;
    policy.step(f.snapshot,{0,2,6},{16,2},out);
    check(out.values().size()==2);
    check(out.values()[0].victim.slot==0 && out.values()[1].victim.slot==1);
    const auto proposed = out.values()[0];
    check(policy.permits_intent(f.snapshot.blocks[0], proposed));
    check(f.mark() == 4);
    check(!policy.permits_intent(f.snapshot.blocks[0], proposed));
    for (auto& b : f.snapshot.blocks) if (b.logical_id < 4) b.mandatory = true;
    ++f.snapshot.round_id;
    out.clear_after_ack();
    policy.step(f.snapshot,{0,2,6},{16,2},out);
    check(out.values().size()==2);
    check(out.values()[0].victim.slot==4 && out.values()[1].victim.slot==5);
    auto queued=out.values()[0];
    check(queued.reason==HistoricalKvDeleteReason::CapacityLru && !queued.evidence_revision);
    check(policy.permits_intent(f.snapshot.blocks[4],queued));
    // Publishing proof leaves the witness available to an already queued LRU
    // intent. Only approval at the owner boundary makes the intent stale.
    f.evidence->publish({{"later","thinking",12*64,14*64,4*64,5*64,1,1,false,true}},1024);
    check(policy.permits_intent(f.snapshot.blocks[4],queued));
    f.evidence->evaluate(f.features, f.snapshot);
    check(f.mark() == 2);
    check(!policy.permits_intent(f.snapshot.blocks[4],queued));
    f.snapshot.blocks[12].mandatory = f.snapshot.blocks[13].mandatory = true;
    out.clear_after_ack();
    policy.step(f.snapshot,{0,2,6,2},{16,2},out);
    check(out.values().size()==2 && out.values()[0].victim.slot==6 && out.values()[1].victim.slot==7);
    out.clear_after_ack();
    // Existing pending credit prevents a second batch from over-deleting.
    policy.step(f.snapshot,{0,2,6,6},{16,2},out);check(out.values().empty());
    // A reservation may demand more than the hysteresis high watermark.
    ++f.snapshot.round_id;
    policy.step(f.snapshot,{9,2,6,0,10},{16,16},out);
    check(out.values().size()==1 && out.values()[0].reason==HistoricalKvDeleteReason::CapacityLru);

    Fixture missing;
    missing.features->clear_question();
    for (auto& b:missing.snapshot.blocks) b.mandatory=b.logical_id>=4;
    HistoricalKvStatisticalPolicy cold(missing.evidence,missing.features,16);
    out.clear_after_ack();cold.step(missing.snapshot,{0,2,6},{16,16},out);
    check(out.values().size()==4);
    for (const auto& intent:out.values()) {
        check(intent.reason==HistoricalKvDeleteReason::CapacityLru);
        const auto witness = missing.evidence->retired(static_cast<uint32_t>(intent.victim.slot), intent.victim);
        check(witness.result_end == 0);
    }
    check(!missing.evidence->result_page(8));
    check(missing.evidence->status()[0].state=="capacity_reclaimed" && missing.evidence->status()[0].retired==0);
    check(cold.metrics().lossy_intents==4);
}

static void independent_groups_and_witness_loss() {
    Fixture f(4, 2, false, 64, {1, 2, 2, 1});
    f.evidence->evaluate(f.features, f.snapshot);
    std::vector<uint8_t> pins(16);
    pins[0] = pins[3] = 2;
    check(f.evidence->commit_selection_boundary(pins, 4, 1) == 2);
    check(f.evidence->eligible(1) && f.evidence->eligible(2) && !f.evidence->eligible(0));
    check(f.evidence->result_page(9) && !f.evidence->result_page(8));
    check(f.evidence->capacity_reclaimable(f.snapshot.blocks[8]));
    check(!f.evidence->capacity_reclaimable(f.snapshot.blocks[9]));
    f.evidence->retired(8, f.snapshot.blocks[8].handle); // Capacity LRU can take the unused witness.
    pins[0] = pins[3] = 0;
    check(f.evidence->commit_selection_boundary(pins, 4, 2) == 0);
    check(f.evidence->retired(1, f.snapshot.blocks[1].handle).page == 9);
    check(f.evidence->result_page(9));
    check(f.evidence->retired(2, f.snapshot.blocks[2].handle).page == 9);
    check(!f.evidence->result_page(9));
    check(f.evidence->permits_selection(0));
}

static void retry_on_new_observation() {
    Fixture f(4, 1, false);
    f.features->clear_question();
    for (uint32_t id = 0; id < 4; ++id) f.features->publish_output_attention(id, .2);
    const auto query = f.features->question_epoch();
    f.evidence->evaluate(f.features, f.snapshot);
    check(f.mark() == 0 && f.evidence->status().front().state == "pending_features");
    f.features->publish_output_attention(7, .5); // Unrelated observation.
    f.evidence->evaluate(f.features, f.snapshot);
    check(f.mark() == 0);
    f.features->publish_output_attention(8, .5);
    f.evidence->evaluate(f.features, f.snapshot);
    check(f.features->question_epoch() == query && f.mark() == 4);
}

static void checkpoint_guard_releases_its_veto() {
    Fixture f(4, 1, false);
    f.evidence->evaluate(f.features, f.snapshot);
    uint32_t blocked = 0;
    const HistoricalKvStepEvidence::SelectionGuard guard{3, &blocked,
        [](void* context, uint32_t page, HistoricalKvStepEvidence::RetainedWitness witness) noexcept {
            return page != *static_cast<const uint32_t*>(context) && witness.page == 8;
        }};
    const std::vector<uint8_t> pins(16);
    check(f.evidence->commit_selection_boundary(pins, 4, 1, &guard) == 3);
    check(!f.evidence->eligible(0) && f.evidence->eligible(1));
    check(f.evidence->commit_selection_boundary(pins, 4, 1, &guard) == 0);
    check(f.evidence->commit_selection_boundary(pins, 4, 1) == 1);
    check(f.evidence->eligible(0));
}
static void observation_keeps_recall_and_pressure() {
    Fixture observer(4, 1, false, 64, {}, true);
    observer.evidence->evaluate(observer.features, observer.snapshot);
    std::vector<uint8_t> pins(16);
    observer.evidence->mark_result_pages(pins, 128);
    check(pins[8] == 0 && observer.evidence->capacity_reclaimable(observer.snapshot.blocks[8]));
    check(observer.evidence->capacity_reclaimable(observer.snapshot.blocks[0]));
    check(observer.mark() == 0 && observer.evidence->permits_selection(0));
    const auto status = observer.evidence->status();
    check(status.size() == 1 && status.front().state == "observed_no_delete");
    check(status.front().selected == 4 && status.front().retired == 0);
}
static void pressure_release_and_reuse() {
    HistoricalKvStepEvidence::Config limits;
    limits.feature_gate=true;limits.min_observations=1;limits.min_cold_observations=0;limits.max_importance=1;
    auto evidence=std::make_shared<HistoricalKvStepEvidence>(128,64,limits);
    HostKvStorage storage(8*64);
    HistoricalKvPageGc::Config cfg;
    cfg.slots=8;cfg.logical_pages=128;cfg.page_tokens=64;cfg.low_ready=2;cfg.high_ready=3;
    cfg.batch_blocks=2;cfg.sink_tokens=cfg.recent_tokens=0;
    cfg.evidence=evidence;cfg.physical_storage=storage;cfg.physical_page_bytes=64;
    HistoricalKvPageGc gc(cfg,std::make_unique<HistoricalKvStatisticalPolicy>(evidence,nullptr,8));
    std::vector<HostKvRecord> records(128);
    auto append=[&](uint32_t p) {
        records[p]=storage.allocate(64);check(bool(records[p]));records[p].seal();
        check(gc.register_page(p,64,p+1));gc.protect_page(p,false);
    };
    for (uint32_t p=0;p<8;++p) append(p);
    auto reader=records[0].lease();
    auto drive=[&] {
        gc.manual_tick();
        for (const auto& page:gc.commit_boundary())
            gc.acknowledge_host_release(page,std::move(records[page.page]),true);
    };
    for (int i=0;i<30;++i) drive();
    check(gc.stats().capacity_lru_committed==3 && gc.stats().semantic_committed==0);
    check(gc.stats().capacity_lru_reclaimed==2); // Tombstones are not physical capacity.
    check(storage.charged_bytes()==6*64);
    reader={};for(int i=0;i<20;++i)drive();
    check(gc.stats().capacity_lru_reclaimed==3 && storage.charged_bytes()==5*64);
    // Repeat pressure and real slot reuse beyond the eight-slot directory.
    for(uint32_t p=8;p<64;++p) {
        if (storage.charged_bytes()>6*64) for(int i=0;i<30;++i)drive();
        append(p);for(int i=0;i<4;++i)drive();
        check(storage.charged_bytes()<=8*64);
    }
    check(gc.stats().capacity_lru_reclaimed>40 && gc.stats().worker_failures==0);
    check(gc.stats().reclaimed==gc.stats().capacity_lru_reclaimed);
}
static void marked_page_outlives_active_view() {
    constexpr uint32_t source = 1, witness = 3;
    constexpr uint8_t active_pin = 4;
    HistoricalKvStepEvidence::Config limits;
    limits.feature_gate = true;
    limits.charge_result_pages = false;
    limits.min_observations = 1;
    limits.min_cold_observations = 0;
    limits.max_importance = 1;
    auto evidence = std::make_shared<HistoricalKvStepEvidence>(16, 64, limits);
    evidence->publish({{"retained-reader", "thinking", 64, 128, 192, 256, 1, 1, false, true}}, 256);
    auto features = std::make_shared<HistoricalKvPageFeatures>(16, 2);
    for (const auto id : {source, witness}) features->publish_page(id, {1, 0});
    features->publish_question_scores(std::vector<double>(16, .5));

    HostKvStorage storage(16 * 64);
    auto source_record = storage.allocate(64);
    auto witness_record = storage.allocate(64);
    source_record.seal();
    witness_record.seal();
    HistoricalKvPageGc::Config cfg;
    cfg.slots = cfg.logical_pages = 16;
    cfg.low_ready = 1;
    cfg.high_ready = 2;
    cfg.sink_tokens = cfg.recent_tokens = 0;
    cfg.evidence = evidence;
    cfg.physical_storage = storage;
    cfg.physical_page_bytes = 64;
    HistoricalKvPageGc gc(cfg, std::make_unique<HistoricalKvStatisticalPolicy>(evidence, features));
    check(gc.register_page(source, 64, 1) && gc.register_page(witness, 64, 1));
    gc.protect_page(source, true);
    gc.manual_tick();
    std::vector<uint8_t> pins(16);
    pins[source] = active_pin;
    check(evidence->commit_selection_boundary(pins, active_pin) == 1);
    check(!evidence->permits_selection(source, false) && gc.is_live(source));
    const auto restore_witness=evidence->selection_witness(source);
    check(restore_witness.page==witness && restore_witness.result_end==256);
    for (int i = 0; i < 8; ++i) {
        gc.manual_tick();
        check(gc.commit_boundary().empty());
    }

    auto pending_reader = source_record.lease();
    gc.protect_page(source, false); // A natural window publication dropped the GPU reader.
    gc.manual_tick();
    const auto& retired = gc.commit_boundary();
    check(retired.size() == 1 && retired.front().page == source && retired.front().witness.page == witness);
    gc.acknowledge_host_release(retired.front(), std::move(source_record));
    for (int i = 0; i < 8; ++i) gc.manual_tick();
    check(gc.stats().semantic_reclaimed == 0 && storage.charged_bytes() == 128);
    pending_reader = {};
    for (int i = 0; i < 8 && !gc.stats().semantic_reclaimed; ++i) gc.manual_tick();
    check(gc.stats().semantic_reclaimed == 1 && storage.charged_bytes() == 64);
}
int main() {
    try {
        relaxed_attention_contract();
        literal_witness_contract();
        duplicate_feedback_contract();
        selection_witness_lifetime();
        pressure_policy_contract();
        independent_groups_and_witness_loss();
        retry_on_new_observation();
        checkpoint_guard_releases_its_veto();
        observation_keeps_recall_and_pressure();
        pressure_release_and_reuse();
        marked_page_outlives_active_view();
        Fixture f;
        check(!f.evidence->handoff_candidate(0)); // Declared scope is not permission.
        f.evidence->evaluate(f.features,f.snapshot);
        check(!f.evidence->eligible(0)); // A worker proposal cannot publish the mark.
        check(f.mark()==4);
        check(f.evidence->eligible(0) && f.evidence->eligible(3));
        check(!f.evidence->eligible(8) && f.evidence->result_page(8));
        HistoricalKvStatisticalPolicy policy(f.evidence,f.features);
        HistoricalKvDecisionBuffer decisions(16);
        policy.step(f.snapshot,{12,1,2},{256,64},decisions); // No pressure needed.
        check(decisions.values().size()==4);
        auto intent=decisions.values()[0];
        auto stale=f.snapshot.blocks[0];++stale.handle.generation;
        check(!policy.permits_intent(stale,intent));
        f.features->clear_question();
        check(f.evidence->eligible(0)); // A committed mark cannot be revived by a new Query.
        check(!f.evidence->permits_selection(0,true));
        check(policy.permits_intent(f.snapshot.blocks[0],intent));
        f.features->publish_question_scores(std::vector<double>(16,.5));
        f.evidence->evaluate(f.features,f.snapshot);
        check(f.evidence->retired(0, stale.handle).result_end == 0);
        for (uint32_t p=0;p<4;++p) {
            const auto witness = f.evidence->retired(p, {7,p,1});
            check(witness.page == 8 && witness.result_end == 576);
        }
        check(f.evidence->status().at(0).state=="retired");
        check(!f.evidence->result_page(8));
        f.evidence->publish({{"owned","thinking",0,256,512,576,1,1,false,true}},1024);
        check(f.evidence->stats().tracked==0); // Completed scopes do not reopen.
        check(!f.evidence->permits_selection(0,true));
        Fixture no_gain(1,1);
        const auto protection_revision = no_gain.evidence->result_protection_revision();
        no_gain.evidence->evaluate(no_gain.features,no_gain.snapshot);
        check(no_gain.evidence->result_protection_revision() == protection_revision);
        check(!no_gain.evidence->eligible(0) && !no_gain.evidence->result_page(8));
        check(no_gain.evidence->status()[0].state=="no_net_gain");
        const auto comparisons = no_gain.evidence->feature_comparisons();
        no_gain.evidence->evaluate(no_gain.features,no_gain.snapshot);
        check(no_gain.evidence->feature_comparisons() == comparisons);
        no_gain.features->publish_question_scores(std::vector<double>(16, .5));
        no_gain.evidence->evaluate(no_gain.features,no_gain.snapshot);
        check(no_gain.evidence->feature_comparisons() > comparisons);
        Fixture existing_analysis(1,1,false);
        existing_analysis.evidence->evaluate(existing_analysis.features,existing_analysis.snapshot);
        existing_analysis.mark();
        check(existing_analysis.evidence->eligible(0)); // No new summary KV was generated.
        Fixture partial(4,1,false);
        partial.evidence->evaluate(partial.features,partial.snapshot);
        partial.mark();
        partial.evidence->retired(0,{7,0,1});
        partial.snapshot.blocks[0].life=HistoricalKvLife::Retired;
        partial.features->publish_question_scores(std::vector<double>(16,.5));
        partial.evidence->evaluate(partial.features,partial.snapshot);
        for (uint32_t p=1;p<4;++p) partial.evidence->retired(p,{7,p,1});
        check(partial.evidence->status()[0].retired==4);
        Fixture importance;
        auto scores=std::vector<double>(16,.1);scores[0]=1;
        importance.features->publish_question_scores(scores);
        importance.evidence->evaluate(importance.features,importance.snapshot);
        importance.mark();
        check(!importance.evidence->eligible(0) && importance.evidence->eligible(1));
        Fixture missing;
        missing.features->clear_question();
        missing.evidence->evaluate(missing.features,missing.snapshot);
        check(!missing.evidence->eligible(0));
        // Independent double-precision geometric boundary. The cosine of
        // (1,0) and (a,sqrt(1-a*a)) is exactly a, not a mean of unrelated mates.
        for (double cosine:{.899,.901}) {
            Fixture geometry;
            geometry.features->truncate_to(512,64);
            geometry.features->publish_page(8,{float(cosine),float(std::sqrt(1-cosine*cosine))});
            geometry.features->publish_question_scores(std::vector<double>(16,.5));
            geometry.evidence->evaluate(geometry.features,geometry.snapshot);
            geometry.mark();
            check(geometry.evidence->eligible(0)==(cosine>.9));
        }
        Fixture rewrite;
        rewrite.evidence->evaluate(rewrite.features,rewrite.snapshot);
        rewrite.mark();
        rewrite.evidence->truncate_to(300);
        check(!rewrite.evidence->eligible(0) && rewrite.evidence->stats().tracked==0);
        Fixture fairness;
        fairness.evidence->publish({{"later","thinking",4*64,7*64,9*64,10*64,1,1,false,true}},1024);
        fairness.features->clear_question();
        for (uint32_t p=0;p<16;++p) if (p!=8) fairness.features->publish_output_attention(p,.5);
        fairness.snapshot.round_id=0;
        fairness.evidence->evaluate(fairness.features,fairness.snapshot); // First witness unknown.
        check(!fairness.evidence->eligible(0));
        ++fairness.snapshot.round_id;
        fairness.evidence->evaluate(fairness.features,fairness.snapshot);
        fairness.mark();
        check(fairness.evidence->eligible(4)); // Later, ready step is not starved.
        Fixture pending;
        pending.evidence->evaluate(pending.features,pending.snapshot);
        pending.features->clear_question();
        check(pending.mark()==0); // Uncommitted proposals still need the current Query.
        Fixture protected_source;
        protected_source.evidence->evaluate(protected_source.features,protected_source.snapshot);
        auto pins=std::vector<uint8_t>(16);pins[0]=2;
        check(protected_source.evidence->commit_selection_boundary(pins,4)==3);
        check(!protected_source.evidence->eligible(0));
        pins[0]=4;
        check(protected_source.evidence->commit_selection_boundary(pins,4)==1);
        Fixture changed_generation;
        changed_generation.evidence->evaluate(changed_generation.features,changed_generation.snapshot);
        changed_generation.evidence->bind(0,{7,0,2});
        check(changed_generation.mark()==3 && !changed_generation.evidence->eligible(0));
        changed_generation.snapshot.blocks[0].handle.generation = 2;
        changed_generation.evidence->evaluate(changed_generation.features, changed_generation.snapshot);
        check(changed_generation.mark()==1);
        Fixture bounded(4,1,false,1);
        bounded.evidence->evaluate(bounded.features,bounded.snapshot);bounded.mark();
        bounded.evidence->publish({{"later","thinking",256,448,576,640,1,1,false,true}},1024);
        check(bounded.evidence->eligible(0) && bounded.evidence->stats().tracked==1);
        check(bounded.evidence->status()[0].state=="metadata_capacity_retained");
        std::cout << "step redundancy: geometry, importance, net gain, epoch, ownership, ack passed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return EXIT_FAILURE; }
}
