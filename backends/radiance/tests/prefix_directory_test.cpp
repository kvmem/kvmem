#include "prefix_directory.hpp"
#include "checkpoint_handoff.hpp"
#include "draft_conditioning.hpp"
#include "canonical_input.hpp"
#include "canonical_boundary.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <deque>
#include "kvmem/spill_io.hpp"
using namespace kvmem;
using namespace kvmem::radiance;
static void check(bool value,const char* why) {if(!value)throw std::runtime_error(why);}
struct HandoffIdentity { bool fallback=false; };
struct HandoffPoint { std::shared_ptr<HandoffIdentity> identity; std::unique_ptr<int> state; };
struct CanonicalMedia {
    uint64_t begin=0,end=0,key=0,position=0;
    bool operator==(const CanonicalMedia& other) const {return begin==other.begin && end==other.end && key==other.key && position==other.position;}
};
static void canonical_boundary_ignores_native_rounding() {
    HostKvStorage account(32768);
    std::vector<int32_t> tokens(144);std::iota(tokens.begin(),tokens.end(),1000);
    const std::vector<std::pair<int32_t,int32_t>> spans{{0,96},{96,128}};
    // All three inputs have the same scheduler-rounded history bucket (96).
    // The native frontend's complete message spans retain the short Query.
    check(canonical_input_boundary(tokens.size(),spans)==128,"canonical boundary used a rounded checkpoint or generation suffix");
    auto a_tokens=tokens,b_tokens=tokens,c_tokens=tokens;
    a_tokens[122]=71;b_tokens[122]=72;c_tokens[122]=73;
    auto root=capture_canonical_input(tokens,96,128,std::vector<CanonicalMedia>{},&account);
    auto capture=[&](const auto& ids,uint64_t accepted) {
        return capture_canonical_input(ids,canonical_input_boundary(ids.size(),spans),accepted,
            std::vector<CanonicalMedia>{},&account);
    };
    auto a=capture(a_tokens,128),b=capture(b_tokens,128),c=capture(c_tokens,128);
    std::vector<CanonicalGroupMember<CanonicalMedia>> groups{{1,root},{1,a}};
    check(!classify_canonical_group(b,groups),"same rounded bucket merged two different full short Queries");
    groups.push_back({2,b});check(!classify_canonical_group(c,groups),"same rounded bucket merged a third short Query");
    groups.push_back({3,c});
    auto extended=a_tokens;extended.resize(160,99);
    const std::vector<std::pair<int32_t,int32_t>> extension{{0,96},{96,128},{128,156}};
    auto next=capture_canonical_input(extended,canonical_input_boundary(extended.size(),extension),156,
        std::vector<CanonicalMedia>{},&account);
    check(classify_canonical_group(next,groups)==1,"complete message extension did not reconnect after an old ancestor restore");
    check(!capture(a_tokens,127),"partially accepted complete message acquired a canonical identity");
    check(canonical_input_boundary(tokens.size(),{})==tokens.size(),"completion or unknown spans must use full native input");
    check(canonical_input_boundary(tokens.size(),{{0,96},{95,128}})==tokens.size(),"overlapping span must conservatively use full native input");
    check(canonical_input_boundary(tokens.size(),{{0,96},{96,145}})==tokens.size(),"invalid last span must not use an earlier partial prefix");
    const std::vector<CanonicalMedia> images{{120,132,17,120}};
    check(!capture_canonical_input(tokens,128,144,images,&account),"canonical message boundary split a media expansion");
    const auto fallback=canonical_input_boundary(tokens.size(),{{0,96},{96,145}});
    check(!capture_canonical_input(tokens,fallback,143,images,&account),"fallback full input was not actually accepted");
    auto complete=capture_canonical_input(tokens,fallback,144,images,&account);
    auto moved=images;moved[0].position++;
    auto other=capture_canonical_input(tokens,fallback,144,moved,&account);
    check(complete && other && !complete->comparable(*other),"fallback canonical ignored original media position");
}
static void canonical_groups_require_complete_input() {
    HostKvStorage account(16384);std::vector<int32_t> doc(128);std::iota(doc.begin(),doc.end(),1000);
    const std::vector<CanonicalMedia> media{{20,24,7,19}};
    auto root=capture_canonical_input(doc,96,128,media,&account);
    auto a_tokens=doc,b_tokens=doc,c_tokens=doc;a_tokens[112]=71;b_tokens[112]=72;c_tokens[112]=73;
    auto a=capture_canonical_input(a_tokens,128,128,media,&account);
    auto b=capture_canonical_input(b_tokens,128,128,media,&account);
    auto c=capture_canonical_input(c_tokens,128,128,media,&account);
    std::vector<CanonicalGroupMember<CanonicalMedia>> groups{{1,root},{1,a}};
    check(!classify_canonical_group(b,groups),"two short-question siblings inherited one ancestor group");
    groups.push_back({2,b});check(!classify_canonical_group(c,groups),"third short-question sibling joined a conflicting group");
    groups.push_back({3,c});check(classify_canonical_group(a,groups)==1,"old-ancestor restore did not reconnect with its exact canonical group");
    groups.push_back({1,{}});check(!classify_canonical_group(a,groups),"unknown completed member did not invalidate its group");
    const auto before=account.charged_bytes();
    check(!capture_canonical_input(a_tokens,128,127,media,&account) && account.charged_bytes()==before,"cancelled/rounded input was treated as canonical accepted proof");
    auto moved=media;moved[0].position++;
    auto different=capture_canonical_input(a_tokens,128,128,moved,&account);
    check(!a->comparable(*different),"canonical group ignored original media positions");
    HostKvStorage tight(64);check(!capture_canonical_input(a_tokens,128,128,media,&tight) && !tight.charged_bytes(),"canonical preflight failed to preserve a tight quota");
    StorageIoQueue queue(1,1);
    const auto before_pending=account.charged_bytes();
    auto refused=capture_canonical_input_reserved(a_tokens,128,128,media,[&](uint64_t bytes) {
        queue.submit(2,[]{});return std::make_shared<HostKvReservation>(account.reserve(bytes));
    });
    check(!refused && !queue.pending_jobs() && !queue.pinned_bytes() && account.charged_bytes()==before_pending,
        "optional canonical queue refusal leaked a reservation");
    bool validation=false;
    try {capture_canonical_input_reserved(a_tokens,128,128,media,[](uint64_t)->std::shared_ptr<HostKvReservation>{throw std::invalid_argument("validation");});}
    catch(const std::invalid_argument&){validation=true;}
    check(validation,"optional metadata boundary swallowed a validation error");
    auto lease=a;a.reset();check(account.charged_bytes()>=lease->bytes(),"shared canonical lease lost its allocation charge");
}
static void checkpoint_publication_preserves_ownership() {
    std::unique_ptr<HandoffPoint> input,fallback;
    for(bool alternate:{false,true}) {
        HandoffPoint pending{std::make_shared<HandoffIdentity>(HandoffIdentity{alternate}),std::make_unique<int>(alternate ? 29 : 17)};
        const auto* identity=pending.identity.get();
        const auto* state=pending.state.get();
        const auto* other=alternate ? input.get() : fallback.get();
        publish_retained_checkpoint(std::make_unique<HandoffPoint>(std::move(pending)),input,fallback);
        const auto& ready=alternate ? fallback : input;
        check(!pending.identity && !pending.state,"pending ownership must transfer completely");
        check(ready && ready->identity.get()==identity && ready->state.get()==state,"publication must keep the original immutable payload");
        check(*ready->state==(alternate ? 29 : 17),"published state differs from the completed checkpoint");
        check((alternate ? input.get() : fallback.get())==other,"publication must preserve the other checkpoint");
    }
    const auto* before_input=input.get();const auto* before_fallback=fallback.get();
    for(bool absent:{false,true}) {
        bool rejected=false;
        auto bad=absent ? std::unique_ptr<HandoffPoint>{} : std::make_unique<HandoffPoint>();
        try{publish_retained_checkpoint(std::move(bad),input,fallback);}
        catch(const std::invalid_argument&){rejected=true;}
        check(rejected && input.get()==before_input && fallback.get()==before_fallback,"invalid completion must not replace published state");
    }
}
static void drafter_refill_limits_and_early_input_anchors() {
    check(restorable_prefix_limit(1010,-1)==1009,"complete draft state supports short follow-ups");
    check(restorable_prefix_limit(4192,4165)==4165,"new Query remains an independent restore bound");
    check(restorable_prefix_limit(4192,2000)==2000,"restored state must not cross the new Query");
    check(restorable_prefix_limit(4192,-1)==4191,"complete windows do not impose a fixed rewind");
    check(restorable_prefix_limit(0,-1)==0,"empty input has no restorable state");
    // Independent position oracle: protecting N pages must cover every token
    // in the trailing W positions, including a one-token partial final page.
    for (uint32_t window : {1U, 128U, 129U, 2048U}) {
        const auto allocation=draft_keep_allocation(128,4096,128,128,window);
        for (uint64_t frontier=4096; frontier<4352; ++frontier) {
            const auto pages=(frontier+127)/128;
            const auto first=pages>allocation.recent_blocks ? (pages-allocation.recent_blocks)*128 : 0;
            check(first<=frontier-window, "recent pages must cover the full draft window at a partial frontier");
        }
    }
    const auto keep=draft_keep_allocation(128,4096,128,128,2048);
    check(keep.recent_effective_tokens==2176 && keep.sink_effective_tokens==128,"protect draft conditioning pages inside the same budget");
    check(draft_keep_allocation(128,4096,128,3072,2048).recent_effective_tokens==3072,"larger recent allocation is retained");
    check(draft_keep_allocation(128,4096,128,128,0).recent_effective_tokens==128,"non-block drafting keeps the requested recent");
    bool refused=false;
    try{(void)draft_keep_allocation(128,2048,128,128,2048);}catch(const std::runtime_error&){refused=true;}
    check(refused,"a draft ring cannot silently grow the selected budget");
    refused=false;
    try{(void)draft_keep_allocation(128,2176,128,512,2048);}catch(const std::runtime_error&){refused=true;}
    check(refused,"sink plus a full window including partial tail must fit the existing budget");
    check(draft_keep_allocation(128,2304,128,512,2048).recent_blocks==17,
        "minimum valid selected budget preserves sink and every conditioning position");
    std::vector<int32_t> prompt(4192);std::iota(prompt.begin(),prompt.end(),1000);
    const std::vector<int32_t> input(prompt.begin(),prompt.begin()+2048);
    const std::vector<int32_t> terminal(prompt.begin(),prompt.begin()+4160);
    int input_state=1,terminal_state=2;
    std::vector<PrefixCandidate> points{{&terminal,&terminal_state,CheckpointPurpose::Endpoint,true},
        {&input,&input_state,CheckpointPurpose::ReuseAnchor,true}};
    const auto limit=restorable_prefix_limit(prompt.size(),4165);
    check(select_prefix_checkpoint(prompt,points,limit)==&terminal_state,
        "complete native draft state permits the later terminal checkpoint");
    check(select_prefix_checkpoint(prompt,points,restorable_prefix_limit(prompt.size(),4165))==&terminal_state,
        "spec-off keeps its later exact terminal prefix");
    points.pop_back();
    check(select_prefix_checkpoint(prompt,points,limit)==&terminal_state,"restoring the actual terminal state does not require an earlier checkpoint");
    check(retain_drafter_input_anchor(2048,2048,3638,2048,-1),
        "first-pass natural input state before Query and its future checkpoint is reusable");
    check(retain_drafter_input_anchor(2048,2048,3638,2048,3584),
        "the 4K document anchor stays before its proven rewind boundary");
    check(retain_drafter_input_anchor(10240,10240,11776,2048,-1),
        "a 12K native input anchor may retain the accepted sparse history after Pressure");
    check(retain_drafter_input_anchor(10240,10240,11776,2048,11520),
        "a later Query replay cannot change the retained 12K input prefix");
    check(retain_drafter_input_anchor(10240,10240,11776,2048,10240),
        "a checkpoint exactly at the retained frontier preserves that state");
    check(!retain_drafter_input_anchor(2048,2048,3638,0,-1),"this exception is only for a declared block drafter");
    check(!retain_drafter_input_anchor(2048,-1,3638,2048,-1),"an unknown input boundary grants no checkpoint");
    check(!retain_drafter_input_anchor(2047,2048,3638,2048,-1),"only the scheduler's exact natural anchor may be retained");
    check(!retain_drafter_input_anchor(2048,2048,2047,2048,-1),"Query probe rows must not enter a retained input prefix");
    check(!retain_drafter_input_anchor(2048,2048,-1,2048,-1),"a missing Query bound does not prove a safe probe anchor");
    check(!retain_drafter_input_anchor(10240,10240,11776,2048,10239),"a rewind that crosses the anchor disqualifies retention");
    check(!retain_drafter_input_anchor(2048,2048,3638,2048,-2),"an invalid checkpoint sentinel grants no retention");
}
struct FailingDirectory : std::deque<HandoffPoint> {
    bool fail=false;
    void push_back(HandoffPoint&& point) {
        if(fail)throw std::bad_alloc();
        std::deque<HandoffPoint>::push_back(std::move(point));
    }
};
static void directory_commit_waits_for_native_completion() {
    FailingDirectory directory;
    directory.push_back({std::make_shared<HandoffIdentity>(),std::make_unique<int>(11)});
    auto* previous=directory.front().state.get();
    ContentGroupPublicationPlan plan;plan.slot=0;plan.retired_slots={0};
    plan.reason=ContentPublicationReason::Replaced;
    HandoffPoint incoming{std::make_shared<HandoffIdentity>(),std::make_unique<int>(23)};
    // Pending and cancelled captures cannot spend public directory quotas.
    check(!publish_content_checkpoint(directory,std::move(incoming),plan,false),"incomplete native capture published");
    check(directory.size()==1 && directory.front().state.get()==previous && incoming.state,"pending capture retired the old directory");
    directory.fail=true;bool failed=false;
    try{publish_content_checkpoint(directory,std::move(incoming),plan,true);}catch(const std::bad_alloc&){failed=true;}
    check(failed && directory.size()==1 && directory.front().state.get()==previous && incoming.state,"publication allocation failure consumed a source");
    directory.fail=false;
    auto invalid=plan;invalid.retired_slots={99};failed=false;
    try{publish_content_checkpoint(directory,std::move(incoming),invalid,true);}catch(const std::invalid_argument&){failed=true;}
    check(failed && directory.front().state.get()==previous && incoming.state,"invalid publication plan changed the directory");
    check(publish_content_checkpoint(directory,std::move(incoming),plan,true),"complete native snapshot did not publish");
    check(directory.size()==1 && *directory.front().state==23 && !incoming.state,"complete publication did not atomically replace its quota victim");
}
int main() {
    canonical_boundary_ignores_native_rounding();
    canonical_groups_require_complete_input();
    checkpoint_publication_preserves_ownership();
    drafter_refill_limits_and_early_input_anchors();
    directory_commit_waits_for_native_completion();
    std::vector<int32_t> input(128);std::iota(input.begin(),input.end(),1000);
    auto terminal=input;terminal.insert(terminal.end(),{21,22,23,24});
    int before=1,after=2;
    std::vector<PrefixCandidate> points{{&terminal,&after,CheckpointPurpose::Endpoint,true},
        {&input,&before,CheckpointPurpose::InputFallback,true}};
    auto next=terminal;next.push_back(40);
    check(select_prefix_checkpoint(next,points)==&after,"complete answer continuation must prefer terminal state");
    next=input;next.insert(next.end(),{90,91,92});
    check(select_prefix_checkpoint(next,points)==&before,"dropped generated answer must reuse input state");
    check(select_prefix_checkpoint(terminal,points)==&before,"same prompt requires logits; earlier state stays reusable");
    check(!select_prefix_checkpoint(input,points),"cannot restore end without an unevaluated token");
    next[100]=-1;
    check(!select_prefix_checkpoint(next,points),"shared header cannot stand in for a complete native snapshot");
    next=input;next.push_back(90);
    check(!select_prefix_checkpoint(next,points,127),"new Query cannot start inside restored history");
    check(select_prefix_checkpoint(next,points,128)==&before,"new Query may start at restored frontier");
    check(!select_prefix_checkpoint(next,points,UINT64_MAX,false),"tool continuation must not discover an arbitrary shared header");
    points[1].continuation=true;
    check(select_prefix_checkpoint(next,points,UINT64_MAX,false)==&before,"proven tool continuation may restore an earlier native checkpoint");
    points[1].continuation=false;
    points[0].continuation=true;
    auto tool=terminal;tool.push_back(81);
    check(select_prefix_checkpoint(tool,points,UINT64_MAX,false)==&after,"accepted endpoint continuation must survive identity-free lookup");
    tool[20]=-1;
    check(!select_prefix_checkpoint(tool,points,UINT64_MAX,false),"continuation classification cannot bypass exact token proof");
    points[1].eligible=false;
    check(!select_prefix_checkpoint(next,points),"native/media rejection is a hard constraint");
    points[1].eligible=true;points[1].storage=nullptr;
    check(!select_prefix_checkpoint(next,points),"LCP alone cannot fabricate missing native state");
    // Exhaustive small frontiers: the returned state must be the largest real
    // matching checkpoint before logits/Query, never a computed LCP position.
    for(size_t length=1;length<150;++length)for(size_t mismatch=0;mismatch<=length;++mismatch) {
        std::vector<int32_t> prompt(length);
        for(size_t i=0;i<length;++i)prompt[i]=i<terminal.size() ? terminal[i] : 7;
        if(mismatch<length)prompt[mismatch]=-1;
        points[1].storage=&before;
        const void* expected=nullptr;
        if(length>input.size() && mismatch>=input.size())expected=&before;
        if(length>terminal.size() && mismatch>=terminal.size())expected=&after;
        check(select_prefix_checkpoint(prompt,points)==expected,"selected state violates frontier/prefix contract");
    }
    std::cout<<"PASS prefix directory: terminal/input, dropped output, LCP, logits, Query and isolation\n";
}
