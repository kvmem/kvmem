#include "prefix_directory.hpp"
#include "checkpoint_handoff.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace kvmem;
using namespace kvmem::radiance;
static void check(bool value,const char* why) {if(!value)throw std::runtime_error(why);}
struct HandoffIdentity { bool fallback=false; };
struct HandoffPoint { std::shared_ptr<HandoffIdentity> identity; std::unique_ptr<int> state; };
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
int main() {
    checkpoint_publication_preserves_ownership();
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
    points[1].eligible=false;
    check(!select_prefix_checkpoint(next,points),"scope/media rejection is a hard constraint");
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
