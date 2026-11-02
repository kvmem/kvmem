#include "prefix_directory.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace kvmem;
using namespace kvmem::radiance;
static void check(bool value,const char* why) {if(!value)throw std::runtime_error(why);}
int main() {
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
