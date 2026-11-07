#include "attention_capture.hpp"
#include "rad_device.h"
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(false)
struct Stream { std::deque<std::function<void()>> queue; };
struct Event { Stream* stream=nullptr; bool done=false; };
static int stream_syncs=0, allocations=0, events=0;
static bool fail_event=false;
static void next(Stream* s) { auto f=std::move(s->queue.front());s->queue.pop_front();f(); }
extern "C" {
int rad_dev_set(int device) { return device==2 ? RAD_OK : RAD_E_INVAL; }
void* rad_dev_alloc(int64_t bytes,int) { ++allocations;return std::malloc(size_t(bytes)); }
void rad_dev_free(void* p,int) { --allocations;std::free(p); }
int rad_stream_sync(RadStream stream) {
    ++stream_syncs;auto* s=static_cast<Stream*>(stream);while(!s->queue.empty())next(s);return RAD_OK;
}
int rad_event_create(RadEvent* event) { ++events;*event=new Event;return RAD_OK; }
void rad_event_destroy(RadEvent event) { --events;delete static_cast<Event*>(event); }
int rad_event_record(RadEvent event,RadStream stream) {
    if(fail_event)return RAD_E_STATE;
    auto* e=static_cast<Event*>(event);e->stream=static_cast<Stream*>(stream);e->done=false;
    e->stream->queue.push_back([e]{e->done=true;});return RAD_OK;
}
int rad_event_sync(RadEvent event) {
    auto* e=static_cast<Event*>(event);
    while(!e->done && !e->stream->queue.empty())next(e->stream);
    return e->done ? RAD_OK : RAD_E_STATE;
}
int rad_memcpy_async(void* dst,const void* src,int64_t bytes,RadStream stream) {
    static_cast<Stream*>(stream)->queue.push_back([=]{std::memcpy(dst,src,size_t(bytes));});return RAD_OK;
}
}
int kvmem::radiance::attention_page_masses(const AttentionKernelArgs& args,RadStream stream) {
    CHECK(args.page_count==1);
    static_cast<Stream*>(stream)->queue.push_back([=]{
        for(uint32_t i=0;i<args.samples;++i)args.masses[i]=1.f;
    });return RAD_OK;
}
template<class F> void rejects(F f) { bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}CHECK(caught); }
int main() {
    Stream stream;
    RadTensor q{},cache{};q.shape[0]=1;cache.shape[2]=4;
    int32_t table=0;RadKVGroupBatch kv{};kv.group=1;kv.block_table=&table;kv.block_table_pitch=1;kv.block_size=4;
    RadBatch batch{};batch.n_tok=1;batch.n_seq=1;batch.n_kv_groups=1;batch.kv=&kv;
    {
        AttentionCapture capture(2,{{0,1,0,1,1,1},{1,1,1,1,1,1},{2,1,2,1,1,1}},1,1);
        for(int repeat=0;repeat<3;++repeat) {
            capture.begin(0,{0},{{0,4,0}},&stream);
            for(int layer=0;layer<3;++layer)CHECK(capture.observe(layer,1,1,q,cache,batch,&stream)==RAD_OK);
            bool later=false;stream.queue.push_back([&]{later=true;});
            const auto rows=capture.collect(1,{0});
            CHECK(rows.size()==1 && rows[0].mass.size()==1 && rows[0].mass[0].mass==1.);
            CHECK(!later && stream_syncs==repeat);
            CHECK(rad_stream_sync(&stream)==RAD_OK && later);
        }
        // An empty sample frame submits no GPU work and must not fence draft work.
        capture.begin(0,{},{{0,4,0}},&stream);
        bool later=false;stream.queue.push_back([&]{later=true;});
        CHECK(capture.collect(1,{}).empty() && !later);rad_stream_sync(&stream);
        capture.begin(0,{0},{{0,4,0}},&stream);
        CHECK(capture.observe(0,1,1,q,cache,batch,&stream)==RAD_OK);
        rejects([&]{capture.collect(1,{0});});capture.discard();
        // A partially submitted failed frame is drained before staging reuse.
        capture.begin(0,{0},{{0,4,0}},&stream);
        CHECK(capture.observe(0,1,1,q,cache,batch,&stream)==RAD_OK);
        CHECK(capture.observe(1,1,1,q,cache,batch,&stream)==RAD_OK);
        fail_event=true;CHECK(capture.observe(2,1,1,q,cache,batch,&stream)<0);
        rejects([&]{capture.collect(1,{0});});capture.discard();
    }
    CHECK(stream.queue.empty() && allocations==0 && events==0);
    std::cout<<"PASS attention completion fence, empty frames, reuse and failed-frame drain\n";
}
