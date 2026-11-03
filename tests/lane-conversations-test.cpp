// Deterministic catalog interleavings; no model or GPU is needed.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "llama-kvmem-driver.h"
#include "llama-kvmem-diag.h"
#include "llama-kvmem-execution.h"
#include "kvmem-session-files.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include <cassert>
#include <iostream>
#include <limits>
#include <set>
#include <thread>

using json = nlohmann::json;
struct ServerState : LlamaEngineState {
    std::mutex mu;
    bool fail_finish = false;
};
static void llama_driver_finish(ServerState& state) {
    if (state.fail_finish) throw std::runtime_error("injected finish failure");
    llama_driver_finish(static_cast<LlamaEngineState&>(state));
}
#include "kvmem-lane-conversations.h"

struct kvmem_lane_conversations_test_access {
    static kvmem_lane_conversations::operation handover(kvmem_lane_conversations& catalog) {
        // Exact published state of attach: lane 0 switched from owner 0 to 2;
        // the outgoing owner is idle and can be reclaimed by lane 1.
        auto& incoming = catalog.entries_[2];
        incoming.lane = 0;
        incoming.busy = true;
        catalog.entries_.at(0).lane = -1;
        catalog.resident_[0] = 2;
        catalog.next_id_ = 3;
        kvmem_lane_conversations::operation op;
        op.lane = 0; op.id = 2; op.outgoing = 0;
        op.added = op.attached = op.ready = true;
        return op;
    }
    static void finish_other_lane(kvmem_lane_conversations& catalog, ServerState& state) {
        kvmem_lane_conversations::operation op;
        op.lane = op.id = op.outgoing = 1;
        op.attached = op.ready = true;
        catalog.entries_.at(1).busy = true;
        catalog.finish(op, state);
    }
    static bool exists(const kvmem_lane_conversations& c, int id) { return c.entries_.count(id); }
    static auto& entry(kvmem_lane_conversations& c, int id) { return c.entries_.at(id); }
    static void check_bindings(const kvmem_lane_conversations& c) {
        std::set<int> ids;
        for(size_t lane=0;lane<c.resident_.size();++lane) {
            const auto id=c.resident_[lane];
            assert(ids.insert(id).second);
            assert(c.entries_.at(id).lane==int(lane));
        }
        for(const auto& item:c.entries_)
            if(item.second.lane>=0) assert(c.resident_.at(item.second.lane)==item.first);
    }
};

int main() {
    using access = kvmem_lane_conversations_test_access;
    {
        // A per-session spill can release private attachments but must leave
        // shared checkpoint vectors and their accounting intact for siblings.
        const auto folder=std::filesystem::temp_directory_path()/"kvmem-shared-checkpoint-spill";
        std::filesystem::create_directories(folder);
        auto disk=std::make_shared<kvmem::SpillFile>(folder,4096);
        auto account=std::make_shared<MultimodalCheckpointAccounting>();
        kvmem_conversation parent;
        auto shared=std::make_shared<MultimodalCheckpointData>();
        shared->recurrent.assign(128,7);shared->draft_carry.assign(16,9);shared->tail_mean.values.assign(4,2.f);
        shared->accounting=account;account->add(shared->resident_bytes());
        MultimodalCheckpoint cp;cp.row=64;cp.data=shared;parent.mm_checkpoints.remember(cp);
        auto child=llama_driver_fork_conversation(parent,64);
        auto own=std::make_shared<MultimodalCheckpointData>();own->recurrent.assign(32,5);
        own->accounting=account;account->add(own->resident_bytes());
        cp.row=96;cp.data=own;parent.mm_checkpoints.remember(cp);
        const auto charge=account->live_bytes.load();
        std::vector<kvmem::SnapshotBuffer> buffers;llama_driver_conversation_buffers(parent,buffers);
        kvmem_session_payload manifest(1,1,buffers);kvmem_session_files files(folder,disk);
        for(uint32_t i=0;i<manifest.chunks.size();++i)manifest.spill(files,i);
        assert(files.contains(1) && own->recurrent.empty());
        assert(shared->recurrent==std::vector<uint8_t>(128,7));
        assert(shared->draft_carry==std::vector<uint8_t>(16,9) && shared->tail_mean.values==std::vector<float>(4,2.f));
        assert(account->live_bytes==charge-32 && !child.payload && !shared->cold);
        for(uint32_t i=0;i<manifest.chunks.size();++i)manifest.restore(files,i);
        assert(own->recurrent==std::vector<uint8_t>(32,5) && account->live_bytes==charge);
        parent.mm_checkpoints.clear();shared.reset();cp.data.reset();
        assert(child.mm_checkpoints.back().data->recurrent.size()==128);
    }
    {
        // All cold restore callers use one materializer. Two readers get
        // private buffers; releasing or failing one restore preserves source.
        const auto folder=std::filesystem::temp_directory_path()/"kvmem-shared-checkpoint-load";
        std::filesystem::create_directories(folder);
        auto disk=std::make_shared<kvmem::SpillFile>(folder,4096);
        auto account=std::make_shared<MultimodalCheckpointAccounting>();
        auto source=std::make_shared<MultimodalCheckpointData>();
        source->tail_mean.values.assign(4,2.f);
        source->accounting=account;account->add(source->resident_bytes());
        const std::vector<uint8_t> recurrent(128,7),carry(16,9);
        source->cold_recurrent=recurrent.size();source->cold_carry=carry.size();
        source->cold=disk->reserve(kvmem::checkpoint_image_size(144,2));
        kvmem::write_checkpoint_image(source->cold,uint64_t(reinterpret_cast<uintptr_t>(source.get())),
            {{recurrent.data(),recurrent.size()},{carry.data(),carry.size()}});
        kvmem_conversation parent;
        MultimodalCheckpoint cp;cp.row=64;cp.data=source;parent.mm_checkpoints.remember(cp);
        auto child=llama_driver_fork_conversation(parent,64);
        const auto charge=account->live_bytes.load(),disk_charge=disk->charged_bytes();
        LlamaEngineState a,b;
        kvmem::CheckpointPolicy policy;policy.enabled=false;
        a.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        b.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        std::shared_ptr<const MultimodalCheckpointData> first,second;
        std::exception_ptr error_a,error_b;
        std::thread one([&]{try{first=llama_driver_materialize_checkpoint(a,child.mm_checkpoints.back().data);}catch(...){error_a=std::current_exception();}});
        std::thread two([&]{try{second=llama_driver_materialize_checkpoint(b,source);}catch(...){error_b=std::current_exception();}});
        one.join();two.join();assert(!error_a && !error_b);
        assert(first!=source && second!=source && first!=second);
        assert(first->recurrent==recurrent && second->draft_carry==carry && first->tail_mean.values==source->tail_mean.values);
        assert(source->cold && source->recurrent.empty() && source->draft_carry.empty());
        assert(disk->charged_bytes()==disk_charge && account->live_bytes==charge+first->resident_bytes()+second->resident_bytes());
        assert(llama_driver_materialize_checkpoint(a,first)==first); // rollback reuses its private result
        first.reset();second.reset();assert(account->live_bytes==charge);
        // A real checksum failure releases only this read's temporary charge.
        const uint8_t bad=8,good=7;source->cold.write(40,&bad,1);
        bool rejected=false;
        try{auto invalid=llama_driver_materialize_checkpoint(a,source);(void)invalid;}catch(const kvmem::SnapshotCorrupt&){rejected=true;}
        assert(rejected && source->cold && source->recurrent.empty() && account->live_bytes==charge);
        source->cold.write(40,&good,1);
        auto recovered=llama_driver_materialize_checkpoint(a,source);
        assert(recovered->recurrent==recurrent && source->cold && disk->charged_bytes()==disk_charge);
        recovered.reset();assert(account->live_bytes==charge);
        policy.host_bytes=charge; a.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        rejected=false;
        try{auto denied=llama_driver_materialize_checkpoint(a,source);(void)denied;}catch(const std::runtime_error&){rejected=true;}
        assert(rejected && account->live_bytes==charge && source->cold);
        // The exclusive path still publishes in place and charges only growth.
        auto private_source=std::make_shared<MultimodalCheckpointData>();
        private_source->cold=source->cold;private_source->cold_recurrent=128;private_source->cold_carry=16;
        private_source->cold_identity=uint64_t(reinterpret_cast<uintptr_t>(source.get()));
        private_source->accounting=account;account->add(private_source->resident_bytes());
        a.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>();
        auto loaded=llama_driver_materialize_checkpoint(a,private_source);
        assert(loaded==private_source && !loaded->cold && loaded->recurrent==recurrent && source->cold);
    }
    // An empty resident slot and an LRU resident victim cannot be moved onto
    // the source lane. Extra catalog capacity still permits a shared branch.
    for(bool retained_limit:{false,true}) for(bool empty_target:{false,true}) {
        ServerState a,b;
        kvmem_lane_conversations catalog({&a,&b},2,0,retained_limit);
        auto prefix=std::make_shared<kvmem_prompt>(std::vector<llama_token>(128,11));
        auto& source=access::entry(catalog,0);
        source.client_id="A";source.committed.prompt=prefix;
        source.committed.match.rows=source.committed.match.live_row=128;
        source.committed.match.ckpt_rows={128};
        if(!empty_target) {
            auto& target=access::entry(catalog,1);target.client_id="B";
            target.committed.prompt=std::make_shared<kvmem_prompt>(std::vector<llama_token>(128,22));
            target.committed.match.rows=target.committed.match.live_row=128;
            target.committed.match.ckpt_rows={128};
        }
        auto ids=prefix->tokens;ids.push_back(33);kvmem_prompt prompt(ids);
        auto queued_a=catalog.pin("A"),queued_b=catalog.pin("B"),request=catalog.pin("C");
        assert(catalog.preview(prompt,"C",false,request)==128);
        kvmem_lane_conversations::operation op;
        assert(catalog.select(prompt,"C",false,{false,false},true,op)>=0);
        access::check_bindings(catalog);
        if(retained_limit) {
            assert(op.added && op.source==0 && op.lane==0 && op.prepare_begin==128);
            assert(access::entry(catalog,op.id).lane==-1);
        } else {
            assert(op.fresh && op.source==-1 && op.id==1 && op.lane==1 && op.prepare_begin==0);
            assert(!source.switching && !source.busy && source.committed.prompt==prefix);
        }
        assert(catalog.select(prompt,"D",false,{true,true},false,op)==-1);
    }
    // Both queued owners edit before their old suffix. Neither request may
    // depend on the other releasing its preview pin before it can execute.
    {
        ServerState a,b;
        kvmem_lane_conversations catalog({&a,&b},2,0,false);
        for(int id=0;id<2;++id) {
            auto& held=access::entry(catalog,id);held.client_id=id?"B":"A";
            held.committed.prompt=std::make_shared<kvmem_prompt>(std::vector<llama_token>(256,id?22:11));
            held.committed.match.rows=held.committed.match.live_row=256;
            held.committed.match.ckpt_rows={128,256};
        }
        auto pa=catalog.pin("A"),pb=catalog.pin("B");
        std::vector<llama_token> aa(128,11),bb(128,22);aa.push_back(33);bb.push_back(44);
        kvmem_prompt qa(aa),qb(bb);kvmem_lane_conversations::operation oa,ob;
        assert(catalog.preview(qa,"A",false,pa)==128 && catalog.preview(qb,"B",false,pb)==128);
        assert(catalog.select(qa,"A",false,{false,false},true,oa)==0);
        assert(catalog.select(qb,"B",false,{true,false},true,ob)==1);
        assert(oa.fresh && ob.fresh && oa.source==-1 && ob.source==-1);
        assert(oa.prepare_begin==0 && ob.prepare_begin==0);
        access::check_bindings(catalog);
    }
    for (bool retained_limit : {false, true}) {
        ServerState a, b;
        kvmem_lane_conversations catalog({&a,&b},2,0,retained_limit);
        auto ap=std::make_shared<kvmem_prompt>(std::vector<llama_token>(128,11));
        auto bp=std::make_shared<kvmem_prompt>(std::vector<llama_token>(128,22));
        for(int id=0;id<2;++id) {
            auto& entry=access::entry(catalog,id);
            entry.client_id=id==0?"A":"B";
            entry.committed.prompt=id==0?ap:bp;
            entry.committed.match.rows=entry.committed.match.live_row=128;
            entry.committed.match.ckpt_rows={128};
        }
        auto ids=ap->tokens;ids.push_back(33);kvmem_prompt edited(ids);
        auto pin=catalog.pin("B"); // Keep the actual HTTP request's pin alive.
        kvmem_lane_conversations::operation op;
        assert(catalog.preview(edited,"B",false,pin,true)==128);
        assert(catalog.select(edited,"B",false,{false,false},false,op,true)>=0);
        const int lane=catalog.select(edited,"B",false,{false,false},true,op,true);
        if(retained_limit) {
            assert(lane==0 && op.source==0 && op.added && op.prepare_begin==128);
        } else {
            assert(lane==1 && op.id==1 && op.outgoing==1 && op.source==-1);
            assert(op.fresh && !op.added && op.prepare_begin==0);
            assert(!access::entry(catalog,0).switching && !access::entry(catalog,0).busy);
        }
        assert(access::entry(catalog,0).committed.prompt==ap);
        assert(access::entry(catalog,0).committed.match.rows==128);
    }
    {
        ServerState state;
        kvmem_lane_conversations catalog({&state}, 8, 0, true);
        kvmem_lane_conversations::operation failed, own, shared;
        failed.prepare_begin = 128; failed.source = 7;
        own.prepare_begin = 64;
        shared.prepare_begin = 128; shared.source = 7;
        catalog.observe_reuse(failed, 0);
        catalog.observe_reuse(failed, 0); // callbacks cannot double-charge
        catalog.observe_reuse(own, 64);
        catalog.observe_reuse(shared, 96);
        const auto status = catalog.status();
        assert(status.at("history_hits") == 2 && status.at("history_misses") == 1);
        assert(status.at("own_history_hits") == 1 && status.at("shared_history_hits") == 1);
        assert(status.at("planned_reuse_tokens") == 320 && status.at("actual_reuse_tokens") == 160);
    }
    {
        kvmem_conversation source;
        auto data = std::make_shared<MultimodalCheckpointData>();
        data->recurrent.assign(16, 7);
        MultimodalCheckpoint cp; cp.row=64; cp.data=data;
        source.mm_checkpoints.remember(cp);
        auto child=llama_driver_fork_conversation(source,64);
        assert(child.mm_checkpoints.back().data==data && data->shared.load());
        source.mm_checkpoints.clear(); cp.data.reset(); data.reset();
        assert(child.mm_checkpoints.back().data->recurrent.size()==16);
    }
    {
        ServerState a, b;
        kvmem_lane_conversations catalog({&a, &b}, 8, 0, true);
        auto op = access::handover(catalog);
        access::finish_other_lane(catalog, b);
        assert(!access::exists(catalog, 0));
        a.cached_tokens = {11, 12, 13};
        catalog.finish(op, a);
        assert(access::entry(catalog, 2).committed.match.rows == 3);
        assert(!access::entry(catalog, 2).busy);
        assert(catalog.status()["retained_sessions"] == 1);
        catalog.finish(op, a); // completion remains idempotent
        assert(access::entry(catalog, 2).committed.match.rows == 3);
    }
    for (bool fail : {false, true}) {
        ServerState a, b;
        kvmem_lane_conversations catalog({&a, &b}, 8, 0);
        auto op = access::handover(catalog);
        // A newer operation has reserved the old owner. A's successful or
        // exceptional completion must not release that operation's protection.
        auto& outgoing = access::entry(catalog, 0);
        outgoing.switching = true;
        outgoing.committed.match.rows = 7;
        a.cached_tokens = {21, 22};
        a.fail_finish = fail;
        catalog.finish(op, a);
        assert(outgoing.switching && outgoing.committed.match.rows == 7);
        assert(!access::entry(catalog, 2).busy);
        assert(access::entry(catalog, 2).committed.match.rows == (fail ? 0 : 2));
    }
    {
        ServerState a, b;
        kvmem_lane_conversations catalog({&a, &b}, 8, 0);
        auto op = access::handover(catalog);
        op.ready = false;
        auto& outgoing = access::entry(catalog, 0);
        outgoing.switching = true;
        outgoing.committed.match.rows = 7;
        a.cached_tokens = {31};
        catalog.finish(op, a);
        assert(!outgoing.switching && outgoing.committed.match.rows == 0);
        assert(!access::entry(catalog, 2).busy);
        assert(access::entry(catalog, 2).committed.match.rows == 0);
    }
    for(int protection=0;protection<3;++protection) {
        ServerState a, b;
        const auto account=a.mm_checkpoint_accounting;
        {
            kvmem_lane_conversations catalog({&a,&b},8,0);
            auto op=access::handover(catalog);
            auto data=std::make_shared<MultimodalCheckpointData>();
            data->recurrent.resize(128);
            const auto charged=data->resident_bytes();
            account->limit=charged;
            data->accounting=account;account->add(charged);
            auto& outgoing=access::entry(catalog,0);
            outgoing.payload.mm_live_checkpoint=std::move(data);
            outgoing.busy=protection==0;outgoing.switching=protection==1;outgoing.refs=protection==2;
            a.mm_checkpoint_reclaim(account,64);
            assert(access::exists(catalog,0) && account->live_bytes==charged);
            outgoing.busy=outgoing.switching=false;outgoing.refs=0;
            a.mm_checkpoint_reclaim(account,64);
            assert(!access::exists(catalog,0) && account->live_bytes==0);
            catalog.finish(op,a); // evicting the old owner cannot invalidate handover completion
            assert(!access::entry(catalog,2).busy);
        }
        assert(!a.mm_checkpoint_reclaim && !b.mm_checkpoint_reclaim);
    }
    std::cout << "PASS lane conversation eviction, ownership, checkpoint admission and failed handover\n";
}
