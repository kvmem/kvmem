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
    static void seed_version(kvmem_lane_conversations& c,int id,uint64_t group,uint64_t used) {
        auto& held=c.entries_[id];held.cache_group=group;held.used=used;
        held.committed.match.rows=held.committed.match.live_row=128;
    }
    static void retain(kvmem_lane_conversations& c,int lane) { c.enforce_sessions(lane); }
    static void classify(kvmem_lane_conversations& c,int id) { c.classify_content_group_locked(id,c.entries_.at(id)); }
    static auto payload_bytes(const kvmem_conversation& c) { return kvmem_lane_conversations::payload_bytes(c); }
    static void cpu_parked(kvmem_lane_conversations& c,int id) { c.entries_[id].cpu_arena_parked=true; }
    static void seed_empty(kvmem_lane_conversations& c,int id) { c.entries_.try_emplace(id); }
    template<class Run> static void reserve_cpu(kvmem_lane_conversations& c,
            const kvmem_lane_conversations::operation& op, Run run) {
        kvmem_lane_conversations::cpu_arena_reservation reservation(c,op);
        run();
    }
    static void seed_checkpoint(kvmem_lane_conversations& c,int id,uint64_t group,uint64_t used,
            const std::shared_ptr<MultimodalCheckpointAccounting>& account) {
        seed_version(c,id,group,used);
        auto data=std::make_shared<MultimodalCheckpointData>();data->tail_mean.values.resize(32);
        data->accounting=account;account->add(data->resident_bytes());
        c.entries_.at(id).payload.mm_live_checkpoint=std::move(data);
    }
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
    const auto check_fork_coverage = [](auto & source) {
        kvmem::CheckpointPolicy policy;
        policy.enabled = true; policy.count = 12;
        source.mm_archive = kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        source.cached_tokens.resize(256);
        std::vector<std::shared_ptr<MultimodalCheckpointData>> data;
        for (int row : {32, 96, 160, 224}) {
            auto state = std::make_shared<MultimodalCheckpointData>();
            data.push_back(state);
            MultimodalCheckpoint checkpoint;
            checkpoint.row = row; checkpoint.data = state;
            assert(source.mm_archive.remember(checkpoint, kvmem::CoveragePurpose::Periodic));
        }
        auto branch = llama_driver_fork_conversation(source, 160);
        std::set<int> rows;
        for (const auto & entry : branch.mm_archive) {
            if (entry.info.identity) rows.insert(entry.record.row);
        }
        assert((rows == std::set<int>{32, 96, 160}));
        assert(branch.mm_archive.policy().count == policy.count);
        assert(data[0]->shared && data[1]->shared && data[2]->shared && !data[3]->shared);
        branch.mm_archive.truncate(96);
        assert(source.mm_archive.size() == 4);
        assert(branch.mm_archive.size() == 2);
    };
    kvmem_conversation parked;
    check_fork_coverage(parked);
    LlamaEngineState live;
    check_fork_coverage(live);

    using access = kvmem_lane_conversations_test_access;
    for(bool exact_tip:{false,true}) {
        ServerState state;kvmem_lane_conversations catalog({&state},8,0,true);
        auto& source=access::entry(catalog,0);
        source.committed.prompt=std::make_shared<kvmem_prompt>(std::vector<llama_token>(256,11));
        source.committed.match.rows=source.committed.match.live_row=256;
        source.committed.match.ckpt_rows={192};source.committed.input_end=220;
        if(exact_tip)source.committed.match.ckpt_rows.push_back(256);
        state.cached_tokens=source.committed.prompt->tokens;
        state.cached_prompt=std::make_shared<kvmem_prompt>(state.cached_tokens);
        for(int row:source.committed.match.ckpt_rows) {
            auto data=std::make_shared<MultimodalCheckpointData>();data->tail_mean.values.resize(4);
            MultimodalCheckpoint cp;cp.row=row;cp.data=data;state.mm_checkpoints.remember(cp);
        }
        auto tokens=source.committed.prompt->tokens;tokens.push_back(22);
        kvmem_prompt request(tokens);kvmem_lane_conversations::operation op;
        assert(catalog.select(request,"",false,{false},true,op)==0);
        assert(op.resident_reuse==exact_tip && op.source==0 && op.prepare_begin==(exact_tip?256:192));
        assert(source.committed.input_end==220 && source.committed.match.live_row==256);
        assert(source.committed.match.ckpt_rows.front()==192); // source remains complete and leased
        if(!exact_tip) {
            auto child=llama_driver_fork_conversation(state,int(op.prepare_begin));
            assert(child.mm_live_row==192 && child.mm_checkpoints.front().row==192);
            assert(state.cached_tokens.size()==256 && state.mm_checkpoints.front().row==192);
            bool refused_tip=false;
            try { auto invalid=llama_driver_snapshot_conversation(state,256); }
            catch(const std::exception&) { refused_tip=true; }
            assert(refused_tip && state.cached_tokens.size()==256 && state.mm_checkpoints.size()==1);
            // Cancellation may roll the private child back before its planned
            // canonical boundary. An unknown boundary must not borrow a group
            // or overwrite the source's complete canonical input.
            auto& rolled=access::entry(catalog,op.id);
            rolled.committed.prompt=child.cached_prompt->prefix(192);
            rolled.committed.match.rows=rolled.committed.match.live_row=192;
            rolled.committed.input_end=0;access::classify(catalog,op.id);
            assert(rolled.cache_group!=source.cache_group && source.committed.input_end==220);
        }
    }
    {
        // Archive holes also occupy a copied vector allocation, independently
        // of native shared payload charges and fixed container object size.
        kvmem_conversation archived,empty;
        kvmem::CheckpointPolicy policy;policy.enabled=true;policy.count=8;
        archived.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        for(int row=1;row<=4;++row) {
            auto data=std::make_shared<MultimodalCheckpointData>();data->tail_mean.values.resize(4);
            MultimodalCheckpoint cp;cp.row=row;cp.data=data;
            assert(archived.mm_archive.remember(cp,kvmem::CoveragePurpose::Periodic));
        }
        archived.mm_archive.truncate(0);assert(archived.mm_archive.size()==0);
        assert(access::payload_bytes(archived)-access::payload_bytes(empty)==
            archived.mm_archive.capacity()*sizeof(kvmem::CheckpointArchive<MultimodalCheckpoint>::Entry));
    }
    {
        // Two siblings can both prepare against the same accepted document.
        // Publication rechecks the whole group after the first sibling commits.
        ServerState state;kvmem_lane_conversations catalog({&state},8,0,true);
        for(int id=0;id<4;++id)access::seed_version(catalog,id,1,uint64_t(id+1));
        const auto set=[&](int id,std::vector<llama_token> tokens) {
            auto& e=access::entry(catalog,id);e.committed.input_end=uint32_t(tokens.size());
            e.committed.prompt=std::make_shared<kvmem_prompt>(std::move(tokens));
        };
        set(0,std::vector<llama_token>(128,11));
        auto a=std::vector<llama_token>(128,11);a.push_back(22);set(1,a);
        auto b=std::vector<llama_token>(128,11);b.push_back(33);set(2,b);
        access::entry(catalog,2).committed.match.rows=0;
        access::entry(catalog,3).committed.match.rows=0;
        access::classify(catalog,1);const auto ga=access::entry(catalog,1).cache_group;
        access::entry(catalog,2).committed.match.rows=129;access::classify(catalog,2);
        const auto gb=access::entry(catalog,2).cache_group;assert(ga!=gb);
        a.push_back(44);set(3,a);access::entry(catalog,3).committed.match.rows=130;
        access::entry(catalog,3).cache_group=gb; // prepare used the old document/other hint
        access::classify(catalog,3);assert(access::entry(catalog,3).cache_group==ga);
    }
    {
        // A resident source retains every complete restore boundary, not just
        // its endpoint: the next Chat template can rewrite that endpoint tail.
        ServerState state;state.cached_tokens.assign(128,11);
        state.cached_prompt=std::make_shared<kvmem_prompt>(state.cached_tokens);
        kvmem::CheckpointPolicy policy;policy.enabled=true;policy.count=8;
        state.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        for(int row:{64,96,128,160}) {
            auto data=std::make_shared<MultimodalCheckpointData>();data->tail_mean.values.assign(4,float(row));
            MultimodalCheckpoint cp;cp.row=row;cp.data=data;
            state.mm_checkpoints.remember(cp);state.mm_archive.remember(cp,kvmem::CoveragePurpose::Input);
        }
        auto snapshot=llama_driver_snapshot_conversation(state,128);
        kvmem_store_match match;match.rows=match.live_row=128;match.lcp=127;
        for(const auto& cp:snapshot.mm_checkpoints) {assert(cp.row<=128 && cp.data->shared.load());match.ckpt_rows.push_back(cp.row);}
        for(const auto& cp:snapshot.mm_archive)if(cp.info.identity)assert(cp.record.row<=128 && cp.record.data->shared.load());
        int keep=0;assert(kvmem_store_checkpoint(match,127,keep) && keep==96);
        state.mm_checkpoints.clear();state.mm_archive.clear();
        assert(snapshot.mm_checkpoints.size()==3 && snapshot.mm_archive.size()==3);
    }
    for(bool matches_full_input:{false,true}) {
        ServerState state;kvmem_lane_conversations catalog({&state},8,0,true);
        auto prefix=std::make_shared<kvmem_prompt>(std::vector<llama_token>(256,11));
        auto& source=access::entry(catalog,0);source.committed.prompt=prefix;
        source.committed.match.rows=source.committed.match.live_row=256;
        source.committed.match.ckpt_rows={128,192,256};source.committed.input_end=220;
        std::vector<llama_token> tokens(matches_full_input?256:200,11);tokens.push_back(22);
        kvmem_prompt request(tokens);kvmem_lane_conversations::operation op;
        assert(catalog.select(request,"shared-parent",false,{false},true,op)>=0);
        assert(op.content_continuation==matches_full_input); // 64-token restore gate is unchanged
    }
    {
        ServerState state;kvmem_lane_conversations catalog({&state},8,0,true);
        auto prefix=std::make_shared<kvmem_prompt>(std::vector<llama_token>(128,11));
        auto& source=access::entry(catalog,0);
        source.committed.prompt=prefix;
        source.committed.match.rows=source.committed.match.live_row=128;
        source.committed.match.ckpt_rows={128};source.switching=true;
        auto tokens=prefix->tokens;tokens.push_back(22);kvmem_prompt request(tokens);
        auto preview=catalog.pin("arbitrary");kvmem_lane_conversations::operation op;
        assert(catalog.preview(request,"arbitrary",false,preview)==128);
        assert(catalog.select(request,"arbitrary",false,{false},false,op)==-1);
        source.switching=false;
        assert(catalog.select(request,"arbitrary",false,{false},false,op)==0 && op.prepare_begin==128);
    }
    {
        // The arena quota is independent of raw H/D and counts in-flight work
        // before a native fork can construct another full runtime. Model-free
        // commitments let us deterministically hold a concurrent reservation.
        ServerState state;state.kparams.cpu_bytes=1024;
        kvmem_lane_conversations catalog({&state},2,0,true);
        access::seed_empty(catalog,1);
        for(int id=2;id<=4;++id)access::cpu_parked(catalog,id);
        kvmem_lane_conversations::operation op;
        op.id=1;op.outgoing=0;op.source=0;op.resident_reuse=true;
        assert(catalog.status().at("cpu_arena_budget_bytes")==5120);
        access::reserve_cpu(catalog,op,[&] {
            assert(catalog.status().at("cpu_arena_live_charge_bytes")==4096);
            assert(catalog.status().at("cpu_arena_reserved_bytes")==1024);
            bool refused=false;
            std::thread other([&] {
                try {access::reserve_cpu(catalog,op,[]{assert(false);});}
                catch(const std::runtime_error&) {refused=true;}
            });
            other.join();assert(refused);
            assert(catalog.status().at("cpu_arena_charged_bytes")==5120);
        });
        assert(catalog.status().at("cpu_arena_reserved_bytes")==0);
        bool injected=false;
        try {access::reserve_cpu(catalog,op,[]{throw std::runtime_error("injected allocation failure");});}
        catch(const std::runtime_error&) {injected=true;}
        assert(injected && catalog.status().at("cpu_arena_reserved_bytes")==0);
        // Exhaust the independent arena quota. attach must refuse before its
        // native fork (this fixture deliberately has no forkable checkpoint).
        auto original=std::make_shared<kvmem_prompt>(std::vector<llama_token>(128,11));
        access::entry(catalog,0).committed.prompt=original;
        access::entry(catalog,0).committed.match.rows=128;
        access::cpu_parked(catalog,5);
        bool denied_before_native=false;
        try {catalog.attach(op,state,"same-parent-key");}
        catch(const std::runtime_error& error) {
            denied_before_native=std::string(error.what()).find("legacy CPU arena capacity")!=std::string::npos;
        }
        assert(denied_before_native && access::entry(catalog,0).committed.prompt==original);
        assert(access::entry(catalog,0).committed.match.rows==128 && !op.attached);
        assert(catalog.status().at("cpu_arena_reserved_bytes")==0);
        assert(catalog.status().at("cpu_arena_charged_bytes")==5120);
        assert(catalog.status().at("cpu_arena_reservation_failures")==2);
        // An existing parked arena is reused by a fresh handover and does not
        // need an additional commitment merely because the quota is full.
        op.resident_reuse=false;op.source=-1;
        access::entry(catalog,1).cpu_arena_parked=true;
        access::entry(catalog,5).cpu_arena_parked=false;
        access::reserve_cpu(catalog,op,[&] {assert(catalog.status().at("cpu_arena_reserved_bytes")==0);});
    }
    {
        // Four physical directory slots retain two independent content groups.
        // Repeated versions from the hot group must not displace the only
        // root version, despite its older LRU stamp or a shared client label.
        ServerState state;
        kvmem_lane_conversations catalog({&state},2,0,true);
        access::seed_version(catalog,0,2,100);
        access::seed_version(catalog,1,1,1);
        access::seed_version(catalog,2,2,20);
        access::seed_version(catalog,3,2,30);
        access::seed_version(catalog,4,2,40);
        for(int id=0;id<5;++id)access::entry(catalog,id).client_id="same-parent-key";
        access::retain(catalog,0);
        assert(access::exists(catalog,1) && !access::exists(catalog,2));
        assert(catalog.status().at("retained_groups")==2);
        assert(catalog.status().at("retained_snapshot_count")==4);
    }
    {
        // The native checkpoint-host pool must use the same fairness as the
        // directory. This reproduces a parent miss with only four histories:
        // a single root was previously reclaimed before a hot group's older
        // version even though the directory count was below its quota.
        ServerState state;const auto account=state.mm_checkpoint_accounting;
        kvmem_lane_conversations catalog({&state},8,0,true);
        access::seed_checkpoint(catalog,1,1,1,account);
        access::seed_checkpoint(catalog,2,2,20,account);
        access::seed_checkpoint(catalog,3,2,30,account);
        account->limit=account->live_bytes.load();
        const auto reclaim=access::entry(catalog,2).payload.mm_live_checkpoint->resident_bytes();
        state.mm_checkpoint_reclaim(account,reclaim);
        assert(access::exists(catalog,1) && !access::exists(catalog,2) && access::exists(catalog,3));
        assert(account->live_bytes.load()+reclaim==account->limit);
    }
    {
        // Session manifests serialize private statistics only. Immutable state
        // records retain one public owner, even when a checkpoint is shared.
        const auto folder=std::filesystem::temp_directory_path()/"kvmem-shared-checkpoint-spill";
        std::filesystem::create_directories(folder);
        auto disk=std::make_shared<kvmem::SpillFile>(folder,4096);
        auto account=std::make_shared<MultimodalCheckpointAccounting>();
        auto storage=std::make_shared<kvmem::HostKvStorage>(4096,disk);
        std::atomic_store(&account->storage_owner,storage);
        const auto make_state=[&](size_t n,uint8_t value) {
            auto data=std::make_shared<MultimodalCheckpointData>();
            const std::vector<uint8_t> payload(n,value);
            data->native_image=storage->allocate(kvmem::checkpoint_image_size(n,2));
            kvmem::write_checkpoint_image(data->native_image,uint64_t(reinterpret_cast<uintptr_t>(data.get())),
                {{payload.data(),n},{nullptr,0}});
            data->native_image.seal();data->residency=data->native_image.lease();data->recurrent_bytes=n;
            return data;
        };
        kvmem_conversation parent;
        auto shared=make_state(128,7);shared->tail_mean.values.assign(4,2.f);
        shared->accounting=account;account->add(shared->metadata_bytes());
        MultimodalCheckpoint cp;cp.row=64;cp.data=shared;parent.mm_checkpoints.remember(cp);
        auto child=llama_driver_fork_conversation(parent,64);
        auto own=make_state(32,5);own->tail_mean.values.assign(8,5.f);
        own->accounting=account;account->add(own->metadata_bytes());
        cp.row=96;cp.data=own;parent.mm_checkpoints.remember(cp);
        const auto charge=account->used(),payload_charge=storage->charged_bytes();
        std::vector<kvmem::SnapshotBuffer> buffers;llama_driver_conversation_buffers(parent,buffers);
        kvmem_session_payload manifest(1,1,buffers);kvmem_session_files files(folder,disk);
        for(uint32_t i=0;i<manifest.chunks.size();++i)manifest.spill(files,i);
        assert(files.contains(1) && own->tail_mean.values.empty());
        assert(shared->tail_mean.values==std::vector<float>(4,2.f));
        assert(storage->charged_bytes()==payload_charge && account->used()==charge-32 && !child.payload);
        for(uint32_t i=0;i<manifest.chunks.size();++i)manifest.restore(files,i);
        assert(own->tail_mean.values==std::vector<float>(8,5.f) && account->used()==charge);
        parent.mm_checkpoints.clear();shared.reset();cp.data.reset();
        LlamaEngineState reader;
        const auto read=llama_driver_read_checkpoint(reader,child.mm_checkpoints.back().data);
        assert(read->recurrent==std::vector<uint8_t>(128,7));
    }
    {
        // Concurrent readers own separate execution-local Q staging. Neither
        // promotes/mutates the immutable source nor charges the parent's Q.
        const auto folder=std::filesystem::temp_directory_path()/"kvmem-shared-checkpoint-load";
        std::filesystem::create_directories(folder);
        auto disk=std::make_shared<kvmem::SpillFile>(folder,4096);
        auto account=std::make_shared<MultimodalCheckpointAccounting>();
        auto storage=std::make_shared<kvmem::HostKvStorage>(4096,disk);
        std::atomic_store(&account->storage_owner,storage);
        const std::vector<uint8_t> recurrent(128,7),carry(16,9);
        const auto make_state=[&] {
            auto data=std::make_shared<MultimodalCheckpointData>();
            data->native_image=storage->allocate(kvmem::checkpoint_image_size(144,2));
            kvmem::write_checkpoint_image(data->native_image,uint64_t(reinterpret_cast<uintptr_t>(data.get())),
                {{recurrent.data(),recurrent.size()},{carry.data(),carry.size()}});
            data->native_image.seal();data->recurrent_bytes=128;data->carry_bytes=16;
            data->accounting=account;account->add(data->metadata_bytes());
            return data;
        };
        auto source=make_state();assert(storage->demote(source->native_image));
        const auto charge=account->used(),disk_charge=disk->charged_bytes();
        LlamaEngineState a,b;
        kvmem::CheckpointPolicy policy;policy.enabled=true;policy.host_bytes=4096;
        a.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        b.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        std::unique_ptr<CheckpointRead> first,second;
        std::exception_ptr error_a,error_b;
        std::thread one([&]{try{first=llama_driver_read_checkpoint(a,source);}catch(...){error_a=std::current_exception();}});
        std::thread two([&]{try{second=llama_driver_read_checkpoint(b,source);}catch(...){error_b=std::current_exception();}});
        one.join();two.join();assert(!error_a && !error_b);
        assert(first.get()!=second.get() && first->recurrent==recurrent && second->carry==carry);
        assert(!source->native_image.resident_bytes() && disk->charged_bytes()==disk_charge && account->used()==charge);
        assert(a.mm_checkpoint_accounting->used()==144 && b.mm_checkpoint_accounting->used()==144);
        first.reset();second.reset();
        assert(!a.mm_checkpoint_accounting->used() && !b.mm_checkpoint_accounting->used());
        // Fault injection changes a private image, not the retained source.
        auto corrupt=make_state();const uint8_t bad=8;corrupt->native_image.write(40,&bad,1);corrupt->native_image.seal();
        bool rejected=false;
        try{auto invalid=llama_driver_read_checkpoint(a,corrupt);(void)invalid;}catch(const kvmem::SnapshotCorrupt&){rejected=true;}
        assert(rejected && !a.mm_checkpoint_accounting->used());
        corrupt.reset();assert(account->used()==charge);
        auto recovered=llama_driver_read_checkpoint(a,source);
        assert(recovered->recurrent==recurrent && !source->native_image.resident_bytes() && disk->charged_bytes()==disk_charge);
        recovered.reset();
        policy.host_bytes=128;a.mm_archive=kvmem::CheckpointArchive<MultimodalCheckpoint>(policy);
        rejected=false;
        try{auto denied=llama_driver_read_checkpoint(a,source);(void)denied;}catch(const std::runtime_error&){rejected=true;}
        assert(rejected && !a.mm_checkpoint_accounting->used() && account->used()==charge);
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
        assert(op.added && op.source==0 && op.lane==0 && op.prepare_begin==128);
        assert(op.resident_reuse && !op.fresh);
        assert(access::entry(catalog,op.id).lane==-1);
        assert(source.committed.prompt==prefix);
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
        assert(!oa.fresh && !ob.fresh && oa.source==0 && ob.source==1);
        assert(oa.prepare_begin==128 && ob.prepare_begin==128);
        assert(oa.added && ob.added && !oa.resident_reuse && !ob.resident_reuse);
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
        assert(lane==0 && op.source==0 && op.added && op.prepare_begin==128);
        assert(op.resident_reuse && !op.fresh);
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
        const auto unknown=kvmem_lane_conversations::reuse_observation(shared);
        assert(unknown.at("cache_semantics_revision")==2 && unknown.at("actual_cached_tokens").is_null());
        assert(unknown.at("restore_path").is_null());
        assert(kvmem_lane_conversations::reuse_observation(shared,96).at("restore_path")=="snapshot");
        assert(kvmem_lane_conversations::reuse_observation(own,64).at("restore_path")=="continuation");
        assert(kvmem_lane_conversations::reuse_observation(shared,0).at("restore_path")=="miss");
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
        data->tail_mean.values.assign(4, 7);
        MultimodalCheckpoint cp; cp.row=64; cp.data=data;
        source.mm_checkpoints.remember(cp);
        auto child=llama_driver_fork_conversation(source,64);
        assert(child.mm_checkpoints.back().data==data && data->shared.load());
        source.mm_checkpoints.clear(); cp.data.reset(); data.reset();
        assert(child.mm_checkpoints.back().data->tail_mean.values.size()==4);
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
    for (bool fail : {false, true}) {
        ServerState a, b;
        kvmem_lane_conversations catalog({&a, &b}, 8, 0);
        auto op = access::handover(catalog);
        const auto account = a.mm_checkpoint_accounting;
        access::seed_checkpoint(catalog, 0, 1, 1, account);
        account->limit = account->used();
        auto & parent = access::entry(catalog, 0);
        // Model the two read pins after successful handovers. Finishing one
        // branch, including an exceptional finish, cannot release its sibling.
        parent.refs = 2; op.source = 0; op.source_pinned = true;
        a.mm_checkpoint_reclaim(account, 1);
        assert(access::exists(catalog, 0));
        a.fail_finish = fail;
        catalog.finish(op, a);
        assert(!op.source_pinned && parent.refs == 1);
        catalog.finish(op, a);
        assert(parent.refs == 1);
        --parent.refs;
        a.mm_checkpoint_reclaim(account, 1);
        assert(!access::exists(catalog, 0));
    }
    for(int protection=0;protection<3;++protection) {
        ServerState a, b;
        const auto account=a.mm_checkpoint_accounting;
        {
            kvmem_lane_conversations catalog({&a,&b},8,0);
            auto op=access::handover(catalog);
            auto data=std::make_shared<MultimodalCheckpointData>();
            data->tail_mean.values.resize(32);
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
