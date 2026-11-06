// Real MemorySession -> Radiance allocator -> GPU copy -> sparse table roundtrip.
#include "native_memory.hpp"
#include "native_pages.hpp"
#include "page_transfer.hpp"
#include "state_checkpoint.hpp"
#include "prefix_lease.hpp"
#include "kvmem/host_kv_storage.hpp"
#include "kvmem/spill_io.hpp"
#include "storage_fault_fixture.hpp"
#include "mem/kv.h"
#include "mem/pools.h"
#include "rad_internal.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <random>
#include <future>

using namespace kvmem;
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)

// Exercise bytewise archival/checkpoints for both packed widths and f16 state.
static unsigned packed_row_half = 8;
static uint32_t state_type = RAD_F32;
struct Engine {
    rad::Pools pools;
    rad::KVManager kv;
    RadStream stream = nullptr;
    std::vector<rad::KVGroupInfo> infos{3};
    Engine(bool hybrid, int max_sequences = 1, uint32_t dtype = RAD_BF16, bool speculative = false) {
        rad::Config c;
        c.vram_weights_mib = c.vram_kv_mib = c.gpu_headroom_mib = 1;
        c.host_pool_mib = 0; c.max_ctx = 256; c.max_tok = 32; c.max_seqs = max_sequences; c.n_spec = speculative ? 7 : 0;
        // Without elastic VMM the fallback uses the supplied allocator. A GPU
        // fixture must not silently back its KV pool with ordinary CPU malloc.
        CHECK(pools.configure(c, 0, &rad::rad_device_allocator(), &rad::rad_device_allocator()) == RAD_OK);
        for (int g = 1; g <= 2; ++g) {
            auto& info = infos[g];
            info.name = "full-" + std::to_string(g);
            info.decl.kind = RAD_KV_FULL; info.decl.dtype = dtype;
            info.decl.n_head_kv = 1; info.decl.head_dim = packed_row_half;
            info.block_size = 16 * g; info.layers = {3 * g};
        }
        if (hybrid) {
            infos.resize(5);
            auto& linear = infos[3];
            linear.name = "linear"; linear.layers = {1, 4};
            linear.decl.kind = RAD_KV_LINEAR; linear.decl.dtype = state_type;
            linear.decl.n_head_kv = 2; linear.decl.state_dim[0] = 8; linear.decl.state_dim[1] = 8;
            auto& conv = infos[4];
            conv.name = "conv"; conv.layers = {1, 4};
            conv.decl.kind = RAD_KV_CONV; conv.decl.dtype = RAD_BF16;
            conv.decl.n_head_kv = 2; conv.decl.head_dim = 8; conv.decl.conv_width = 4;
        }
        if (speculative) {
            infos.resize(6);
            auto& draft = infos[5];
            draft.name = "drafter-window"; draft.layers = {8}; draft.drafter = true;
            draft.decl.kind = RAD_KV_WINDOW; draft.decl.dtype = dtype;
            draft.decl.n_head_kv = 1; draft.decl.head_dim = 8;
            draft.decl.window = 64; draft.block_size = 16;
        }
        CHECK(kv.configure(infos, c, pools.kv()) == RAD_OK);
        for (int g = 1; g <= 2; ++g) infos[g].bytes_per_block = kv.plan(g)->bytes_per_block;
        CHECK(kv.add_sequence(17) == RAD_OK);
        CHECK(kv.ensure(17, 128) == RAD_OK);
        CHECK(rad_stream_create(&stream, 0) == RAD_OK);
    }
    ~Engine() { if (stream) { rad_stream_sync(stream); rad_stream_destroy(stream); } }
};

int main(int argc, char** argv) {
    try {
        // Independent history oracle: accepted convolution state is the final
        // width-1 inputs of old history followed by the accepted token prefix.
        for (uint32_t accepted = 1; accepted <= 4; ++accepted) {
            std::vector<uint16_t> snapshot = {12,13,20,21,22,23, 32,33,40,41,42,43};
            canonical_conv_window(reinterpret_cast<uint8_t*>(snapshot.data()), snapshot.size()*2, 6, 4, 2, accepted);
            for (size_t channel = 0; channel < 2; ++channel) {
                std::vector<uint16_t> history = {uint16_t(11+20*channel),uint16_t(12+20*channel),uint16_t(13+20*channel)};
                for (uint32_t t = 0; t < accepted; ++t) history.push_back(uint16_t(20+20*channel+t));
                for (size_t t = 0; t < 3; ++t) CHECK(snapshot[channel*6+t] == history[history.size()-3+t]);
                for (size_t t = 3; t < 6; ++t) CHECK(snapshot[channel*6+t] == 0);
            }
        }
        int device = -1;
        const bool host_test = argc == 2 && std::string(argv[1]) == "--host-test";
        if (argc == 2 && !host_test) device = std::stoi(argv[1]);
        else for (int i = 0; i < rad_dev_count(); ++i) {
            RadDeviceProps p{};
            if (rad_dev_props(i, &p) == RAD_OK && bool(p.is_host_backend) == host_test) { device = i; break; }
        }
        CHECK(device >= 0 && rad_dev_set(device) == RAD_OK);
        RadDeviceProps props{}; CHECK(rad_dev_props(device, &props) == RAD_OK);
        CHECK(bool(props.is_host_backend) == host_test);
        std::cout << "device=" << props.name << " arch=" << props.arch
                  << (host_test ? " HOST-ONLY TEST; NOT GPU QUALIFICATION" : " GPU TEST") << '\n';
        {
            auto leader=std::make_shared<Engine>(true,2),follower=std::make_shared<Engine>(true,2);
            const auto* p=follower->kv.plan(1);void* base=p->base;
            const std::vector<GroupPool> pools{{1,p->base,uint64_t(p->pool_bytes)}};
            const std::vector<CopySpan> spans{{1,0,0,uint64_t(p->pool_bytes)}};
            std::vector<uint8_t> bytes(size_t(p->pool_bytes),71);
            {PageTransfer copy(device,bytes.size(),follower->stream);copy.restore(spans,pools,follower->stream,bytes);CHECK(copy.drain());}
            CHECK(leader->kv.rollback(17,64)==RAD_OK);
            CHECK(leader->kv.add_sequence(29)==RAD_OK && leader->kv.ensure(29,32)==RAD_OK);
            CHECK(follower->kv.mirror_adapter_metadata(leader->kv)==RAD_OK);
            CHECK(follower->kv.seq_tokens(17)==64 && follower->kv.seq_tokens(29)==32 && follower->kv.plan(1)->base==base);
            for(const auto& g:leader->kv.plans()) {
                if(g.kind==RAD_KV_FULL) {
                    CHECK(follower->kv.block_table(17,g.index)==leader->kv.block_table(17,g.index));
                    CHECK(follower->kv.block_table(29,g.index)==leader->kv.block_table(29,g.index));
                    CHECK(follower->kv.free_blocks(g.index)==leader->kv.free_blocks(g.index));
                } else if(g.stateful())CHECK(follower->kv.state_slot(29,g.index)==leader->kv.state_slot(29,g.index));
            }
            {PageTransfer copy(device,bytes.size(),follower->stream);copy.spill(spans,pools,follower->stream);CHECK(copy.drain());CHECK(copy.payload()==bytes);}
            int32_t a=-1,b=-1;
            CHECK(leader->kv.reserve_block(1,&a)==RAD_OK && follower->kv.reserve_block(1,&b)==RAD_OK && a==b);
            leader->kv.release_block(1,a);follower->kv.release_block(1,b);
            leader->kv.free_sequence(17);CHECK(follower->kv.mirror_adapter_metadata(leader->kv)==RAD_OK);
            CHECK(!follower->kv.has_sequence(17) && follower->kv.has_sequence(29));
            auto incompatible=std::make_shared<Engine>(false,2);
            CHECK(follower->kv.mirror_adapter_metadata(incompatible->kv)==RAD_E_INVAL);
            CHECK(follower->kv.seq_tokens(29)==32 && follower->kv.plan(1)->base==base);
            std::cout<<"rank metadata mirror: index ownership, allocation order, slot reuse, local bytes and shape rejection PASS\n";
        }
        // Split prefill collects trunk statistics before the DFlash sidecars.
        // The native manager's frontier includes RESERVED lookahead, while a
        // NativeMemory transaction owns only accepted trunk history. Pressure
        // must start after the sidecars drain and their future rows are trimmed.
        {
            auto sidecar=std::make_shared<Engine>(true,2,RAD_BF16,true);
            auto& kv=sidecar->kv;
            kv.clear_state_clears();
            for(const auto& group:kv.plans())if(group.base && group.pool_bytes)
                CHECK(rad_memset_async(group.base,37+group.index,group.pool_bytes,sidecar->stream)==RAD_OK);
            CHECK(rad_stream_sync(sidecar->stream)==RAD_OK);
            kv.trim_window(17,128);
            const auto epoch=kv.window_epoch(17);
            const auto floor=kv.window_floor(17);
            const auto window=kv.block_table(17,5);
            const auto window_begin=kv.first_block_pos(17,5);
            auto read_pages=[&](uint64_t sequence,int group) {
                const auto& plan=*kv.plan(group);
                const auto& table=kv.block_table(sequence,group);
                std::vector<CopySpan> spans;
                for(size_t index=0;index<table.size();++index)
                    spans.push_back({group,uint64_t(table[index]*plan.layer_bytes_per_block()),
                        uint64_t(index*plan.layer_bytes_per_block()),uint64_t(plan.layer_bytes_per_block())});
                PageTransfer read(device,table.size()*plan.layer_bytes_per_block(),sidecar->stream);
                read.spill(spans,{{group,plan.base,uint64_t(plan.pool_bytes)}},sidecar->stream);
                CHECK(read.drain());return read.payload();
            };
            const auto full1=read_pages(17,1),full2=read_pages(17,2),accepted_window=read_pages(17,5);
            std::vector<int32_t> tokens(128,9);
            CHECK(WindowCheckpoint::required_bytes(kv,17,128)==accepted_window.size());
            auto anchor=std::make_unique<PrefixLease>(sidecar,kv,17,tokens,device,sidecar->stream);
            const MemoryStamp mark{true,{17,444},1,0,128,1};
            std::vector<BlockDescriptor> catalog;
            for(uint64_t index=0;index<4;++index)catalog.push_back({mark.session,index,index*32,32,1,0});
            auto memory=std::make_shared<NativeMemory>(MemoryBinding{sidecar,&kv,sidecar->stream,device,1},
                sidecar->infos,BackendIdentity{"radiance","unit","sidecar-boundary","test"},
                mark,catalog,32,256,1024*1024);
            MemorySession transaction(memory);
            transaction.prepare(memory->stamp(),{0,3},{3},{64,32,128,1024*1024});
            CHECK(kv.ensure(17,136,128)==RAD_OK && kv.seq_tokens(17)==136);
            bool refused=false;
            try{(void)memory->snapshot();}
            catch(const MemoryError& error){
                refused=error.code()==MemoryErrorCode::Stale &&
                    std::string(error.what()).find("engine frontier changed outside the memory boundary")!=std::string::npos;
            }
            CHECK(refused); // Do not weaken the transaction's accepted-frontier guard.
            CHECK(rad_stream_sync(sidecar->stream)==RAD_OK); // no-row prefill has no collect_step fence
            CHECK(kv.rollback(17,128)==RAD_OK);
            CHECK(transaction.abort()==AbortResult::PreviousViewPreserved);
            CHECK(memory->snapshot().stamp.frontier==128);
            CHECK(read_pages(17,1)==full1 && read_pages(17,2)==full2);
            CHECK(kv.block_table(17,5)==window && kv.first_block_pos(17,5)==window_begin);
            CHECK(read_pages(17,5)==accepted_window);
            CHECK(kv.window_epoch(17)==epoch && kv.window_floor(17)==floor);
            // Correct boundary: the trunk accepted 32 rows; the sidecar reserves
            // 8 more, finishes, and trims those 8 BEFORE preparing Pressure.
            CHECK(kv.ensure(17,160,128)==RAD_OK);
            CHECK(kv.ensure(17,168,128)==RAD_OK);
            CHECK(rad_stream_sync(sidecar->stream)==RAD_OK);
            CHECK(kv.rollback(17,160)==RAD_OK);
            memory->accept_append(160,{{mark.session,4,128,32,2,2}});
            const auto window160=kv.block_table(17,5);
            const auto bytes160=read_pages(17,5);
            transaction.prepare(memory->stamp(),{0,4},{4},{64,32,128,1024*1024});
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            while(transaction.phase()!=MemoryPhase::Ready){
                CHECK(std::chrono::steady_clock::now()<deadline);transaction.advance();
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            CHECK(transaction.publish().visible_tokens==64);
            CHECK(kv.seq_tokens(17)==160 && memory->snapshot().stamp.frontier==160);
            CHECK(kv.window_epoch(17)==epoch && kv.window_floor(17)==floor);
            CHECK(kv.block_table(17,5)==window160 && read_pages(17,5)==bytes160);
            for(int group=1;group<=2;++group)
                for(auto byte:read_pages(17,group))CHECK(byte==uint8_t(37+group));
            CHECK(kv.add_sequence(29)==RAD_OK && kv.ensure(29,1)==RAD_OK);
            kv.clear_state_clears();
            tokens.push_back(10);anchor->restore(kv,29,tokens,device,sidecar->stream);
            CHECK(kv.seq_tokens(29)==128 && read_pages(29,1)==full1 && read_pages(29,2)==full2);
            CHECK(kv.first_block_pos(29,5)==window_begin && read_pages(29,5)==accepted_window);
            CHECK(kv.window_epoch(29)>0 && kv.window_floor(29)==window_begin);
            // Tiered/forked prefixes must retain their own immutable draft bytes.
            anchor->demote(device,sidecar->stream,1024*1024);
            auto window_storage=std::make_shared<HostKvStorage>(1024*1024);
            anchor->park(window_storage,{});
            auto forked=anchor->fork();
            auto payload=forked->load_cpu();
            kv.free_sequence(29);
            CHECK(kv.add_sequence(31)==RAD_OK && kv.ensure(31,1)==RAD_OK);
            kv.clear_state_clears();
            auto truncated=payload;
            truncated.window.pop_back();
            bool truncated_refused=false;
            try{forked->restore(kv,31,tokens,device,sidecar->stream,&truncated);}
            catch(const std::runtime_error&){truncated_refused=true;}
            CHECK(truncated_refused && kv.seq_tokens(31)==1 && kv.window_epoch(31)==0);
            forked->restore(kv,31,tokens,device,sidecar->stream,&payload);
            CHECK(kv.seq_tokens(31)==128 && read_pages(31,5)==accepted_window);
            WindowCheckpoint query_window;
            query_window.save(kv,31,128,device,sidecar->stream);
            std::vector<rad::KVManager::FullTable> query_tables;
            for(int group=1;group<=2;++group)query_tables.push_back({group,kv.block_table(31,group)});
            CHECK(kv.ensure(31,160,128)==RAD_OK);
            CHECK(kv.rewind_full_tables(31,160,128,128,query_tables)==RAD_OK);
            CHECK(kv.block_table(31,5).empty() && kv.window_floor(31)==128);
            query_window.restore(kv,31,device,sidecar->stream);
            CHECK(read_pages(31,5)==accepted_window && kv.window_floor(31)==window_begin);
            const auto restored_epoch=kv.window_epoch(31);
            auto duplicate=kv.block_table(31,5);
            CHECK(duplicate.size()>1);
            duplicate[1]=duplicate[0];
            CHECK(kv.restore_window_tables(31,128,window_begin,{{5,window_begin,duplicate}})==RAD_E_INVAL);
            CHECK(kv.window_epoch(31)==restored_epoch && read_pages(31,5)==accepted_window);
            std::cout<<"DFlash split-prefill boundary: stale reserved lookahead rejected; drained trim, Pressure, window bytes/epoch and immutable anchor PASS\n";
        }
        const auto disk_directory=std::filesystem::temp_directory_path()/
            ("kvmem-radiance-native-"+std::to_string(std::random_device{}()));
        CHECK(std::filesystem::create_directory(disk_directory));
        for (bool disk_backed : {false,true})
        for (unsigned format : {0u,1u,2u,3u})
        for (bool hybrid : {false, true})
        for (bool speculative : {false, true}) {
        const uint32_t dtype = format ? RAD_F8E4M3 : RAD_BF16;
        packed_row_half = format==2 ? 200 : format==3 ? 176 : 8;
        state_type = format>=2 ? RAD_F16 : RAD_F32;
        if (speculative && !hybrid) continue; // This fixture's drafter follows its recurrent groups.
        auto engine = std::make_shared<Engine>(hybrid, 1, dtype, speculative);
        const auto initial_window = speculative ? engine->kv.block_table(17, 5) : std::vector<int32_t>{};
        const auto initial_window_epoch = engine->kv.window_epoch(17);
        std::vector<GroupPool> state_pools;
        std::vector<CopySpan> state_spans;
        std::vector<uint8_t> state_expected;
        for (const auto& p : engine->kv.plans()) if (p.stateful()) {
            const auto offset = state_expected.size();
            state_pools.push_back({p.index, p.base, uint64_t(p.pool_bytes)});
            state_spans.push_back({p.index, 0, offset, uint64_t(p.pool_bytes)});
            state_expected.resize(offset + size_t(p.pool_bytes));
            for (size_t n = offset; n < state_expected.size(); ++n) state_expected[n] = uint8_t(n * 13 + p.index);
        }
        if (hybrid) {
            PageTransfer write(device, state_expected.size(), engine->stream);
            write.restore(state_spans, state_pools, engine->stream, state_expected);
            CHECK(write.drain());
        }
        StateCheckpoint checkpoint;
        checkpoint.save(engine->kv, 17, device, engine->stream);
        if (hybrid) {
            for (const auto& p : engine->kv.plans()) if (p.stateful())
                for (int32_t layer = 0; layer < p.n_layers; ++layer)
                    CHECK(rad_memset_async(engine->kv.state_ptr(p.index, layer, engine->kv.state_slot(17, p.index)),
                                           99, p.layer_bytes_per_state(), engine->stream) == RAD_OK);
            CHECK(rad_stream_sync(engine->stream) == RAD_OK);
        }
        bool wrong_sequence = false;
        try { checkpoint.restore(engine->kv, 999, device, engine->stream); }
        catch (const std::exception&) { wrong_sequence = true; }
        CHECK(wrong_sequence);
        checkpoint.restore(engine->kv, 17, device, engine->stream);
        NativePages pages(engine->kv.plans(), engine->infos, 32, "test-native-unit-scales-v1");
        std::vector<GroupPool> pools;
        for (int g = 1; g <= 2; ++g) {
            const auto& p = *engine->kv.plan(g);
            pools.push_back({g, p.base, uint64_t(p.pool_bytes)});
        }
        auto bindings_at = [&](size_t compact_block) {
            std::vector<GroupPages> result;
            for (int g = 1; g <= 2; ++g) {
                const auto count = 32 / engine->kv.plan(g)->block_size;
                const auto& table = engine->kv.block_table(17, g);
                const auto start = compact_block * count;
                CHECK(start + count <= table.size());
                result.push_back({g, uint64_t(engine->kv.live_blocks(g)),
                    {table.begin() + start, table.begin() + start + count}});
            }
            return result;
        };
        std::vector<std::vector<uint8_t>> originals;
        for (uint64_t b = 0; b < 4; ++b) {
            std::vector<uint8_t> bytes(pages.layout().record_bytes);
            for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(31 * b + 17 * i);
            PageTransfer write(device, bytes.size(), engine->stream);
            write.restore(pages.spans(engine->kv.plans(), bindings_at(b)), pools, engine->stream, bytes);
            CHECK(write.drain());
            CHECK(write.poll().status == TransferStatus::Complete);
            originals.push_back(std::move(bytes));
        }
        MemoryStamp stamp{true, {17, 1}, 1, 0, 128, 1};
        std::vector<BlockDescriptor> blocks;
        for (uint64_t b = 0; b < 4; ++b) blocks.push_back({stamp.session, b, b * 32, 32, 1, 0});
        const auto shared_storage=disk_backed ? std::make_shared<HostKvStorage>(2*pages.layout().record_bytes,
            std::make_shared<SpillFile>(disk_directory,16*pages.layout().record_bytes)) : nullptr;
        const auto shared_io=disk_backed ? std::make_shared<StorageIoQueue>(2,64*pages.layout().record_bytes) : nullptr;
        auto backend = std::make_shared<NativeMemory>(MemoryBinding{engine, &engine->kv, engine->stream, device, 1},
            engine->infos, BackendIdentity{"radiance", "unit", "byte-fixture", "test-bf16-v1"},
            stamp, blocks, 32, 256, 8 * pages.layout().record_bytes,shared_storage,shared_io);
        CHECK(!backend->descriptor().capabilities.pre_rope_statistics);
        CHECK(!backend->descriptor().capabilities.state_checkpoint);
        MemorySession session(backend);
        auto budget = [&](uint64_t visible) {
            return MemoryBudget{visible, 32, visible + 32, 8 * pages.layout().record_bytes};
        };
        auto finish = [&]() {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (session.phase() != MemoryPhase::Ready) {
                CHECK(std::chrono::steady_clock::now() < deadline);
                session.advance();
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        };
        auto verify = [&](const std::vector<uint64_t>& selected) {
            if (hybrid) {
                PageTransfer read(device, state_expected.size(), engine->stream);
                read.spill(state_spans, state_pools, engine->stream);
                CHECK(read.drain()); CHECK(read.payload() == state_expected);
            }
            const auto frontier = backend->stamp().frontier;
            CHECK(engine->kv.seq_tokens(17) == int64_t(frontier));
            CHECK(engine->kv.execution_offset(17, 1) == int64_t(frontier - selected.size() * 32));
            for (size_t i = 0; i < selected.size(); ++i) {
                PageTransfer read(device, pages.layout().record_bytes, engine->stream);
                read.spill(pages.spans(engine->kv.plans(), bindings_at(i)), pools, engine->stream);
                CHECK(read.drain());
                CHECK(read.payload() == originals.at(selected[i]));
            }
        };
        if(shared_io)for(bool full:{true,false}) {
            std::promise<void> started,release;
            const auto gate=release.get_future().share();
            auto hold=shared_io->submit(0,[&]{started.set_value();gate.wait();});
            started.get_future().wait();
            StorageTicket second;
            if(full)second=shared_io->submit(0,[]{});
            const auto before=backend->stamp();
            session.prepare(before,{0,3},{3},budget(64));
            session.advance();CHECK(rad_stream_sync(engine->stream)==RAD_OK);
            session.advance(); // queue full or queued private work; no publication
            CHECK(session.phase()==MemoryPhase::Spilling && shared_io->pending_jobs()==2);
            if(full) {
                CHECK(session.abort()==AbortResult::PreviousViewPreserved);
                release.set_value();hold.wait();second.wait();
            } else {
                // Drain must retain all source/CPU/page ownership until the
                // cancelled queued job releases its captures.
                auto unblock=std::async(std::launch::async,[&]{
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));release.set_value();});
                CHECK(session.abort()==AbortResult::PreviousViewPreserved);
                unblock.get();hold.wait();
            }
            CHECK(backend->stamp()==before && shared_io->pending_jobs()==0 && shared_io->pinned_bytes()==0);
            verify({0,1,2,3});
        }
        // Selection originates in the common planner, never the engine allocator.
        for (const std::vector<uint64_t>& selected : {std::vector<uint64_t>{0, 3}, {1, 3}, {2, 3}, {0, 1, 2, 3}}) {
            session.prepare(backend->stamp(), selected, {3}, budget(selected.size() * 32));
            finish();
            const auto view = session.publish();
            CHECK(view.visible_tokens == selected.size() * 32);
            verify(selected);
            if (speculative) CHECK(engine->kv.block_table(17, 5) == initial_window);
            CHECK(engine->kv.window_epoch(17) == initial_window_epoch);
        }
        // Roll back a transaction after GPU restore has completed, before publication.
        session.prepare(backend->stamp(), {0, 3}, {3}, budget(64)); finish(); session.publish();
        const auto before = backend->snapshot();
        const auto free1 = engine->kv.free_blocks(1), free2 = engine->kv.free_blocks(2);
        session.prepare(before.stamp, {1, 3}, {3}, budget(64)); finish();
        CHECK(session.abort() == AbortResult::PreviousViewPreserved);
        CHECK(backend->stamp() == before.stamp);
        CHECK(engine->kv.free_blocks(1) == free1 && engine->kv.free_blocks(2) == free2);
        verify({0, 3});
        // Constrain the native allocator to append slack only. A third GPU view
        // cannot fit: restore must recycle the evicted live block after spill.
        // External prefix refs are authoritative and MUST prevent recycling.
        {
            std::vector<std::pair<int32_t, int32_t>> held, shared;
            for (int g = 1; g <= 2; ++g) {
                const auto tail = 32 / engine->kv.plan(g)->block_size;
                while (engine->kv.free_blocks(g) > tail) {
                    int32_t id = -1;
                    CHECK(engine->kv.reserve_block(g, &id) == RAD_OK);
                    held.emplace_back(g, id);
                }
                const auto table = engine->kv.block_table(17, g);
                for (int i = 0; i < tail; ++i) {
                    CHECK(engine->kv.retain_block(g, table[i]) == RAD_OK);
                    shared.emplace_back(g, table[i]);
                }
            }
            bool leased_refused = false;
            try { session.prepare(backend->stamp(), {1, 3}, {3}, budget(64)); }
            catch (const MemoryError& e) { leased_refused = e.code() == MemoryErrorCode::BudgetExceeded; }
            CHECK(leased_refused && backend->stamp().valid);
            verify({0, 3});
            for (const auto& [g, id] : shared) engine->kv.release_block(g, id);
            const auto before_reuse = backend->reuse_stats();
            // Cancellation before any copies cannot touch old bytes or leak IDs.
            session.prepare(backend->stamp(), {1, 3}, {3}, budget(64));
            CHECK(session.abort() == AbortResult::PreviousViewPreserved);
            verify({0, 3});
            // Cancellation during spill and after restore both retain the old
            // view; only the latter needs the preallocated host rollback journal.
            session.prepare(backend->stamp(), {1, 3}, {3}, budget(64));
            session.advance();
            CHECK(session.abort() == AbortResult::PreviousViewPreserved);
            verify({0, 3});
            session.prepare(backend->stamp(), {1, 3}, {3}, budget(64)); finish();
            CHECK(session.abort() == AbortResult::PreviousViewPreserved);
            verify({0, 3});
            CHECK(backend->reuse_stats().rollbacks == before_reuse.rollbacks + 1);
            for (int cycle = 0; cycle < 16; ++cycle) {
                const uint64_t pick = cycle % 2 ? 0 : 1;
                session.prepare(backend->stamp(), {pick, 3}, {3}, budget(64)); finish(); session.publish();
                verify({pick, 3});
                for (int g = 1; g <= 2; ++g)
                    CHECK(engine->kv.free_blocks(g) == 32 / engine->kv.plan(g)->block_size);
            }
            CHECK(backend->reuse_stats().transactions == before_reuse.transactions + 19);
            CHECK(backend->reuse_stats().recycled_pages >= before_reuse.recycled_pages + 19 * 3);
            CHECK(backend->reuse_stats().journal_bytes_peak == pages.layout().record_bytes);
            for (const auto& [g, id] : held) engine->kv.release_block(g, id);
            CHECK(engine->kv.free_blocks(1) == free1 && engine->kv.free_blocks(2) == free2);
        }
        // An accepted append grows the compact tail without changing old pages.
        CHECK(engine->kv.ensure(17, 160) == RAD_OK);
        std::vector<uint8_t> appended(pages.layout().record_bytes, 213);
        {
            PageTransfer write(device, appended.size(), engine->stream);
            write.restore(pages.spans(engine->kv.plans(), bindings_at(2)), pools, engine->stream, appended);
            CHECK(write.drain());
        }
        originals.push_back(appended);
        backend->accept_append(160, {{stamp.session, 4, 128, 32, 1, 1}});
        CHECK(backend->stamp().history_revision == before.stamp.history_revision + 1);
        CHECK(backend->snapshot().blocks.size() == 5);
        verify({0, 3, 4});
        session.prepare(backend->stamp(), {1, 4}, {4}, budget(64)); finish(); session.publish();
        verify({1, 4});
        // A prompt can end between native page boundaries. Keep its mutable
        // tail resident while historical complete blocks are spilled/restored.
        CHECK(engine->kv.ensure(17, 163) == RAD_OK);
        std::vector<CopySpan> tail_spans;
        uint64_t tail_bytes = 0;
        auto describe_tail = [&]() {
            tail_spans.clear(); tail_bytes = 0;
            for (int g = 1; g <= 2; ++g) {
                const auto& p = *engine->kv.plan(g);
                const auto id = engine->kv.block_table(17, g).back();
                const auto fragment = uint64_t(p.bytes_per_block / p.n_layers);
                const auto stride = uint64_t(p.pool_bytes / p.n_layers);
                for (int64_t layer = 0; layer < p.n_layers; ++layer) {
                    tail_spans.push_back({g, layer * stride + uint64_t(id) * fragment, tail_bytes, fragment});
                    tail_bytes += fragment;
                }
            }
        };
        describe_tail();
        const std::vector<uint8_t> tail_expected(tail_bytes, 211);
        { PageTransfer write(device, tail_bytes, engine->stream);
          write.restore(tail_spans, pools, engine->stream, tail_expected); CHECK(write.drain()); }
        backend->accept_append(163, {{stamp.session, 5, 160, 3, 1, 1}});
        bool lost_tail_refused = false;
        try { session.prepare(backend->stamp(), {0, 1}, {}, budget(64)); }
        catch (const MemoryError& e) { lost_tail_refused = e.code() == MemoryErrorCode::InvalidPlan; }
        CHECK(lost_tail_refused && backend->stamp().valid);
        for (auto pick : {uint64_t(0), uint64_t(2)}) {
            session.prepare(backend->stamp(), {pick, 5}, {5}, budget(64)); finish();
            const auto view = session.publish();
            CHECK(view.visible_tokens == 35);
            CHECK(engine->kv.execution_offset(17, 1) == 128);
            CHECK(backend->snapshot().blocks.back().host_version == 0);
            describe_tail();
            PageTransfer read(device, tail_bytes, engine->stream);
            read.spill(tail_spans, pools, engine->stream); CHECK(read.drain());
            CHECK(read.payload() == tail_expected);
        }
        CHECK(engine->kv.ensure(17, 175) == RAD_OK);
        backend->accept_append(175, {{stamp.session, 5, 160, 15, 2, 2}});
        session.prepare(backend->stamp(), {1, 5}, {5}, budget(64)); finish();
        CHECK(session.publish().visible_tokens == 47);
        describe_tail();
        { PageTransfer read(device, tail_bytes, engine->stream);
          read.spill(tail_spans, pools, engine->stream); CHECK(read.drain()); CHECK(read.payload() == tail_expected); }
        // The same tail closes and becomes an ordinary immutable archive block.
        CHECK(engine->kv.ensure(17, 192) == RAD_OK);
        originals.emplace_back(pages.layout().record_bytes, 211);
        { PageTransfer write(device, originals.back().size(), engine->stream);
          write.restore(pages.spans(engine->kv.plans(), bindings_at(1)), pools, engine->stream, originals.back());
          CHECK(write.drain()); }
        backend->accept_append(192, {{stamp.session, 5, 160, 32, 3, 3}});
        verify({1, 5});
        session.prepare(backend->stamp(), {0, 4}, {4}, budget(64)); finish(); session.publish();
        verify({0, 4});
        session.prepare(backend->stamp(), {2, 5}, {5}, budget(64)); finish(); session.publish();
        verify({2, 5});
        // Replay drops stale query payloads even if they were archived, keeps
        // pre-query pages, and changes execution identity. New bytes at reused
        // logical block IDs must never restore an old query's host payload.
        const auto before_rewind = backend->stamp();
        const auto before_window_epoch = engine->kv.window_epoch(17);
        backend->rewind(128);
        CHECK(engine->kv.window_epoch(17) == before_window_epoch + (speculative ? 1 : 0));
        if (speculative) CHECK(engine->kv.window_floor(17) == 128);
        if (speculative) CHECK(engine->kv.block_table(17, 5).empty());
        CHECK(backend->stamp().execution_history == before_rewind.execution_history + 1);
        CHECK(backend->snapshot().blocks.size() == 4);
        CHECK(engine->kv.seq_tokens(17) == 128);
        verify({2});
        CHECK(engine->kv.ensure(17, 192) == RAD_OK);
        for (int index = 4; index <= 5; ++index) {
            originals[index].assign(pages.layout().record_bytes, uint8_t(230 + index));
            PageTransfer write(device, originals[index].size(), engine->stream);
            write.restore(pages.spans(engine->kv.plans(), bindings_at(index - 3)), pools,
                          engine->stream, originals[index]);
            CHECK(write.drain());
        }
        backend->accept_append(192, {{stamp.session, 4, 128, 32, 10, 10}, {stamp.session, 5, 160, 32, 10, 10}});
        session.prepare(backend->stamp(), {0, 4}, {4}, budget(64)); finish(); session.publish();
        session.prepare(backend->stamp(), {2, 5}, {5}, budget(64)); finish(); session.publish();
        verify({2, 5});
        // Engine changes outside the boundary must be caught, not silently published.
        if(disk_backed) {
            CHECK(shared_storage->io_stats().written_bytes>0 && shared_storage->io_stats().read_bytes>0);
            CHECK(shared_storage->charged_bytes()<=shared_storage->capacity_bytes());
#ifdef __linux__
            shared_storage->trim(0);
            const auto before=backend->stamp();
            const auto free1=engine->kv.free_blocks(1),free2=engine->kv.free_blocks(2);
            truncate_fixture_spill(disk_directory);
            bool failed=false;
            try{session.prepare(before,{1,5},{5},budget(64));finish();}
            catch(const MemoryError& error){failed=error.code()==MemoryErrorCode::TransferFailed;}
            CHECK(failed && backend->stamp()==before);
            CHECK(engine->kv.free_blocks(1)==free1 && engine->kv.free_blocks(2)==free2);
            verify({2,5}); // failing SSD read never changed native pages or state
#endif
        } else {
        session.prepare(backend->stamp(), {1, 5}, {5}, budget(64));
        CHECK(engine->kv.ensure(17, 193) == RAD_OK);
        bool stale = false;
        try { session.advance(); } catch (const MemoryError& e) { stale = e.code() == MemoryErrorCode::Stale; }
        CHECK(stale);
        CHECK(!backend->stamp().valid);
        }
        for (bool host_prefix : {false, true}) {
            auto leased_engine = std::make_shared<Engine>(hybrid, 2, dtype, speculative);
            auto& kv = leased_engine->kv;
            CHECK(kv.ensure(17, 133) == RAD_OK);
            for (const auto& p : kv.plans()) if (p.kind >= 0)
                CHECK(rad_memset_async(p.base, 47, p.pool_bytes, leased_engine->stream) == RAD_OK);
            CHECK(rad_stream_sync(leased_engine->stream) == RAD_OK);
            kv.clear_state_clears();
            std::vector<rad::KVManager::FullTable> sparse;
            for (int g = 1; g <= 2; ++g) {
                const auto table = kv.block_table(17, g);
                std::vector<int32_t> selected;
                const int pages_per_block = 32 / kv.plan(g)->block_size;
                for (int logical : {0, 3, 4}) for (int page = 0; page < pages_per_block; ++page) {
                    const int index = logical * pages_per_block + page;
                    if (size_t(index) < table.size()) selected.push_back(table[index]);
                }
                sparse.push_back({g, selected});
            }
            CHECK(kv.publish_full_tables(17, 133, 69, sparse) == RAD_OK);
            const auto saved_window_first = speculative ? std::max<int64_t>(kv.first_block_pos(17, 5),
                (133 / kv.plan(5)->block_size - (kv.plan(5)->window_blocks - 1)) * kv.plan(5)->block_size) : 0;
            std::vector<int32_t> prefix(133);
            for (size_t i = 0; i < prefix.size(); ++i) prefix[i] = int32_t(i);
            auto lease = std::make_unique<PrefixLease>(leased_engine, kv, 17, prefix, device, leased_engine->stream);
            auto next = prefix; next.push_back(200);
            CHECK(lease->matches(next) && !lease->matches(prefix));
            auto wrong = next; wrong[64] = 400;
            CHECK(!lease->matches(wrong));
            if (host_prefix) {
                bool too_small = false;
                try { lease->demote(device, leased_engine->stream, 1); }
                catch (const MemoryError& error) { too_small = error.code() == MemoryErrorCode::BudgetExceeded; }
                CHECK(too_small && lease->host_bytes() == 0);
                for (const auto& t : sparse) for (auto page : t.blocks) CHECK(kv.block_refcount(t.group, page) == 2);
                lease->demote(device, leased_engine->stream, 1024 * 1024);
                if(disk_backed) {
                    auto storage=std::make_shared<HostKvStorage>(std::max<uint64_t>(512,pages.layout().record_bytes),std::make_shared<SpillFile>(disk_directory,1024*1024));
                    lease->park(storage,shared_io);
                    CHECK(storage->charged_bytes()<=storage->capacity_bytes() && storage->disk_bytes()>0);
                }
                CHECK(lease->host_bytes() > 0);
                for (const auto& t : sparse) {
                    for (auto page : t.blocks) CHECK(kv.block_refcount(t.group, page) == 1);
                    CHECK(rad_memset_async(kv.plan(t.group)->base, 99, kv.plan(t.group)->pool_bytes, leased_engine->stream) == RAD_OK);
                }
                CHECK(rad_stream_sync(leased_engine->stream) == RAD_OK);
            }
            kv.free_sequence(17);
            CHECK(kv.add_sequence(29) == RAD_OK && kv.ensure(29, 1) == RAD_OK);
            kv.clear_state_clears(); // the test explicitly restores all committed states below
            bool refused_prefix = false;
            try { lease->restore(kv, 29, wrong, device, leased_engine->stream); }
            catch (const std::exception&) { refused_prefix = true; }
            CHECK(refused_prefix && kv.seq_tokens(29) == 1);
            lease->restore(kv, 29, next, device, leased_engine->stream);
            if (speculative) {
                CHECK(!kv.block_table(29, 5).empty());
                CHECK(kv.first_block_pos(29, 5) == saved_window_first);
                CHECK(kv.window_epoch(29) == 2 && kv.window_floor(29) == saved_window_first);
                const auto& p = *kv.plan(5);
                for (auto page : kv.block_table(29, 5)) {
                    PageTransfer read(device, p.layer_bytes_per_block(), leased_engine->stream);
                    read.spill({{5, uint64_t(page) * p.layer_bytes_per_block(), 0,
                                 uint64_t(p.layer_bytes_per_block())}},
                               {{5, p.base, uint64_t(p.pool_bytes)}}, leased_engine->stream);
                    CHECK(read.drain());
                    for (auto value : read.payload()) CHECK(value == 47);
                }
            }
            CHECK(kv.seq_tokens(29) == 133 && !kv.canonical(29));
            for (const auto& t : sparse) {
                const auto& actual = kv.block_table(29, t.group);
                CHECK(actual.size() == t.blocks.size());
                if (!host_prefix) CHECK(actual == t.blocks);
                CHECK(kv.execution_offset(29, t.group) == 64);
                for (auto page : actual) {
                    CHECK(kv.block_refcount(t.group, page) == (host_prefix ? 1 : 2));
                    const auto& p = *kv.plan(t.group);
                    for (int32_t layer = 0; layer < p.n_layers; ++layer) {
                        PageTransfer read(device, p.layer_bytes_per_block(), leased_engine->stream);
                        read.spill({{p.index, uint64_t(layer * p.layer_stride() + page * p.layer_bytes_per_block()),
                            0, uint64_t(p.layer_bytes_per_block())}}, {{p.index, p.base, uint64_t(p.pool_bytes)}}, leased_engine->stream);
                        CHECK(read.drain());
                        for (auto value : read.payload()) CHECK(value == 47);
                    }
                }
            }
            // The new request must copy-on-write a retained partial last page.
            CHECK(kv.ensure(29, 134) == RAD_OK);
            const auto copies = kv.take_pending_copies();
            CHECK(copies.size() == (host_prefix ? 0 : 2));
            for (const auto& copy : copies) {
                const auto& p = *kv.plan(copy.group);
                CHECK(copy.src_block != copy.dst_block);
                for (int32_t layer = 0; layer < p.n_layers; ++layer) {
                    auto* base = static_cast<char*>(p.base) + layer * p.layer_stride();
                    CHECK(rad_memcpy_async(base + copy.dst_block * p.layer_bytes_per_block(),
                        base + copy.src_block * p.layer_bytes_per_block(), p.layer_bytes_per_block(),
                        leased_engine->stream) == RAD_OK);
                    CHECK(rad_memset_async(base + copy.dst_block * p.layer_bytes_per_block(), 91,
                        p.layer_bytes_per_block(), leased_engine->stream) == RAD_OK);
                    PageTransfer read(device, p.layer_bytes_per_block(), leased_engine->stream);
                    read.spill({{p.index, uint64_t(layer * p.layer_stride() + copy.src_block * p.layer_bytes_per_block()),
                        0, uint64_t(p.layer_bytes_per_block())}}, {{p.index, p.base, uint64_t(p.pool_bytes)}}, leased_engine->stream);
                    CHECK(read.drain());
                    for (auto value : read.payload()) CHECK(value == 47);
                }
            }
            kv.free_sequence(29);
            lease.reset();
            for (int g = 1; g <= 2; ++g) CHECK(kv.free_blocks(g) == kv.total_blocks(g));
        }
        // Rebinding preserves the logical archive while physical request IDs
        // change. A later restore must publish to the NEW request, not the dead ID.
        for (bool host_prefix : {false, true}) {
            auto resumed_engine = std::make_shared<Engine>(hybrid, 1, dtype, speculative);
            auto& kv = resumed_engine->kv;
            for (const auto& p : kv.plans()) if (p.base && p.pool_bytes)
                CHECK(rad_memset_async(p.base, 53, p.pool_bytes, resumed_engine->stream) == RAD_OK);
            CHECK(rad_stream_sync(resumed_engine->stream) == RAD_OK);
            const MemoryStamp initial{true, {17, 77}, 1, 0, 128, 1};
            std::vector<BlockDescriptor> catalog;
            for (uint64_t b = 0; b < 4; ++b) catalog.push_back({initial.session, b, b * 32, 32, 1, 0});
            auto retained = std::make_shared<NativeMemory>(MemoryBinding{resumed_engine, &kv,
                resumed_engine->stream, device, 1}, resumed_engine->infos,
                BackendIdentity{"radiance", "unit", "resume-fixture", "test-bf16-v1"},
                initial, catalog, 32, 256, 1024 * 1024);
            auto select = [&](const std::vector<uint64_t>& selected) {
                MemorySession transaction(retained);
                transaction.prepare(retained->stamp(), selected, {3}, {64, 32, 128, 1024 * 1024});
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (transaction.phase() != MemoryPhase::Ready) {
                    CHECK(std::chrono::steady_clock::now() < deadline);
                    transaction.advance();
                }
                transaction.publish();
            };
            select({0, 3});
            std::vector<int32_t> tokens(128, 7);
            auto lease = std::make_unique<PrefixLease>(resumed_engine, kv, 17, tokens, device, resumed_engine->stream);
            retained->retain_view();
            std::vector<std::pair<int32_t, int32_t>> held;
            if (host_prefix) {
                for (int group = 1; group <= 2; ++group) while (kv.free_blocks(group) > 32 / kv.plan(group)->block_size) {
                    int32_t page = -1;
                    CHECK(kv.reserve_block(group, &page) == RAD_OK);
                    held.emplace_back(group, page);
                }
                CHECK(retained->needs_unleased_pages({1, 3}, {3}, {64, 32, 128, 1024 * 1024}));
                lease->demote(device, resumed_engine->stream, 1024 * 1024 - retained->archive_bytes());
                CHECK(!retained->needs_unleased_pages({1, 3}, {3}, {64, 32, 128, 1024 * 1024}));
            }
            select({1, 3}); // blockwise prefill may change pages AFTER checkpoint
            if (host_prefix) CHECK(retained->reuse_stats().recycled_pages > 0);
            for (const auto& [group, page] : held) kv.release_block(group, page);
            CHECK(kv.ensure(17, 160) == RAD_OK);
            retained->accept_append(160, {{initial.session, 4, 128, 32, 2, 2}});
            const auto after_append = retained->stamp();
            kv.free_sequence(17);
            CHECK(kv.add_sequence(29) == RAD_OK && kv.ensure(29, 1) == RAD_OK);
            bool invalid = false;
            try { retained->rebind_sequence(29); } catch (const MemoryError&) { invalid = true; }
            CHECK(invalid);
            tokens.push_back(9);
            lease->restore(kv, 29, tokens, device, resumed_engine->stream);
            kv.clear_state_clears();
            retained->rebind_sequence(29, host_prefix);
            lease.reset();
            CHECK(retained->snapshot().stamp.session == initial.session);
            CHECK(retained->stamp().frontier == 128);
            CHECK(retained->stamp().execution_history == after_append.execution_history + 1);
            CHECK(retained->stamp().view_version > after_append.view_version);
            CHECK(retained->snapshot().resident == std::vector<uint64_t>({0, 3}));
            select({1, 3}); // block1 comes back from the host archive
            CHECK(!kv.has_sequence(17) && kv.seq_tokens(29) == 128 && kv.execution_offset(29, 1) == 64);
            // Native drafter lookahead can exceed the last prefill chunk.
            CHECK(kv.ensure(29, 168) == RAD_OK);
            CHECK(kv.rollback(29, 160) == RAD_OK);
            retained->accept_append(160, {{initial.session, 4, 128, 32, 2, 2}});
            CHECK(retained->snapshot().stamp.frontier == 160);
        }
        // Two branches restore the same parked prefix independently. Their
        // archive bytes share physical H/D, while logical identities, selected
        // pages and subsequent accepted tails must never alias.
        {
            auto branched=std::make_shared<Engine>(hybrid,2,dtype,speculative);
            auto& kv=branched->kv;
            for(const auto& p:kv.plans())if(p.base && p.pool_bytes)
                CHECK(rad_memset_async(p.base,61,p.pool_bytes,branched->stream)==RAD_OK);
            CHECK(rad_stream_sync(branched->stream)==RAD_OK);kv.clear_state_clears();
            auto storage=std::make_shared<HostKvStorage>(disk_backed ? std::max<uint64_t>(4096,pages.layout().record_bytes) : 1024*1024,
                disk_backed ? std::make_shared<SpillFile>(disk_directory,1024*1024) : nullptr);
            const MemoryStamp mark{true,{17,333},1,0,128,1};
            std::vector<BlockDescriptor> blocks;
            for(uint64_t i=0;i<4;++i)blocks.push_back({mark.session,i,i*32,32,1,0});
            auto root=std::make_shared<NativeMemory>(MemoryBinding{branched,&kv,branched->stream,device,1},
                branched->infos,BackendIdentity{"radiance","unit","branches","test"},mark,blocks,32,256,1024*1024,storage,shared_io);
            auto select_branch=[&](const std::shared_ptr<NativeMemory>& memory,const std::vector<uint64_t>& selected) {
                MemorySession transaction(memory);
                transaction.prepare(memory->stamp(),selected,{3},{64,32,256,1024*1024});
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
                while(transaction.phase()!=MemoryPhase::Ready) {
                    CHECK(std::chrono::steady_clock::now()<deadline);transaction.advance();
                }
                transaction.publish();
            };
            select_branch(root,{0,3});root->retain_view();const auto root_stamp=root->stamp();
            std::vector<int32_t> prompt(128,7);
            auto prefix=std::make_unique<PrefixLease>(branched,kv,17,prompt,device,branched->stream);
            prefix->demote(device,branched->stream,1024*1024);prefix->park(storage,shared_io);
            const auto charged=storage->charged_bytes()+storage->disk_bytes();
            auto a=root->fork_retained({29,334}),c=root->fork_retained({41,335});
            CHECK(root->shared_archive_bytes(*a)==root->archive_bytes());
            CHECK(a->shared_archive_bytes(*c)==root->archive_bytes());
            auto pa=prefix->fork(),pc=prefix->fork();
            CHECK(storage->charged_bytes()+storage->disk_bytes()==charged);
            kv.free_sequence(17);prompt.push_back(9);
            for(auto id:{29u,41u})CHECK(kv.add_sequence(id)==RAD_OK && kv.ensure(id,1)==RAD_OK);
            kv.clear_state_clears();
            auto cpu_read=std::async(std::launch::async,[&]{return pa->load_cpu();});
            const auto prepared=cpu_read.get();
            PrefixLease::Payload foreign;foreign.kv=prepared.kv;foreign.state=prepared.state;
            bool foreign_refused=false;
            try {pa->restore(kv,29,prompt,device,branched->stream,&foreign);}
            catch(const std::exception&){foreign_refused=true;}
            CHECK(foreign_refused && kv.seq_tokens(29)==1);
            auto truncated=prepared;truncated.kv.pop_back();bool refused=false;
            try {pa->restore(kv,29,prompt,device,branched->stream,&truncated);}
            catch(const std::exception&){refused=true;}
            CHECK(refused && kv.seq_tokens(29)==1);
            pa->restore(kv,29,prompt,device,branched->stream,&prepared);a->rebind_sequence(29,true);
            pc->restore(kv,41,prompt,device,branched->stream);c->rebind_sequence(41,true);
            pa.reset();pc.reset();
            for(int g=1;g<=2;++g)for(auto page:kv.block_table(29,g))
                CHECK(std::find(kv.block_table(41,g).begin(),kv.block_table(41,g).end(),page)==kv.block_table(41,g).end());
            select_branch(a,{1,3});select_branch(c,{2,3});
            CHECK(root->stamp()==root_stamp);
            const auto c_before=c->snapshot();
            CHECK(kv.ensure(29,145)==RAD_OK);
            a->accept_append(145,{{a->stamp().session,4,128,17,2,2}});
            CHECK(c->stamp()==c_before.stamp && kv.seq_tokens(41)==128);
            // Preserve an input checkpoint while a second, later accepted
            // endpoint gets different recurrent bytes. Restore both after the
            // producing sequence is gone; neither frontier may borrow the other.
            for(const auto& p:kv.plans())if(p.stateful())for(int layer=0;layer<p.n_layers;++layer)
                CHECK(rad_memset_async(kv.state_ptr(p.index,layer,kv.state_slot(29,p.index)),73,
                    p.layer_bytes_per_state(),branched->stream)==RAD_OK);
            CHECK(rad_stream_sync(branched->stream)==RAD_OK);
            std::vector<int32_t> terminal_tokens(145,7);
            auto terminal=std::make_unique<PrefixLease>(branched,kv,29,terminal_tokens,device,branched->stream);
            a->retain_view();terminal->demote(device,branched->stream,1024*1024);terminal->park(storage,shared_io);
            kv.free_sequence(29);
            CHECK(kv.add_sequence(53)==RAD_OK && kv.ensure(53,1)==RAD_OK);kv.clear_state_clears();
            terminal_tokens.push_back(9);terminal->restore(kv,53,terminal_tokens,device,branched->stream);
            a->rebind_sequence(53,true);CHECK(a->stamp().frontier==145 && c->stamp().frontier==128);
            for(const auto& p:kv.plans())if(p.stateful())for(int layer=0;layer<p.n_layers;++layer) {
                const auto* ptr=static_cast<char*>(kv.state_ptr(p.index,layer,kv.state_slot(53,p.index)));
                PageTransfer read(device,p.layer_bytes_per_state(),branched->stream);
                read.spill({{p.index,uint64_t(ptr-static_cast<char*>(p.base)),0,uint64_t(p.layer_bytes_per_state())}},
                    {{p.index,p.base,uint64_t(p.pool_bytes)}},branched->stream);
                CHECK(read.drain());for(auto byte:read.payload())CHECK(byte==73);
            }
            terminal.reset();kv.free_sequence(53);
            a.reset();prefix.reset();root.reset();
            select_branch(c,{0,3}); // source and sibling no longer exist
            const SessionIdentity expected_branch{41,335};CHECK(c->stamp().session==expected_branch);
            for(const auto& p:kv.plans())if(p.stateful())for(int layer=0;layer<p.n_layers;++layer) {
                const auto* ptr=static_cast<char*>(kv.state_ptr(p.index,layer,kv.state_slot(41,p.index)));
                PageTransfer read(device,p.layer_bytes_per_state(),branched->stream);
                read.spill({{p.index,uint64_t(ptr-static_cast<char*>(p.base)),0,uint64_t(p.layer_bytes_per_state())}},
                    {{p.index,p.base,uint64_t(p.pool_bytes)}},branched->stream);
                CHECK(read.drain());for(auto byte:read.payload())CHECK(byte==61);
            }
        }
        // A complete output snapshot can end in a partial selection/native page.
        // Restore, then append: native ownership and the mutable-tail version
        // must remain consistent with the common memory catalog.
        {
            auto partial_engine = std::make_shared<Engine>(hybrid);
            auto& kv = partial_engine->kv;
            CHECK(kv.has_sequence(17) && kv.seq_tokens(17)==128);
            CHECK(rad_stream_sync(partial_engine->stream) == RAD_OK);
            const MemoryStamp mark{true,{17,91},1,0,128,1};
            std::vector<BlockDescriptor> blocks;
            for (uint64_t i=0;i<4;++i) blocks.push_back({mark.session,i,i*32,32,1,0});
            auto memory = std::make_shared<NativeMemory>(MemoryBinding{partial_engine,&kv,partial_engine->stream,device,1},
                partial_engine->infos,BackendIdentity{"radiance","unit","partial","test"},mark,blocks,32,256,1024*1024);
            memory->rewind(123);
            CHECK(memory->snapshot().blocks.back().block.valid_tokens==27);
            CHECK(memory->snapshot().blocks.back().block.content_version==2);
            memory->retain_view();
            std::vector<int32_t> tokens(123,7);
            auto lease=std::make_unique<PrefixLease>(partial_engine,kv,17,tokens,device,partial_engine->stream);
            kv.free_sequence(17);
            CHECK(kv.add_sequence(29)==RAD_OK && kv.ensure(29,1)==RAD_OK);
            kv.clear_state_clears();
            tokens.push_back(9);
            lease->restore(kv,29,tokens,device,partial_engine->stream);
            memory->rebind_sequence(29);
            lease.reset();
            CHECK(kv.ensure(29,145)==RAD_OK);
            memory->accept_append(145,{{mark.session,3,96,32,3,3},{mark.session,4,128,17,3,3}});
            CHECK(memory->stamp().frontier==145 && kv.seq_tokens(29)==145);
            CHECK(memory->snapshot().blocks.back().block.valid_tokens==17);
        }
        // A retained checkpoint can target a DIFFERENT owned state slot in the
        // same engine. Strict replay still rejects that sequence; a different
        // engine/pool is never an implicit destination.
        {
            Engine continuation(hybrid, 2, dtype, speculative);
            CHECK(continuation.kv.add_sequence(29) == RAD_OK);
            CHECK(continuation.kv.ensure(29, 32) == RAD_OK);
            for (const auto& p : continuation.kv.plans()) if (p.stateful()) {
                CHECK(continuation.kv.state_slot(17, p.index) != continuation.kv.state_slot(29, p.index));
                for (int32_t layer = 0; layer < p.n_layers; ++layer) {
                    CHECK(rad_memset_async(continuation.kv.state_ptr(p.index, layer, continuation.kv.state_slot(17, p.index)),
                        37, p.layer_bytes_per_state(), continuation.stream) == RAD_OK);
                    CHECK(rad_memset_async(continuation.kv.state_ptr(p.index, layer, continuation.kv.state_slot(29, p.index)),
                        99, p.layer_bytes_per_state(), continuation.stream) == RAD_OK);
                }
            }
            CHECK(rad_stream_sync(continuation.stream) == RAD_OK);
            continuation.kv.clear_state_clears();
            StateCheckpoint retained;
            retained.save(continuation.kv, 17, device, continuation.stream);
            if(disk_backed) {
                auto storage=std::make_shared<HostKvStorage>(std::max<uint64_t>(512,pages.layout().record_bytes),std::make_shared<SpillFile>(disk_directory,1024*1024));
                retained.park(storage,shared_io);
                CHECK(storage->charged_bytes()<=storage->capacity_bytes());
            }
            bool strict_refused = false, other_engine_refused = false;
            try { retained.restore(continuation.kv, 29, device, continuation.stream); }
            catch (const std::exception&) { strict_refused = true; }
            try { retained.restore_rebound(engine->kv, 17, device, engine->stream); }
            catch (const std::exception&) { other_engine_refused = true; }
            CHECK(strict_refused && other_engine_refused);
            continuation.kv.free_sequence(17);
            retained.restore_rebound(continuation.kv, 29, device, continuation.stream);
            for (const auto& p : continuation.kv.plans()) if (p.stateful()) {
                for (int32_t layer = 0; layer < p.n_layers; ++layer) {
                    const auto* state = static_cast<char*>(continuation.kv.state_ptr(p.index, layer,
                        continuation.kv.state_slot(29, p.index)));
                    PageTransfer read(device, p.layer_bytes_per_state(), continuation.stream);
                    read.spill({{p.index, uint64_t(state - static_cast<char*>(p.base)), 0,
                                  uint64_t(p.layer_bytes_per_state())}},
                               {{p.index, p.base, uint64_t(p.pool_bytes)}}, continuation.stream);
                    CHECK(read.drain());
                    for (const auto value : read.payload()) CHECK(value == 37);
                }
            }
        }
        std::cout << "PASS common MemorySession/native " << (host_test ? "HOST" : "GPU")
                  << ": select, spill, restore, pressure-recycle, protected-prefix, rollback, append, stale-frontier; hybrid="
                  << hybrid << " speculative=" << speculative << " dtype=" << dtype << " disk=" << disk_backed << " state_bytes=" << state_expected.size() << '\n';
        }
        {
            Engine persistent(true);
            CHECK(rad_stream_sync(persistent.stream)==RAD_OK);
            for(int iteration=0;iteration<1100;++iteration) {
                StateCheckpoint saved;
                saved.save(persistent.kv,17,device,persistent.stream);
                saved.restore(persistent.kv,17,device,persistent.stream);
            }
            std::cout << "PASS 1100 checkpoint save/restore cycles on one engine stream\n";
        }
        CHECK(std::filesystem::is_empty(disk_directory));std::filesystem::remove(disk_directory);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
