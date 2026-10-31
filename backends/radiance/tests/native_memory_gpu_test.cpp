// Real MemorySession -> Radiance allocator -> GPU copy -> sparse table roundtrip.
#include "native_memory.hpp"
#include "native_pages.hpp"
#include "page_transfer.hpp"
#include "state_checkpoint.hpp"
#include "prefix_lease.hpp"
#include "mem/kv.h"
#include "mem/pools.h"
#include "rad_internal.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace kvmem;
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)

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
            info.decl.n_head_kv = 1; info.decl.head_dim = 8;
            info.block_size = 16 * g; info.layers = {3 * g};
        }
        if (hybrid) {
            infos.resize(5);
            auto& linear = infos[3];
            linear.name = "linear"; linear.layers = {1, 4};
            linear.decl.kind = RAD_KV_LINEAR; linear.decl.dtype = RAD_F32;
            linear.decl.n_head_kv = 2; linear.decl.state_dim[0] = 8; linear.decl.state_dim[1] = 8;
            auto& conv = infos[4];
            conv.name = "conv"; conv.layers = {1, 4};
            conv.decl.kind = RAD_KV_CONV; conv.decl.dtype = RAD_BF16;
            conv.decl.n_head_kv = 2; conv.decl.head_dim = 8; conv.decl.conv_width = 4;
        }
        if (speculative) {
            infos.resize(6);
            auto& draft = infos[5];
            draft.name = "drafter-window"; draft.layers = {8};
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
        for (uint32_t dtype : {uint32_t(RAD_BF16), uint32_t(RAD_F8E4M3)})
        for (bool hybrid : {false, true})
        for (bool speculative : {false, true}) {
        if (speculative && !hybrid) continue; // This fixture's drafter follows its recurrent groups.
        auto engine = std::make_shared<Engine>(hybrid, 1, dtype, speculative);
        const auto initial_window = speculative ? engine->kv.block_table(17, 5) : std::vector<int32_t>{};
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
            PageTransfer write(device, state_expected.size());
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
            PageTransfer write(device, bytes.size());
            write.restore(pages.spans(engine->kv.plans(), bindings_at(b)), pools, engine->stream, bytes);
            CHECK(write.drain());
            CHECK(write.poll().status == TransferStatus::Complete);
            originals.push_back(std::move(bytes));
        }
        MemoryStamp stamp{true, {17, 1}, 1, 0, 128, 1};
        std::vector<BlockDescriptor> blocks;
        for (uint64_t b = 0; b < 4; ++b) blocks.push_back({stamp.session, b, b * 32, 32, 1, 0});
        auto backend = std::make_shared<NativeMemory>(MemoryBinding{engine, &engine->kv, engine->stream, device, 1},
            engine->infos, BackendIdentity{"radiance", "unit", "byte-fixture", "test-bf16-v1"},
            stamp, blocks, 32, 256, 8 * pages.layout().record_bytes);
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
                PageTransfer read(device, state_expected.size());
                read.spill(state_spans, state_pools, engine->stream);
                CHECK(read.drain()); CHECK(read.payload() == state_expected);
            }
            const auto frontier = backend->stamp().frontier;
            CHECK(engine->kv.seq_tokens(17) == int64_t(frontier));
            CHECK(engine->kv.execution_offset(17, 1) == int64_t(frontier - selected.size() * 32));
            for (size_t i = 0; i < selected.size(); ++i) {
                PageTransfer read(device, pages.layout().record_bytes);
                read.spill(pages.spans(engine->kv.plans(), bindings_at(i)), pools, engine->stream);
                CHECK(read.drain());
                CHECK(read.payload() == originals.at(selected[i]));
            }
        };
        // Selection originates in the common planner, never the engine allocator.
        for (const std::vector<uint64_t>& selected : {std::vector<uint64_t>{0, 3}, {1, 3}, {2, 3}, {0, 1, 2, 3}}) {
            session.prepare(backend->stamp(), selected, {3}, budget(selected.size() * 32));
            finish();
            const auto view = session.publish();
            CHECK(view.visible_tokens == selected.size() * 32);
            verify(selected);
            if (speculative) CHECK(engine->kv.block_table(17, 5) == initial_window);
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
            PageTransfer write(device, appended.size());
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
        { PageTransfer write(device, tail_bytes);
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
            PageTransfer read(device, tail_bytes);
            read.spill(tail_spans, pools, engine->stream); CHECK(read.drain());
            CHECK(read.payload() == tail_expected);
        }
        CHECK(engine->kv.ensure(17, 175) == RAD_OK);
        backend->accept_append(175, {{stamp.session, 5, 160, 15, 2, 2}});
        session.prepare(backend->stamp(), {1, 5}, {5}, budget(64)); finish();
        CHECK(session.publish().visible_tokens == 47);
        describe_tail();
        { PageTransfer read(device, tail_bytes);
          read.spill(tail_spans, pools, engine->stream); CHECK(read.drain()); CHECK(read.payload() == tail_expected); }
        // The same tail closes and becomes an ordinary immutable archive block.
        CHECK(engine->kv.ensure(17, 192) == RAD_OK);
        originals.emplace_back(pages.layout().record_bytes, 211);
        { PageTransfer write(device, originals.back().size());
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
        backend->rewind(128);
        if (speculative) CHECK(engine->kv.block_table(17, 5).empty());
        CHECK(backend->stamp().execution_history == before_rewind.execution_history + 1);
        CHECK(backend->snapshot().blocks.size() == 4);
        CHECK(engine->kv.seq_tokens(17) == 128);
        verify({2});
        CHECK(engine->kv.ensure(17, 192) == RAD_OK);
        for (int index = 4; index <= 5; ++index) {
            originals[index].assign(pages.layout().record_bytes, uint8_t(230 + index));
            PageTransfer write(device, originals[index].size());
            write.restore(pages.spans(engine->kv.plans(), bindings_at(index - 3)), pools,
                          engine->stream, originals[index]);
            CHECK(write.drain());
        }
        backend->accept_append(192, {{stamp.session, 4, 128, 32, 10, 10}, {stamp.session, 5, 160, 32, 10, 10}});
        session.prepare(backend->stamp(), {0, 4}, {4}, budget(64)); finish(); session.publish();
        session.prepare(backend->stamp(), {2, 5}, {5}, budget(64)); finish(); session.publish();
        verify({2, 5});
        // Engine changes outside the boundary must be caught, not silently published.
        session.prepare(backend->stamp(), {1, 5}, {5}, budget(64));
        CHECK(engine->kv.ensure(17, 193) == RAD_OK);
        bool stale = false;
        try { session.advance(); } catch (const MemoryError& e) { stale = e.code() == MemoryErrorCode::Stale; }
        CHECK(stale);
        CHECK(!backend->stamp().valid);
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
            if (speculative) CHECK(kv.block_table(29, 5).empty());
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
                        PageTransfer read(device, p.layer_bytes_per_block());
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
                    PageTransfer read(device, p.layer_bytes_per_block());
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
                    PageTransfer read(device, p.layer_bytes_per_state());
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
                  << hybrid << " speculative=" << speculative << " dtype=" << dtype << " state_bytes=" << state_expected.size() << '\n';
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
