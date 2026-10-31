#include "statistics_capture.hpp"
#include "statistics_observer.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "runtime/ctx.h"
#include "rad_device.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace kvmem;
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
struct Buffer {
    void* p; int kind;
    Buffer(int64_t n, int k) : p(rad_dev_alloc(n, k)), kind(k) { CHECK(p); }
    ~Buffer() { rad_dev_free(p, kind); }
};
static void arch_step(RadCtx* c, const RadBatch* b) {
    CHECK(bool(rad_kv_observing(c)) == (b->draft_pass == 0));
    const RadOperand q{RAD_OPK_BUF, RAD_DT_INVALID, 1, nullptr, 0, b->n_tok, 8};
    const RadOperand k{RAD_OPK_BUF, RAD_DT_INVALID, 1, nullptr, 8, b->n_tok, 4};
    rad_kv_observe(c, 3, 1, 4, 1, q, k);
    // Simulate the producer's next in-place operation. The capture MUST retain
    // the bytes from before this overwrite, not read this buffer after the pass.
    rad_memset_async(rad_buf_ptr(c, 1), 0, b->n_tok * 16 * 4, rad_stream(c));
}
int main(int argc, char** argv) {
    try {
        CHECK(!StatisticsObserver{}.retain_history_boundary());
        SelectionOptions selection_options;
        selection_options.budget_tokens = 512;
        CHECK(StatisticsObserver{selection_options}.retain_history_boundary());
        rad::StepPlan sampling_plan;
        sampling_plan.e.resize(1);
        sampling_plan.e[0].is_prefill = true;
        CHECK(StatisticsObserver{}.split_sampling(sampling_plan));
        sampling_plan.e[0].is_prefill = false;
        sampling_plan.e[0].n_spec = 2;
        CHECK(!StatisticsObserver{}.split_sampling(sampling_plan));
        int device = -1;
        if (argc == 2) device = std::stoi(argv[1]);
        else for (int i = 0; i < rad_dev_count(); ++i) {
            RadDeviceProps p{};
            if (rad_dev_props(i, &p) == RAD_OK && !p.is_host_backend) { device = i; break; }
        }
        CHECK(device >= 0 && rad_dev_set(device) == RAD_OK);
        StatisticsCapture capture(device, {{3, 1, 0, 4, 2, 1}}, 3);
        Buffer positions(12, RAD_MEM_DEVICE), host_positions(12, RAD_MEM_HOST_PINNED);
        Buffer upload(3 * 16 * 4, RAD_MEM_HOST_PINNED);
        auto* hp = static_cast<int32_t*>(host_positions.p);
        auto* data = static_cast<float*>(upload.p);
        for (int row = 0; row < 3; ++row) {
            hp[row] = row;
            for (int col = 0; col < 16; ++col) data[row * 16 + col] = float(row * 16 + col);
        }
        rad::Program program;
        program.meta.n_layers = 4;
        program.arena_bytes = 4096;
        program.weights.resize(1); program.buffers.resize(2); program.ops.resize(2);
        auto& buffer = program.buffers[1];
        buffer.name = "qkv"; buffer.decl.dtype = RAD_F32; buffer.decl.rank = 2;
        buffer.decl.domain = RAD_DOMAIN_DEVICE; buffer.decl.shape[0] = 3; buffer.decl.shape[1] = 16;
        buffer.bytes = 3 * 16 * 4; buffer.arena_offset = 0; buffer.first_def = buffer.last_use = 1;
        program.ops[1].op = "observation-lifetime-anchor"; program.ops[1].index = 1;
        rad::CtxDesc desc;
        desc.program = &program; desc.device = device; desc.arch_step = arch_step;
        desc.kv_observer = StatisticsCapture::observe; desc.kv_observer_data = &capture;
        desc.kv.resize(2); desc.kv[1].layer_slot = {-1, -1, -1, 0};
        rad::Ctx ctx;
        CHECK(ctx.init(desc) == RAD_OK);
        CHECK(!rad_kv_observing(&ctx)); // no active trunk pass
        CHECK(rad_memcpy_async(positions.p, hp, 12, ctx.stream()) == RAD_OK);
        CHECK(rad_memcpy_async(ctx.buf_ptr(1), data, buffer.bytes, ctx.stream()) == RAD_OK);
        RadBatch batch{};
        batch.n_seq = 1; batch.n_tok = 3; batch.step = 11; batch.phase = RAD_PHASE_PREFILL;
        batch.n_spec = 2; // A native verify pass; only its accepted prefix is committed.
        batch.positions = static_cast<const int32_t*>(positions.p);
        capture.begin({17, 1}, 0, 3);
        CHECK(ctx.run_step(&batch) == RAD_OK);
        CHECK(ctx.host_syncs() == 0);
        // Serial MTP history and chained proposals reuse the same attention
        // component. Neither may overwrite/duplicate the target's capture.
        batch.draft_pass = -1;
        CHECK(ctx.run_step(&batch) == RAD_OK);
        batch.draft_pass = 2;
        CHECK(ctx.run_step(&batch) == RAD_OK);
        batch.draft_pass = 0;
        capture.seal();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (capture.poll().status == TransferStatus::Pending) {
            CHECK(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        RawKvStoreConfig cfg;
        cfg.n_layer = 1; cfg.n_embd_k = 4; cfg.n_embd_v = 4; cfg.block_tokens = 4;
        RawKvStore store(cfg);
        bool refused = false;
        try { capture.commit(store, {17, 2}, 2, 1); }
        catch (const MemoryError&) { refused = true; }
        CHECK(refused && !store.has_block(0));
        const auto query = capture.commit(store, {17, 1}, 2, 1);
        CHECK(query.token_counts == std::vector<uint32_t>{1});
        for (int col = 0; col < 8; ++col) CHECK(query.sums[col] == float(16 + col));
        float mean[4]{};
        store.mean_k(0, 0, mean);
        for (int col = 0; col < 4; ++col) CHECK(mean[col] == float(16 + col));
        CHECK(store.n_tokens(0) == 2); // rejected third row never enters the statistics
        CHECK(ctx.pass_counts().played == 0);
        batch.n_spec = 0;
        // Query is a bounded half-open span; rows after it remain committed K
        // but must never change the frozen query with assistant/decode tokens.
        CHECK(rad_memcpy_async(ctx.buf_ptr(1), data, buffer.bytes, ctx.stream()) == RAD_OK);
        capture.begin({18, 1}, 0, 3, 1, 2);
        CHECK(ctx.run_step(&batch) == RAD_OK); capture.seal();
        while (capture.poll().status == TransferStatus::Pending) {
            CHECK(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        RawKvStore bounded(cfg);
        bool outside = false;
        try { capture.commit(bounded, {18, 1}, 3, 0, 2); }
        catch (const MemoryError&) { outside = true; }
        CHECK(outside && !bounded.has_block(0));
        const auto inside = capture.commit(bounded, {18, 1}, 3, 1, 2);
        CHECK(inside.token_counts == std::vector<uint32_t>{1});
        for (int col = 0; col < 8; ++col) CHECK(inside.sums[col] == float(16 + col));
        CHECK(bounded.n_tokens(0) == 3);
        // A frozen query needs no Q transfers on replay/decode, but all K rows
        // remain necessary. Stale staging contents must never be credited.
        CHECK(rad_memcpy_async(ctx.buf_ptr(1), data, buffer.bytes, ctx.stream()) == RAD_OK);
        capture.begin({19, 1}, 0, 3, UINT32_MAX, UINT32_MAX);
        CHECK(ctx.run_step(&batch) == RAD_OK); capture.seal();
        while (capture.poll().status == TransferStatus::Pending) {
            CHECK(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        RawKvStore frozen(cfg);
        const auto skipped = capture.commit(frozen, {19, 1}, 3, UINT32_MAX, UINT32_MAX);
        CHECK(skipped.token_counts == std::vector<uint32_t>{0});
        for (const auto v : skipped.sums) CHECK(v == 0);
        CHECK(frozen.n_tokens(0) == 3);
        // Missing layer coverage fails closed.
        capture.begin({17, 1}, 2, 1);
        bool incomplete = false;
        try { capture.seal(); } catch (const MemoryError&) { incomplete = true; }
        CHECK(incomplete && capture.discard());
        // The early event must never publish a partial multi-layer frame, nor
        // skip explicit seal(), and must precede later in-place overwrites.
        StatisticsCapture multi(device, {{3,1,0,4,2,1},{7,1,1,4,2,1}}, 3);
        CHECK(rad_memcpy_async(ctx.buf_ptr(1), data, buffer.bytes, ctx.stream()) == RAD_OK);
        RadTensor q{}, k{};
        q.data=ctx.buf_ptr(1);q.dtype=RAD_F32;q.rank=2;q.shape[0]=3;q.shape[1]=8;q.stride[0]=16;q.stride[1]=1;
        k=q;k.data=static_cast<float*>(ctx.buf_ptr(1))+8;k.shape[1]=4;
        multi.begin({20,1},0,3,1,2);
        CHECK(StatisticsCapture::observe(&multi,3,1,4,q,k,batch,ctx.stream())==RAD_OK);
        CHECK(rad_stream_sync(ctx.stream())==RAD_OK);
        CHECK(multi.poll().status==TransferStatus::Pending);
        incomplete=false;try{multi.seal();}catch(const MemoryError&){incomplete=true;}CHECK(incomplete);
        CHECK(StatisticsCapture::observe(&multi,7,1,4,q,k,batch,ctx.stream())==RAD_OK);
        CHECK(rad_memset_async(ctx.buf_ptr(1),0,buffer.bytes,ctx.stream())==RAD_OK);
        CHECK(rad_stream_sync(ctx.stream())==RAD_OK);
        CHECK(multi.poll().status==TransferStatus::Pending); // explicit seal remains required
        multi.seal();
        const char* early_mode=std::getenv("KVMEM_CAPTURE_EARLY_SEAL");
        if(early_mode&&std::strcmp(early_mode,"1")==0)
            CHECK(multi.poll().status==TransferStatus::Complete); // earlier stream sync fenced this event
        const auto multi_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(multi.poll().status==TransferStatus::Pending){
            CHECK(std::chrono::steady_clock::now()<multi_deadline);std::this_thread::yield();
        }
        bool double_seal=false;try{multi.seal();}catch(const MemoryError&){double_seal=true;}CHECK(double_seal);
        auto two_cfg=cfg;two_cfg.n_layer=2;RawKvStore two(two_cfg);
        const auto both=multi.commit(two,{20,1},2,1,2);
        CHECK(both.token_counts==std::vector<uint32_t>({1,1}));
        for(int layer=0;layer<2;++layer){
            for(int col=0;col<8;++col)CHECK(both.sums[layer*8+col]==float(16+col));
            two.mean_k(0,layer,mean);for(int col=0;col<4;++col)CHECK(mean[col]==float(16+col));
        }
        std::cout << "PASS pre-RoPE stream ordering, column stride, GQA widths, accepted rows, identity, coverage, early-event lifecycle\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
