#include "page_transfer.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <stdexcept>

using namespace kvmem;
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
template<class F> void rejects(F fn) {
    bool caught = false;
    try { fn(); } catch (const MemoryError&) { caught = true; }
    CHECK(caught);
}
struct Stream { std::deque<std::function<void()>> queue; };
struct Event { bool done = false; Stream* stream = nullptr; };
static int copies = 0, fail_copy = -1, host_allocs = 0, streams = 0, events = 0;
static bool fail_wait = false;
static void flush(Stream* s) {
    while (!s->queue.empty()) { auto f = std::move(s->queue.front()); s->queue.pop_front(); f(); }
}
extern "C" {
const char* rad_dev_last_error() { return "injected test error"; }
int rad_dev_set(int n) { return n == 2 ? 0 : -1; } // catches hardcoded device 0
void* rad_dev_alloc(int64_t n, int kind) {
    CHECK(kind == RAD_MEM_HOST_PINNED); ++host_allocs; return std::malloc(size_t(n));
}
void rad_dev_free(void* p, int) { --host_allocs; std::free(p); }
int rad_stream_create(RadStream* out, int) { ++streams; *out = new Stream; return 0; }
int rad_stream_sync(RadStream s) { flush(static_cast<Stream*>(s)); return 0; }
void rad_stream_destroy(RadStream s) { CHECK(static_cast<Stream*>(s)->queue.empty()); --streams; delete static_cast<Stream*>(s); }
int rad_event_create(RadEvent* e) { ++events; *e = new Event; return 0; }
void rad_event_destroy(RadEvent e) { --events; delete static_cast<Event*>(e); }
int rad_event_record(RadEvent e, RadStream stream) {
    auto* ev = static_cast<Event*>(e); auto* s = static_cast<Stream*>(stream);
    ev->stream = s; ev->done = false; s->queue.push_back([ev] { ev->done = true; }); return 0;
}
int rad_event_wait(RadStream stream, RadEvent e) {
    if (fail_wait) return -1;
    auto* ev = static_cast<Event*>(e);
    static_cast<Stream*>(stream)->queue.push_back([ev] { flush(ev->stream); CHECK(ev->done); }); return 0;
}
int rad_event_query(RadEvent e) { return static_cast<Event*>(e)->done ? 1 : 0; }
int rad_memcpy_async(void* d, const void* s, int64_t n, RadStream stream) {
    if (++copies == fail_copy) return -1;
    static_cast<Stream*>(stream)->queue.push_back([=] { std::memcpy(d, s, size_t(n)); }); return 0;
}
}
int main() {
    Stream compute;
    std::vector<uint8_t> pool(64, 0xcd), payload;
    std::vector<CopySpan> spans{{0, 32, 0, 8}, {0, 8, 8, 8}};
    std::vector<GroupPool> pools{{0, pool.data(), pool.size()}};
    compute.queue.push_back([&] {
        std::fill(pool.begin() + 32, pool.begin() + 40, 0x17);
        std::fill(pool.begin() + 8, pool.begin() + 16, 0x29);
    });
    {
        PageTransfer t(2, 16);
        rejects([&] { t.poll(); });
        auto bad = spans; bad[0].pool_offset = 100;
        rejects([&] { t.spill(bad, pools, &compute); });
        CHECK(copies == 0); CHECK(compute.queue.size() == 1);
        bad = spans; bad[1].record_offset = 0;
        rejects([&] { t.spill(bad, pools, &compute); });
        t.spill(spans, pools, &compute);
        CHECK(t.poll().status == TransferStatus::Pending);
        rejects([&] { t.payload(); });
        CHECK(t.drain());
        CHECK(t.poll().status == TransferStatus::Complete);
        payload = t.payload();
        CHECK(std::all_of(payload.begin(), payload.begin() + 8, [](auto x) { return x == 0x17; }));
        CHECK(std::all_of(payload.begin() + 8, payload.end(), [](auto x) { return x == 0x29; }));
        rejects([&] { t.spill(spans, pools, &compute); });
    }
    CHECK(host_allocs == 0 && streams == 0 && events == 0);
    std::fill(pool.begin(), pool.end(), 0xcd);
    {
        PageTransfer t(2, 16);
        const std::vector<CopySpan> dst{{0, 0, 0, 8}, {0, 48, 8, 8}};
        t.restore(dst, pools, &compute, payload);
        CHECK(t.poll().status == TransferStatus::Pending);
        CHECK(t.drain());
        CHECK(t.poll().status == TransferStatus::Complete);
        for (size_t i = 0; i < pool.size(); ++i)
            CHECK(pool[i] == (i < 8 ? 0x17 : i >= 48 && i < 56 ? 0x29 : 0xcd));
    }
    // Enqueue failure after a successful copy must drain before freeing staging.
    {
        PageTransfer t(2, 16);
        fail_copy = copies + 2;
        rejects([&] { t.restore(spans, pools, &compute, payload); });
        CHECK(t.poll().status == TransferStatus::Failed);
    }
    CHECK(compute.queue.empty());
    CHECK(host_allocs == 0 && streams == 0 && events == 0);
    // The source-ready fence is already queued on compute when wait fails.
    // Cleanup must drain compute as well, before destroying that fence.
    {
        PageTransfer t(2, 16);
        fail_wait = true;
        rejects([&] { t.spill(spans, pools, &compute); });
        CHECK(t.poll().status == TransferStatus::Failed);
    }
    CHECK(compute.queue.empty());
    CHECK(host_allocs == 0 && streams == 0 && events == 0);
    // Radiance AQL queue IDs are lifetime-limited, not merely concurrent.
    // Thousands of transactions on a borrowed lane must create/destroy none.
    fail_wait = false; fail_copy = -1;
    for (int i = 0; i < 2048; ++i) {
        PageTransfer t(2, 16, &compute);
        t.restore(spans, pools, &compute, payload);
        CHECK(t.drain());
        CHECK(t.poll().status == TransferStatus::Complete);
        CHECK(streams == 0);
    }
    CHECK(host_allocs == 0 && streams == 0 && events == 0 && compute.queue.empty());
    std::cout << "Radiance transfer ordering, pending state and failure cleanup: PASS\n";
}
