// Production dispatch, host/HIP backends and original upstream blit kernels.
#include "device/device.h"
#include "rad_internal.h"
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
int main(int argc, char** argv) {
    try {
        auto* b = rad::hip_runtime_backend_if_present();
        CHECK(b && b->count() > 0);
        CHECK(std::strcmp(rad::device_backend_name(), "hip-runtime") == 0);
        CHECK(rad_dev_count() == b->count());
        CHECK(rad_stream_sync(nullptr) == RAD_E_INVAL);
        auto* host = rad::host_backend();
        const auto granule = host->vmem_granularity();
        CHECK(granule > 0 && granule <= (16ll << 20));
        auto* region = host->vmem_reserve(2*granule,granule);
        CHECK(region && reinterpret_cast<uintptr_t>(region) % granule == 0);
        CHECK(host->vmem_commit(region,granule) == RAD_OK);
        std::memset(region,0x91,4096);
        CHECK(host->vmem_decommit(region,granule) == RAD_OK);
        CHECK(host->vmem_commit(region,granule) == RAD_OK);
        for(int i=0;i<4096;++i) CHECK(static_cast<unsigned char*>(region)[i] == 0);
        CHECK(host->vmem_protect(region,granule) == RAD_OK);
        CHECK(host->vmem_decommit(region,granule) == RAD_OK);
        host->vmem_release(region,2*granule);
        int device = -1;
        int64_t largest = -1;
        for (int i = 0; i < b->count(); ++i) {
            RadDeviceProps p{}; CHECK(b->props(i, &p) == RAD_OK);
            std::printf("device %d %s %s free=%lld\n", i, p.name, p.arch, (long long)p.vram_free);
            if (p.vram_free > largest) { device = i; largest = p.vram_free; }
        }
        if (argc > 1) device = std::stoi(argv[1]);
        CHECK(device >= 0 && device < b->count());
        CHECK(b->set(device) == RAD_OK);
        std::printf("backend=%s selected_device=%d\n", b->name, device);
        constexpr int n = 65536;
        auto* h = static_cast<unsigned char*>(b->alloc(n, RAD_MEM_HOST_PINNED));
        auto* d = b->alloc(n, RAD_MEM_DEVICE);
        auto* r = static_cast<RadCopyRange*>(b->alloc(2*sizeof(RadCopyRange), RAD_MEM_HOST_MAPPED));
        CHECK(h && d && r && b->device_ptr(r));
        auto* pageable = b->alloc(128, RAD_MEM_HOST); CHECK(pageable);
        CHECK((reinterpret_cast<uintptr_t>(pageable) & 4095) == 0);
        b->free(pageable, RAD_MEM_HOST);
        for (int i = 0; i < n; ++i) h[i] = static_cast<unsigned char>(i*17+3);
        std::array<unsigned char,n> expected{}; std::memcpy(expected.data(),h,n);
        RadStream a{}, c{}; RadEvent event{}, local{};
        CHECK(b->stream_create(&a,0) == RAD_OK && b->stream_create(&c,1) == RAD_OK);
        CHECK(b->event_create(&event,RAD_EVENT_HOST_WAIT) == RAD_OK);
        CHECK(b->event_create(&local,RAD_EVENT_LOCAL) == RAD_OK);
        CHECK(b->stream_sync(nullptr) == RAD_E_INVAL);
        CHECK(b->memcpy_async(d,h,n,nullptr) == RAD_E_INVAL);
        CHECK(b->memcpy_async(d,h,-1,a) == RAD_E_INVAL);
        CHECK(b->event_sync(local) == RAD_E_INVAL);
        CHECK(b->memcpy_async(d,h,n,a) == RAD_OK);
        CHECK(b->event_record(event,a) == RAD_OK);
        CHECK(b->event_wait(c,event) == RAD_OK);
        CHECK(b->memcpy_async(h,d,n,c) == RAD_OK && b->stream_sync(c) == RAD_OK);
        CHECK(std::memcmp(h,expected.data(),n) == 0);
        CHECK(b->event_sync(event) == RAD_OK && b->event_query(event) == 1);
        for (int run = 0; run < 16; ++run) {
            CHECK(b->memset_async(d,0,n,a) == RAD_OK);
            r[0] = {64,128,128}; r[1] = {512,1024,256};
            b->host_wrote();
            CHECK(b->memcpy_ranges_async(d,h,32,static_cast<const RadCopyRange*>(b->device_ptr(r)),2,a) == RAD_OK);
            CHECK(b->stream_sync(a) == RAD_OK);
            std::array<unsigned char,n> got{};
            CHECK(b->memcpy_async(got.data(),d,n,a) == RAD_OK && b->stream_sync(a) == RAD_OK);
            for (int i=0;i<n;++i) {
                unsigned char want = i < 32 ? h[i] : i>=128 && i<256 ? h[i-64] :
                                     i>=1024 && i<1280 ? h[i-512] : 0;
                CHECK(got[i] == want);
            }
        }
        CHECK(!b->tape_begin && !b->vmem_reserve); // explicitly unqualified optional features
        b->event_destroy(local); b->event_destroy(event);
        b->stream_destroy(c); b->stream_destroy(a);
        b->free(r,RAD_MEM_HOST_MAPPED); b->free(d,RAD_MEM_DEVICE); b->free(h,RAD_MEM_HOST_PINNED);
        std::puts("PASS: 64KiB roundtrip, cross-stream ordering, 16 real GPU ranged copies, null/error contracts");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
