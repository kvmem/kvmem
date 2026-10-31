// Real runtime roundtrip using the adapter, two pools and non-dense layer IDs.
// This is a bounded byte-identity gate, not an inference or retrieval benchmark.
#include "page_transfer.hpp"
#include "mem/kv.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace kvmem;
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
struct Buffer {
    void* ptr; int kind;
    Buffer(int64_t n, int k) : ptr(rad_dev_alloc(n, k)), kind(k) { CHECK(ptr); }
    ~Buffer() { rad_dev_free(ptr, kind); }
};
struct Stream {
    RadStream ptr = nullptr;
    Stream() { CHECK(rad_stream_create(&ptr, 0) == RAD_OK); }
    ~Stream() { rad_stream_sync(ptr); rad_stream_destroy(ptr); }
};
int main(int argc, char** argv) {
    try {
        int device = -1;
        if (argc == 2) device = std::stoi(argv[1]);
        else for (int i = 0; i < rad_dev_count(); ++i) {
            RadDeviceProps p{};
            if (rad_dev_props(i, &p) == RAD_OK && !p.is_host_backend) { device = i; break; }
        }
        CHECK(device >= 0 && rad_dev_set(device) == RAD_OK);
        RadDeviceProps props{}; CHECK(rad_dev_props(device, &props) == RAD_OK);
        CHECK(!props.is_host_backend);
        std::cout << "device=" << props.name << " arch=" << props.arch << '\n';
        std::vector<rad::KVGroupInfo> infos(2);
        std::vector<rad::KVGroupPlan> plans(2);
        for (int i = 0; i < 2; ++i) {
            auto& p = plans[i]; auto& g = infos[i];
            p.index = i; p.kind = RAD_KV_FULL; p.block_size = 4 * (i + 1);
            p.n_layers = 2; p.n_blocks = 8;
            p.bytes_per_block = 2 * p.block_size * 2 * 16 * 2;
            p.pool_bytes = p.bytes_per_block * p.n_blocks;
            g.decl.kind = p.kind; g.decl.dtype = RAD_BF16;
            g.decl.n_head_kv = 1; g.decl.head_dim = 16;
            g.block_size = p.block_size; g.bytes_per_block = p.bytes_per_block;
            g.layers = {i * 5 + 1, i * 5 + 3};
        }
        Buffer a(plans[0].pool_bytes, RAD_MEM_DEVICE), b(plans[1].pool_bytes, RAD_MEM_DEVICE);
        Buffer ha(plans[0].pool_bytes, RAD_MEM_HOST_PINNED), hb(plans[1].pool_bytes, RAD_MEM_HOST_PINNED);
        // Stream is destroyed/drained before backing allocations on exceptional paths.
        Stream compute;
        plans[0].base = a.ptr; plans[1].base = b.ptr;
        std::vector<GroupPool> pools{{0, a.ptr, uint64_t(plans[0].pool_bytes)},
                                      {1, b.ptr, uint64_t(plans[1].pool_bytes)}};
        std::vector<void*> hosts{ha.ptr, hb.ptr};
        for (int i = 0; i < 2; ++i) {
            auto* h = static_cast<uint8_t*>(hosts[i]);
            for (int64_t j = 0; j < plans[i].pool_bytes; ++j) h[j] = uint8_t(j * 17 + i * 7);
            CHECK(rad_memcpy_async(pools[i].base, h, plans[i].pool_bytes, compute.ptr) == RAD_OK);
        }
        NativePages layout(plans, infos, 8, "BF16/no-external-scales/test");
        auto from = layout.spans(plans, {{0, 8, {5, 1}}, {1, 8, {6}}});
        std::vector<uint8_t> payload;
        {
            PageTransfer transfer(device, layout.layout().record_bytes);
            transfer.spill(from, pools, compute.ptr); // must wait for the queued uploads
            CHECK(transfer.drain());
            CHECK(transfer.poll().status == TransferStatus::Complete);
            payload = transfer.payload();
        }
        for (auto s : from)
            CHECK(std::memcmp(payload.data() + s.record_offset,
                              static_cast<uint8_t*>(hosts[s.group]) + s.pool_offset, s.bytes) == 0);
        for (const auto& p : pools)
            CHECK(rad_memset_async(p.base, 0xcd, int64_t(p.bytes), compute.ptr) == RAD_OK);
        auto to = layout.spans(plans, {{0, 8, {0, 7}}, {1, 8, {2}}});
        {
            PageTransfer transfer(device, layout.layout().record_bytes);
            transfer.restore(to, pools, compute.ptr, payload);
            CHECK(transfer.drain());
            CHECK(transfer.poll().status == TransferStatus::Complete);
        }
        for (int i = 0; i < 2; ++i)
            CHECK(rad_memcpy_async(hosts[i], pools[i].base, plans[i].pool_bytes, compute.ptr) == RAD_OK);
        CHECK(rad_stream_sync(compute.ptr) == RAD_OK);
        size_t checked = 0;
        for (int i = 0; i < 2; ++i) {
            std::vector<uint8_t> expected(size_t(plans[i].pool_bytes), 0xcd);
            for (auto s : to) if (s.group == i)
                std::memcpy(expected.data() + s.pool_offset, payload.data() + s.record_offset, s.bytes);
            CHECK(std::memcmp(expected.data(), hosts[i], expected.size()) == 0);
            checked += expected.size();
        }
        std::cout << "PASS native GPU spill/restore and untouched bytes: " << checked << " bytes\n";
        // Cross the native AQL lifetime queue limit with transaction count,
        // while reusing one engine-owned stream. No model inference required.
        for (int i = 0; i < 1100; ++i) {
            PageTransfer transfer(device, layout.layout().record_bytes, compute.ptr);
            transfer.spill(to, pools, compute.ptr);
            CHECK(transfer.drain());
            CHECK(transfer.poll().status == TransferStatus::Complete);
            CHECK(transfer.payload() == payload);
        }
        std::cout << "PASS 1100 borrowed-lane transfers beyond AQL queue-ID lifetime limit\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
