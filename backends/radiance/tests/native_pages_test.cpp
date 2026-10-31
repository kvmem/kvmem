#include "native_pages.hpp"
#include "mem/kv.h"
#include <algorithm>
#include <cstring>
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

int main() {
    // Two real engine plans, unequal native page sizes, non-dense model layer
    // identities. Each pool has four physical slots; selection spans two pages
    // in group 1 and one in group 2. No GPU is needed to test byte addressing.
    std::vector<rad::KVGroupInfo> info(4);
    std::vector<rad::KVGroupPlan> plans;
    for (int id : {1, 2}) {
        auto& g = info[id];
        g.decl.kind = RAD_KV_FULL;
        g.decl.dtype = RAD_BF16;
        g.decl.n_head_kv = 1;
        g.decl.head_dim = 2;
        g.layers = id == 1 ? std::vector<int>{2, 7} : std::vector<int>{11};
        g.block_size = 2 * id;
        g.bytes_per_block = 2 * g.block_size * 2 * 2 * g.layers.size();
        rad::KVGroupPlan p;
        p.index = id; p.kind = g.decl.kind; p.n_layers = g.layers.size();
        p.block_size = g.block_size; p.bytes_per_block = g.bytes_per_block;
        p.n_blocks = 4; p.pool_bytes = p.bytes_per_block * p.n_blocks;
        plans.push_back(p);
    }
    rad::KVGroupPlan recurrent;
    recurrent.index = 3; recurrent.kind = RAD_KV_LINEAR;
    plans.push_back(recurrent); // must not turn recurrent state into attention pages
    NativePages layout(plans, info, 4, "pinned-engine/unit-scales");
    CHECK(layout.layout().planes.size() == 3);
    CHECK(layout.layout().planes[1].layer == 7);
    CHECK(layout.layout().record_bytes == 96);
    std::vector<GroupPages> src{{1, 4, {3, 1}}, {2, 4, {2}}};
    const auto spans = layout.spans(plans, src);
    CHECK(spans.size() == 5);
    CHECK(spans[0].pool_offset == 48);
    CHECK(spans[2].pool_offset == 112); // second layer, NOT page * bundled-layer size
    std::vector<std::vector<uint8_t>> pools(3), output(3);
    for (int id : {1, 2}) {
        pools[id].resize(plans[id - 1].pool_bytes);
        output[id].resize(pools[id].size(), 0xcd);
        for (size_t i = 0; i < pools[id].size(); ++i) pools[id][i] = uint8_t(i * 13 + id);
    }
    std::vector<uint8_t> record(layout.layout().record_bytes);
    for (auto s : spans)
        std::memcpy(record.data() + s.record_offset, pools[s.group].data() + s.pool_offset, s.bytes);
    // Restore to different physical IDs. Check selected AND untouched bytes.
    const auto restore = layout.spans(plans, {{1, 4, {0, 2}}, {2, 4, {1}}});
    std::vector<std::vector<bool>> touched(3);
    for (int id : {1, 2}) touched[id].resize(pools[id].size(), false);
    for (size_t i = 0; i < restore.size(); ++i) {
        const auto s = restore[i], before = spans[i];
        std::memcpy(output[s.group].data() + s.pool_offset, record.data() + s.record_offset, s.bytes);
        CHECK(std::memcmp(output[s.group].data() + s.pool_offset,
                          pools[before.group].data() + before.pool_offset, s.bytes) == 0);
        std::fill(touched[s.group].begin() + s.pool_offset,
                  touched[s.group].begin() + s.pool_offset + s.bytes, true);
    }
    for (int id : {1, 2}) for (size_t i = 0; i < output[id].size(); ++i)
        if (!touched[id][i]) CHECK(output[id][i] == 0xcd);
    rejects([&] { NativePages bad(plans, info, 3, "encoding"); });
    rejects([&] { layout.spans(plans, {{1, 2, {3, 1}}, {2, 4, {2}}}); }); // loaned tail
    rejects([&] { layout.spans(plans, {{1, 4, {1, 1}}, {2, 4, {2}}}); });
    rejects([&] { layout.spans(plans, {{1, 4, {-1, 1}}, {2, 4, {2}}}); });
    rejects([&] { layout.spans(plans, {{1, 4, {0, 1}}}); });
    auto changed = plans; changed[0].n_blocks = 5;
    rejects([&] { layout.spans(changed, src); });
    changed = plans; changed[0].kind = RAD_KV_WINDOW;
    rejects([&] { layout.spans(changed, src); });
    changed = plans; changed[0].bytes_per_block -= 1;
    rejects([&] { NativePages bad(changed, info, 4, "encoding"); });
    auto wrong_info = info; wrong_info[2].layers[0] = 7;
    rejects([&] { NativePages bad(plans, wrong_info, 4, "encoding"); });
    NativePages other_format(plans, info, 4, "different-scale-binding");
    CHECK(!(other_format.layout() == layout.layout()));
    // The native FP8 cache has half-width pages; do not expand it in the host
    // archive or accept a BF16 payload just because the token geometry matches.
    auto fp8_plans = plans;
    auto fp8_info = info;
    for (int id : {1, 2}) {
        fp8_info[id].decl.dtype = RAD_F8E4M3;
        fp8_info[id].bytes_per_block /= 2;
        fp8_plans[id - 1].bytes_per_block /= 2;
        fp8_plans[id - 1].pool_bytes /= 2;
    }
    NativePages fp8(fp8_plans, fp8_info, 4, "pinned-engine/unit-scales");
    CHECK(fp8.layout().record_bytes * 2 == layout.layout().record_bytes);
    CHECK(!(fp8.layout() == layout.layout()));
    const auto fp8_spans = fp8.spans(fp8_plans, src);
    CHECK(fp8_spans.size() == spans.size());
    for (size_t i = 0; i < spans.size(); ++i) {
        CHECK(fp8_spans[i].pool_offset * 2 == spans[i].pool_offset);
        CHECK(fp8_spans[i].record_offset * 2 == spans[i].record_offset);
        CHECK(fp8_spans[i].bytes * 2 == spans[i].bytes);
    }
    rejects([&] { fp8.spans(plans, src); });
    auto zero_rank = plans;
    zero_rank[0].zero = true; zero_rank[1].zero = true;
    rejects([&] { NativePages bad(zero_rank, info, 4, "encoding"); });
    std::cout << "Radiance native page addressing: PASS\n";
}
