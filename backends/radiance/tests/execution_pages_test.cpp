#include "execution_pages.hpp"
#include <iostream>
#include <limits>
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
    ExecutionView v;
    v.stamp = {true, {7, 3}, 5, 6, 22, 9};
    BlockDescriptor a{{7, 3}, 0, 0, 8, 1, 1}, b{{7, 3}, 2, 16, 6, 2, 2};
    v.blocks = {{a, 0}, {b, 8}};
    v.visible_tokens = 14;
    std::vector<BlockPages> bindings{{a, {5, 1}}, {b, {7, 3}}};
    ExecutionPages pages(v, 8, 4, 10, bindings, {9});
    CHECK((pages.table() == std::vector<int32_t>{5, 1, 7, 3, 9}));
    CHECK(pages.append_offset() == 8);
    CHECK(pages.append_capacity() == 6);
    CHECK(pages.slot(0) == 20);
    CHECK(pages.slot(7) == 7);
    for (uint64_t pos = 8; pos < 16; ++pos) CHECK(!pages.slot(pos));
    CHECK(pages.slot(16) == 28);
    CHECK(pages.slot(21) == 13);
    CHECK(pages.slot(22) == 14); // append in the actual retained partial tail
    CHECK(pages.slot(24) == 36);
    CHECK(!pages.slot(28));
    CHECK(!pages.slot(std::numeric_limits<uint64_t>::max()));
    CHECK(pages.used_after(22) == 14);
    CHECK(pages.used_after(28) == 20);
    rejects([&] { pages.used_after(21); });
    rejects([&] { pages.used_after(29); });

    // Copy identity matters, not just the logical ID or the physical page count.
    auto changed = bindings; ++changed[0].block.content_version;
    rejects([&] { ExecutionPages p(v, 8, 4, 10, changed); });
    changed = bindings; changed[1].pages[0] = 1;
    rejects([&] { ExecutionPages p(v, 8, 4, 10, changed); });
    changed = bindings; changed[1].pages[0] = 10;
    rejects([&] { ExecutionPages p(v, 8, 4, 10, changed); });
    changed = bindings; changed[0].pages.pop_back();
    rejects([&] { ExecutionPages p(v, 8, 4, 10, changed); });
    rejects([&] { ExecutionPages p(v, 8, 4, 10, bindings, {5}); });
    rejects([&] { ExecutionPages p(v, 8, 3, 10, bindings); });
    auto bad = v; bad.blocks[1].compact_start = 9;
    rejects([&] { ExecutionPages p(bad, 8, 4, 10, bindings); });
    bad = v; ++bad.stamp.session.incarnation;
    rejects([&] { ExecutionPages p(bad, 8, 4, 10, bindings); });
    bad = v; bad.stamp.frontier = 26; // same alignment, but partial tail is old
    rejects([&] { ExecutionPages p(bad, 8, 4, 10, bindings); });
    bad = v; bad.stamp.frontier = 23;
    rejects([&] { ExecutionPages p(bad, 8, 4, 10, bindings); });
    bad = v; --bad.visible_tokens;
    rejects([&] { ExecutionPages p(bad, 8, 4, 10, bindings); });

    // Dense is the identity case; physical IDs need not be monotone.
    v.stamp.frontier = 16; v.visible_tokens = 16;
    b.original_start = 8; b.valid_tokens = 8;
    v.blocks = {{a, 0}, {b, 8}};
    bindings = {{a, {5, 1}}, {b, {7, 3}}};
    ExecutionPages dense(v, 8, 4, 10, bindings, {9});
    CHECK(dense.append_offset() == 0);
    for (uint64_t pos = 0; pos < 20; ++pos)
        CHECK(dense.slot(pos) == int32_t(dense.table()[pos / 4] * 4 + pos % 4));

    // Empty selection at an aligned frontier: all slots are future writes.
    v.blocks.clear(); v.visible_tokens = 0;
    ExecutionPages empty(v, 8, 4, 10, {}, {6});
    CHECK(!empty.slot(15)); CHECK(empty.slot(16) == 24);
    CHECK(empty.used_after(17) == 1);
    std::cout << "Radiance original/compact page coordinates: PASS\n";
}
