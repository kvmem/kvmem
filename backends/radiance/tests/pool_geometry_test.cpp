#include "pool_geometry.hpp"
#include <iostream>
#include <stdexcept>
using namespace kvmem::radiance;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
template<class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    CHECK(caught);
}
int main() {
    auto p = pool_geometry(16, 64, 128, {16});
    CHECK(p.block_tokens == 16 && p.resident_tokens == 352);
    p = pool_geometry(16, 64, 129, {16});
    CHECK(p.resident_tokens == 384);
    p = pool_geometry(8, 192, 100, {16, 24});
    CHECK(p.block_tokens == 48 && p.resident_tokens == 576);
    CHECK(pool_geometry(128, 0, 128, {16}).resident_tokens == 0);
    const auto leased = pool_geometry(128, 1536, 128, {16});
    CHECK(leased_pool_tokens(leased, 1536, 1024) == 3072);
    CHECK(leased_pool_tokens(leased, 1536, 1025) == 3200);
    CHECK(leased_pool_tokens(pool_geometry(128, 7680, 128, {16}), 7680, 1024) == 9216);
    CHECK(leased_pool_tokens(pool_geometry(128, 36864, 128, {16}), 36864, 1024) == 38400);
    rejects([&] { leased_pool_tokens(leased, 1536, 0); });
    rejects([&] { leased_pool_tokens(leased, 1536, 2147483647u); });
    CHECK(output_fits_reserve(512, 128, 1024));
    CHECK(output_fits_reserve(896, 128, 1024));
    CHECK(!output_fits_reserve(897, 128, 1024));
    CHECK(!output_fits_reserve(-1, 128, 1024));
    CHECK(!output_fits_reserve(INT64_MAX, 128, 1024));
    rejects([] { pool_geometry(0, 64, 128, {16}); });
    rejects([] { pool_geometry(16, 64, 0, {16}); });
    rejects([] { pool_geometry(16, 64, 128, {}); });
    rejects([] { pool_geometry(16, 64, 128, {0}); });
    rejects([] { pool_geometry(16, 16, 128, {16}); });
    rejects([] { pool_geometry(16, 65, 128, {16}); });
    rejects([] { pool_geometry(65537, 0, 128, {65539}); });
    rejects([] { pool_geometry(16, 64, 2147483647u, {16}); });
    std::cout << "pool geometry: alignment, bounded residency and overflow PASS\n";
}
