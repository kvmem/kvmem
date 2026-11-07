#include "kvmem/layerwise_query.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace kvmem;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
template<class F> void rejects(F f, MemoryErrorCode code = MemoryErrorCode::InvalidPlan) {
    try { f(); } catch (const MemoryError& e) { CHECK(e.code() == code); return; }
    throw std::runtime_error("expected rejection");
}
LayerwiseQueryIdentity identity() {
    return {{true, {7, 2}, 11, 13, 4096, 17}, 19, {4096, 4707}, {4100, 4700}};
}
void successful_partitions(uint64_t chunk) {
    auto i = identity(); LayerwiseQueryProgress p(i, {3, 7, 11});
    for (uint32_t l : {3u, 7u, 11u}) {
        p.begin_layer(i, l);
        for (auto at = i.evaluation.begin; at < i.evaluation.end;) {
            const auto n = std::min(chunk, i.evaluation.end - at);
            p.projected(i, l, at, n); at += n;
        }
        p.begin_restore(i, l); p.restored(i, l, 2048);
        for (auto at = i.evaluation.begin; at < i.evaluation.end;) {
            const auto n = std::min(chunk, i.evaluation.end - at);
            p.attended(i, l, at, n); at += n;
        }
        p.complete_layer(i, l);
    }
    CHECK(p.phase() == LayerwiseQueryPhase::Complete);
    CHECK(p.work().projected_rows == 611 * 3);
    CHECK(p.work().attended_rows == 611 * 3);
    CHECK(p.work().completed_layers == 3);
    rejects([&] { p.begin_layer(i, 3); });
}
int main() {
    try {
        for (auto chunk : {1u, 127u, 256u, 512u, 2048u}) successful_partitions(chunk);
        auto i = identity();
        for (int field = 0; field < 4; ++field) {
            auto zero = i;
            if (field == 0) zero.history.session.id = 0;
            if (field == 1) zero.history.session.incarnation = 0;
            if (field == 2) zero.history.history_revision = 0;
            if (field == 3) zero.history.execution_history = 0;
            rejects([&] { LayerwiseQueryProgress p(zero, {3}); });
        }
        rejects([&] { LayerwiseQueryProgress p(i, {}); });
        rejects([&] { LayerwiseQueryProgress p(i, {7, 3}); });
        rejects([&] { LayerwiseQueryProgress p(i, {3, 3}); });
        auto bad = i; bad.query.end = i.evaluation.end + 1;
        rejects([&] { LayerwiseQueryProgress p(bad, {3}); });
        bad = i; bad.evaluation.begin--;
        rejects([&] { LayerwiseQueryProgress p(bad, {3}); });
        LayerwiseQueryProgress p(i, {3, 7});
        rejects([&] { p.begin_layer(i, 7); });
        bad = i; bad.history.execution_history++;
        rejects([&] { p.begin_layer(bad, 3); }, MemoryErrorCode::Stale);
        bad = i; bad.request++;
        rejects([&] { p.begin_layer(bad, 3); }, MemoryErrorCode::Stale);
        p.begin_layer(i, 3);
        rejects([&] { p.begin_restore(i, 3); });
        rejects([&] { p.projected(i, 3, 4097, 1); });
        rejects([&] { p.projected(i, 3, 4096, UINT64_MAX); });
        p.projected(i, 3, 4096, 256);
        rejects([&] { p.projected(i, 3, 4096, 256); });
        rejects([&] { p.begin_restore(i, 3); });
        p.projected(i, 3, 4352, 355);
        rejects([&] { p.projected(i, 3, 4096, 611); });
        p.begin_restore(i, 3);
        rejects([&] { p.attended(i, 3, 4096, 611); });
        rejects([&] { p.restored(i, 3, 4097); });
        p.restored(i, 3, 2048);
        p.attended(i, 3, 4096, 256);
        rejects([&] { p.complete_layer(i, 3); });
        p.attended(i, 3, 4352, 355); p.complete_layer(i, 3);
        p.begin_layer(i, 7); p.projected(i, 7, 4096, 611); p.begin_restore(i, 7);
        rejects([&] { p.restored(i, 7, 2047); });
        p.abort(); CHECK(p.phase() == LayerwiseQueryPhase::Aborted);
        rejects([&] { p.restored(i, 7, 2048); });
        rejects([&] { p.begin_layer(i, 7); });
        std::cout << "layerwise Query progress tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return EXIT_FAILURE;
    }
}
