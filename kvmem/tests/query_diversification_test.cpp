#include "kvmem/mean_key_retrieval.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/kvmem_store.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class Function> bool invalid(Function f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}
int main() {
    try {
        const std::vector<double> single{-0.2, 0.9, 0.4, -0.7};
        require(kvmem::diversify_retrieval_scores({single}, {1}) == single,
                "one-query scoring changed");
        const auto opposite = kvmem::diversify_retrieval_scores({{1, .8, -.2, -1}, {-1, -.2, .8, 1}});
        require(opposite[0] > opposite[3] && opposite[3] > opposite[1] && opposite[1] > opposite[2],
                "independent opposed queries were averaged or ranking order changed");
        require(kvmem::diversify_retrieval_scores({single, {0, 0, 0, 0}}) == single,
                "zero-query gained arbitrary page-ID votes");
        require(kvmem::diversify_retrieval_scores({single, {.4, .4, .4, .4}}) == single,
                "constant-score query displaced a useful query");
        const auto tied = kvmem::diversify_retrieval_scores({{1, 1, 0, 0}, {0, 0, 1, 1}});
        require(tied[0] > tied[2] && tied[2] > tied[1] && tied[1] > tied[3],
                "equal-score ties are not deterministic");
        const auto excluded = kvmem::diversify_retrieval_scores({{4, 3, 2, 1}, {4, 1, 2, 3}}, {0});
        require(excluded[0] == 0 && excluded[1] > excluded[3] && excluded[3] > excluded[2],
                "mandatory pages consumed the query interleave");
        const auto flat = kvmem::diversify_retrieval_scores({{0, 0, 0}, {1, 1, 1}});
        require(flat[2] > flat[1] && flat[1] > flat[0], "uninformative queries lost recency fallback");
        require(invalid([] { kvmem::diversify_retrieval_scores({{1}, {1, 2}}); }), "invalid geometry accepted");
        require(invalid([] { kvmem::diversify_retrieval_scores({{std::numeric_limits<double>::quiet_NaN()}}); }),
                "nonfinite query score accepted");
        // Production cosine -> production shared-budget selector. Clock-only
        // retrieval loses task history; independent task + clock retains both.
        kvmem::RawKvStore keys({1, 2, 0, 64});
        for (unsigned block = 0; block < 17; ++block) {
            const float row[]{block == 2 ? 64.F : 0.F, block == 2 ? 0.F : 64.F};
            keys.write_layer_mean_sum(block * 64, 64, 0, row);
        }
        const float task[]{1, 0}, clock[]{0, 1}; const unsigned count[]{1};
        const auto a = kvmem::mean_key_retrieval_scores(keys, 17, {task, count, 1, 2, 1, 1});
        const auto b = kvmem::mean_key_retrieval_scores(keys, 17, {clock, count, 1, 2, 1, 1});
        kvmem::KvMemStoreConfig config;
        config.block_tokens = 64; config.select_budget = 256; config.sink_blocks = 1; config.recent_blocks = 1;
        kvmem::KvMemStore selector(config); selector.register_append(1088);
        selector.set_retrieval_scores(b); const auto old = selector.pick_topk_blocks({16});
        require(std::find(old.begin(), old.end(), 2) == old.end(), "clock-only control did not reproduce loss");
        selector.set_retrieval_scores(kvmem::diversify_retrieval_scores({a, b}, {0, 16}));
        const auto selected = selector.pick_topk_blocks({16});
        require(selected.size() == 4 && std::find(selected.begin(), selected.end(), 2) != selected.end(),
                "multiple queries expanded the budget or lost task history");
        require(selected.front() == 0 && selected.back() == 16, "mandatory pages lost");
        const float negative[]{-1, 0};
        const auto c = kvmem::mean_key_retrieval_scores(keys, 17, {negative, count, 1, 2, 1, 1});
        const auto separated = kvmem::diversify_retrieval_scores({a, c}, {0, 16});
        require(separated[2] > 0, "Q and -Q canceled before scoring");
        std::cout << "QUERY_DIVERSIFICATION PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
