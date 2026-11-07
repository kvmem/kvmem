#include "kvmem/layerwise_scoring.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void close(double actual, double expected) {
    require(std::isfinite(actual) && std::abs(actual - expected) < 2e-12,
            "layerwise score differs from independent reference");
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid layerwise scoring input was accepted");
}

void current_layer_and_gqa() {
    kvmem::RawKvStore keys({3, 4, 0, 1});
    const float means[3][4]{{1,0,0,2}, {0,1,-1,0}, {-1,-1,2,1}};
    const float other[4]{30,-20,10,40};
    for (uint32_t block = 0; block < 3; ++block) {
        keys.write_layer_mean_sum(block,1,1,means[block]);
        keys.write_layer_mean_sum(block,1,0,other);
        keys.write_layer_mean_sum(block,1,2,other);
    }
    const float query[2][4][2]{{{2,0},{0,1},{-1,1},{2,-2}},
                             {{-1,0},{1,2},{0,2},{-2,1}}};
    const auto frame = kvmem::query_score_frame(3,1,2,0,0,{});
    for (float scale : {0.f, .25f, 1.f}) {
        const auto actual = kvmem::layerwise_mean_key_scores(keys,
            {&query[0][0][0],1,2,4,2,2,scale}, frame);
        long double expected[3]{};
        const long double factor = scale ? scale : 1/std::sqrt(2.);
        for (uint32_t row = 0; row < 2; ++row) {
            for (uint32_t head = 0; head < 4; ++head) {
                long double weights[3]{}, denominator = 0;
                for (uint32_t block = 0; block < 3; ++block) {
                    const uint32_t offset = (head/2)*2;
                    const long double dot = query[row][head][0]*means[block][offset] +
                                            query[row][head][1]*means[block][offset+1];
                    denominator += weights[block] = std::exp(dot*factor);
                }
                for (uint32_t block = 0; block < 3; ++block)
                    expected[block] += weights[block]/denominator/4;
            }
        }
        for (uint32_t block = 0; block < 3; ++block)
            close(actual[block],double(expected[block]));
        close(std::accumulate(actual.begin(),actual.end(),0.),2);
    }
}

void domain_and_mandatory() {
    kvmem::RawKvStore keys({2,2,0,4});
    const float sums[3][2]{{8,0},{0,4},{40,0}};
    for (uint32_t block = 0; block < 3; ++block)
        keys.write_layer_mean_sum(block*4,4,1,sums[block]);
    const float q[]{2,0,0,2};
    const kvmem::CurrentLayerQuery query{q,1,2,1,1,2};
    auto frame = kvmem::query_score_frame(12,4,2,0,0,{2});
    require(frame.score_mask[2] && !frame.transport_mask[2],
            "mandatory fixture does not distinguish scoring and transport");
    const auto actual = kvmem::layerwise_mean_key_scores(keys,query,frame);
    long double expected[3]{};
    const long double logits[2][3]{{4/std::sqrt(2.),0,20/std::sqrt(2.)},
                                  {0,2/std::sqrt(2.),0}};
    for (const auto& row : logits) {
        long double denominator = 0;
        for (auto value : row) denominator += std::exp(value);
        for (uint32_t block = 0; block < 3; ++block)
            expected[block] += std::exp(row[block])/denominator;
    }
    for (uint32_t block = 0; block < 3; ++block)
        close(actual[block],double(expected[block]));
    require(actual[1] > actual[0], "mandatory denominator regression");
    frame.score_mask[2] = 0;
    const auto masked = kvmem::layerwise_mean_key_scores(keys,query,frame);
    require(masked[0] > masked[1], "fixture cannot detect a changed denominator");
    close(masked[2],0);
    close(std::accumulate(masked.begin(),masked.end(),0.),2);
    frame.score_mask.assign(3,0);
    const auto empty = kvmem::layerwise_mean_key_scores(keys,query,frame);
    close(std::accumulate(empty.begin(),empty.end(),0.),0);
}

void token_tiles_preserve_the_complete_page_domain() {
    kvmem::RawKvStore keys({1,1,0,1});
    const float a[]{1}, b[]{-1}, c[]{.5};
    keys.write_layer_mean_sum(0,1,0,a);
    keys.write_layer_mean_sum(1,1,0,b);
    keys.write_layer_mean_sum(2,1,0,c);
    const float q[]{1,2,-1};
    const auto frame = kvmem::query_score_frame(3,1,2,0,0,{});
    const auto full = kvmem::layerwise_mean_key_scores(keys,{q,0,3,1,1,1},frame);
    const auto first = kvmem::layerwise_mean_key_scores(keys,{q,0,1,1,1,1},frame);
    const auto rest = kvmem::layerwise_mean_key_scores(keys,{q+1,0,2,1,1,1},frame);
    for (uint32_t block = 0; block < 3; ++block)
        close(full[block],first[block]+rest[block]);
}

void target_layer_coverage() {
    kvmem::RawKvStore keys({2,1,0,4});
    const float full[]{4}, one[]{1};
    keys.write_layer_mean_sum(0,4,0,full);
    keys.write_layer_mean_sum(4,2,0,full);
    const float q[]{1};
    const kvmem::CurrentLayerQuery query{q,1,1,1,1,1};
    const auto frame = kvmem::query_score_frame(6,4,2,0,0,{});
    require(keys.has_block(0) && keys.mean_k_tokens(0,1) == 0,
            "fixture does not distinguish block existence and target-layer statistics");
    require(keys.mean_k_tokens(20,1) == 0 && keys.mean_k_tokens(0,20) == 0,
            "absent Mean-K coverage must be zero");
    rejects([&]{kvmem::layerwise_mean_key_scores(keys,query,frame);});
    keys.write_layer_mean_sum(0,4,1,full);
    keys.write_layer_mean_sum(4,1,1,one);
    require(keys.mean_k_tokens(0,1) == 4 && keys.mean_k_tokens(1,1) == 1,
            "Mean-K coverage used another layer or a block maximum");
    rejects([&]{kvmem::layerwise_mean_key_scores(keys,query,frame);});
    // A masked page is still part of the frozen index and must have the same
    // accepted coverage; its absence cannot be hidden by a scoring mask.
    auto masked = frame; masked.score_mask[1] = 0;
    rejects([&]{kvmem::layerwise_mean_key_scores(keys,query,masked);});
    keys.write_layer_mean_sum(5,1,1,one);
    const auto valid = kvmem::layerwise_mean_key_scores(keys,query,frame);
    close(valid[0],.5); close(valid[1],.5);
    // A probe or append that updated the partial block beyond the accepted
    // frontier must restore the exact mean checkpoint before selection.
    keys.write_layer_mean_sum(6,1,1,one);
    require(keys.mean_k_tokens(1,1) == 3, "future-tail fixture did not advance");
    rejects([&]{kvmem::layerwise_mean_key_scores(keys,query,frame);});
}

void invalid_inputs() {
    kvmem::RawKvStore keys({2,2,0,1});
    const float k[]{1,0}; keys.write_layer_mean_sum(0,1,1,k);
    const float q[]{1,0};
    const auto frame = kvmem::query_score_frame(1,1,1,0,0,{});
    const kvmem::CurrentLayerQuery valid{q,1,1,1,1,2};
    auto run = [&](kvmem::CurrentLayerQuery v) {
        return kvmem::layerwise_mean_key_scores(keys,v,frame);
    };
    auto bad = valid; bad.data = nullptr; rejects([&]{run(bad);});
    bad = valid; bad.layer = 2; rejects([&]{run(bad);});
    bad = valid; bad.rows = 0; rejects([&]{run(bad);});
    bad = valid; bad.query_heads = 0; rejects([&]{run(bad);});
    bad = valid; bad.kv_heads = 0; rejects([&]{run(bad);});
    bad = valid; bad.query_heads = 3; bad.kv_heads = 2; rejects([&]{run(bad);});
    bad = valid; bad.head_dim = 1; rejects([&]{run(bad);});
    for (float scale : {-1.f,std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        bad = valid; bad.attention_scale = scale; rejects([&]{run(bad);});
    }
    for (float value : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(),65536.f}) {
        const float broken[]{value,0};
        bad = valid; bad.data = broken; rejects([&]{run(bad);});
    }
    auto broken = frame; broken.counts.clear();
    rejects([&]{kvmem::layerwise_mean_key_scores(keys,valid,broken);});
    broken = frame; broken.starts[0] = 1;
    rejects([&]{kvmem::layerwise_mean_key_scores(keys,valid,broken);});
    auto empty = frame; empty.score_mask[0] = 0;
    const float nan[]{std::numeric_limits<float>::quiet_NaN(),0};
    bad = valid; bad.data = nan;
    rejects([&]{kvmem::layerwise_mean_key_scores(keys,bad,empty);});
}
}

int main() {
    try {
        current_layer_and_gqa();
        domain_and_mandatory();
        token_tiles_preserve_the_complete_page_domain();
        target_layer_coverage();
        invalid_inputs();
        std::cout << "layerwise_scoring_test: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
