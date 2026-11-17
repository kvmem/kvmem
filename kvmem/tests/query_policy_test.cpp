#include "kvmem/query_policy.hpp"
#include "kvmem/graph_recall.hpp"
#include "kvmem/raw_kv_store.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)

int main() {
    using namespace kvmem;
    auto q = bounded_query(3, 129784, 129796, 512);
    CHECK(q.begin == 129275 && q.end == 129787);
    q = bounded_query(100, std::numeric_limits<uint64_t>::max(), 120, 512);
    CHECK(q.begin == 100 && q.end == 120);
    CHECK(bounded_query(140, 20, 120, 512).empty());
    CHECK(!query_replay_plan({}, 0, 100, 2048, 64, 1, 2).admitted);
    CHECK(query_replay_plan({100,120}, 100, 140, 2048, 64, 2, 3, 4).admitted);
    CHECK(!query_replay_plan({100,120}, 100, 140, 2048, 64, 2, 2, 4).admitted);
    CHECK(!query_replay_plan({100,120}, 100, 2200, 2048, 64, 2, 100).admitted);
    CHECK(!query_replay_plan({100,120}, 101, 140, 2048, 64, 2, 3).admitted);
    CHECK(query_replay_plan({64,128},64,128,64,64,1,2).admitted);
    CHECK(!query_replay_plan({64,128},64,128,63,64,1,2).admitted);
    CHECK((query_prefix_required(321,64,1,1,{2,20}) == std::vector<uint32_t>{0,2,5}));
    CHECK((query_prefix_required(321,64,1,0,{}) == std::vector<uint32_t>{0,5}));
    // A short Q does not permit dropping the tool/template dependency suffix.
    CHECK(!query_replay_plan({100,110},100,3000,2048,64,1,100).admitted);
    CHECK(!query_replay_plan({UINT64_MAX-1,UINT64_MAX},UINT64_MAX-1,UINT64_MAX,
        2048,64,1,100,1).admitted);

    // A full 256K Query is replayed through a 36K+16K physical window.
    const auto full_replay = streaming_query_replay_plan({66,262139},66,262148,0,64,2,832,34);
    CHECK(full_replay.admitted && full_replay.begin==66 && full_replay.end==262148);
    CHECK(std::strcmp(full_replay.reason,"streaming_query_replay")==0);
    CHECK(!query_replay_plan({66,262139},66,262148,UINT32_MAX,64,2,832,34).admitted);
    CHECK(!streaming_query_replay_plan({66,262139},66,262148,2048,64,2,832,34).admitted);
    CHECK(streaming_query_replay_plan({100,120},100,140,0,64,2,3,4).admitted);
    CHECK(!streaming_query_replay_plan({100,120},101,140,0,64,2,3).admitted);
    CHECK(!streaming_query_replay_plan({64,256},64,256,0,0,1,8).admitted);
    CHECK(!streaming_query_replay_plan({64,256},64,256,0,64,8,8).admitted);
    CHECK(!streaming_query_replay_plan({64,256},64,256,0,64,7,8,64).admitted);
    CHECK(!streaming_query_replay_plan({64,256},64,256,0,64,9,8).admitted);
    CHECK(!streaming_query_replay_plan({},0,256,0,64,0,8).admitted);

    // Independent prefix oracle: a probe grows the partially filled history
    // page, but neither its new K nor its extra page belongs in Query scoring.
    RawKvStoreConfig config;
    config.n_layer=1;config.n_embd_k=1;config.block_tokens=4;
    RawKvStore keys(config);
    const float history[]{1,2,3,4,5,6,7};
    keys.write_layer_mean_k(0,7,0,history);
    const auto prefix=keys.mean_checkpoint(7);
    const float probe[]{100,200};
    keys.write_layer_mean_k(7,2,0,probe);
    const auto full=keys.mean_checkpoint(7);
    keys.restore_mean_checkpoint(7,prefix);
    const uint32_t rows=1;
    const float query=1;
    const auto frame=query_score_frame(7,4,1,0,0,{});
    const auto scores=page_softmax_scores(keys,{&query,&rows,1,1,1,1,1},frame);
    const double expected=std::exp(2.5)/(std::exp(2.5)+std::exp(6.0));
    CHECK(scores.size()==2 && std::abs(scores[0]-expected)<1e-12 &&
        std::abs(scores[1]-(1-expected))<1e-12);
    keys.restore_mean_checkpoint(7,full);
    float mean=0;
    keys.mean_k(1,0,&mean);
    CHECK(mean==29.5f && keys.mean_checkpoint(7)==full);
}
