#include "hipfire_kvmem_shim.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__,         \
                         __LINE__, #cond);                                     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

namespace {

constexpr uint32_t kBlock = 4;
constexpr uint32_t kWidth = 8;
constexpr uint32_t kLayers = 2;
constexpr uint64_t kRowBytes = 36;

hk_raw_store_config raw_config() {
    hk_raw_store_config cfg{};
    cfg.n_layer = kLayers;
    cfg.n_embd_k = kWidth;
    cfg.n_embd_v = kWidth;
    cfg.block_tokens = kBlock;
    cfg.k_row_bytes = 0;
    cfg.k_gpu_row_bytes = kRowBytes;
    cfg.v_gpu_row_bytes = kRowBytes;
    return cfg;
}

hk_kv_store_config kv_config() {
    hk_kv_store_config cfg{};
    cfg.block_tokens = kBlock;
    cfg.select_budget = 2 * kBlock;
    cfg.gen_budget = 32768;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    cfg.lrfu_lambda = 0.5;
    cfg.lrfu_top_fraction = 0.9;
    cfg.gpu_memory_ratio = 0.5;
    cfg.gpu_high_watermark = 0.95;
    cfg.gpu_low_watermark = 0.85;
    return cfg;
}

} // namespace

int main() {
    CHECK(hk_abi_version() == HK_ABI_VERSION);
    CHECK(hk_version() != nullptr && *hk_version() != '\0');
    CHECK(hk_kvmem_commit() != nullptr && *hk_kvmem_commit() != '\0');

    hk_storage * storage = hk_storage_create(1u << 20, nullptr, 0);
    CHECK(storage != nullptr);

    const hk_raw_store_config raw_cfg = raw_config();
    hk_raw_store * raw = hk_raw_store_create(&raw_cfg, storage);
    CHECK(raw != nullptr);

    std::vector<float> k(kBlock * kWidth);
    for (uint32_t t = 0; t < kBlock; ++t) {
        for (uint32_t d = 0; d < kWidth; ++d) {
            k[t * kWidth + d] = static_cast<float>(t * kWidth + d);
        }
    }
    hk_raw_store_write_layer_mean_k(raw, 0, kBlock, 0, k.data());
    CHECK(hk_last_error() == nullptr);
    CHECK(hk_raw_store_mean_k_tokens(raw, 0, 0) == kBlock);

    float mean[kWidth] = {};
    hk_raw_store_mean_k(raw, 0, 0, mean);
    for (uint32_t d = 0; d < kWidth; ++d) {
        float expect = 0.0f;
        for (uint32_t t = 0; t < kBlock; ++t) {
            expect += static_cast<float>(t * kWidth + d);
        }
        expect /= static_cast<float>(kBlock);
        CHECK(std::fabs(mean[d] - expect) < 1e-6f);
    }

    // Query-span page-softmax contract. The second block is deliberately close
    // to the first so a reduction-order change can flip the ranking instead of
    // hiding behind a wide margin.
    hk_raw_store_config soft_cfg = raw_config();
    soft_cfg.n_layer = 1;
    soft_cfg.k_gpu_row_bytes = 0;
    soft_cfg.v_gpu_row_bytes = 0;
    hk_raw_store * soft_raw = hk_raw_store_create(&soft_cfg, storage);
    CHECK(soft_raw != nullptr);
    std::vector<float> full(kBlock * kWidth, 1.0f);
    std::vector<float> near(kBlock * kWidth, 0.999f);
    hk_raw_store_write_layer_mean_k(soft_raw, 0, kBlock, 0, full.data());
    hk_raw_store_write_layer_mean_k(soft_raw, kBlock, kBlock, 0, near.data());
    const float query[kWidth] = {1, 1, 1, 1, 1, 1, 1, 1};
    const uint32_t layer_rows[] = {1};
    double softmax[2] = {};
    const int soft_rc = hk_page_softmax_scores(soft_raw, 2, 2 * kBlock, kBlock, 2, 0, 0,
                                               query, layer_rows, 1, 1, 1, 1, kWidth,
                                               1.0f / std::sqrt(static_cast<float>(kWidth)),
                                               0, softmax);
    if (soft_rc != HK_OK) {
        std::fprintf(stderr, "page-softmax: %s\n", hk_last_error());
    }
    CHECK(soft_rc == HK_OK);
    CHECK(hk_last_error() == nullptr);
    CHECK(softmax[0] > softmax[1]);
    CHECK(std::fabs(softmax[0] + softmax[1] - 1.0) < 1e-12);
    hk_raw_store_destroy(soft_raw);

    const hk_kv_store_config kv_cfg = kv_config();
    hk_kv_store * kv = hk_kv_store_create(&kv_cfg);
    CHECK(kv != nullptr);
    hk_kv_store_register_append(kv, 12);
    CHECK(hk_last_error() == nullptr);
    CHECK(hk_kv_store_block_count(kv) == 3);
    CHECK(hk_kv_store_total_tokens(kv) == 12);

    uint32_t picked[4] = {};
    const uint32_t picked_count =
        hk_kv_store_pick_prefill_pressure_blocks(kv, nullptr, 0, picked, 4);
    CHECK(picked_count == 2);
    CHECK(picked[0] == 0 && picked[1] == 2);

    hk_kv_store_destroy(kv);
    hk_raw_store_destroy(raw);
    hk_storage_destroy(storage);

    hk_clear_error();
    CHECK(hk_raw_store_create(nullptr, nullptr) == nullptr);
    CHECK(hk_last_error() != nullptr && *hk_last_error() != '\0');
    return 0;
}
