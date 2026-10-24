#include "kvmem-chat-sampling.h"
#include "reasoning-budget.h"

#include <climits>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

static llama_token forced_token(llama_sampler * sampler) {
    llama_token_data data[] = {{10, 1, 0}, {11, 1, 0}, {12, 1, 0}, {13, 1, 0}, {20, 1, 0}};
    llama_token_data_array cur{data, 5, -1, false};
    llama_sampler_apply(sampler, &cur);
    llama_token result = -1;
    int n = 0;
    for (const auto & x : data) if (std::isfinite(x.logit)) { result = x.id; ++n; }
    return n == 1 ? result : -1;
}

static void test_prefill_and_rollback() {
    for (int budget : {0, 1, 128}) {
        auto * sampler = common_reasoning_budget_init(nullptr, {{10}}, {{11, 12}}, {13, 11, 12}, budget);
        common_reasoning_budget_accept_prefill(sampler, 10);
        // Template whitespace and assistant continuation may exhaust the budget.
        // They must not consume any of the forced output, even if its message matches.
        for (int i = 0; i < budget + 3; ++i) common_reasoning_budget_accept_prefill(sampler, 13);
        CHECK(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_FORCING);
        CHECK(forced_token(sampler) == 13);
        auto * checkpoint = llama_sampler_clone(sampler);
        // Only generated, accepted tokens advance the forced sequence.
        llama_sampler_accept(sampler, 13);
        CHECK(forced_token(sampler) == 11);
        auto * rejected = llama_sampler_clone(sampler);
        llama_sampler_accept(rejected, 11);
        llama_sampler_accept(rejected, 12);
        CHECK(common_reasoning_budget_get_state(rejected) == REASONING_BUDGET_DONE);
        CHECK(forced_token(sampler) == 11);
        CHECK(forced_token(checkpoint) == 13);
        llama_sampler_free(rejected);
        llama_sampler_free(sampler);
        sampler = checkpoint;
        for (llama_token t : {13, 11, 12}) {
            CHECK(forced_token(sampler) == t);
            llama_sampler_accept(sampler, t);
        }
        CHECK(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);
        CHECK(forced_token(sampler) == -1);
        llama_sampler_free(sampler);

        // An already closed thinking block in the prompt needs no forced output.
        sampler = common_reasoning_budget_init(nullptr, {{10}}, {{11, 12}}, {11, 12}, budget);
        for (llama_token t : {10, 20, 11, 12, 20}) common_reasoning_budget_accept_prefill(sampler, t);
        CHECK(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);
        CHECK(forced_token(sampler) == -1);
        llama_sampler_free(sampler);
    }
    auto * sampler = common_reasoning_budget_init(nullptr, {{10}}, {{11}}, {11}, 128);
    common_reasoning_budget_accept_prefill(sampler, 10);
    common_reasoning_budget_accept_prefill(sampler, 20); // existing template prefix consumes one
    for (int i = 0; i < 126; ++i) {
        CHECK(forced_token(sampler) == -1);
        llama_sampler_accept(sampler, 20);
    }
    CHECK(forced_token(sampler) == -1);
    llama_sampler_accept(sampler, 20);
    CHECK(forced_token(sampler) == 11);
    llama_sampler_free(sampler);
}

static void test_request_validation() {
    using json = nlohmann::json;
    std::string err;
    for (const char * key : {"reasoning_budget_tokens", "thinking_budget_tokens"}) {
        for (int value : {0, 1, 128, INT_MAX}) {
            int budget = 4096;
            CHECK(kvmem_chat_reasoning_budget_override({{key, value}}, budget, err));
            CHECK(budget == value);
        }
        for (const json value : {json(-1), json(nullptr)}) {
            int budget = 4096;
            CHECK(kvmem_chat_reasoning_budget_override({{key, value}}, budget, err));
            CHECK(budget == 4096);
        }
        for (const json value : {json("128"), json(1.5), json(true), json(-2),
                                json(uint64_t(INT_MAX) + 1), json(UINT64_MAX)}) {
            int budget = 4096;
            CHECK(!kvmem_chat_reasoning_budget_override({{key, value}}, budget, err));
            CHECK(budget == 4096 && !err.empty());
        }
    }
    int budget = 4096;
    CHECK(kvmem_chat_reasoning_budget_override(json::object(), budget, err) && budget == 4096);
    CHECK(kvmem_chat_reasoning_budget_override({{"reasoning_budget_tokens", 1}, {"thinking_budget_tokens", 128}}, budget, err));
    CHECK(budget == 1); // canonical field takes precedence
    common_params_sampling sp;
    sp.reasoning_budget_tokens = 0;
    // 禁用思考时无论模板标记如何都忽略 budget（Octop disabled 模式残留字段容错）
    CHECK(kvmem_chat_reasoning_budget_supported(sp, false, err));
    // 思考启用但模板没 budget 标记——降级用 CLI 默认值，不再 400
    CHECK(kvmem_chat_reasoning_budget_supported(sp, true, err));
    sp.reasoning_budget_start = {10}; sp.reasoning_budget_end = {{11}}; sp.reasoning_budget_forced = {11};
    CHECK(kvmem_chat_reasoning_budget_supported(sp, true, err));
}

static void test_effective_reasoning_budget() {
    // A thinking budget must leave room for the final answer.
    CHECK(kvmem_effective_reasoning_budget(true, 4096, 4096, 1024) == 3072);
    CHECK(kvmem_effective_reasoning_budget(true, 4096, 4096, 2048) == 2048);
    // -1 inherits/unlimits and must not be clamped.
    CHECK(kvmem_effective_reasoning_budget(true, -1, 4096, 1024) == -1);
    // Without thinking the budget is irrelevant.
    CHECK(kvmem_effective_reasoning_budget(false, 4096, 4096, 1024) == 4096);
    CHECK(kvmem_effective_reasoning_budget(false, -1, 512, 1024) == -1);
    // max_tokens below the reserve leaves no thinking room at all.
    CHECK(kvmem_effective_reasoning_budget(true, 4096, 512, 1024) == 0);
    // Unlimited max_tokens keeps the requested budget.
    CHECK(kvmem_effective_reasoning_budget(true, 1024, 8192, 1024) == 1024);
    CHECK(kvmem_effective_reasoning_budget(true, 4096, 0, 1024) == 4096);
    CHECK(kvmem_effective_reasoning_budget(true, 4096, -1, 1024) == 4096);
    // Defensive: negative reserve behaves as zero.
    CHECK(kvmem_effective_reasoning_budget(true, 4096, 4096, -5) == 4096);
}

int main() {
    test_request_validation();
    test_effective_reasoning_budget();
    test_prefill_and_rollback();
    std::puts("reasoning budget request, clamp, prefill and rollback tests PASS");
}
