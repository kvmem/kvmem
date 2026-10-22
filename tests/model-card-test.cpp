#include "kvmem-model-card.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

static void test_text_model() {
    using nlohmann::json;
    const auto card = kvmem_model_card_json("text-model", 8192, 2048, 4096, false, true, false, 1024, "");
    CHECK(card["id"] == "text-model");
    CHECK(card["name"] == "text-model");
    CHECK(card["object"] == "model");
    CHECK(card["status"]["value"] == "loaded");
    CHECK(card["context_length"] == 8192);
    CHECK(card["max_output_tokens"] == 2048);
    CHECK(card["api_mode"] == "chat_completions");
    CHECK(card["input"] == json::array({"text"}));
    CHECK(card["kvmem"]["generation_limit"] == 4096);
    CHECK(card["kvmem"]["capabilities"]["tools"] == true);
    CHECK(card["kvmem"]["capabilities"]["thinking"] == true);
    CHECK(card["kvmem"]["capabilities"]["vision"] == false);
    CHECK(card["kvmem"]["reasoning"]["supported"] == true);
    CHECK(card["kvmem"]["reasoning"]["toggle"] == true);
    CHECK(card["kvmem"]["reasoning"]["default_mode"] == "disabled");
    CHECK(card["kvmem"]["reasoning"]["efforts"] == json::array({"low", "medium", "xhigh"}));
    CHECK(card["kvmem"]["reasoning"]["default_effort"].is_null());
    CHECK(card["kvmem"]["reasoning"]["effort_type"] == "enum");
    CHECK(card["kvmem"]["reasoning"]["adapter"] == "llamacpp");
    CHECK(card["kvmem"]["reasoning"]["budget_tokens"] == 1024);
}

static void test_vision_model() {
    using nlohmann::json;
    const auto card = kvmem_model_card_json("vision-model", 32768, -1, 32768, true, false, true, -1, "medium");
    CHECK(card["input"] == json::array({"text", "image"}));
    CHECK(card["kvmem"]["capabilities"]["vision"] == true);
    CHECK(card["kvmem"]["capabilities"]["tools"] == false);
    CHECK(card["kvmem"]["reasoning"]["default_mode"] == "enabled");
    CHECK(card["kvmem"]["reasoning"]["default_effort"] == "medium");
    CHECK(card["kvmem"]["reasoning"]["budget_tokens"] == -1);
    CHECK(card["max_output_tokens"] == 32768);
    CHECK(card["context_length"] == 32768);
    CHECK(card["kvmem"]["generation_limit"] == 32768);
}

int main() {
    test_text_model();
    test_vision_model();
    std::puts("model card PASS");
}
