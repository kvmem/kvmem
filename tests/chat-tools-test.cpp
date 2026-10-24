#include "kvmem-chat-tools.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

static std::vector<common_chat_tool> two_tools() {
    return {
        { "read_file",  "read a file",  "{}" },
        { "write_file", "write a file", "{}" },
    };
}

static bool err_contains(const std::string & err, const std::string & needle) {
    return err.find(needle) != std::string::npos;
}

static void test_named_function_narrows_tools() {
    const nlohmann::json tc = {
        { "type", "function" },
        { "function", { { "name", "read_file" } } },
    };
    auto tools = two_tools();
    auto choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    std::string err;
    CHECK(kvmem_apply_named_tool_choice(tc, tools, choice, err));
    CHECK(err.empty());
    CHECK(tools.size() == 1);
    CHECK(tools[0].name == "read_file");
    CHECK(tools[0].description == "read a file");
    CHECK(choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED);
}

static void test_unknown_function_rejected() {
    const nlohmann::json tc = {
        { "type", "function" },
        { "function", { { "name", "delete_file" } } },
    };
    auto tools = two_tools();
    auto choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    std::string err;
    CHECK(!kvmem_apply_named_tool_choice(tc, tools, choice, err));
    CHECK(err_contains(err, "delete_file"));
    CHECK(tools.size() == 2);
    CHECK(choice == COMMON_CHAT_TOOL_CHOICE_AUTO);
}

static void test_empty_tools_rejected() {
    const nlohmann::json tc = {
        { "type", "function" },
        { "function", { { "name", "read_file" } } },
    };
    std::vector<common_chat_tool> tools;
    auto choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    std::string err;
    CHECK(!kvmem_apply_named_tool_choice(tc, tools, choice, err));
    CHECK(err_contains(err, "read_file"));
    CHECK(tools.empty());
}

static void test_missing_function_name_rejected() {
    const nlohmann::json no_function = { { "type", "function" } };
    auto tools = two_tools();
    auto choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    std::string err;
    CHECK(!kvmem_apply_named_tool_choice(no_function, tools, choice, err));
    CHECK(err == "tool_choice.function.name is required");

    const nlohmann::json empty_name = {
        { "type", "function" },
        { "function", { { "name", "" } } },
    };
    err.clear();
    CHECK(!kvmem_apply_named_tool_choice(empty_name, tools, choice, err));
    CHECK(err == "tool_choice.function.name is required");
    CHECK(tools.size() == 2);
}

static void test_string_choices_untouched() {
    for (const char * mode : { "auto", "required", "none" }) {
        const nlohmann::json tc = mode;
        auto tools = two_tools();
        auto choice = std::string(mode) == "none" ? COMMON_CHAT_TOOL_CHOICE_NONE
                                                  : COMMON_CHAT_TOOL_CHOICE_AUTO;
        const auto before = choice;
        std::string err;
        CHECK(kvmem_apply_named_tool_choice(tc, tools, choice, err));
        CHECK(err.empty());
        CHECK(tools.size() == 2);
        CHECK(tools[0].name == "read_file");
        CHECK(tools[1].name == "write_file");
        CHECK(choice == before);
    }
}

static void test_non_object_untouched() {
    const nlohmann::json tc = nlohmann::json::array({ "not", "an", "object" });
    auto tools = two_tools();
    auto choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    std::string err;
    CHECK(kvmem_apply_named_tool_choice(tc, tools, choice, err));
    CHECK(err.empty());
    CHECK(tools.size() == 2);
    CHECK(choice == COMMON_CHAT_TOOL_CHOICE_NONE);
}

int main() {
    test_named_function_narrows_tools();
    test_unknown_function_rejected();
    test_empty_tools_rejected();
    test_missing_function_name_rejected();
    test_string_choices_untouched();
    test_non_object_untouched();
    std::puts("chat tools PASS");
}
