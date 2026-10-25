#pragma once

#include "common.h"
#include "chat.h"
#include "nlohmann/json.hpp"

#include <string>
#include <vector>

// OpenAI named-function tool_choice: {"type":"function","function":{"name":"x"}}.
// Narrow the advertised tools to the named function and force a tool call.
inline bool kvmem_apply_named_tool_choice(
        const nlohmann::json & tool_choice,
        std::vector<common_chat_tool> & tools,
        common_chat_tool_choice & choice,
        std::string & err) {
    if (!tool_choice.is_object()) {
        return true;
    }
    if (!tool_choice.contains("function") || !tool_choice["function"].is_object()) {
        err = "tool_choice.function.name is required";
        return false;
    }
    const auto & fn = tool_choice["function"];
    if (!fn.contains("name") || !fn["name"].is_string() || fn["name"].get<std::string>().empty()) {
        err = "tool_choice.function.name is required";
        return false;
    }
    const std::string name = fn["name"].get<std::string>();
    if (tools.empty()) {
        err = "tool_choice names \"" + name + "\" but no tools were provided";
        return false;
    }
    for (const auto & tool : tools) {
        if (tool.name == name) {
            tools = { tool };
            choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            return true;
        }
    }
    err = "tool_choice function \"" + name + "\" not found in tools";
    return false;
}
