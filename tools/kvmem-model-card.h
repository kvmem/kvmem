#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>

// OpenAI-compatible model card with kvmem capability metadata.
// 中文：/v1/models 的能力声明，供客户端自动配置思考、工具与上下文。
inline nlohmann::json kvmem_model_card_json(
    const std::string & model_name,
    int context_length,
    int default_max_tokens,
    int generation_limit,
    bool vision,
    bool supports_tools,
    bool enable_thinking,
    int reasoning_budget_tokens,
    const std::string & default_reasoning_effort) {
    using json = nlohmann::json;
    // 未显式设置 --n-predict 时服务端以 -1 表示“不限制”，对外应上报生成上限，
    // 否则客户端会把 max_tokens=-1 当成非法值。
    const int max_output_tokens = default_max_tokens > 0
        ? std::min(default_max_tokens, generation_limit)
        : generation_limit;
    return {
        {"id", model_name},
        {"name", model_name},
        {"object", "model"},
        {"status", {{"value", "loaded"}}},
        {"context_length", context_length},
        {"max_output_tokens", max_output_tokens},
        {"api_mode", "chat_completions"},
        {"input", vision ? json::array({"text", "image"}) : json::array({"text"})},
        {"kvmem", {
            {"generation_limit", generation_limit},
            {"capabilities", {{"tools", supports_tools}, {"thinking", true}, {"vision", vision}}},
            {"reasoning", {
                {"supported", true},
                {"toggle", true},
                {"default_mode", enable_thinking ? "enabled" : "disabled"},
                {"efforts", json::array({"low", "medium", "xhigh"})},
                {"default_effort", default_reasoning_effort.empty() ? json(nullptr) : json(default_reasoning_effort)},
                {"effort_type", "enum"},
                {"adapter", "llamacpp"},
                {"budget_tokens", reasoning_budget_tokens},
            }},
        }},
    };
}
