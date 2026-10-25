#pragma once

#include "nlohmann/json.hpp"

#include <map>
#include <string>

// Keep llama.cpp's JSON-serialized kwargs representation. Templates render and
// validate model-specific effort levels; the server does not invent prompts.
inline bool kvmem_chat_template_override(const nlohmann::json & body, bool & thinking,
        std::map<std::string, std::string> & kwargs, std::string & err) {
    using json = nlohmann::json;
    bool enabled = thinking;
    auto merged = kwargs;
    const auto read_enable = [&](const json & obj) {
        const auto it = obj.find("enable_thinking");
        if (it == obj.end()) return true;
        if (!it->is_boolean()) {
            err = "enable_thinking must be a boolean";
            return false;
        }
        enabled = it->get<bool>();
        return true;
    };
    if (!body.is_object()) { err = "request must be a JSON object"; return false; }
    if (body.contains("kvmem") && body["kvmem"].is_object() && !read_enable(body["kvmem"])) return false;
    if (!read_enable(body)) return false;
    const json * effort = nullptr;
    const auto kw = body.find("chat_template_kwargs");
    if (kw != body.end() && !kw->is_null()) {
        if (!kw->is_object()) { err = "chat_template_kwargs must be a JSON object"; return false; }
        if (!read_enable(*kw)) return false;
        for (const auto & item : kw->items()) merged[item.key()] = item.value().dump();
        if (kw->contains("reasoning_effort")) effort = &(*kw)["reasoning_effort"];
    }
    if (body.contains("reasoning_effort") && !body["reasoning_effort"].is_null()) effort = &body["reasoning_effort"];
    if (effort) {
        if (!effort->is_string() || effort->get_ref<const std::string &>().empty()) {
            err = "reasoning_effort must be a nonempty string";
            return false;
        }
        const auto & level = effort->get_ref<const std::string &>();
        if (level == "none") {
            enabled = false;
            merged.erase("reasoning_effort");
        } else if (level == "default") {
            merged.erase("reasoning_effort");
        } else {
            // 跨 provider effort 名兼容：有些 harness 发 high/max，
            // 但 llama.cpp 本地模板只认 xhigh/medium/low。
            std::string normalized = level;
            if (normalized == "high" || normalized == "max") normalized = "xhigh";
            merged["reasoning_effort"] = json(normalized).dump();
        }
    }
    // Avoid a stale default kwarg overriding a request's explicit switch during rendering.
    merged["enable_thinking"] = json(enabled).dump();
    thinking = enabled;
    kwargs = std::move(merged);
    return true;
}
