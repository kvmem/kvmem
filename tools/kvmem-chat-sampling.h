#pragma once

#include "common.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

// Match llama-server: omitted/null/-1 inherits the process budget.
inline bool kvmem_chat_reasoning_budget_override(
        const nlohmann::json & body, int & budget, std::string & err) {
    if (!body.is_object()) {
        err = "request must be a JSON object";
        return false;
    }
    const nlohmann::json * selected = nullptr;
    for (const char * key : {"reasoning_budget_tokens", "thinking_budget_tokens"}) {
        const auto it = body.find(key);
        if (it == body.end() || it->is_null()) continue;
        if (!it->is_number_integer()) {
            err = std::string(key) + " must be an integer (-1 to inherit, 0 or greater to limit thinking)";
            return false;
        }
        // Check before narrowing: unsigned JSON integers can exceed INT64_MAX.
        const double value = it->get<double>();
        if (value < -1 || value > std::numeric_limits<int32_t>::max()) {
            err = std::string(key) + " must be between -1 and 2147483647";
            return false;
        }
        if (!selected) selected = &*it;
    }
    if (selected && selected->get<int>() != -1) budget = selected->get<int>();
    return true;
}

inline bool kvmem_chat_reasoning_budget_supported(
        const common_params_sampling & sp, bool thinking, std::string & err) {
    if (!thinking) {
        // 思考被禁用，budget_tokens 没有意义——直接忽略，不报错。
        // 这样 harness 在 disabled 模式下残留的 budget 字段不会 400。
        return true;
    }
    if (sp.reasoning_budget_tokens >= 0 &&
            (sp.reasoning_budget_start.empty() || sp.reasoning_budget_end.empty() ||
             sp.reasoning_budget_forced.empty())) {
        // 思考启用但模板没 budget 标记——降级：用 CLI --reasoning-budget
        // 的全局默认值，不再让整条请求 400。
        err.clear();
        return true;
    }
    return true;
}

// Reserve room for the final answer/tool payload when a thinking budget would
// otherwise consume the entire max_tokens allowance.
inline int kvmem_effective_reasoning_budget(
        bool thinking, int requested_budget, int max_tokens, int answer_reserve) {
    if (!thinking || requested_budget <= 0) return requested_budget;
    if (answer_reserve < 0) answer_reserve = 0;
    if (max_tokens <= 0) return requested_budget;
    const int cap = std::max(0, max_tokens - answer_reserve);
    return std::min(requested_budget, cap);
}

// Qwen3.8-27B model card, Best Practices (2026-09-13).
inline common_params_sampling kvmem_chat_sampling_defaults(bool thinking) {
    common_params_sampling sp;
    sp.temp = thinking ? 1.0f : 0.7f;
    sp.top_p = thinking ? 0.95f : 0.8f;
    sp.top_k = 20;
    sp.min_p = 0.0f;
    sp.penalty_present = thinking ? 0.0f : 1.5f;
    sp.penalty_repeat = 1.0f;
    sp.penalty_freq = 0.0f;
    return sp;
}

// Shared validation for HTTP requests and process defaults. Null means inherit.
inline bool kvmem_chat_sampling_override(
        const nlohmann::json & body, common_params_sampling & sp, std::string & err) {
    if (!body.is_object()) {
        err = "request must be a JSON object";
        return false;
    }
    auto number = [&](const char * key, double lo, double hi, bool integer, double & value) {
        const auto it = body.find(key);
        if (it == body.end() || it->is_null()) {
            return true;
        }
        if (!it->is_number() || (integer && !it->is_number_integer())) {
            err = std::string(key) + (integer ? " must be an integer" : " must be a number");
            return false;
        }
        value = it->get<double>();
        if (!std::isfinite(value) || value < lo || value > hi) {
            err = std::string(key) + " out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]";
            return false;
        }
        return true;
    };
    auto real = [&](const char * key, float & target, double lo, double hi) {
        double value = target;
        if (!number(key, lo, hi, false, value)) {
            return false;
        }
        target = static_cast<float>(value);
        return true;
    };
    if (!real("temperature", sp.temp, 0, 2) ||
        !real("top_p", sp.top_p, 0, 1) ||
        !real("min_p", sp.min_p, 0, 1) ||
        !real("presence_penalty", sp.penalty_present, -2, 2) ||
        !real("frequency_penalty", sp.penalty_freq, -2, 2)) {
        return false;
    }
    const auto has = [&](const char * key) { return body.contains(key) && !body[key].is_null(); };
    if (has("repeat_penalty") && has("repetition_penalty") && body["repeat_penalty"] != body["repetition_penalty"]) {
        err = "repeat_penalty and repetition_penalty must agree when both are provided";
        return false;
    }
    if (!real("repeat_penalty", sp.penalty_repeat, std::numeric_limits<float>::min(), std::numeric_limits<float>::max()) ||
        !real("repetition_penalty", sp.penalty_repeat, std::numeric_limits<float>::min(), std::numeric_limits<float>::max())) {
        return false;
    }
    double top_k = sp.top_k;
    double seed = sp.seed;
    if (!number("top_k", 0, std::numeric_limits<int32_t>::max(), true, top_k) ||
        !number("seed", 0, std::numeric_limits<uint32_t>::max(), true, seed)) {
        return false;
    }
    sp.top_k = static_cast<int32_t>(top_k);
    sp.seed = static_cast<uint32_t>(seed);
    return true;
}

inline void kvmem_chat_sampling_normalize(common_params_sampling & sp) {
    if (sp.temp == 0.0f) {
        sp.top_k = 0;
        sp.top_p = 1.0f;
        sp.min_p = 0.0f;
        // Explicit penalties still apply before greedy selection.
        sp.samplers = { COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TEMPERATURE };
    }
}

inline std::string kvmem_chat_sampling_cli_key(const std::string & flag) {
    if (flag == "--temp" || flag == "--temperature") return "temperature";
    if (flag == "--top-p") return "top_p";
    if (flag == "--top-k") return "top_k";
    if (flag == "--min-p") return "min_p";
    if (flag == "--presence-penalty") return "presence_penalty";
    if (flag == "--frequency-penalty") return "frequency_penalty";
    if (flag == "--repeat-penalty" || flag == "--repetition-penalty") return "repetition_penalty";
    if (flag == "--seed") return "seed";
    return {};
}
