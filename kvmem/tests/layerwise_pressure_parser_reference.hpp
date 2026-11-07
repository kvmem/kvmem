#pragma once
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace qw3::detail {
// Exposure-normalized EMA: repeated identical attention does not raise a key's
// score solely because it was visible in more prefill chunks.
struct PrefillAttentionScore {
    double mass = 0, exposure = 0;
    void observe(double value, double decay) {
        mass = decay * mass + value;
        exposure = decay * exposure + 1;
    }
    double mean() const { return exposure > 0 ? mass / exposure : 0; }
};
// Transactional parser: speculative rows do not mutate committed state.
// Only consumed generated tokens in the configured scope are sampled.
struct ActionAttentionParser {
    enum class Scope { ToolCall, NonThinking };
    Scope scope = Scope::ToolCall;
    static Scope parse_scope(std::string_view value) {
        if (value == "tool-call") return Scope::ToolCall;
        if (value == "non-thinking") return Scope::NonThinking;
        throw std::invalid_argument("--kvmem-action-scope must be tool-call|non-thinking");
    }
    const char *scope_name() const {
        return scope == Scope::NonThinking ? "non-thinking" : "tool-call";
    }
    struct State { bool thinking = false, action = false; uint64_t tokens = 0; };
    int32_t open = -1, close = -1, think_open = -1, think_close = -1;
    uint32_t interval = 16;
    State committed;
    std::vector<State> candidate;
    uint64_t mask = 0;
    uint32_t accepted = 0;

    void prepare(const uint32_t *tokens, uint32_t count) {
        State s = committed;
        candidate.clear(); mask = 0; accepted = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const auto t = static_cast<int32_t>(tokens[i]);
            if (t == think_open) { s.thinking = true; s.action = false; }
            else if (t == think_close) s.thinking = false;
            else if (t == close) s.action = false;
            else if (t == open && !s.thinking) s.action = true;
            else if (!s.thinking && (scope == Scope::NonThinking || s.action) &&
                     s.tokens++ % interval == 0) {
                if (i >= 64) throw std::runtime_error("action capture supports verifier batches up to 64 rows");
                mask |= uint64_t(1) << i;
            }
            candidate.push_back(s);
        }
    }
    // Returns only newly accepted sampled rows. Repeated commit is idempotent.
    uint64_t commit(uint32_t count) {
        if (count > candidate.size() || count < accepted)
            throw std::logic_error("invalid action attention acceptance prefix");
        auto prefix = [](uint32_t n) { return n >= 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1; };
        const uint64_t newly = mask & prefix(count) & ~prefix(accepted);
        if (count) committed = candidate[count-1];
        accepted = count;
        return newly;
    }
};
}
