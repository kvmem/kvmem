#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kvmem {

// Explicit native grammar binding; never a replacement set of Qwen token IDs.
struct LayerwiseExternalOutputScope {};
struct LayerwiseOutputScopeState {
    bool thinking = false, tool_call = false;
};
struct LayerwiseOutputScopeRow {
    bool thinking = false, tool_call = false, body = false;
};
struct LayerwiseOutputGrammarConfig {
    int32_t channel_open = -1, channel_close = -1;
    int32_t turn_open = -1, turn_close = -1;
    int32_t tool_open = -1, tool_close = -1;
    std::string thinking_channel = "thought";
    uint32_t max_token_bytes = 65536;
};
enum class LayerwiseOutputHeader : uint8_t { None, Channel, Turn };
struct LayerwiseOutputGrammarState {
    LayerwiseOutputScopeState scope;
    LayerwiseOutputHeader header = LayerwiseOutputHeader::None;
    uint32_t matched = 0;
    bool mismatch = false;
};
struct LayerwiseOutputGrammarToken {
    uint32_t id = 0;
    // Exact bytes from the bound native tokenizer, valid for this call only.
    std::string_view bytes;
};

// Bounded accepted-prefix lexer for native channel/turn/tool grammars. Header
// fragments are excluded immediately, including near-miss or unfinished names.
// A row containing header bytes is a control row even if its bytes also contain
// the header's terminating newline. No header text or borrowed pointer is kept.
class LayerwiseOutputGrammar {
public:
    explicit LayerwiseOutputGrammar(LayerwiseOutputGrammarConfig);
    void reset();
    // Scan the complete trusted prompt before output begins. This does not
    // advance the OutputHistory sampling clock or its accepted KV frontier.
    void prompt(LayerwiseOutputGrammarToken);
    std::vector<LayerwiseOutputScopeRow> prepare(const std::vector<LayerwiseOutputGrammarToken>&);
    // Called only after the same prefix's native fence and OutputHistory accept.
    // Repeated acceptance is idempotent; a rejected suffix never changes state.
    void accept(uint32_t accepted);
    LayerwiseOutputGrammarState snapshot() const noexcept { return state_; }
    void restore(LayerwiseOutputGrammarState);
    const LayerwiseOutputGrammarConfig& config() const noexcept { return config_; }
    uint64_t metadata_bytes() const noexcept;
private:
    LayerwiseOutputScopeRow step(LayerwiseOutputGrammarState&, LayerwiseOutputGrammarToken) const;
    void validate(LayerwiseOutputGrammarState) const;
    LayerwiseOutputGrammarConfig config_;
    LayerwiseOutputGrammarState state_;
    std::array<LayerwiseOutputGrammarState,64> candidates_{};
    uint32_t count_ = 0, accepted_ = 0;
    bool prepared_ = false;
};

} // namespace kvmem
