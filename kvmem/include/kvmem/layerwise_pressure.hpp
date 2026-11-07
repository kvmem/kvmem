#pragma once

#include "layerwise_memory.hpp"
#include "layerwise_output_scope.hpp"
#include <array>
#include <utility>
#include <vector>

namespace kvmem {

struct LayerwisePrefillOptions {
    uint32_t block_tokens = 0;
    uint64_t select_tokens = 0, reserve_tokens = 0;
    uint64_t interval = 1024;
    uint32_t stride = 32; // zero samples the last row only, as upstream
    double half_life = 8192;
};
struct LayerwisePrefillValue {
    uint64_t original_start = 0;
    double mass = 0, exposure = 0;
};
struct LayerwisePrefillState {
    uint32_t model_layer = 0;
    std::vector<LayerwisePrefillValue> values; // ordered original logical block starts
    uint64_t end = 0, samples = 0, chunks = 0;
};
struct LayerwisePrefillSnapshot {
    SessionIdentity session;
    std::vector<LayerwisePrefillState> layers;
};
struct LayerwisePrefillMass {
    uint32_t model_layer = 0;
    std::vector<uint64_t> starts; // physical segment starts in this layer's dense window
    std::vector<uint64_t> history; // original logical block start; duplicates are meaningful
    std::vector<double> masses; // sampled rows x history.size(), native values unchanged
};
struct LayerwisePrefillAcceptance {
    MemoryStamp history; // successful, complete native acceptance; frontier is end
    uint64_t count = 0, accepted = 0, local_begin = 0;
    std::vector<uint32_t> rows; // offsets in this accepted ordinary prefill chunk
    std::vector<LayerwisePrefillMass> layers; // exactly the configured public target layers
};
enum class LayerwisePressureSource { Prefill, Output };
struct LayerwisePressureChoice {
    uint32_t model_layer = 0;
    std::vector<uint64_t> candidates, fixed, selected; // native canonical IDs, chronological
    std::vector<double> scores; // candidate-aligned mass / exposure
    uint64_t accepted_score_end = 0, samples = 0, chunks = 0;
    uint64_t selected_tokens = 0, charged_tokens = 0;
    LayerwisePressureSource source = LayerwisePressureSource::Prefill;
    uint64_t output_samples = 0, output_rounds = 0;
};

// Host reference of 72a81cc PrefillHistory and LayerwisePrefillPressure.
// Input is accepted native per-segment probability mass, not Mean-K scores.
// No graph/guidance, historical Q, draft layer or decode rule enters this API.
// The owner waits for native score readback and whole-model acceptance before
// accept(); this class neither executes nor certifies that native work.
class LayerwisePrefillPressure {
public:
    LayerwisePrefillPressure(SessionIdentity, std::vector<uint32_t> model_layers,
                            LayerwisePrefillOptions);
    std::vector<uint32_t> rows(uint64_t begin, uint32_t count, uint64_t local_begin,
                               bool replay = false) const;
    // Partial acceptance, old end and unsampled chunks return false unchanged.
    // Every target layer validates into private state before one atomic publish.
    bool accept(const LayerwisePrefillAcceptance&);
    const LayerwisePrefillSnapshot& snapshot() const noexcept { return state_; }
    const LayerwisePrefillOptions& options() const noexcept { return options_; }
    // Matches upstream rewind: a sampled history beyond position is cleared.
    void restore(LayerwisePrefillSnapshot, uint64_t position);
    // Candidate identities come exclusively from each layer's accepted resident
    // map. Sink/mandatory IDs must also be resident: no cold recall is permitted.
    // Positive scores rank first, ties/remaining capacity use resident recency.
    std::vector<LayerwisePressureChoice> select(const LayerwiseMemoryCatalog&,
        const std::vector<uint64_t>& mandatory, uint64_t budget_tokens,
        uint32_t sink_blocks = 0) const;
private:
    SessionIdentity session_;
    std::vector<uint32_t> layers_;
    LayerwisePrefillOptions options_;
    LayerwisePrefillSnapshot state_;
};

enum class LayerwiseOutputScope { ToolCall, NonThinking };
struct LayerwiseOutputOptions {
    LayerwiseOutputScope scope = LayerwiseOutputScope::ToolCall;
    uint32_t interval = 16;
    double decay = .5;
    uint64_t commit_max_bytes = 16777216;
};
struct LayerwiseOutputParserState {
    bool thinking = false, action = false;
    uint64_t tokens = 0;
};
struct LayerwiseOutputState {
    uint32_t model_layer = 0;
    std::vector<std::pair<uint64_t,double>> scores, pending; // original block starts
    uint64_t samples = 0, rounds = 0;
};
struct LayerwiseOutputSnapshot {
    SessionIdentity session;
    std::vector<LayerwiseOutputState> layers;
    LayerwiseOutputParserState parser;
    uint64_t request = 0, accepted_end = 0;
    bool active = false;
};
struct LayerwiseOutputMass {
    uint32_t model_layer = 0;
    std::vector<uint64_t> history; // native physical segment -> original logical start
    std::vector<double> masses; // captured rows x history.size(), no renormalization
};
struct LayerwiseOutputAcceptance {
    MemoryStamp history; // native consumed-row frontier, never predicted-token (+1)
    uint64_t request = 0;
    uint32_t accepted = 0; // scheduler-confirmed prefix of the prepared verifier batch
    std::vector<uint32_t> rows; // captured native row offsets; rejected rows are ignored
    std::vector<LayerwiseOutputMass> layers;
};

// New independent-layer application of upstream's ORDINARY ActionHistory rule.
// Upstream's layerwise experiment itself qualifies prefill only. This reference
// does not advertise native decode execution. One accepted-token scope parser
// is shared by all target layers; layer scores/mass and publication are separate.
class LayerwiseOutputPressure {
public:
    // Exact model/tokenizer-provided IDs: tool-open, tool-close, think-open,
    // think-close, message-start. Missing/aliased markers are unsupported;
    // callers must not guess replacements for another model's template.
    LayerwiseOutputPressure(SessionIdentity, std::vector<uint32_t> model_layers,
        LayerwisePrefillOptions, std::array<int32_t,5> markers, LayerwiseOutputOptions = {});
    // Explicit model-native semantic scope. No Qwen marker IDs are installed,
    // and the token-marker begin()/prepare() entry points reject this owner.
    LayerwiseOutputPressure(SessionIdentity, std::vector<uint32_t> model_layers,
        LayerwisePrefillOptions, LayerwiseExternalOutputScope, LayerwiseOutputOptions = {});
    void begin(const MemoryStamp& accepted_history, uint64_t request,
               const std::vector<uint32_t>& prompt);
    void begin_scope(const MemoryStamp& accepted_history, uint64_t request, LayerwiseOutputScopeState);
    // Prompt rows advance native acceptance without becoming Output samples.
    void accept_prefill(const MemoryStamp& accepted_history, uint64_t request, uint64_t begin);
    uint64_t prepare(uint64_t request, uint64_t begin, const std::vector<uint32_t>& consumed_tokens);
    uint64_t prepare_scope(uint64_t request, uint64_t begin, const std::vector<LayerwiseOutputScopeRow>&);
    // Atomically validates all target layers before committing parser or mass.
    // Returns only newly accepted sampled row bits. Repeated prefix is idempotent.
    uint64_t accept(const LayerwiseOutputAcceptance&);
    // Literal ordinary reference: finish clears pending mass, retains sample N
    // until begin(), and publishes S'=decay*S+mass/N. A closed-window preview
    // still applies decay*S while N remains nonzero, exactly as pressure_seed.
    void finish(uint64_t request, const LayerwiseMemoryCatalog& accepted_history);
    const LayerwiseOutputSnapshot& snapshot() const noexcept { return state_; }
    // Accepted pending mass/parser are retained; speculative candidates are not.
    // Restore only into the same configured owner, after native rollback/drain.
    void restore(LayerwiseOutputSnapshot, uint64_t position);
    std::vector<LayerwisePressureChoice> select(const LayerwiseMemoryCatalog&,
        const LayerwisePrefillPressure&, const std::vector<uint64_t>& mandatory,
        uint64_t budget_tokens, uint32_t sink_blocks = 0) const;
private:
    void owner(uint64_t) const;
    void begin_state(const MemoryStamp&, uint64_t request, LayerwiseOutputScopeState);
    SessionIdentity session_;
    std::vector<uint32_t> layers_;
    LayerwisePrefillOptions prefill_options_;
    std::array<int32_t,5> markers_;
    LayerwiseOutputOptions options_;
    LayerwiseOutputSnapshot state_;
    std::vector<LayerwiseOutputParserState> candidate_;
    uint64_t mask_ = 0, begin_ = 0;
    uint32_t accepted_ = 0;
    bool prepared_ = false;
};

} // namespace kvmem
