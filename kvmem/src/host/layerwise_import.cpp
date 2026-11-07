#include "kvmem/layerwise_import.hpp"
#include <algorithm>

namespace kvmem {

LayerwiseImportStart plan_layerwise_import(const LayerwiseWindowPlan & plan,
                                          LayerwiseWindowState state) {
    const auto & e = plan.envelope();
    const auto capacity = plan.capacity();
    const uint64_t block = e.block_tokens;
    const auto startup = capacity - e.generation_tokens;
    // action validates bounded coordinates/physical extent, without performing
    // its block-boundary dispatch. Import Pressure is an independent decision.
    (void) plan.action(state);
    if (state.history > e.prompt_end || startup % block ||
        state.visible_tokens % block != state.history % block)
        throw MemoryError(MemoryErrorCode::InvalidPlan,
                          "import requires an exact accepted native prefix and aligned startup budget");
    const auto output = e.rolling_active_window ? 0 : e.output_tokens;
    const auto remaining = e.prompt_end - state.history;
    const auto evaluation = plan.query_evaluation();
    LayerwiseImportStart result;
    result.budget_tokens = startup;
    result.restore_tokens = state.visible_tokens;
    result.terminal_reserve_overflow = !e.query && state.history > plan.generation_checkpoint() &&
        state.visible_tokens + remaining + output > capacity - block;
    result.prefill_probe_overflow = e.prefill_window == LayerwiseWindowPolicy::Pressure &&
        (!evaluation || state.history < evaluation->begin) &&
        int64_t(state.visible_tokens) > plan.prefill_boundary(state.history);
    result.query_replay_overflow = evaluation && state.history == evaluation->begin &&
        state.visible_tokens + evaluation->end - state.history > capacity - block;
    if (!result.pressure()) return result;
    if (result.terminal_reserve_overflow) {
        // Python's signed floor division must reject a nonpositive available
        // budget; unsigned subtraction here would turn it into a huge window.
        const auto available = int64_t(capacity) - int64_t(output) - int64_t(remaining) - int64_t(block);
        if (available <= 0)
            throw MemoryError(MemoryErrorCode::BudgetExceeded,
                              "continuation output reserve exceeds active capacity");
        result.budget_tokens = std::min(startup, uint64_t(available) / block * block);
    }
    if (!result.budget_tokens)
        throw MemoryError(MemoryErrorCode::BudgetExceeded,
                          "continuation output reserve exceeds active capacity");
    const auto padding = (block - state.history % block) % block;
    // A positive aligned budget always contains at least one block.
    result.restore_tokens = std::min(state.history, result.budget_tokens - padding);
    result.selection_budget = (result.restore_tokens + block - 1) / block * block;
    return result;
}

} // namespace kvmem
