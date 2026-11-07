#pragma once

#include "layerwise_window.hpp"

namespace kvmem {

// Fenced import of an exact accepted checkpoint can start between ordinary
// Pressure boundaries. These are the three independent 72a81cc SessionLedger
// import tests, before any native append or private Query probe allocation.
struct LayerwiseImportStart {
    bool terminal_reserve_overflow = false;
    bool prefill_probe_overflow = false;
    bool query_replay_overflow = false;
    uint64_t budget_tokens = 0; // startup K, reduced only by terminal output reserve
    uint64_t restore_tokens = 0; // exact native extent, including accepted partial tail
    uint64_t selection_budget = 0; // ceil(restore_tokens / block_tokens) * block_tokens
    bool pressure() const noexcept {
        return terminal_reserve_overflow || prefill_probe_overflow || query_replay_overflow;
    }
};

// The caller proves the checkpoint token/state identity and actual resident
// maps, then executes resident-only Pressure with selection_budget and checks
// restore_tokens after its native fence. This pure function does not restore,
// cold-recall, reset OutputHistory, or mutate the supplied WindowPlan.
// Imports contain complete pages plus the one accepted partial tail, so L and
// H must have the same block remainder. The configured startup K is aligned.
LayerwiseImportStart plan_layerwise_import(const LayerwiseWindowPlan &,
                                          LayerwiseWindowState);

} // namespace kvmem
