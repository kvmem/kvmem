// Pinned upstream reference, unmodified body: kvmem/kvmem-vllm
// 72a81cc99aa78d85004ecf1364fc9f3094db8816, blob 993ea21defcd36bdac3f98cd738cd34da7ef579b
// third_party/qw3/src/kvmem_prefix_reuse_policy.hpp (Apache-2.0).
#pragma once

#include <cstdint>

namespace qw3::detail {

// A speculative cycle commits the current token plus up to `drafts` tokens.
// Stop at the next physical block boundary when a replay snapshot is due;
// otherwise repeated MTP jumps can skip every safe post-selection origin.
constexpr uint32_t kvmem_checkpoint_draft_limit(
        uint64_t position, uint32_t block_tokens, uint32_t drafts,
        bool checkpoint_due) {
    if (!checkpoint_due || block_tokens == 0) return drafts;
    const uint32_t gap = block_tokens - position % block_tokens;
    return drafts < gap ? drafts : gap - 1;
}

// Warm sessions need an aligned checkpoint in the current selection. A
// periodic reselect between that checkpoint and EOS invalidates P while M can
// remain unaligned. Schedule periodic updates at the first block boundary at
// or after their interval; explicit pressure recovery remains independent.
constexpr bool kvmem_periodic_reselect_boundary_due(
        uint64_t position, uint64_t last_reselect, uint32_t interval,
        uint32_t block_tokens) {
    const uint32_t bt = block_tokens ? block_tokens : 1;
    const uint32_t gap = interval ? interval : 1;
    return position >= last_reselect && position - last_reselect >= gap &&
        position % bt == 0;
}

// A speculative acceptance can otherwise jump over the scheduled boundary.
// Clamp when the next boundary is due or this cycle could cross the
// interval threshold (including configurations smaller than a draft cycle).
constexpr bool kvmem_periodic_reselect_align_drafts(
        uint64_t position, uint64_t last_reselect, uint32_t interval,
        uint32_t block_tokens, uint32_t requested_drafts) {
    if (position < last_reselect || block_tokens == 0) return false;
    const uint64_t elapsed = position - last_reselect;
    const uint32_t gap = interval ? interval : 1;
    const uint32_t boundary_distance = block_tokens - position % block_tokens;
    return elapsed >= gap || gap - elapsed <= boundary_distance ||
        gap - elapsed <= uint64_t(requested_drafts) + 1;
}

// The rendered message header can begin in the block BEFORE its scoring Q.
// A fresh Q needs a checkpoint no later than both boundaries; otherwise an
// unaligned M can skip the only reachable state for the complete live replay.
constexpr uint64_t kvmem_replay_capture_begin(uint64_t query_begin,
                                             uint64_t replay_begin,
                                             uint64_t replay_end) {
    return replay_end > replay_begin && replay_begin < query_begin
        ? replay_begin : query_begin;
}

// A live tool-result header can follow Q by only a few template tokens. Its
// aligned block may therefore start INSIDE Q. Replay the complete preceding
// block instead, so the post-selection boundary is also a legal future P.
// A boundary at zero is handled by the existing initial-query path.
constexpr uint64_t kvmem_complete_query_replay_boundary(
        uint64_t requested_begin, uint64_t query_begin,
        uint64_t query_end, uint32_t block_tokens, bool minimal_replay = false) {
    const uint32_t bt = block_tokens ? block_tokens : 1;
    const uint64_t boundary = requested_begin / bt * bt;
    const uint64_t before_query = query_begin / bt * bt;
    return !minimal_replay && before_query > 0 && boundary > before_query && boundary < query_end
        ? before_query : boundary;
}

// A cached scoring Q does not make all tokens after its old P/M checkpoint a
// live transaction. Ingest completed tool history before capturing the current
// live suffix boundary, keeping the restored checkpoint as the earliest
// reachable origin. Otherwise an old P can turn several bounded tool results
// into one mandatory suffix larger than the selection budget. The complete
// source remains indexed; only the post-selection replay origin advances.
constexpr uint64_t kvmem_cached_query_replay_boundary(
        uint64_t requested_begin, uint64_t query_begin,
        uint64_t query_end, uint32_t block_tokens,
        uint64_t resume_position, bool minimal_replay = false) {
    const uint64_t live_boundary = kvmem_complete_query_replay_boundary(
        requested_begin, query_begin, query_end, block_tokens, minimal_replay);
    return live_boundary > resume_position ? live_boundary : resume_position;
}

// Stage P before one complete branch-sensitive tail block.  Agent harnesses
// rewrite a small suffix of the prior live assistant turn when it becomes
// history (for example, the generated empty-think marker may disappear).  The
// exact rewrite is tokenizer/template dependent and has exceeded the old
// four-token allowance.  Keeping one full KVMem block after P ensures that a
// rewrite of at most block_tokens still leaves P no later than the next turn's
// block-aligned query replay boundary.  If that conservative boundary is
// already behind a warm resume point, retain the first aligned boundary that
// is still reachable from the appended suffix.  It may lie in the rewritten
// tail, but exact-LCP admission will reject it if that tail actually changes;
// unlike an unaligned prompt-end fallback, it remains a legal semantic replay
// origin when the tail is unchanged.
constexpr uint64_t kvmem_branch_safe_prompt_checkpoint_split(
        uint64_t prompt_end, uint64_t prefill_base,
        uint32_t block_tokens) {
    constexpr uint32_t kShortTailGuard = 4;
    if (prompt_end <= prefill_base) return 0;
    const uint32_t bt = block_tokens == 0 ? 1 : block_tokens;
    uint64_t split = 0;
    if (prompt_end > bt) {
        const uint64_t last_boundary = ((prompt_end - 1) / bt) * bt;
        // The last boundary can itself fall inside the rewritten branch tail.
        // Retreat exactly one complete block to make it branch-safe.
        split = last_boundary >= bt ? last_boundary - bt : 0;
    } else {
        split = prompt_end > kShortTailGuard
            ? prompt_end - kShortTailGuard : 0;
    }
    // Equality is useful on a P->P continuation: the executor is already at
    // this exact safe boundary, so the caller can carry the restored checkpoint
    // forward without re-prefilling or replacing it with an unaligned prompt-end
    // fallback.
    if (split < prefill_base) {
        const uint64_t gap = (bt - prefill_base % bt) % bt;
        return gap < prompt_end - prefill_base ? prefill_base + gap : 0;
    }
    if (split >= prompt_end) return 0;
    return split;
}

// A branch-safe checkpoint can still be unusable for query-conditioned
// continuation when it falls after the aligned replay boundary but before the
// end of Q.  Such a snapshot contains only a prefix of the retrieval query:
// it is too late to replay Q from its boundary and too early to use the
// complete-query snapshot.  Clamp that interval back to the replay boundary.
//
// `prefill_base == replay_boundary` is intentional.  A warm continuation has
// already restored that state, so the caller can capture it again without
// consuming another token and keep it as the next turn's safe P checkpoint.
constexpr uint64_t kvmem_query_safe_prompt_checkpoint_split(
        uint64_t prompt_end, uint64_t prefill_base,
        uint32_t block_tokens, uint64_t query_begin,
        uint64_t query_end) {
    const uint64_t split = kvmem_branch_safe_prompt_checkpoint_split(
        prompt_end, prefill_base, block_tokens);
    if (query_end <= query_begin) return split;
    const uint32_t bt = block_tokens == 0 ? 1 : block_tokens;
    const uint64_t replay_boundary = (query_begin / bt) * bt;
    if (split > replay_boundary && split < query_end &&
        replay_boundary >= prefill_base && replay_boundary < prompt_end) {
        if (replay_boundary == 0 && query_end < prompt_end) {
            // A short system prefix can put the first message header in block
            // zero. There is no useful pre-query P; use a complete-Q aligned
            // boundary in the final template tail if one actually exists.
            const uint64_t gap = (bt - query_end % bt) % bt;
            if (gap < prompt_end - query_end && query_end + gap >= prefill_base)
                return query_end + gap;
        }
        return replay_boundary;
    }
    return split;
}

// Prefix reuse above the semantic budget is an incremental source-index
// operation even when the current request has reselection disabled (for
// example, an agent tool-result continuation).  Such a request must therefore
// resume only from a checkpoint whose sparse state and source-index cursor are
// both durable.  Query capture has the same requirement below the budget so a
// prepared checkpoint remains safe when the following turn crosses it.
struct KvmemPrefixReuseSafety {
    bool above_selection_budget = false;
    bool source_index_required = false;
    bool source_index_layout_incompatible = false;
    bool block_end_checkpoint = false;
    bool block_prompt_checkpoint = false;
};

constexpr KvmemPrefixReuseSafety kvmem_prefix_reuse_safety(
        uint64_t logical_prompt_tokens, uint32_t selection_budget,
        bool query_capture_active, bool per_token_source_index,
        bool end_resumable, bool end_source_index_ready,
        bool prompt_resumable, bool prompt_source_index_ready) {
    KvmemPrefixReuseSafety out;
    out.above_selection_budget =
        selection_budget > 0 && logical_prompt_tokens > selection_budget;
    out.source_index_required =
        query_capture_active || out.above_selection_budget;
    // The raw per-token index is laid out using the current request length and
    // cannot be incrementally extended after an above-budget prefix restore.
    out.source_index_layout_incompatible =
        out.above_selection_budget && per_token_source_index;
    out.block_end_checkpoint = out.source_index_layout_incompatible ||
        (out.source_index_required &&
         (!end_resumable || !end_source_index_ready));
    out.block_prompt_checkpoint = out.source_index_layout_incompatible ||
        (out.source_index_required &&
         (!prompt_resumable || !prompt_source_index_ready));
    return out;
}

// A no-query above-budget continuation still incrementally extends the
// position-invariant history index.  Preserve it when a safe warm checkpoint
// was selected; a cold request intentionally rebuilds it from token zero.
constexpr bool kvmem_preserve_warm_history_index(
        bool warm_reuse, bool history_index_capture) {
    return warm_reuse && history_index_capture;
}

// A StateSnapshot is meaningful only for the exact assembled working-set
// generation that produced it.  Token LCP/source-index readiness cannot make a
// pre-selection checkpoint safe after a mid-decode semantic selection changed
// page ordering and RoPE bake positions.
constexpr bool kvmem_checkpoint_selection_compatible(
        uint64_t checkpoint_generation, uint64_t current_generation) {
    return checkpoint_generation == current_generation;
}

// A server may replay from the live tool suffix, whose aligned beginning
// falls inside an earlier scoring query. Such an origin is valid for the
// current replay but cannot replace a reusable P: it contains only part of Q.
constexpr bool kvmem_checkpoint_outside_query(
        uint64_t position, uint64_t query_begin, uint64_t query_end,
        uint32_t block_tokens) {
    const uint32_t bt = block_tokens == 0 ? 1 : block_tokens;
    return query_end <= query_begin || position <= (query_begin / bt) * bt ||
        position >= query_end;
}

// A durable query snapshot is valid only for the same public task span and
// while every prompt token used to derive it remains inside the exact LCP.  A
// normal query snapshot depends only on query_end.  A private guided rewrite
// depends on the whole prompt visible to that rewrite, so source_end is the
// prompt end of the refresh request.
constexpr bool kvmem_warm_query_snapshot_coordinates_match(
        bool stashed, uint64_t stored_begin, uint64_t stored_end,
        uint64_t stored_source_end, uint64_t requested_begin,
        uint64_t requested_end) {
    return stashed && stored_begin == requested_begin &&
        stored_end == requested_end && stored_source_end >= stored_end;
}

constexpr bool kvmem_warm_query_snapshot_reusable(
        bool stashed, uint64_t stored_begin, uint64_t stored_end,
        uint64_t stored_source_end, uint64_t requested_begin,
        uint64_t requested_end, uint64_t common_prefix) {
    return kvmem_warm_query_snapshot_coordinates_match(
               stashed, stored_begin, stored_end, stored_source_end,
               requested_begin, requested_end) &&
        stored_source_end <= common_prefix;
}

constexpr bool kvmem_warm_query_snapshot_capture_ready(
        bool guided_query, bool inherited_snapshot,
        bool scorer_ready, bool original_query_complete) {
    return (guided_query || inherited_snapshot)
        ? scorer_ready : original_query_complete;
}

// Query replay normally proves that every row of the public query span was
// captured in the current pass.  A post-query checkpoint can instead restore a
// complete private guided-query snapshot whose row count intentionally differs
// from that public span.  In that case scorer readiness is the relevant proof;
// retaining the public-span equality check would reject every short rewrite.
constexpr bool kvmem_query_replay_capture_ready(
        bool inherited_snapshot, bool scorer_ready,
        bool public_query_complete) {
    return inherited_snapshot ? scorer_ready : public_query_complete;
}

// The public query span describes which prompt tokens define the retrieval
// intent.  A private guided rewrite is a detached scorer input and can have a
// different row count.  When that snapshot is restored on a later tool turn,
// the per-layer query tensor must be wide enough for both lengths; unused rows
// are capacity only and never contribute to scoring (the scorer consumes the
// separately tracked captured-row count).
constexpr uint32_t kvmem_query_storage_rows(
        uint32_t public_query_rows, uint32_t restored_query_rows) {
    const uint32_t rows = public_query_rows > restored_query_rows
        ? public_query_rows : restored_query_rows;
    return rows == 0 ? 1 : rows;
}

// Semantic query replay restores a block/page boundary snapshot. A post-query
// M checkpoint commonly ends a few chat-template tokens past a block boundary;
// even with a durable Q snapshot it cannot be used as a replay origin. Let the
// caller fall through to the aligned P checkpoint instead.
//
// A prepare-only stable tool continuation is different: it restores the exact
// current-selection M snapshot and only appends the strict-extension suffix.
// It does not rewind/replay the query or truncate pages to an earlier replay
// boundary, so an unaligned M is both necessary and safe when a guided refresh
// ended less than one block before the turn stopped. `semantic_replay` makes
// that distinction explicit instead of treating every query-capture request as
// a replay request.
constexpr bool kvmem_post_query_replay_checkpoint_aligned(
        bool semantic_replay, uint64_t checkpoint,
        uint64_t query_boundary_ceiling, uint32_t block_tokens) {
    const uint32_t bt = block_tokens == 0 ? 1 : block_tokens;
    return !semantic_replay || checkpoint <= query_boundary_ceiling ||
        checkpoint % bt == 0;
}

constexpr uint64_t kvmem_warm_query_snapshot_source_end(
        bool guided_query, bool inherited_snapshot,
        uint64_t prompt_tokens, uint64_t inherited_source_end,
        uint64_t query_end) {
    if (guided_query) return prompt_tokens;
    if (inherited_snapshot) return inherited_source_end;
    return query_end;
}

}  // namespace qw3::detail
