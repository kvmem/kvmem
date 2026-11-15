#include "kvmem/historical_kv_semantic.hpp"
#include "kvmem/historical_kv_audit.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace kvmem {
namespace {
bool same(HistoricalKvHandle a, HistoricalKvHandle b) {
    return a.incarnation == b.incarnation && a.slot == b.slot && a.generation == b.generation;
}
bool overlap(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return a < d && c < b; }
bool same_update(const HistoricalKvStepEvidence::Update& a, const HistoricalKvStepEvidence::Update& b) {
    return a.id == b.id && a.kind == b.kind && a.source_begin == b.source_begin &&
        a.source_end == b.source_end && a.result_begin == b.result_begin && a.result_end == b.result_end &&
        a.observation_id == b.observation_id && a.importance == b.importance &&
        a.cold == b.cold && a.complete == b.complete && a.witness_masks == b.witness_masks &&
        a.duplicate_begin == b.duplicate_begin;
}
}

std::vector<uint8_t> historical_kv_literal_witnesses(const std::vector<int32_t>& source,
    const std::vector<uint8_t>& source_digits, const std::vector<int32_t>& result,
    uint32_t page_tokens) {
    if (!page_tokens || source.empty() || result.empty() || source.size() != source_digits.size() ||
        source.size() % page_tokens || result.size() % page_tokens ||
        source.size() / page_tokens > historical_kv_max_source_pages ||
        result.size() / page_tokens > historical_kv_max_witness_pages)
        throw std::invalid_argument("Invalid step literal witness range");

    constexpr size_t context = 2;
    const size_t witness_pages = result.size() / page_tokens;
    const auto all_witnesses = static_cast<uint8_t>((1u << witness_pages) - 1);
    std::vector<uint8_t> masks(source.size() / page_tokens, all_witnesses);
    size_t begin = 0;
    while (begin < source_digits.size()) {
        if (!source_digits[begin]) {
            ++begin;
            continue;
        }
        size_t end = begin + 1;
        while (end < source_digits.size() && source_digits[end]) ++end;
        uint8_t matches = 0;
        // A truncated run/context cannot establish a replacement. Inside the
        // source span, read across page seams so all affected pages are checked.
        if (begin >= context && source.size() - end >= context) {
            const auto first = source.begin() + begin - context;
            const auto last = source.begin() + end + context;
            for (size_t page = 0; page < witness_pages; ++page) {
                const auto from = result.begin() + page * page_tokens;
                const auto to = from + page_tokens;
                if (std::search(from, to, first, last) != to)
                    matches |= static_cast<uint8_t>(1u << page);
            }
        }
        for (size_t page = begin / page_tokens; page <= (end - 1) / page_tokens; ++page)
            masks[page] &= matches;
        begin = end;
    }
    return masks;
}

HistoricalKvStepEvidence::HistoricalKvStepEvidence(uint32_t pages, uint32_t tokens, Config config)
    : pages_(pages), tokens_(tokens), config_(config), bindings_(pages),
      marks_(config.feature_gate ? pages : 0), witness_refs_(marks_.size()), retired_pages_(pages) {
    if (!pages || !tokens || !config.min_observations || !config.max_steps || config.max_steps > 64 ||
        !std::isfinite(config.max_importance) || config.max_importance < 0 || config.max_importance > 1 ||
        !std::isfinite(config.similarity) || config.similarity < 0 || config.similarity > 1 ||
        !config.max_source_pages || config.max_source_pages > historical_kv_max_source_pages ||
        !config.max_result_pages || config.max_result_pages > historical_kv_max_witness_pages)
        throw std::invalid_argument("Invalid semantic GC limits");
    steps_.reserve(config.max_steps);
    finished_.reserve(config.max_steps);
}
uint32_t HistoricalKvStepEvidence::first_full_page(uint32_t begin) const noexcept {
    return static_cast<uint32_t>((uint64_t(begin) + tokens_ - 1) / tokens_);
}
bool HistoricalKvStepEvidence::source(const Step& s, uint32_t page) const {
    const uint64_t begin = uint64_t(page) * tokens_;
    return begin >= std::max(s.value.source_begin, config_.source_floor_tokens) && begin + tokens_ <= s.value.source_end;
}
bool HistoricalKvStepEvidence::result(const Step& s, uint32_t page) const {
    const uint64_t begin = uint64_t(page) * tokens_;
    return begin >= s.value.result_begin && begin + tokens_ <= s.value.result_end;
}
void HistoricalKvStepEvidence::publish(const std::vector<Update>& updates, uint32_t accepted) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (updates.size() > config_.max_steps || uint64_t(accepted) > uint64_t(pages_) * tokens_)
        throw std::invalid_argument("Semantic GC batch exceeds its bounded domain");
    auto next = steps_; auto revision = next_revision_;
    auto finished=finished_;
    auto evicted = evicted_evidence_, ignored = ignored_retired_updates_;
    std::vector<std::string> ids;
    for (const auto& u : updates) {
        const bool duplicate = u.kind == "duplicate_tool_feedback";
        if (u.id.empty() || u.id.size() > 128 ||
            (u.kind != "thinking" && u.kind != "completed_step" && u.kind != "tool_detail" &&
                u.kind != "superseded" && !duplicate) ||
            u.source_begin >= u.source_end || u.result_begin >= u.result_end ||
            u.source_end > accepted || u.result_end > accepted ||
            !std::isfinite(u.importance) || u.importance < 0 || u.importance > 1 ||
            std::find(ids.begin(), ids.end(), u.id) != ids.end())
            throw std::invalid_argument("Invalid semantic GC span or observation");
        ids.push_back(u.id);
        if (config_.feature_gate && std::any_of(finished.begin(), finished.end(),
                [&](const Status& s) { return s.id == u.id; })) { ++ignored; continue; }
        // Round inward. Never retire a mixed boundary/partial tail page.
        const auto full = [&](uint32_t begin, uint32_t end) {
            return (uint64_t(begin) + tokens_ - 1) / tokens_ < uint64_t(end) / tokens_;
        };
        if (!full(u.source_begin, u.source_end) || !full(u.result_begin, u.result_end) ||
            overlap(u.source_begin, u.source_end, u.result_begin, u.result_end))
            throw std::invalid_argument("Semantic GC requires disjoint full source and retained-result pages");
        if (config_.feature_gate && ((u.kind != "thinking" && !duplicate) ||
            (u.source_end - u.source_begin) / tokens_ > config_.max_source_pages ||
            (u.result_end - u.result_begin) / tokens_ > config_.max_result_pages))
            throw std::invalid_argument("Step redundancy scope exceeds its bounded thinking/result domain");
        constexpr auto margin = historical_kv_duplicate_context_tokens;
        if (duplicate && (!config_.feature_gate || !u.complete || !u.witness_masks.empty() ||
                u.source_begin % tokens_ || u.source_end % tokens_ ||
                u.result_begin % tokens_ || u.result_end % tokens_ || u.source_end > u.result_begin ||
                u.duplicate_begin < uint64_t(u.result_begin) + margin ||
                uint64_t(u.duplicate_begin) + (u.source_end - u.source_begin) + margin > u.result_end))
            throw std::invalid_argument("Invalid verified duplicate feedback scope");
        if (!duplicate && u.duplicate_begin)
            throw std::invalid_argument("Unexpected duplicate feedback witness");
        if (!u.witness_masks.empty()) {
            const auto source_pages = u.source_end / tokens_ - (uint64_t(u.source_begin) + tokens_ - 1) / tokens_;
            const auto result_pages = u.result_end / tokens_ - (uint64_t(u.result_begin) + tokens_ - 1) / tokens_;
            if (!config_.feature_gate || u.witness_masks.size() != source_pages ||
                std::any_of(u.witness_masks.begin(), u.witness_masks.end(),
                    [&](uint8_t mask) { return (mask >> result_pages) != 0; }))
                throw std::invalid_argument("Invalid step literal witness mask");
        }
        uint32_t remaining = 0;
        for (uint32_t id = first_full_page(std::max(u.source_begin, config_.source_floor_tokens));
             id < u.source_end / tokens_; ++id)
            remaining += !retired_pages_[id];
        // Replaying old metadata must not reopen a completed entry or retain
        // its witness forever. A source entirely inside the sink is also a no-op.
        if (!remaining) {
            ++ignored;
            if (config_.feature_gate) {
                if (finished.size()==config_.max_steps) finished.erase(finished.begin());
                finished.push_back({u.id,"no_live_unprotected_source",0,0});
            }
            continue;
        }
        auto it = std::find_if(next.begin(), next.end(), [&](const Step& s) { return s.value.id == u.id; });
        if (it != next.end()) {
            const auto& old = it->value;
            if (config_.feature_gate && !same_update(old,u))
                throw std::invalid_argument("Step redundancy scope IDs are immutable");
            if (old.kind != u.kind || old.source_begin != u.source_begin || old.source_end != u.source_end ||
                old.result_begin != u.result_begin || old.result_end != u.result_end || (old.complete && !u.complete) ||
                u.observation_id < old.observation_id)
                throw std::invalid_argument("Stale or conflicting semantic GC evidence");
            if (u.observation_id == old.observation_id) {
                if (!same_update(old, u)) throw std::invalid_argument("A duplicate observation cannot change evidence");
                continue;
            }
        } else {
            const auto result_begin = retired_pages_.begin() + first_full_page(u.result_begin);
            const auto result_end = retired_pages_.begin() + u.result_end / tokens_;
            if (config_.feature_gate) {
                if (std::all_of(result_begin, result_end, [](uint8_t retired) { return retired != 0; })) {
                    if (finished.size() == config_.max_steps) finished.erase(finished.begin());
                    finished.push_back({u.id, "witness_retired", 0, 0});
                    ++ignored;
                    continue;
                }
            } else if (std::any_of(result_begin, result_end, [](uint8_t retired) { return retired != 0; })) {
                throw std::invalid_argument("Semantic GC witness has already retired");
            }
            if (next.size() == config_.max_steps) {
                if (config_.feature_gate && std::all_of(next.begin(),next.end(),
                        [](const Step& s) { return s.marked != 0; })) {
                    // Bounded metadata cannot revoke an already published
                    // selection tombstone. Retain the new source instead.
                    if (finished.size()==config_.max_steps) finished.erase(finished.begin());
                    finished.push_back({u.id,"metadata_capacity_retained",0,0});
                    ++ignored;continue;
                }
                // This evicts advisory evidence, NOT a KV block. Old queued
                // intents fail revision validation once their evidence is gone.
                const auto oldest = std::min_element(next.begin(), next.end(), [&](const Step& a, const Step& b) {
                    if ((a.marked != 0) != (b.marked != 0)) return !a.marked;
                    return a.revision < b.revision;
                });
                next.erase(oldest); ++evicted;
            }
            for (const auto& s : next)
                if (overlap(u.source_begin,u.source_end,s.value.source_begin,s.value.source_end) ||
                    overlap(u.source_begin,u.source_end,s.value.result_begin,s.value.result_end) ||
                    overlap(u.result_begin,u.result_end,s.value.source_begin,s.value.source_end))
                    throw std::invalid_argument("Semantic GC source overlaps a source or retained result");
            next.push_back({}); it = next.end() - 1;
            // Bind a new witness once, rather than walking its entire span for
            // every source candidate and every boundary revalidation.
            for (uint32_t id = first_full_page(u.result_begin);
                 id < u.result_end / tokens_; ++id)
                it->result_bound += bindings_[id].generation != 0;
        }
        if (it->observations != UINT32_MAX) ++it->observations;
        it->cold = u.cold ? (it->cold == UINT32_MAX ? UINT32_MAX : it->cold + 1) : 0;
        it->value = u; it->revision = revision++;
        it->source_remaining = remaining;
    }
    steps_ = std::move(next);
    if (!config_.feature_gate && next_revision_ != revision)
        result_protection_revision_.fetch_add(1, std::memory_order_release);
    next_revision_ = revision;
    finished_=std::move(finished);
    evicted_evidence_ = evicted; ignored_retired_updates_ = ignored;
}
void HistoricalKvStepEvidence::reset_bindings() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(bindings_.begin(), bindings_.end(), HistoricalKvHandle{});
    for (auto& s : steps_) {
        s.result_bound = 0;
        s.evaluated = false;
        s.selected.clear();
        s.revision = next_revision_++;
    }
}

void HistoricalKvStepEvidence::invalidate_proposals(Step& step, uint32_t page) {
    step.evaluated = false;
    step.checked_protection_epoch = 0;
    step.revision = next_revision_++;
    step.selected.erase(std::remove_if(step.selected.begin(), step.selected.end(), [&](const Selection& selection) {
        return selection.page == page || (page >= selection.witness &&
            page - selection.witness < selection.witness_pages);
    }), step.selected.end());
}

void HistoricalKvStepEvidence::bind(uint32_t page, HistoricalKvHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (page >= pages_ || !handle.generation || retired_pages_[page])
        throw std::invalid_argument("Invalid semantic GC page binding");
    const auto previous = bindings_[page];
    if (same(previous, handle)) return;
    bindings_[page] = handle;
    for (auto& s : steps_) {
        const bool is_result = result(s, page);
        if (is_result && !previous.generation) ++s.result_bound;
        if (config_.feature_gate && (is_result || source(s, page))) invalidate_proposals(s, page);
    }
}

bool HistoricalKvStepEvidence::release_mark(uint32_t page) noexcept {
    auto& mark = marks_[page];
    if (!mark.revision) return false;
    bool changed = false;
    for (uint32_t id = mark.witness.page; id < mark.witness.page + mark.witness.page_count; ++id)
        changed |= --witness_refs_[id] == 0;
    mark = {};
    return changed;
}
HistoricalKvStepEvidence::RetainedWitness HistoricalKvStepEvidence::retired(
    uint32_t page, HistoricalKvHandle handle) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (page >= pages_ || retired_pages_[page] || !same(bindings_[page], handle)) return {};
    retired_pages_[page] = 1;
    bindings_[page] = {};
    const auto witness = config_.feature_gate ? marks_[page].witness : RetainedWitness{};
    const bool pin_changed = config_.feature_gate && release_mark(page);
    for (auto& s : steps_) {
        const bool is_source = source(s, page), is_result = result(s, page);
        if (is_result) --s.result_bound;
        if (config_.feature_gate && (is_source || is_result)) invalidate_proposals(s, page);
        if (is_source && s.source_remaining) {
            --s.source_remaining;
            if (!config_.feature_gate || witness.result_end) ++s.retired_count;
            if (witness.result_end) --s.marked;
        }
    }
    const auto before = steps_.size();
    steps_.erase(std::remove_if(steps_.begin(), steps_.end(), [&](Step& s) {
        if (s.source_remaining) {
            if (!config_.feature_gate || s.marked || s.result_bound) return false;
            for (auto id = first_full_page(s.value.result_begin); id < s.value.result_end / tokens_; ++id)
                if (!retired_pages_[id]) return false;
        }
        if (config_.feature_gate) {
            // Capacity was reserved at construction; retirement never allocates.
            if (finished_.size() == config_.max_steps) finished_.erase(finished_.begin());
            const char* state = s.source_remaining ? (s.retired_count ? "retired_partial" : "witness_retired") :
                (s.retired_count ? "retired" : "capacity_reclaimed");
            finished_.push_back({std::move(s.value.id), state, s.retired_count, s.retired_count});
        }
        return true;
    }), steps_.end());
    completed_ += before - steps_.size();
    if (pin_changed || (!config_.feature_gate && before != steps_.size()))
        result_protection_revision_.fetch_add(1, std::memory_order_release);
    return witness;
}
void HistoricalKvStepEvidence::truncate_to(uint32_t frontier) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto before = steps_.size();
    bool pin_changed = false;
    steps_.erase(std::remove_if(steps_.begin(),steps_.end(),[&](const Step& s) {
        if (s.value.source_end <= frontier && s.value.result_end <= frontier) return false;
        if (s.marked)
            for (auto id = first_full_page(s.value.source_begin); id < s.value.source_end / tokens_; ++id)
                pin_changed |= release_mark(id);
        return true;
    }),steps_.end());
    std::fill(bindings_.begin(),bindings_.end(),HistoricalKvHandle{});
    std::fill(retired_pages_.begin() + std::min<size_t>(pages_, frontier / tokens_), retired_pages_.end(), 0);
    for (auto& s : steps_) {
        s.result_bound = 0;
        s.evaluated = false;
        s.selected.clear();
        s.revision = next_revision_++;
    }
    // A rewritten suffix must not inherit a completed ID from its old lineage.
    finished_.clear();
    if (pin_changed || (!config_.feature_gate && before != steps_.size()))
        result_protection_revision_.fetch_add(1, std::memory_order_release);
}
bool HistoricalKvStepEvidence::result_page(uint32_t page) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.feature_gate) return page < pages_ && witness_refs_[page] != 0;
    return page < pages_ && std::any_of(steps_.begin(),steps_.end(),[&](const Step& s) {return result(s,page);});
}
bool HistoricalKvStepEvidence::capacity_reclaimable(const HistoricalKvBlockMeta& b) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (b.logical_id>=pages_ || retired_pages_[b.logical_id] || !same(bindings_[b.logical_id],b.handle)) return false;
    if (config_.observe_only) return true;
    const auto page=static_cast<uint32_t>(b.logical_id);
    if (config_.feature_gate) return !witness_refs_[page] && !marks_[page].revision;
    return std::none_of(steps_.begin(), steps_.end(), [&](const Step& s) { return result(s, page); });
}
void HistoricalKvStepEvidence::mark_result_pages(std::vector<uint8_t>& flags, uint8_t bit) const {
    if (config_.observe_only) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& s : steps_) {
        if (config_.feature_gate && !s.marked) continue;
        const auto end = std::min<uint64_t>(std::min<size_t>(pages_,flags.size()),s.value.result_end/tokens_);
        for (auto page = first_full_page(s.value.result_begin); page < end; ++page)
            if (!config_.feature_gate || witness_refs_[page]) flags[page] |= bit;
    }
}
bool HistoricalKvStepEvidence::eligible_locked(uint32_t page, bool source_bound, bool result_bound) const {
    if (page >= pages_ || retired_pages_[page] || (source_bound && !bindings_[page].generation)) return false;
    if (config_.feature_gate) {
        const auto& mark = marks_[page];
        if (!mark.revision) return false;
        if (!source_bound || !result_bound) return true;
        for (uint32_t id = mark.witness.page; id < mark.witness.page + mark.witness.page_count; ++id)
            if (!bindings_[id].generation) return false;
        return true;
    }
    for (const auto& s : steps_) if (source(s,page)) {
        if (!s.value.complete || s.observations < config_.min_observations || s.cold < config_.min_cold_observations ||
            s.value.importance > config_.max_importance) return false;
        const auto result_pages = uint64_t(s.value.result_end) / tokens_ -
            (uint64_t(s.value.result_begin) + tokens_ - 1) / tokens_;
        if (result_bound && s.result_bound != result_pages) return false;
        return true;
    }
    return false;
}
bool HistoricalKvStepEvidence::eligible(uint32_t page) const {
    std::lock_guard<std::mutex> lock(mutex_); return eligible_locked(page);
}
bool HistoricalKvStepEvidence::handoff_candidate(uint32_t page) const {
    std::lock_guard<std::mutex> lock(mutex_); return eligible_locked(page, false, false);
}
bool HistoricalKvStepEvidence::selection_excluded(uint32_t page) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return page < pages_ && (retired_pages_[page] || eligible_locked(page, false, true));
}
bool HistoricalKvStepEvidence::permits_selection(uint32_t page, bool mandatory) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.feature_gate && eligible_locked(page,false,true)) return false;
    return page < pages_ && !retired_pages_[page] && (mandatory || !eligible_locked(page, false, true));
}
HistoricalKvStepEvidence::RetainedWitness HistoricalKvStepEvidence::selection_witness(uint32_t page) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.feature_gate || page >= pages_ || retired_pages_[page] || !marks_[page].revision) return {};
    return marks_[page].witness;
}
HistoricalKvStepEvidence::Stats HistoricalKvStepEvidence::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {steps_.size(), completed_, evicted_evidence_, ignored_retired_updates_};
}
uint64_t HistoricalKvStepEvidence::revision(const HistoricalKvBlockMeta& block) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block.logical_id >= pages_ || !same(bindings_[block.logical_id],block.handle) ||
        !eligible_locked(static_cast<uint32_t>(block.logical_id))) return 0;
    if (config_.feature_gate) return marks_[block.logical_id].revision;
    for (const auto& s : steps_) if (source(s,static_cast<uint32_t>(block.logical_id))) return s.revision;
    return 0;
}

std::vector<HistoricalKvStepEvidence::Status> HistoricalKvStepEvidence::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto out=finished_;
    for (const auto& s:steps_) out.push_back({s.value.id,
        s.marked ? "marked_unselectable" : !s.selected.empty() ? "awaiting_boundary" :
        s.evaluated && s.missing_observations.empty() ? "no_net_gain" : "pending_features",
        static_cast<uint32_t>(s.selected.size()) + s.marked + s.retired_count,
        s.retired_count});
    return out;
}

void HistoricalKvStepEvidence::evaluate(const std::shared_ptr<HistoricalKvPageFeatures>& features,
                                        const HistoricalKvSnapshot& snapshot) {
    if (!config_.feature_gate || !features) return;
    const auto started=std::chrono::steady_clock::now();
    const auto epoch=features->question_epoch();
    const auto observation_revision = features->observation_revision();
    Step candidate;
    std::array<HistoricalKvHandle, historical_kv_max_source_pages> source_bindings{};
    std::array<HistoricalKvHandle, historical_kv_max_witness_pages> result_bindings{};
    uint32_t first_source = 0, first_witness = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        features_=features;
        // One ready scope per repository round, with a rotating start. A
        // failed score stays dormant until new evidence could change it.
        for (size_t offset = 0; offset < steps_.size(); ++offset) {
            auto& s = steps_[(snapshot.round_id + offset) % steps_.size()];
            if (s.source_remaining == s.marked || !s.result_bound || !s.value.complete ||
                s.observations < config_.min_observations || s.cold < config_.min_cold_observations ||
                s.value.importance > config_.max_importance) continue;
            if (s.evaluated && (s.value.duplicate_begin || s.feature_epoch == epoch)) {
                if (s.missing_observations.empty() || s.observation_revision == observation_revision) continue;
                s.observation_revision = observation_revision;
                if (!features->any_observed(s.missing_observations, config_.relaxed_attention_redundancy)) continue;
            }
            candidate.value = s.value;
            candidate.revision = s.revision;
            candidate.marked = s.marked;
            candidate.retired_count = s.retired_count;
            first_source = first_full_page(s.value.source_begin);
            first_witness = first_full_page(s.value.result_begin);
            for (auto id = first_source; id < s.value.source_end / tokens_; ++id)
                if (source(s, id) && !marks_[id].revision) source_bindings[id - first_source] = bindings_[id];
            for (auto id = first_witness; id < s.value.result_end / tokens_; ++id)
                result_bindings[id - first_witness] = bindings_[id];
            break;
        }
    }
    if (!candidate.revision) return;
    const bool duplicate = candidate.value.duplicate_begin != 0;
    if (!duplicate) features->begin_round(snapshot.incarnation);
    const auto dims=features->dimensions();
    struct Feature { uint32_t page; HistoricalKvHandle handle; std::vector<float> key; double importance; };
    std::vector<Feature> sources,witnesses;
    std::vector<Selection> selected;
    std::vector<uint32_t> missing_observations;
    std::array<HistoricalKvHandle, historical_kv_max_witness_pages> sealed_witnesses{};
    size_t sealed_results = 0;
    for (const auto& b:snapshot.blocks) {
        if (b.logical_id >= pages_ || b.life != HistoricalKvLife::Live || !b.sealed || b.valid_tokens != tokens_)
            continue;
        const auto page=static_cast<uint32_t>(b.logical_id);
        const bool is_source=source(candidate,page), is_result=result(candidate,page);
        if (!is_source && !is_result) continue;
        auto& expected = is_source ? source_bindings[page - first_source] : result_bindings[page - first_witness];
        if (!expected.generation || !same(expected, b.handle)) continue;
        expected = {};
        if (is_result) {
            sealed_witnesses[page - first_witness] = b.handle;
            ++sealed_results;
        }
        if (duplicate) {
            if (is_source) {
                constexpr auto margin = historical_kv_duplicate_context_tokens;
                const auto copy = candidate.value.duplicate_begin + uint64_t(page) * tokens_ - candidate.value.source_begin;
                const auto first = (copy - margin) / tokens_;
                const auto last = (copy + tokens_ + margin - 1) / tokens_;
                selected.push_back({page, static_cast<uint32_t>(first), b.handle,
                    static_cast<uint32_t>(last - first + 1)});
            }
            continue;
        }
        Feature f{page,b.handle,std::vector<float>(dims),0};
        if (!features->read_observed(b,f.key.data(),f.importance,config_.relaxed_attention_redundancy)) {
            missing_observations.push_back(page);
            continue;
        }
        (is_source?sources:witnesses).push_back(std::move(f));
    }
    selected.erase(std::remove_if(selected.begin(), selected.end(), [&](const Selection& s) {
        const auto begin = sealed_witnesses.begin() + (s.witness - first_witness);
        return std::any_of(begin, begin + s.witness_pages, [](HistoricalKvHandle h) { return !h.generation; });
    }), selected.end());
    const auto result_count = candidate.value.result_end / tokens_ - first_witness;
    const auto charged_results=config_.charge_result_pages ? result_count : 0;
    const auto cosine=[](const std::vector<float>& a,const std::vector<float>& b) {
        double aa=0,bb=0,ab=0;
        for (size_t i=0;i<a.size();++i) { aa+=double(a[i])*a[i]; bb+=double(b[i])*b[i]; ab+=double(a[i])*b[i]; }
        return aa>0 && bb>0 ? ab/std::sqrt(aa*bb) : -1.0;
    };
    uint64_t comparisons=0;
    const auto& masks = candidate.value.witness_masks;
    const bool audit = historical_kv_audit_enabled();
    size_t literal_vetoes = 0;
    size_t importance_rejections = 0, similarity_rejections = 0;
    for (const auto& s:sources) {
        bool vetoed = false;
        const auto witness = std::find_if(witnesses.begin(), witnesses.end(), [&](const Feature& w) {
            if (w.importance<s.importance) { ++importance_rejections; return false; }
            const bool covered = config_.relaxed_attention_redundancy || masks.empty() ||
                (masks[s.page - first_source] & (1u << (w.page - first_witness)));
            if (!covered && !audit) return false;
            ++comparisons;
            if (!(cosine(s.key,w.key)>=config_.similarity)) { ++similarity_rejections; return false; }
            vetoed |= !covered;
            return covered;
        });
        if (witness != witnesses.end()) selected.push_back({s.page, witness->page, s.handle});
        else if (vetoed) {
            ++literal_vetoes;
            std::fprintf(stderr, "KVMEM_STEP_LITERAL_VETO id=%s page=%u\n", candidate.value.id.c_str(), s.page);
        }
    }
    comparisons_.fetch_add(comparisons,std::memory_order_relaxed);
    if (!duplicate && features->question_epoch()!=epoch) return;
    const bool net_gain = selected.size() + candidate.marked + candidate.retired_count > charged_results;
    if (audit && duplicate) std::fprintf(stderr,
        "KVMEM_STEP_DUPLICATE id=%s matched=%zu witness_pages=%zu\n",
        candidate.value.id.c_str(),selected.size(),sealed_results);
    if (audit && !duplicate) std::fprintf(stderr,
        "KVMEM_STEP_REDUNDANCY id=%s observed_sources=%zu witnesses=%zu matched=%zu literal_vetoes=%zu approved=%zu epoch=%llu worker_us=%llu\n",
        candidate.value.id.c_str(),sources.size(),witnesses.size(),selected.size(),literal_vetoes,
        net_gain ? selected.size() : 0,(unsigned long long)epoch,
        (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-started).count());
    if (!net_gain) selected.clear();
    if (audit && !duplicate) std::fprintf(stderr,
        "KVMEM_STEP_FILTER id=%s mode=%s similarity=%.3f observed_sources=%zu observed_witnesses=%zu missing_pages=%zu importance_rejected_pairs=%zu similarity_rejected_pairs=%zu literal_vetoed_sources=%zu selected_sources=%zu\n",
        candidate.value.id.c_str(),config_.relaxed_attention_redundancy ? "attention_redundancy" : "strict",
        config_.similarity,sources.size(),witnesses.size(),missing_observations.size(),
        importance_rejections,similarity_rejections,literal_vetoes,selected.size());
    std::sort(selected.begin(), selected.end(), [](const Selection& a, const Selection& b) {
        return std::tie(a.witness, a.page) < std::tie(b.witness, b.page);
    });
    const auto unseen = [](HistoricalKvHandle h) { return h.generation != 0; };
    const bool fresh_snapshot = std::none_of(source_bindings.begin(), source_bindings.end(), unseen) &&
        std::none_of(result_bindings.begin(), result_bindings.end(), unseen);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it=std::find_if(steps_.begin(),steps_.end(),[&](const Step& s) {
        return s.revision==candidate.revision && s.value.id==candidate.value.id;
    });
    if (it==steps_.end() || (!duplicate && features->question_epoch()!=epoch)) return;
    it->selected = std::move(selected);
    it->witnesses = sealed_witnesses;
    it->missing_observations = std::move(missing_observations);
    it->evaluated = fresh_snapshot;
    it->feature_epoch = epoch;
    it->observation_revision = observation_revision;
    it->checked_protection_epoch = 0;
}

size_t HistoricalKvStepEvidence::commit_selection_boundary(const std::vector<uint8_t>& protections,
                                                           uint8_t allowed, uint64_t protection_epoch,
                                                           const SelectionGuard* guard) {
    if (!config_.feature_gate) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto features=features_.lock();
    if (!features) return 0;
    size_t marked = 0;
    bool pin_changed = false;
    const auto guard_revision = guard ? guard->revision : 0;
    for (auto it = steps_.begin(); it != steps_.end();) {
        auto current = it++;
        auto& s = *current;
        if (s.selected.empty() || (!s.value.duplicate_begin && s.feature_epoch != features->question_epoch()) ||
            (protection_epoch && s.checked_protection_epoch == protection_epoch &&
             s.checked_guard_revision == guard_revision)) continue;
        s.checked_protection_epoch = protection_epoch;
        s.checked_guard_revision = guard_revision;
        const auto first_witness = first_full_page(s.value.result_begin);
        std::array<const Selection*, historical_kv_max_source_pages> ready{};
        size_t count = 0;
        for (const auto& source : s.selected) {
            const auto p = source.page;
            if (p >= protections.size() || (protections[p] & uint8_t(~allowed)) ||
                marks_[p].revision || !same(bindings_[p], source.handle) ||
                source.witness + source.witness_pages > protections.size()) continue;
            bool bound = true;
            for (uint32_t id = source.witness; id < source.witness + source.witness_pages; ++id)
                bound &= bindings_[id].generation && same(bindings_[id], s.witnesses[id - first_witness]);
            if (!bound || (guard && !guard->permits(guard->context, p,
                    {source.witness, s.value.result_end, source.witness_pages}))) continue;
            ready[count++] = &source;
        }
        size_t approved = 0;
        for (size_t begin = 0; begin < count;) {
            const auto first = ready[begin]->witness;
            auto end = first + ready[begin]->witness_pages;
            size_t next = begin + 1;
            while (next < count && ready[next]->witness < end) {
                end = std::max(end, ready[next]->witness + ready[next]->witness_pages);
                ++next;
            }
            // Overlapping witness ranges form a single independent group.
            // Charge each newly protected page once, after dropping blocked sources.
            size_t extra_pins = 0;
            for (auto id = first; id < end; ++id) extra_pins += !protections[id] && !witness_refs_[id];
            if (extra_pins <= next - begin) {
                approved += next - begin;
                if (!config_.observe_only) {
                    for (auto index = begin; index < next; ++index) {
                        const auto& source = *ready[index];
                        marks_[source.page] = {next_revision_++, {source.witness, s.value.result_end, source.witness_pages}};
                        for (uint32_t id = source.witness; id < source.witness + source.witness_pages; ++id)
                            pin_changed |= witness_refs_[id]++ == 0;
                    }
                }
            }
            begin = next;
        }
        if (!approved) continue;
        if (config_.observe_only) {
            if (finished_.size() == config_.max_steps) finished_.erase(finished_.begin());
            finished_.push_back({std::move(s.value.id), "observed_no_delete", static_cast<uint32_t>(approved), 0});
            it = steps_.erase(current);
            continue;
        }
        s.marked += static_cast<uint32_t>(approved);
        s.revision = next_revision_++;
        s.selected.erase(std::remove_if(s.selected.begin(), s.selected.end(),
            [&](const Selection& selection) { return marks_[selection.page].revision != 0; }), s.selected.end());
        marked += approved;
    }
    if (pin_changed) result_protection_revision_.fetch_add(1, std::memory_order_release);
    return marked;
}
HistoricalKvStatisticalPolicy::HistoricalKvStatisticalPolicy(std::shared_ptr<HistoricalKvStepEvidence> evidence,
    std::shared_ptr<HistoricalKvPageFeatures> features, uint32_t pressure_capacity)
    : evidence_(std::move(evidence)), features_(std::move(features)), pressure_capacity_(pressure_capacity),
      audit_(historical_kv_audit_enabled()) {
    if (!evidence_) throw std::invalid_argument("Missing semantic evidence");
    capacity_candidates_.reserve(pressure_capacity_);
}
bool HistoricalKvStatisticalPolicy::permits(const HistoricalKvBlockMeta& b) const noexcept {
    return b.life == HistoricalKvLife::Live && b.sealed && !b.mandatory &&
        (evidence_->revision(b) != 0 || (pressure_capacity_ && evidence_->capacity_reclaimable(b)));
}
HistoricalKvPolicy::Metrics HistoricalKvStatisticalPolicy::metrics() const noexcept {
    Metrics out;out.comparisons=evidence_->feature_comparisons();
    out.lossy_intents=lossy_intents_.load(std::memory_order_relaxed);return out;
}
bool HistoricalKvStatisticalPolicy::permits_intent(const HistoricalKvBlockMeta& b,
                                                  const HistoricalKvDeleteIntent& intent) const noexcept {
    if (b.life!=HistoricalKvLife::Live || !b.sealed || b.mandatory) return false;
    if (intent.reason==HistoricalKvDeleteReason::CapacityLru)
        return pressure_capacity_ && !intent.evidence_revision && evidence_->capacity_reclaimable(b);
    return intent.reason==HistoricalKvDeleteReason::Semantic && intent.evidence_revision &&
        *intent.evidence_revision && evidence_->revision(b)==*intent.evidence_revision;
}
HistoricalKvStepResult HistoricalKvStatisticalPolicy::step(const HistoricalKvSnapshot& snapshot,
    const HistoricalKvPressure& pressure, HistoricalKvWorkBudget budget, HistoricalKvDecisionBuffer& out) {
    if (!budget.scan || !budget.intents) throw std::invalid_argument("Invalid semantic GC work budget");
    if (pressure_capacity_ && (snapshot.blocks.size()>pressure_capacity_ || !pressure.low ||
        pressure.low>=pressure.high || pressure.high>pressure_capacity_ || pressure.requested_ready>pressure_capacity_))
        throw std::invalid_argument("Invalid semantic GC pressure capacity/watermarks");
    if (incarnation_ != snapshot.incarnation || round_ != snapshot.round_id) {
        if (features_) evidence_->evaluate(features_,snapshot);
        if (incarnation_!=snapshot.incarnation) pressure_mode_=false;
        incarnation_ = snapshot.incarnation; round_ = snapshot.round_id; cursor_ = 0;
        capacity_candidates_.clear();capacity_cursor_=0;ranked_=false;
    }
    size_t examined = 0, emitted = 0;
    while (cursor_ < snapshot.blocks.size() && examined < budget.scan && emitted < budget.intents && out.remaining()) {
        const auto index=cursor_++;
        const auto& b = snapshot.blocks[index]; ++examined;
        if (b.life!=HistoricalKvLife::Live || !b.sealed || b.mandatory) continue;
        const auto revision=evidence_->revision(b);
        if (!revision) {
            if (pressure_capacity_ && evidence_->capacity_reclaimable(b)) capacity_candidates_.push_back(index);
            continue;
        }
        HistoricalKvDeleteIntent intent{b.handle,snapshot.round_id,HistoricalKvDeleteReason::Semantic};
        intent.expected_content_version = b.content_version;
        intent.evidence_revision = revision;
        out.push(intent); ++emitted;
        if (audit_) std::fprintf(stderr,
            "KVMEM_GC_AUDIT event=propose policy=statistical incarnation=%llu slot=%u generation=%llu round=%llu original_begin=%llu valid_tokens=%u ready=%u low=%u high=%u evidence_revision=%llu\n",
            (unsigned long long)b.handle.incarnation, b.handle.slot, (unsigned long long)b.handle.generation,
            (unsigned long long)snapshot.round_id, (unsigned long long)b.original_begin, b.valid_tokens,
            pressure.ready, pressure.low, pressure.high, (unsigned long long)*intent.evidence_revision);
    }
    if (cursor_<snapshot.blocks.size()) return {HistoricalKvStepStatus::Yielded,examined,emitted};
    if (!pressure_capacity_) return {emitted ? HistoricalKvStepStatus::Complete : HistoricalKvStepStatus::NoCandidate,examined,emitted};
    if (pressure.ready<pressure.low) pressure_mode_=true;
    if (pressure.ready>=pressure.high) pressure_mode_=false;
    const uint64_t target=std::max(pressure_mode_ ? pressure.high : 0u,pressure.requested_ready);
    const uint64_t planned=uint64_t(pressure.ready)+pressure.pending_slot_credit+emitted;
    uint64_t wanted=target>planned ? target-planned : 0;
    if (!wanted) return {HistoricalKvStepStatus::Complete,examined,emitted};
    if (!ranked_) {
        const auto key=[&](size_t i) {
            const auto& b=snapshot.blocks[i];
            return std::make_tuple(b.last_use_seq,b.birth_seq,b.logical_id,b.handle.slot,b.handle.generation);
        };
        std::sort(capacity_candidates_.begin(),capacity_candidates_.end(),[&](size_t a,size_t b) {return key(a)<key(b);});
        ranked_=true;
    }
    // Retain this bounded ranking across batches. Re-scanning a full Host
    // directory per 64 deletes can otherwise exhaust the prefill wait budget.
    while (wanted && capacity_cursor_<capacity_candidates_.size() && emitted<budget.intents && out.remaining()) {
        const auto& b=snapshot.blocks[capacity_candidates_[capacity_cursor_++]];
        if (!evidence_->capacity_reclaimable(b)) continue;
        HistoricalKvDeleteIntent intent{b.handle,snapshot.round_id,HistoricalKvDeleteReason::CapacityLru};
        intent.expected_content_version=b.content_version;
        out.push(intent);++emitted;--wanted;++lossy_intents_;
        if (audit_) std::fprintf(stderr,
            "KVMEM_GC_AUDIT event=propose policy=capacity_lru incarnation=%llu slot=%u generation=%llu round=%llu original_begin=%llu valid_tokens=%u ready=%u low=%u high=%u requested_ready=%u\n",
            (unsigned long long)b.handle.incarnation,b.handle.slot,(unsigned long long)b.handle.generation,
            (unsigned long long)snapshot.round_id,(unsigned long long)b.original_begin,b.valid_tokens,
            pressure.ready,pressure.low,pressure.high,pressure.requested_ready);
    }
    return {wanted && capacity_cursor_<capacity_candidates_.size() ? HistoricalKvStepStatus::Yielded :
        (emitted ? HistoricalKvStepStatus::Complete : HistoricalKvStepStatus::NoCandidate),examined,emitted};
}
} // namespace kvmem
