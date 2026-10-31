#include "native_memory.hpp"
#include "native_pages.hpp"
#include "execution_pages.hpp"
#include "page_transfer.hpp"
#include "kvmem/native_kv_payload.hpp"
#include "mem/kv.h"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace kvmem::radiance {
namespace {
[[noreturn]] void fail(MemoryErrorCode code, const char* msg) { throw MemoryError(code, msg); }
uint64_t checked_bytes(uint64_t count, uint64_t size) {
    if (size && count > uint64_t(std::numeric_limits<int64_t>::max()) / size)
        fail(MemoryErrorCode::BudgetExceeded, "native staging size overflow");
    return count * size;
}
bool same_view(const ExecutionView& a, const ExecutionView& b) {
    if (!(a.stamp == b.stamp) || a.visible_tokens != b.visible_tokens || a.blocks.size() != b.blocks.size())
        return false;
    for (size_t i = 0; i < a.blocks.size(); ++i)
        if (!(a.blocks[i].block == b.blocks[i].block) || a.blocks[i].compact_start != b.blocks[i].compact_start)
            return false;
    return true;
}
using Bindings = std::map<uint64_t, std::vector<GroupPages>>;
}

struct NativeMemory::State {
    MemoryBinding binding;
    NativePages pages;
    MemoryDescriptor descriptor;
    MemorySnapshot current;
    NativeKvArchive archive;
    uint64_t host_capacity;
    Bindings resident;
    std::vector<rad::KVManager::FullTable> tables;
    std::vector<GroupPool> pools;
    std::vector<std::pair<int32_t, int32_t>> state_slots;
    uint64_t visible = 0;
    uint64_t native_sequence = 0;
    bool pending = false;
    NativeMemory::ReuseStats reuse;
    struct RetainedView {
        MemorySnapshot snapshot;
        Bindings resident;
        std::vector<rad::KVManager::FullTable> tables;
        uint64_t visible;
    };
    std::unique_ptr<RetainedView> retained;

    State(MemoryBinding b, const std::vector<rad::KVGroupInfo>& infos, BackendIdentity identity,
          MemoryStamp stamp, uint32_t tokens, uint64_t max_pos, uint64_t capacity)
        : binding(std::move(b)), pages(binding.manager->plans(), infos, tokens, identity.state_schema),
          descriptor{std::move(identity), pages.layout(), {}},
          archive(descriptor.identity, stamp.session, descriptor.layout, capacity), host_capacity(capacity) {
        current.stamp = stamp;
        native_sequence = stamp.session.id;
        uint64_t quantum = 1;
        for (const auto& p : binding.manager->plans()) {
            if (p.kind < 0) continue;
            if (p.stateful()) {
                if (p.zero || p.state_copies < 1 || binding.manager->state_slot(stamp.session.id, p.index) < 0)
                    fail(MemoryErrorCode::Unsupported, "state retention requires one committed nonzero slot");
                state_slots.emplace_back(p.index, binding.manager->state_slot(stamp.session.id, p.index));
                continue; // Recurrent/conv state is continuous, NOT a paged KV payload.
            }
            if (p.kind == RAD_KV_WINDOW) continue; // Native drafter rolling cache, never selected/archive-owned.
            if (p.kind != RAD_KV_FULL || p.zero)
                fail(MemoryErrorCode::Unsupported, "native transaction currently requires nonzero FULL attention groups");
            quantum = std::lcm(quantum, uint64_t(p.block_size));
            tables.push_back({p.index, binding.manager->block_table(stamp.session.id, p.index)});
            pools.push_back({p.index, p.base, uint64_t(p.pool_bytes)});
        }
        descriptor.capabilities = {uint32_t(quantum), max_pos, true, true, true, false, false, true};
        descriptor.validate();
    }
    void check() const {
        if (!current.stamp.valid) fail(MemoryErrorCode::InvalidSession, "native sequence invalidated");
        const auto& kv = *binding.manager;
        const auto seq = native_sequence;
        if (!kv.has_sequence(seq) || kv.seq_tokens(seq) != int64_t(current.stamp.frontier))
            fail(MemoryErrorCode::Stale, "engine frontier changed outside the memory boundary");
        for (const auto& t : tables)
            if (kv.block_table(seq, t.group) != t.blocks ||
                kv.execution_offset(seq, t.group) != int64_t(current.stamp.frontier - visible))
                fail(MemoryErrorCode::Stale, "engine page ownership changed outside the memory boundary");
        for (const auto& [group, slot] : state_slots)
            if (kv.state_slot(seq, group) != slot)
                fail(MemoryErrorCode::Stale, "engine recurrent-state ownership changed during KV transaction");
    }
    std::vector<CopySpan> spans(const std::vector<BlockDescriptor>& blocks, const Bindings& bindings) const {
        std::vector<CopySpan> out;
        uint64_t offset = 0;
        for (const auto& block : blocks) {
            auto group_pages = bindings.at(block.id);
            for (auto& g : group_pages) g.backed_blocks = binding.manager->live_blocks(g.group);
            auto record = pages.spans(binding.manager->plans(), group_pages);
            for (auto span : record) { span.record_offset += offset; out.push_back(span); }
            offset += descriptor.layout.record_bytes;
        }
        return out;
    }
};

class NativeMemory::Transaction final : public MemoryTransfer {
    std::shared_ptr<State> s_;
    WorkingSetPlan plan_;
    MemorySnapshot next_;
    Bindings next_bindings_;
    std::vector<rad::KVManager::FullTable> next_tables_;
    std::vector<std::pair<int32_t, int32_t>> reserved_;
    std::vector<CopySpan> spill_spans_, restore_spans_;
    std::vector<uint8_t> restore_bytes_;
    std::unique_ptr<PageTransfer> spill_, restore_;
    // Only exclusively owned, evicted pages can be reused. Prefix leases and
    // unchanged selected pages stay owned by Radiance and are never overwritten.
    std::vector<std::pair<int32_t, int32_t>> recycled_;
    std::vector<BlockDescriptor> rollback_blocks_;
    std::vector<CopySpan> rollback_spans_;
    std::vector<uint8_t> rollback_bytes_;
    std::unique_ptr<PageTransfer> rollback_;
    bool restore_may_have_written_ = false;
    enum class Phase { Prepared, Spilling, Spilled, Restoring, Ready, Published, Aborted } phase_ = Phase::Prepared;
    bool owns_pending_ = false;
public:
    Transaction(std::shared_ptr<State> state, WorkingSetPlan plan) : s_(std::move(state)), plan_(std::move(plan)) {}
    ~Transaction() override { abort(); }
    void prepare() {
        auto& s = *s_;
        s.check();
        if (s.pending) fail(MemoryErrorCode::Pending, "native transfer already pending");
        if (!(plan_.expected == s.current.stamp)) fail(MemoryErrorCode::Stale, "native plan stamp changed");
        std::vector<uint64_t> selected;
        for (const auto& b : plan_.next.blocks) selected.push_back(b.block.id);
        const auto verified = plan_working_set(s.descriptor, s.current, selected, plan_.mandatory, plan_.budget);
        // An incomplete accepted tail is mutable. Keep it on device until it
        // closes; never archive bytes that a later append will overwrite.
        for (const auto& item : s.current.blocks) if (item.block.valid_tokens < s.descriptor.layout.block_tokens) {
            if (!s.resident.count(item.block.id) ||
                std::find(selected.begin(), selected.end(), item.block.id) == selected.end())
                fail(MemoryErrorCode::InvalidPlan, "partial history tail must remain resident");
        }
        if (!same_view(verified.next, plan_.next) || verified.spill != plan_.spill ||
            verified.evict != plan_.evict || verified.restore != plan_.restore ||
            verified.device_tokens_required != plan_.device_tokens_required ||
            verified.host_payload_bytes_required != plan_.host_payload_bytes_required)
            fail(MemoryErrorCode::InvalidPlan, "native transaction received an inconsistent plan");
        if (plan_.budget.host_payload_bytes > s.host_capacity)
            fail(MemoryErrorCode::BudgetExceeded, "host budget exceeds the native archive capacity");
        next_ = s.current;
        next_.stamp = plan_.next.stamp;
        next_.resident = selected;
        auto& kv = *s.binding.manager;
        // Reserve capacity before taking ownership, so vector allocation cannot
        // lose a page obtained from the engine's allocator.
        uint64_t reserve_count = 0;
        std::map<int32_t, std::vector<std::pair<int32_t, uint64_t>>> recyclable;
        std::set<uint64_t> journal_ids;
        for (const auto& t : s.tables) {
            const auto page_tokens = uint64_t(kv.plan(t.group)->block_size);
            const uint64_t restore_count = checked_bytes(plan_.restore.size(), s.descriptor.layout.block_tokens / page_tokens);
            const uint64_t tail = plan_.budget.reserve_tokens / page_tokens + (plan_.budget.reserve_tokens % page_tokens != 0);
            const auto free = uint64_t(kv.free_blocks(t.group));
            if (free < tail) fail(MemoryErrorCode::BudgetExceeded, "not enough free native append pages");
            const auto shortage = restore_count > free - tail ? restore_count - (free - tail) : 0;
            auto& candidates = recyclable[t.group];
            // Keep the old fast path when there is already room. Under pressure
            // reuse ONLY a page whose previous logical block the common planner
            // evicts and whose sole reference is this live sequence.
            for (const auto& block : plan_.evict) {
                if (candidates.size() >= shortage) break;
                for (const auto& g : s.resident.at(block.id)) if (g.group == t.group) {
                    for (auto page : g.pages) {
                        if (candidates.size() >= shortage) break;
                        if (kv.block_refcount(t.group, page) == 1)
                            candidates.emplace_back(page, block.id);
                    }
                }
            }
            if (candidates.size() != shortage)
                fail(MemoryErrorCode::BudgetExceeded, "native page pressure: leased pages cannot be recycled");
            for (const auto& [page, block] : candidates) journal_ids.insert(block);
            reserve_count += restore_count - shortage + tail;
            if (kv.reserve_release_space(t.group, restore_count + tail + t.blocks.size()) < 0)
                fail(MemoryErrorCode::BudgetExceeded, "reserve native release bookkeeping");
        }
        reserved_.reserve(size_t(reserve_count));
        size_t recycled_count = 0;
        for (const auto& [group, pages] : recyclable) recycled_count += pages.size();
        recycled_.reserve(recycled_count);
        for (const auto& v : plan_.next.blocks) {
            auto found = s.resident.find(v.block.id);
            if (found != s.resident.end()) { next_bindings_.emplace(*found); continue; }
            auto& binding = next_bindings_[v.block.id];
            for (const auto& table : s.tables) {
                GroupPages g{table.group, uint64_t(kv.live_blocks(table.group)), {}};
                const auto count = s.descriptor.layout.block_tokens / kv.plan(table.group)->block_size;
                g.pages.reserve(size_t(count));
                for (int64_t i = 0; i < count; ++i) {
                    int32_t page = -1;
                    auto& reuse = recyclable[table.group];
                    if (!reuse.empty()) {
                        page = reuse.back().first;
                        reuse.pop_back();
                        recycled_.emplace_back(table.group, page);
                    } else {
                        if (kv.reserve_block(table.group, &page) < 0)
                            fail(MemoryErrorCode::BudgetExceeded, "engine page reservation failed");
                        reserved_.emplace_back(table.group, page);
                    }
                    g.pages.push_back(page);
                }
                binding.push_back(std::move(g));
            }
        }
        for (const auto& t : s.tables) {
            const auto page = uint64_t(kv.plan(t.group)->block_size);
            const auto count = plan_.budget.reserve_tokens / page + (plan_.budget.reserve_tokens % page != 0);
            for (uint64_t i = 0; i < count; ++i) {
                int32_t id = -1;
                if (kv.reserve_block(t.group, &id) < 0) fail(MemoryErrorCode::BudgetExceeded, "reserve decode tail");
                reserved_.emplace_back(t.group, id);
            }
            std::vector<BlockPages> blocks;
            for (const auto& v : plan_.next.blocks) {
                for (const auto& g : next_bindings_.at(v.block.id))
                    if (g.group == t.group) blocks.push_back({v.block, g.pages});
            }
            ExecutionPages execution(plan_.next, s.descriptor.layout.block_tokens, uint32_t(page),
                                      kv.live_blocks(t.group), blocks);
            next_tables_.push_back({t.group, execution.table()});
        }
        spill_spans_ = s.spans(plan_.spill, s.resident);
        restore_spans_ = s.spans(plan_.restore, next_bindings_);
        const auto record_bytes = s.descriptor.layout.record_bytes;
        for (const auto& block : plan_.evict) if (journal_ids.count(block.id)) rollback_blocks_.push_back(block);
        if (!rollback_blocks_.empty()) {
            // Preallocate the exceptional abort path before any old bytes can
            // change. The ordinary spill/archive supplies its ORIGINAL bytes;
            // no extra D2H copy and no permanent replacement GPU allocation.
            rollback_spans_ = s.spans(rollback_blocks_, s.resident);
            rollback_bytes_.resize(size_t(checked_bytes(rollback_blocks_.size(), record_bytes)));
            rollback_ = std::make_unique<PageTransfer>(s.binding.device, rollback_bytes_.size(), s.binding.compute);
        }
        if (!plan_.spill.empty()) spill_ = std::make_unique<PageTransfer>(s.binding.device,
            checked_bytes(plan_.spill.size(), record_bytes), s.binding.compute);
        if (!plan_.restore.empty()) {
            restore_bytes_.reserve(size_t(checked_bytes(plan_.restore.size(), record_bytes)));
            for (const auto& b : plan_.restore) {
                const auto& bytes = s.archive.get(b).bytes();
                restore_bytes_.insert(restore_bytes_.end(), bytes.begin(), bytes.end());
            }
            restore_ = std::make_unique<PageTransfer>(s.binding.device, restore_bytes_.size(), s.binding.compute);
        }
        s.pending = true;
        owns_pending_ = true;
        if (!recycled_.empty()) {
            ++s.reuse.transactions;
            s.reuse.recycled_pages += recycled_.size();
            s.reuse.journal_bytes_peak = std::max(s.reuse.journal_bytes_peak, uint64_t(rollback_bytes_.size()));
        }
    }
    void start_spill() override {
        s_->check();
        if (phase_ != Phase::Prepared) fail(MemoryErrorCode::InvalidPlan, "spill phase order");
        phase_ = Phase::Spilling;
        if (spill_) spill_->spill(spill_spans_, s_->pools, s_->binding.compute);
    }
    TransferPoll poll() override {
        s_->check();
        if (phase_ == Phase::Spilling) {
            const auto result = spill_ ? spill_->poll() : TransferPoll{TransferStatus::Complete, {}};
            if (result.status != TransferStatus::Complete) return result;
            if (spill_) {
                const auto bytes = spill_->payload();
                const auto n = s_->descriptor.layout.record_bytes;
                for (size_t i = 0; i < plan_.spill.size(); ++i) {
                    const auto& b = plan_.spill[i];
                    s_->archive.put(NativeKvPayload(s_->descriptor.identity, s_->descriptor.layout, b,
                        {bytes.begin() + i * n, bytes.begin() + (i + 1) * n}));
                    // Completed copies remain useful after a cancelled transaction.
                    for (auto& item : s_->current.blocks) if (item.block.id == b.id) item.host_version = b.content_version;
                    for (auto& item : next_.blocks) if (item.block.id == b.id) item.host_version = b.content_version;
                }
            }
            phase_ = Phase::Spilled;
            return result;
        }
        if (phase_ == Phase::Restoring) {
            const auto result = restore_ ? restore_->poll() : TransferPoll{TransferStatus::Complete, {}};
            if (result.status == TransferStatus::Complete) phase_ = Phase::Ready;
            return result;
        }
        if (phase_ == Phase::Spilled || phase_ == Phase::Ready) return {TransferStatus::Complete, {}};
        fail(MemoryErrorCode::InvalidPlan, "poll phase order");
    }
    void start_restore() override {
        s_->check();
        if (phase_ != Phase::Spilled) fail(MemoryErrorCode::InvalidPlan, "restore before completed spill");
        for (const auto& [group, page] : recycled_)
            if (s_->binding.manager->block_refcount(group, page) != 1)
                fail(MemoryErrorCode::Stale, "recycled page acquired another owner before restore");
        const auto n = s_->descriptor.layout.record_bytes;
        for (size_t i = 0; i < rollback_blocks_.size(); ++i) {
            const auto& bytes = s_->archive.get(rollback_blocks_[i]).bytes();
            if (bytes.size() != n) fail(MemoryErrorCode::InvalidPlan, "rollback payload size mismatch");
            std::copy(bytes.begin(), bytes.end(), rollback_bytes_.begin() + i * n);
        }
        phase_ = Phase::Restoring;
        restore_may_have_written_ = bool(restore_);
        if (restore_) restore_->restore(restore_spans_, s_->pools, s_->binding.compute, restore_bytes_);
    }
    void publish(const ExecutionView& view) override {
        s_->check();
        if (phase_ != Phase::Ready || !same_view(view, plan_.next))
            fail(MemoryErrorCode::InvalidPlan, "native publication is not ready or view changed");
        if (s_->binding.manager->publish_full_tables(s_->native_sequence,
            int64_t(view.stamp.frontier), int64_t(view.visible_tokens), next_tables_) < 0)
            fail(MemoryErrorCode::TransferFailed, "engine rejected native page publication");
        // Everything after native publication is non-allocating.
        std::swap(s_->current, next_);
        s_->resident.swap(next_bindings_);
        s_->tables.swap(next_tables_);
        s_->visible = view.visible_tokens;
        phase_ = Phase::Published;
        release();
    }
    AbortResult abort() noexcept override {
        if (phase_ == Phase::Published || phase_ == Phase::Aborted)
            return s_->current.stamp.valid ? AbortResult::PreviousViewPreserved : AbortResult::SessionInvalidated;
        const bool spill_ok = !spill_ || spill_->drain();
        const bool restore_ok = !restore_ || restore_->drain();
        try { s_->check(); } catch (...) { s_->current.stamp.valid = false; }
        bool rollback_ok = true;
        if (spill_ok && restore_ok && s_->current.stamp.valid && restore_may_have_written_ && rollback_) {
            try {
                rollback_->restore(rollback_spans_, s_->pools, s_->binding.compute, rollback_bytes_);
                rollback_ok = rollback_->drain();
                if (rollback_ok) ++s_->reuse.rollbacks;
            } catch (...) { rollback_ok = false; }
        }
        if (!spill_ok || !restore_ok || !rollback_ok) {
            s_->current.stamp.valid = false;
            // Leave reserved IDs held on a fatal runtime error, just like the
            // staging allocations: the owning engine must stop this session.
            reserved_.clear();
        }
        phase_ = Phase::Aborted;
        release();
        return s_->current.stamp.valid ? AbortResult::PreviousViewPreserved : AbortResult::SessionInvalidated;
    }
private:
    void release() noexcept {
        for (const auto& [g, b] : reserved_) s_->binding.manager->release_block(g, b);
        reserved_.clear();
        if (owns_pending_) { s_->pending = false; owns_pending_ = false; }
    }
};

NativeMemory::NativeMemory(MemoryBinding binding, const std::vector<rad::KVGroupInfo>& infos,
                          BackendIdentity identity, MemoryStamp stamp,
                          const std::vector<BlockDescriptor>& catalog, uint32_t block_tokens,
                          uint64_t max_pos, uint64_t host_capacity) {
    if (!binding.engine_lifetime || !binding.manager || !binding.compute || binding.device < 0 || binding.ranks != 1)
        fail(MemoryErrorCode::Unsupported, "native boundary needs a live single-rank engine owner and stream");
    if (!binding.manager->canonical(stamp.session.id))
        fail(MemoryErrorCode::Unsupported, "initial attachment requires canonical engine pages");
    state_ = std::make_shared<State>(std::move(binding), infos, std::move(identity), stamp,
                                     block_tokens, max_pos, host_capacity);
    auto& s = *state_;
    for (const auto& b : catalog) {
        if (!b.valid_tokens || b.valid_tokens > block_tokens ||
            (b.valid_tokens < block_tokens && b.original_start + b.valid_tokens != stamp.frontier))
            fail(MemoryErrorCode::InvalidPlan, "only the accepted history tail may be partial");
        s.current.blocks.push_back({b, 0});
        s.current.resident.push_back(b.id);
        auto& binding_pages = s.resident[b.id];
        for (const auto& t : s.tables) {
            const auto* p = s.binding.manager->plan(t.group);
            const uint64_t first = b.original_start / p->block_size;
            const uint64_t count = (uint64_t(b.valid_tokens) + p->block_size - 1) / p->block_size;
            if (first > t.blocks.size() || count > t.blocks.size() - first)
                fail(MemoryErrorCode::Stale, "canonical engine pages do not cover accepted history");
            binding_pages.push_back({t.group, uint64_t(s.binding.manager->live_blocks(t.group)),
                {t.blocks.begin() + first, t.blocks.begin() + first + count}});
        }
        if (b.valid_tokens == block_tokens) s.pages.spans(s.binding.manager->plans(), binding_pages);
    }
    s.visible = stamp.frontier;
    s.check();
    const auto quantum = s.descriptor.capabilities.page_tokens;
    const auto rounded = stamp.frontier / quantum * quantum + (stamp.frontier % quantum ? quantum : 0);
    plan_working_set(s.descriptor, s.current, s.current.resident, {},
                     {rounded, 0, std::max<uint64_t>(rounded, quantum), host_capacity});
}
NativeMemory::~NativeMemory() = default;
MemoryDescriptor NativeMemory::descriptor() const { return state_->descriptor; }
MemorySnapshot NativeMemory::snapshot() const { state_->check(); return state_->current; }
MemoryStamp NativeMemory::stamp() const noexcept { return state_->current.stamp; }
NativeMemory::ReuseStats NativeMemory::reuse_stats() const noexcept { return state_->reuse; }
std::unique_ptr<MemoryTransfer> NativeMemory::prepare(const WorkingSetPlan& plan) {
    auto tx = std::make_unique<Transaction>(state_, plan);
    tx->prepare();
    return tx;
}
void NativeMemory::accept_append(uint64_t frontier, const std::vector<BlockDescriptor>& added) {
    auto& s = *state_;
    if (s.pending) fail(MemoryErrorCode::Pending, "cannot accept tokens during a native transfer");
    try {
        auto& kv = *s.binding.manager;
        const auto old = s.current.stamp;
        const auto block = s.descriptor.layout.block_tokens;
        if (!old.valid || frontier <= old.frontier || frontier > s.descriptor.capabilities.max_position ||
            old.history_revision == UINT64_MAX || old.view_version == UINT64_MAX ||
            !kv.has_sequence(s.native_sequence) || kv.seq_tokens(s.native_sequence) != int64_t(frontier))
            fail(MemoryErrorCode::Stale, "accepted append does not match engine frontier");
        MemorySnapshot next = s.current;
        auto bindings = s.resident;
        auto tables = s.tables;
        const uint64_t offset = old.frontier - s.visible;
        for (auto& t : tables) {
            const auto& actual = kv.block_table(s.native_sequence, t.group);
            const uint64_t page = kv.plan(t.group)->block_size;
            if (kv.execution_offset(s.native_sequence, t.group) != int64_t(offset) ||
                actual.size() != (frontier - offset + page - 1) / page || actual.size() < t.blocks.size() ||
                !std::equal(t.blocks.begin(), t.blocks.end(), actual.begin()))
                fail(MemoryErrorCode::Stale, "accepted append changed existing native pages");
            t.blocks = actual;
        }
        uint64_t cursor = old.frontier / block * block;
        // Re-describe the formerly partial tail first. Its physical pages must
        // still be resident, and its immutable archive version must be absent.
        uint64_t previous_tail_version = 0, previous_tail_id = 0;
        if (old.frontier % block) {
            if (next.blocks.empty() || next.blocks.back().block.original_start != cursor ||
                next.blocks.back().host_version != 0 ||
                !bindings.count(next.blocks.back().block.id))
                fail(MemoryErrorCode::Stale, "mutable tail lost its resident ownership");
            const auto tail_id = next.blocks.back().block.id;
            previous_tail_id = tail_id;
            previous_tail_version = next.blocks.back().block.content_version;
            next.blocks.pop_back();
            next.resident.erase(std::remove(next.resident.begin(), next.resident.end(), tail_id), next.resident.end());
            bindings.erase(tail_id);
        }
        for (const auto& b : added) {
            if (!(b.session == old.session) || b.original_start != cursor ||
                !b.valid_tokens || b.valid_tokens > block ||
                b.content_version == 0 || cursor > frontier || b.valid_tokens > frontier - cursor ||
                (b.valid_tokens < block && b.valid_tokens != frontier - cursor) ||
                (previous_tail_version && (b.content_version <= previous_tail_version || b.id != previous_tail_id)) ||
                std::any_of(next.blocks.begin(), next.blocks.end(), [&](const auto& x) { return x.block.id == b.id; }))
                fail(MemoryErrorCode::InvalidPlan, "invalid accepted append catalog");
            previous_tail_version = 0;
            auto& group_pages = bindings[b.id];
            for (const auto& t : tables) {
                const uint64_t page = kv.plan(t.group)->block_size;
                if (cursor < offset) fail(MemoryErrorCode::Stale, "append tail precedes compact offset");
                const uint64_t first = (cursor - offset) / page;
                const uint64_t count = (uint64_t(b.valid_tokens) + page - 1) / page;
                if (first > t.blocks.size() || count > t.blocks.size() - first)
                    fail(MemoryErrorCode::Stale, "engine does not cover appended tail");
                group_pages.push_back({t.group, uint64_t(kv.live_blocks(t.group)),
                    {t.blocks.begin() + first, t.blocks.begin() + first + count}});
            }
            if (b.valid_tokens == block) s.pages.spans(kv.plans(), group_pages);
            next.blocks.push_back({b, 0}); next.resident.push_back(b.id);
            cursor += b.valid_tokens;
        }
        if (cursor != frontier) fail(MemoryErrorCode::InvalidPlan, "append catalog does not cover new history");
        next.stamp.frontier = frontier;
        ++next.stamp.history_revision;
        ++next.stamp.view_version;
        const uint64_t visible = frontier - offset;
        const auto page = s.descriptor.capabilities.page_tokens;
        const auto rounded = (visible + page - 1) / page * page;
        plan_working_set(s.descriptor, next, next.resident, {}, {rounded, 0, rounded, s.host_capacity});
        std::swap(s.current, next); s.resident.swap(bindings); s.tables.swap(tables); s.visible = visible;
        s.check();
    } catch (...) { s.current.stamp.valid = false; throw; }
}
void NativeMemory::rewind(uint64_t frontier) {
    auto& s = *state_;
    s.check();
    if (s.pending) fail(MemoryErrorCode::Pending, "cannot rewind a pending transaction");
    const auto old = s.current.stamp;
    if (frontier > old.frontier || frontier % s.descriptor.layout.block_tokens ||
        old.history_revision == UINT64_MAX || old.view_version == UINT64_MAX || old.execution_history == UINT64_MAX)
        fail(MemoryErrorCode::InvalidPlan, "invalid block-aligned replay boundary");
    auto next = s.current;
    auto bindings = s.resident;
    auto tables = s.tables;
    std::vector<uint64_t> dropped;
    for (const auto& b : next.blocks) if (b.block.original_start >= frontier) dropped.push_back(b.block.id);
    next.blocks.erase(std::remove_if(next.blocks.begin(), next.blocks.end(),
        [&](const auto& b) { return b.block.original_start >= frontier; }), next.blocks.end());
    for (auto id : dropped) {
        bindings.erase(id);
        next.resident.erase(std::remove(next.resident.begin(), next.resident.end(), id), next.resident.end());
    }
    uint64_t visible = 0;
    for (const auto& b : next.blocks) if (bindings.count(b.block.id)) visible += b.block.valid_tokens;
    for (auto& t : tables) {
        t.blocks.clear();
        for (auto id : next.resident) for (const auto& g : bindings.at(id))
            if (g.group == t.group) t.blocks.insert(t.blocks.end(), g.pages.begin(), g.pages.end());
    }
    next.stamp.frontier = frontier;
    ++next.stamp.history_revision; ++next.stamp.view_version; ++next.stamp.execution_history;
    if (s.binding.manager->rewind_full_tables(s.native_sequence, old.frontier, frontier, visible, tables) < 0)
        fail(MemoryErrorCode::Stale, "engine refused replay table publication");
    // All allocation/validation was above the atomic engine publication.
    for (auto id : dropped) s.archive.erase(id);
    std::swap(s.current, next); s.resident.swap(bindings); s.tables.swap(tables); s.visible = visible;
    s.check();
}
void NativeMemory::retain_view() {
    auto& s = *state_;
    s.check();
    if (s.pending || s.current.stamp.frontier % s.descriptor.layout.block_tokens)
        fail(MemoryErrorCode::InvalidPlan, "retained memory view requires a drained full-block boundary");
    s.retained = std::make_unique<State::RetainedView>(State::RetainedView{
        s.current, s.resident, s.tables, s.visible});
}
uint64_t NativeMemory::archive_bytes() const noexcept { return state_->archive.bytes(); }
bool NativeMemory::needs_unleased_pages(const std::vector<uint64_t>& selected,
                                      const std::vector<uint64_t>& mandatory, const MemoryBudget& budget) const {
    const auto& s = *state_;
    s.check();
    if (s.pending) fail(MemoryErrorCode::Pending, "pressure check during pending transfer");
    const auto plan = plan_working_set(s.descriptor, s.current, selected, mandatory, budget);
    const auto& kv = *s.binding.manager;
    for (const auto& t : s.tables) {
        const uint64_t page = kv.plan(t.group)->block_size;
        const uint64_t restore = plan.restore.size() * (s.descriptor.layout.block_tokens / page);
        const uint64_t tail = budget.reserve_tokens / page + (budget.reserve_tokens % page != 0);
        const uint64_t free = kv.free_blocks(t.group);
        uint64_t exclusive = 0;
        for (const auto& b : plan.evict) for (const auto& g : s.resident.at(b.id))
            if (g.group == t.group) for (auto id : g.pages)
                exclusive += kv.block_refcount(g.group, id) == 1;
        if (free < tail || restore > free - tail + exclusive) return true;
    }
    return false;
}
void NativeMemory::rebind_sequence(uint64_t sequence, bool relocated_prefix) {
    auto& s = *state_;
    auto& kv = *s.binding.manager;
    const auto& next = s.retained ? s.retained->snapshot : s.current;
    auto tables = s.retained ? s.retained->tables : s.tables;
    auto bindings = s.retained ? s.retained->resident : s.resident;
    const auto visible = s.retained ? s.retained->visible : s.visible;
    if (s.pending || !s.current.stamp.valid || !sequence ||
        s.current.stamp.view_version == UINT64_MAX || s.current.stamp.history_revision == UINT64_MAX ||
        s.current.stamp.execution_history == UINT64_MAX || !kv.has_sequence(sequence) ||
        kv.seq_tokens(sequence) != int64_t(next.stamp.frontier))
        fail(MemoryErrorCode::Stale, "restored sequence does not match retained frontier");
    if (relocated_prefix && !s.retained)
        fail(MemoryErrorCode::Stale, "relocation requires retained identity");
    if (relocated_prefix) for (auto& t : tables) {
        const auto& actual = kv.block_table(sequence, t.group);
        if (actual.size() != t.blocks.size()) fail(MemoryErrorCode::Stale, "relocated prefix page count changed");
        std::map<int32_t, int32_t> relocation;
        for (size_t i = 0; i < actual.size(); ++i)
            if (!relocation.emplace(t.blocks[i], actual[i]).second || kv.block_refcount(t.group, actual[i]) < 1)
                fail(MemoryErrorCode::Stale, "relocated prefix page ownership invalid");
        for (auto& [id, groups] : bindings) for (auto& group : groups) if (group.group == t.group)
            for (auto& page : group.pages) page = relocation.at(page);
        t.blocks = actual;
    }
    for (const auto& t : tables)
        if (kv.block_table(sequence, t.group) != t.blocks ||
            kv.execution_offset(sequence, t.group) != int64_t(next.stamp.frontier - visible))
            fail(MemoryErrorCode::Stale, "restored sequence does not own retained pages");
    auto slots = s.state_slots;
    for (auto& [group, slot] : slots) {
        slot = kv.state_slot(sequence, group);
        if (slot < 0) fail(MemoryErrorCode::Stale, "restored sequence has no recurrent slot");
    }
    s.native_sequence = sequence;
    s.state_slots.swap(slots);
    if (s.retained) {
        // PrefixLease has already restored native pages/state. Discard only
        // invalidated suffix archive entries, then restore its matching catalog.
        for (const auto& b : s.current.blocks)
            if (b.block.original_start >= next.stamp.frontier) s.archive.erase(b.block.id);
        auto& saved = *s.retained;
        saved.tables.swap(tables);
        saved.resident.swap(bindings);
        saved.snapshot.stamp.history_revision = s.current.stamp.history_revision + 1;
        saved.snapshot.stamp.view_version = s.current.stamp.view_version;
        saved.snapshot.stamp.execution_history = s.current.stamp.execution_history + 1;
        std::swap(s.current, saved.snapshot);
        s.resident.swap(saved.resident);
        s.tables.swap(saved.tables);
        s.visible = saved.visible;
        s.retained.reset();
    }
    ++s.current.stamp.view_version;
    s.check();
}
void NativeMemory::invalidate() noexcept { state_->current.stamp.valid = false; }
} // namespace kvmem::radiance
