#include "kvmem/query_policy.hpp"
#include "llama-kvmem-driver.h"
#include "llama-kvmem-execution.h"
#include "llama-kvmem-diag.h"
#include "kvmem/session_memory.hpp"
#include "common.h"
#include "log.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>

std::vector<llama_token> llama_driver_tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> out;
    if (n <= 0) {
        return out;
    }
    out.resize((size_t) n);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), n, add_special, true);
    return out;
}

static void multimodal_commit(LlamaEngineState & st, const std::vector<llama_token> & gen);
static void checkpoint_tier(LlamaEngineState & st, bool drain = false, bool force = false);

static void begin_history_round(LlamaEngineState & st) {
    if (st.kparams.lrfu) {
        const auto prefix = st.history_input_prompt && st.active_prompt
            ? st.active_prompt->common_prefix(*st.history_input_prompt) : 0;
        const auto count = kvmem::lrfu_prefix_points(st.history_points, prefix,
            st.history_input_end ? st.history_input_end : (st.active_prompt ? st.active_prompt->tokens.size() : 0));
        st.rollback_history = count ? st.history_points[count - 1].state : kvmem::LrfuState{};
        llama_kvmem_restore_history(st.rollback_history);
        kvmem_diag("KVMEM_LRFU_BEGIN parent=%llu inherited_inputs=%zu\n",
            (unsigned long long) st.rollback_history.round, count);
    }
    llama_kvmem_begin_retrieval_round();
}

// Exchange the active conversation's payload with `conv`. Called twice per
// switch: once to park the outgoing conversation, once to install the incoming
// one. Swapping is an involution, so calling it twice on the same entry undoes
// it, which is how a refused switch is rolled back.
void llama_driver_swap_conversation(LlamaEngineState & st, kvmem_conversation & conv) {
    checkpoint_tier(st, true);
    st.cached_tokens.swap(conv.cached_tokens);
    st.cached_prompt.swap(conv.cached_prompt);
    st.history_input_prompt.swap(conv.history_input_prompt);
    st.history_points.swap(conv.history_points);
    st.mm_checkpoints.swap(conv.mm_checkpoints);
    st.mm_archive.swap(conv.mm_archive);
    std::swap(st.mm_live_row, conv.mm_live_row);
    st.mm_live_checkpoint.swap(conv.mm_live_checkpoint);
    st.mm_query.swap(conv.mm_query);
    std::swap(st.last_n_gen, conv.last_n_gen);
}

// Accounted host bytes of one conversation: the adapter's raw K/V store plus
// the driver-owned recurrent checkpoints and token history. RawKvStore walks
// blocks times layers under its own mutex, so this runs once per request, at
// the commit, and once more for an eviction's trace line. Selection reads the
// figure the last commit stored in the table instead of recomputing it.
static uint64_t conversation_bytes(const LlamaEngineState & st, int id) {
    const kvmem_store_table::entry * held = st.conv_table.find(id);
    if (!held) {
        return 0;
    }
    uint64_t bytes = st.kparams.enabled ? llama_kvmem_store_bytes(held->store_id) : 0;
    const auto entry = st.conv.find(id);
    if (entry == st.conv.end()) {
        return bytes;
    }
    const bool active = id == st.conv_active;
    const auto & checkpoints = active ? st.mm_checkpoints : entry->second.mm_checkpoints;
    std::set<const MultimodalCheckpointData *> unique;
    for (const auto & checkpoint : checkpoints) {
        if (checkpoint.data && unique.insert(checkpoint.data.get()).second) {
            bytes += checkpoint.data->resident_bytes();
        }
    }
    for (const auto & archived : (active ? st.mm_archive : entry->second.mm_archive)) {
        const auto & cp = archived.record;
        if (cp.data && unique.insert(cp.data.get()).second) bytes += cp.data->resident_bytes();
    }
    const auto & live = active ? st.mm_live_checkpoint : entry->second.mm_live_checkpoint;
    if (live && unique.insert(live.get()).second) bytes += live->resident_bytes();
    const auto & tokens = active ? st.cached_tokens : entry->second.cached_tokens;
    bytes += (uint64_t) tokens.capacity() * sizeof(llama_token);
    const auto & conv = entry->second;
    bytes += sizeof(kvmem_conversation) + conv.client_id.capacity();
    if (conv.payload) bytes += conv.payload->metadata_bytes();
    bytes += checkpoints.capacity()*sizeof(MultimodalCheckpoint);
    const auto & prompt = active ? st.cached_prompt : conv.cached_prompt;
    if (prompt) bytes += prompt->index_bytes();
    const auto & points = active ? st.history_points : conv.history_points;
    bytes += points.capacity()*sizeof(kvmem::LrfuPoint);
    for (const auto & point : points) bytes += point.state.scores->capacity()*sizeof(double);
    const auto & history_input = active ? st.history_input_prompt : conv.history_input_prompt;
    if (history_input && history_input != prompt) bytes += history_input->index_bytes();
    const auto & query = active ? st.mm_query : conv.mm_query;
    if (query) {
        bytes += sizeof(MultimodalQuery) + query->user.capacity() + query->state.count.capacity()*sizeof(uint32_t);
        bytes += query->state.rows.capacity()*sizeof(std::vector<float>);
        bytes += query->state.ranges.capacity()*sizeof(llama_kvmem_row_range);
        for (const auto & sum : query->state.rows) bytes += sum.capacity()*sizeof(float);
        if (query->prefix && query->prefix != prompt) bytes += query->prefix->index_bytes();
        for (const auto & media : query->media) bytes += sizeof(media) + media.second.capacity();
    }
    return bytes;
}

// Describe one live conversation for the selection policy, reading that
// conversation's own payload: LlamaEngineState's fields when it is the attached one,
// the parked entry otherwise.
static kvmem_store_match conversation_match(const LlamaEngineState & st, int id, const kvmem_prompt & prompt) {
    kvmem_store_match match;
    match.id = id;
    const kvmem_store_table::entry * held = st.conv_table.find(id);
    const auto entry = st.conv.find(id);
    if (!held || entry == st.conv.end()) {
        return match;
    }
    const kvmem_conversation & conv = entry->second;
    if (conv.payload && conv.payload->invalid) return match;
    const bool active = id == st.conv_active;
    match.used = held->used;
    // The accounted figure from that conversation's last commit, not a fresh
    // walk: llama_kvmem_store_bytes takes the store mutex and walks blocks
    // times layers, and this runs once per live store on the prefill latency
    // path. Only the attached conversation's footprint can have moved since,
    // and conversation_commit refreshes exactly that one.
    match.bytes = held->bytes;
    match.client_id = conv.client_id;
    match.allow_bound_edit = (active ? st.mm_archive : conv.mm_archive).policy().enabled;
    match.rows = (int) (active ? st.cached_tokens.size() : conv.cached_tokens.size());
    match.last_n_gen = active ? st.last_n_gen : conv.last_n_gen;
    const auto & cached = active ? st.cached_prompt : conv.cached_prompt;
    match.lcp = cached ? (int) prompt.common_prefix(*cached) : 0;
    match.live_row = active ? st.mm_live_row : conv.mm_live_row;
    for (const auto & checkpoint : (active ? st.mm_checkpoints : conv.mm_checkpoints)) {
        match.ckpt_rows.push_back(checkpoint.row);
    }
    for (const auto & e : (active ? st.mm_archive : conv.mm_archive))
        if (e.info.identity) match.ckpt_rows.push_back(e.record.row);
    return match;
}

// Least recently used conversation that is not the attached one.
static int conversation_lru_victim(const LlamaEngineState & st) {
    for (int id : st.conv_table.lru_order()) {
        if (id != st.conv_active) {
            return id;
        }
    }
    return -1;
}

// Frees one parked conversation and reports whether it actually went. Never
// the attached one: llama_memory_clear reaches the attached host store, so an
// inactive store is released through the adapter handle instead, and a table
// row dropped without that release would leave a host store nothing can reach.
static bool conversation_evict(LlamaEngineState & st, int id, const char * reason) {
    if (id == st.conv_active) {
        return false;
    }
    const kvmem_store_table::entry * held = st.conv_table.find(id);
    if (!held) {
        return false;
    }
    const uint32_t rows = st.kparams.enabled ? llama_kvmem_store_rows(held->store_id) : 0;
    const uint64_t bytes = conversation_bytes(st, id);
    const int32_t store_id = held->store_id;
    // Partial deletion must never leave a selectable cache with missing KV.
    if (st.conv.at(id).payload) st.conv.at(id).payload->invalid = true;
    if (st.session_files && !st.session_files->erase(id)) {
        ++st.conv_counts.disk_errors;
        LOG_WRN("srv    KVMEM cannot remove session file id=%d; retaining quota charge\n", id);
        return false;
    }
    if (st.kparams.enabled && !llama_kvmem_store_destroy(store_id)) {
        // Dropping the row anyway would leave the bundle in the adapter's pool
        // with nothing naming it: one runtime with its pinned arena and two
        // host mirrors, leaked for the life of the process. Report the refusal
        // so the caller's skip path runs.
        LOG_WRN("srv    KVMEM conv=%d store=%d refused destroy; row kept\n", id, (int) store_id);
        return false;
    }
    st.conv_table.erase(id);
    st.conv.erase(id);
    ++st.conv_counts.evictions;
    kvmem_diag("KVMEM_TRACE store_evict id=%d rows=%u bytes=%llu reason=%s\n",
            id, rows, (unsigned long long) bytes, reason);
    return true;
}

// Drop a parked conversation's payload, keeping only its identity. The store
// it described no longer holds those rows, so the next request that selects it
// must take an ordinary cache miss rather than resume a checkpoint against KV
// that was never restored. Assigning a default entry is deliberate: it keeps
// this complete as kvmem_conversation grows.
void llama_driver_drop_conversation(kvmem_conversation & conv) {
    kvmem_conversation kept;
    kept.client_id = conv.client_id;
    conv = std::move(kept);
}

void llama_driver_publish_conversations(LlamaEngineState & st) {
    if (st.conv_limits.max_stores <= 1) {
        return;
    }
    st.conv_counts.count = st.conv_table.count();
    st.conv_counts.max = st.conv_limits.max_stores;
    st.conv_counts.active = st.conv_active;
    st.conv_counts.bytes = st.conv_table.bytes_total();
    st.conv_counts.bytes_max = st.conv_limits.max_bytes;
    if (st.session_files) {
        st.conv_counts.disk_bytes = st.session_files->bytes();
        st.conv_counts.disk_bytes_max = st.session_files->limit();
    }
    st.conv_stats.publish(st.conv_counts);
}

void llama_driver_clear(LlamaEngineState & st);


// Included after LlamaEngineState and conversation bookkeeping. Frozen allocations
// stay in place; failures leave resumable RAM/disk manifests, never attached
// partial stores. No whole-session serialization buffer is allocated.
void llama_driver_conversation_buffers(kvmem_conversation & conv, std::vector<kvmem::SnapshotBuffer> & buffers) {
    std::set<const MultimodalCheckpointData *> unique;
    auto checkpoint = [&](const std::shared_ptr<const MultimodalCheckpointData> & p) {
        if (!p || !unique.insert(p.get()).second || p->shared.load()) return;
        // Created mutable by multimodal_checkpoint(). Live/rollback aliases
        // were cleared after parking. Shared checkpoints stay charged in RAM
        // (or retain their existing cold extent) until their last owner drops
        // them; a per-session manifest cannot clear another owner's vectors.
        auto & data = *const_cast<MultimodalCheckpointData *>(p.get());
        auto add = [&](auto & vector) {
            if (!vector.capacity()) return;
            auto b = kvmem::SnapshotBuffer::bind(vector);
            if (data.accounting) {
                b.accounting = data.accounting.get();
                b.reserve = [](void * p, uint64_t bytes) {
                    auto& a=*static_cast<MultimodalCheckpointAccounting*>(p);
                    return a.reserve(size_t(bytes),a.limit);
                };
                b.account = [](void * p, int64_t delta) {
                    auto & a = *static_cast<MultimodalCheckpointAccounting *>(p);
                    if (delta < 0) a.live_bytes -= size_t(-delta); else a.add(size_t(delta));
                };
            }
            buffers.push_back(b);
        };
        add(data.recurrent); add(data.draft_carry); add(data.tail_mean.values);
    };
    for (const auto & c : conv.mm_checkpoints) checkpoint(c.data);
    for (const auto & e : conv.mm_archive) if (e.info.identity) checkpoint(e.record.data);
    checkpoint(conv.mm_live_checkpoint);
}

static kvmem_session_payload & session_freeze(LlamaEngineState & st, int id) {
    auto & conv = st.conv.at(id);
    if (conv.payload) return *conv.payload;
    if (id == st.conv_active) throw std::runtime_error("cannot freeze active session");
    std::vector<kvmem::SnapshotBuffer> buffers;
    llama_driver_conversation_buffers(conv, buffers);
    const int32_t store = st.conv_table.find(id)->store_id;
    llama_kvmem_store_freeze(store, buffers);
    try {
        conv.payload = std::make_unique<kvmem_session_payload>(id, ++st.session_generation, buffers);
    } catch (...) { llama_kvmem_store_thaw(store); throw; }
    conv.cold = true; // also covers partially migrated and frozen, still-hot stores
    return *conv.payload;
}

static void session_refresh_bytes(LlamaEngineState & st) {
    for (const auto & entry : st.conv_table.entries())
        st.conv_table.set_bytes(entry.id, conversation_bytes(st, entry.id));
}

static void session_warn_budget(LlamaEngineState & st) {
    const bool over = st.conv_limits.max_bytes && st.conv_table.bytes_total() > st.conv_limits.max_bytes;
    if (over && !st.conv_budget_warned)
        LOG_WRN("srv    KVMEM session RAM soft limit exceeded (bytes=%llu cap=%llu); retaining active KV and cold metadata\n",
            (unsigned long long)st.conv_table.bytes_total(), (unsigned long long)st.conv_limits.max_bytes);
    st.conv_budget_warned = over;
}

// Plan the final placement BEFORE choosing a transfer order or applying LRU.
// Retention uses measured RAM and exact restore sizes, just like RAM-only mode.
// Worst-case future generation estimates must not evict reusable sessions.
static void session_balance(LlamaEngineState & st, int target) {
    const auto started = std::chrono::steady_clock::now();
    auto & files = *st.session_files;
    auto * incoming = target >= 0 && target != st.conv_active ? st.conv.at(target).payload.get() : nullptr;
    session_refresh_bytes(st);
    const uint64_t restore = incoming ? incoming->restore_bytes() : 0;
    uint64_t projected = st.conv_table.bytes_total() + restore;
    const uint64_t ram_free = kvmem::session_memory_available();
    uint64_t freed_ram = 0;
    std::vector<kvmem_session_payload *> outgoing;
    for (int id : st.conv_table.lru_order()) {
        if (id == target || id == st.conv_active) continue;
        const bool soft_pressure = st.conv_limits.max_bytes && projected > st.conv_limits.max_bytes;
        if (!soft_pressure && restore <= ram_free + freed_ram) break;
        auto & p = session_freeze(st, id);
        if (p.invalid || !p.ram_bytes()) continue;
        projected -= std::min(projected, p.ram_bytes()); freed_ram += p.ram_bytes();
        outgoing.push_back(&p);
    }
    // Unfinished temp files still have their original RAM source. A failed
    // cleanup remains an I/O error, never an eviction request.
    for (auto * p : outgoing) for (uint32_t i = 0; i < p->chunks.size(); ++i)
        if (p->chunks[i].in_ram && files.contains_chunk(p->id, i) && !files.ready(p->id, i) && !files.erase_chunk(p->id, i))
            throw std::runtime_error("cannot remove unfinished session snapshot");

    const uint64_t disk_available = files.available();
    const uint64_t disk_limit = std::min(files.limit(), files.bytes() + std::min(disk_available, UINT64_MAX - files.bytes()));
    uint64_t final_disk = files.bytes() - (incoming ? files.session_bytes(target) : 0);
    for (auto * p : outgoing) final_disk += p->disk_bytes() - files.session_bytes(p->id);
    uint64_t disk_used = files.bytes(), evicted_ram = 0;
    std::vector<int> evictions;
    // Only FINAL capacity pressure permits LRU. Credit target files that will
    // be freed by restore, including the RAM 10 / target 15 / disk quota 20 case.
    for (int id : st.conv_table.lru_order()) {
        if (final_disk <= disk_limit) break;
        if (id == target || id == st.conv_active) continue;
        const auto it = std::find_if(outgoing.begin(), outgoing.end(), [&](auto * p) { return p->id == id; });
        const uint64_t native_disk = st.kparams.payload_disk_bytes
            ? llama_kvmem_store_payload_bytes(st.conv_table.find(id)->store_id, true) : 0;
        const uint64_t saved = (it != outgoing.end() ? (*it)->disk_bytes() : files.session_bytes(id)) + native_disk;
        if (!saved) continue;
        final_disk -= saved; disk_used -= files.session_bytes(id) + native_disk;
        evicted_ram += conversation_bytes(st, id); evictions.push_back(id);
        if (it != outgoing.end()) outgoing.erase(it);
    }
    if (final_disk > disk_limit) throw std::runtime_error("cannot free final session disk capacity");
    const auto plan = kvmem_plan_session_transfer(files, incoming, outgoing,
        kvmem::session_memory_available() + evicted_ram, disk_used, disk_limit);
    const char * route = plan.route == kvmem_session_plan::path::ram_first ? "ram_first" :
        plan.route == kvmem_session_plan::path::exchange ? "exchange" : "disk_first";
    kvmem_diag("KVMEM_TRACE session_transfer target=%d path=%s moves=%zu peak_disk=%llu final_disk=%llu\n",
        target, route, plan.moves.size(), (unsigned long long)plan.peak_disk, (unsigned long long)final_disk);
    for (int id : evictions)
        if (!conversation_evict(st, id, "disk_lru")) throw std::runtime_error("cannot remove LRU session snapshot");
    try {
        kvmem_execute_session_transfer(files, plan, kvmem::session_memory_available);
    } catch (...) { session_refresh_bytes(st); throw; }
    session_refresh_bytes(st);
    for (auto * p : outgoing) {
        ++st.conv_counts.spills;
        kvmem_diag("KVMEM_TRACE session_spill id=%d disk_bytes=%llu ram_bytes=%llu ms=%.2f\n", p->id,
            (unsigned long long)files.session_bytes(p->id), (unsigned long long)st.conv_table.find(p->id)->bytes,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
    }
    if (incoming) {
        if (!incoming->complete() || files.contains(target)) throw std::runtime_error("incomplete session restore");
        llama_kvmem_store_thaw(st.conv_table.find(target)->store_id);
        auto & conv = st.conv.at(target); conv.payload.reset(); conv.cold = false;
        st.conv_table.set_bytes(target, conversation_bytes(st, target));
        ++st.conv_counts.restores;
        kvmem_diag("KVMEM_TRACE session_restore id=%d rows=%u ms=%.2f\n", target, conv.stored,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
    }
}

// Post-request enforcement must not invalidate the completed active cache.
static void session_make_room(LlamaEngineState & st, int target) {
    try { session_balance(st, target); }
    catch (const std::exception & e) {
        ++st.conv_counts.disk_errors; session_refresh_bytes(st);
        LOG_WRN("srv    KVMEM idle session migration deferred: %s\n", e.what());
    }
    session_warn_budget(st);
}

void llama_driver_begin_disk_request(LlamaEngineState & st, const kvmem_prompt & prompt,
                                  const std::string & client_id, int predict) {
    const uint64_t rows = uint64_t(prompt.tokens.size()) + uint64_t(std::max(0, predict)) +
        (st.spec.ok ? uint64_t(std::max(0, st.spec_n_max)) + 1 : 0);
    if (rows > UINT32_MAX) throw std::invalid_argument("session token count overflow");
    const uint64_t reserve = llama_kvmem_store_capacity(uint32_t(rows));
    std::vector<kvmem_store_match> matches;
    for (const auto & entry : st.conv_table.entries()) matches.push_back(conversation_match(st, entry.id, prompt));
    auto limits = st.conv_limits; limits.max_bytes = 0; limits.attached = st.conv_active;
    const auto selection = kvmem_store_select(matches, int(prompt.tokens.size())-(st.spec.ok ? 1 : 0), st.spec.ok, client_id, limits);
    int target = selection.id;
    if (target != st.conv_active && st.conv_active >= 0) {
        if (!st.mm_committed || st.conv_table.find(st.conv_active)->store_id != llama_kvmem_store_current())
            throw std::runtime_error("session switch requires a committed active conversation");
        const int outgoing = st.conv_active;
        llama_driver_swap_conversation(st, st.conv.at(outgoing));
        if (!llama_kvmem_store_park()) {
            llama_driver_swap_conversation(st, st.conv.at(outgoing));
            if (!llama_kvmem_store_n_tokens() && !st.cached_tokens.empty()) llama_driver_clear(st);
            throw std::runtime_error("could not park active session");
        }
        st.conv_active = -1;
        st.mm_live_checkpoint.reset(); st.mm_rollback.reset(); st.mm_rollback_prompt.reset(); st.mm_pending_query.reset();
        if (!llama_kvmem_store_rows(st.conv_table.find(outgoing)->store_id)) llama_driver_drop_conversation(st.conv.at(outgoing));
        st.conv_table.set_bytes(outgoing, conversation_bytes(st, outgoing));
    }
    for (int victim : st.conv_table.lru_order()) {
        if (st.conv_table.count() + (target < 0 ? 1 : 0) <= st.conv_limits.max_stores) break;
        if (victim != target && victim != st.conv_active) conversation_evict(st, victim, "lru");
    }
    if (st.conv_table.count() + (target < 0 ? 1 : 0) > st.conv_limits.max_stores)
        throw std::runtime_error("cannot free session count capacity");
    try {
        // A hot/fresh request does not depend on optional idle demotion. The
        // RAM cap stays soft even when that background retention work fails.
        if (target < 0 || !st.conv.at(target).payload) session_make_room(st, target);
        else session_balance(st, target);
    } catch (const kvmem_session_corrupt & e) {
        ++st.conv_counts.disk_errors;
        LOG_WRN("srv    KVMEM session id=%d corrupt: %s; cache miss\n", target, e.what());
        if (target < 0 || !conversation_evict(st, target, "invalid_snapshot"))
            throw std::runtime_error("cannot remove invalid session snapshot");
        target = -1;
        session_balance(st, target);
    } catch (...) {
        ++st.conv_counts.disk_errors; session_refresh_bytes(st); throw;
    }
    if (target < 0) {
        const int32_t store = llama_kvmem_store_create();
        if (store < 0) throw std::runtime_error("cannot allocate session store");
        target = st.conv_table.add(store); st.conv.emplace(target, kvmem_conversation{});
    }
    if (target != st.conv_active) {
        const int32_t store = st.conv_table.find(target)->store_id;
        const bool restored = llama_kvmem_store_switch(store);
        if (llama_kvmem_store_current() != store) throw std::runtime_error("cannot attach session store");
        llama_driver_swap_conversation(st, st.conv.at(target)); st.conv_active = target;
        st.mm_live_checkpoint.reset();
        if (!restored) llama_driver_clear(st);
        ++st.conv_counts.switches;
    }
    st.conv_table.touch(target, ++st.conv_clock);
    if (selection.id == target) ++st.conv_counts.extends; else ++st.conv_counts.forks;
    llama_driver_publish_conversations(st);
    kvmem_diag("KVMEM_TRACE session_select id=%d keep=%d reserve=%llu\n", target,
        selection.id == target ? selection.keep : 0, (unsigned long long)reserve);
}

void llama_driver_check_payload_budget(LlamaEngineState & st, const kvmem_prompt & prompt, int predict) {
    if (!st.kparams.payload_disk_bytes) return;
    const uint64_t rows = uint64_t(prompt.tokens.size()) + uint64_t(std::max(0, predict)) +
        (st.spec.ok ? uint64_t(std::max(0, st.spec_n_max)) + 1 : 0);
    if (rows > UINT32_MAX) throw std::invalid_argument("native KV token count overflow");
    const auto required = llama_kvmem_store_payload_capacity(uint32_t(rows));
    const auto budget = llama_kvmem_payload_budget();
    if (required > budget)
        throw std::invalid_argument("request exceeds the combined native KV RAM/SSD history budget");
    const auto fits = [&] {
        const auto physical = llama_kvmem_execution_storage_stats(st.execution.get());
        const auto used = physical.host_bytes + physical.disk_bytes;
        const auto own = llama_kvmem_store_payload_bytes(llama_kvmem_store_current(), false);
        if (own > used) throw std::logic_error("native KV owner exceeds shared account");
        return used - own <= budget && required <= budget - (used - own);
    };
    // The request's history reservation includes its output bound. Retained state
    // attachments count in D too; reclaim whole idle owners before new KV writes.
    for (int victim : st.conv_table.lru_order()) {
        if (fits()) return;
        if (victim != st.conv_active && !conversation_evict(st, victim, "payload_capacity"))
            throw std::runtime_error("cannot reclaim idle native KV capacity");
    }
    if (!fits()) throw std::runtime_error("native KV capacity is still held by another owner");
}


// Runs when the conversation's footprint is final for the committed turn.
static void conversation_enforce_budget(LlamaEngineState & st) {
    if (st.conv_limits.max_stores <= 1) {
        return;
    }
    if (st.session_files) {
        session_make_room(st, st.conv_active);
        return;
    }
    while (st.conv_table.count() > st.conv_limits.max_stores) {
        const int victim = conversation_lru_victim(st);
        if (victim < 0 || !conversation_evict(st, victim, "lru")) {
            break;
        }
    }
    while (st.conv_limits.max_bytes != 0 && st.conv_table.bytes_total() > st.conv_limits.max_bytes) {
        const int victim = conversation_lru_victim(st);
        if (victim < 0) {
            // The attached conversation alone exceeds the cap. Report it and
            // serve the request: that reproduces today's uncapped single-store
            // behavior instead of failing a request the server would serve.
            if (!st.conv_budget_warned) {
                st.conv_budget_warned = true;
                LOG_WRN("srv    KVMEM one conversation exceeds --kvmem-conversations-gb (bytes=%llu cap=%llu); nothing evicted\n",
                        (unsigned long long) st.conv_table.bytes_total(),
                        (unsigned long long) st.conv_limits.max_bytes);
            }
            break;
        }
        if (!conversation_evict(st, victim, "bytes")) {
            break;
        }
    }
}

static void conversation_commit(LlamaEngineState & st, uint32_t stored) {
    if (st.conv_limits.max_stores <= 1) {
        return;
    }
    const auto entry = st.conv.find(st.conv_active);
    if (entry == st.conv.end()) {
        return;
    }
    if (st.session_files) {
        if (st.cached_prompt && st.cached_prompt->has_media()) st.cached_prompt = st.cached_prompt->cache_index();
        if (st.mm_query && st.mm_query->prefix && st.mm_query->prefix->has_media()) {
            auto query = std::make_shared<MultimodalQuery>(*st.mm_query);
            query->prefix = query->prefix->cache_index(); st.mm_query = std::move(query);
        }
    }
    entry->second.stored = stored;
    if (!st.turn_conversation_id.empty()) {
        // One id names one conversation: a client that reuses an id after its
        // own context was compacted must not leave two stores answering to it.
        for (auto & other : st.conv) {
            if (other.first != st.conv_active && other.second.client_id == st.turn_conversation_id) {
                other.second.client_id.clear();
            }
        }
        entry->second.client_id = st.turn_conversation_id;
    }
    st.conv_table.touch(st.conv_active, ++st.conv_clock);
    st.conv_table.set_bytes(st.conv_active, conversation_bytes(st, st.conv_active));
    conversation_enforce_budget(st);
    llama_driver_publish_conversations(st);
}

void llama_driver_clear(LlamaEngineState & st) {
    llama_memory_t mem = llama_get_memory(st.ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    if (st.spec.ctx_dft) {
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            llama_memory_clear(md, true);
        }
    }
    st.cached_tokens.clear();
    st.cached_prompt.reset();
    st.history_input_prompt.reset();
    st.history_points.clear();
    st.rollback_history = {};
    st.mm_checkpoints.clear();
    st.mm_archive.clear();
    st.mm_live_row = 0;
    st.mm_live_checkpoint.reset();
    st.mm_query.reset();
    st.mm_pending_query.reset();
    if (st.spec.ok) {
        std::vector<uint8_t> carry;
        common_speculative_get_state(st.spec.spec, 0, carry);
        std::fill(carry.begin(), carry.end(), 0);
        common_speculative_set_state(st.spec.spec, 0, carry);
    }
    st.last_n_gen = 0;
    // Clears the attached conversation only: llama_memory_clear reaches the
    // host store that is bound right now, and every st.* field above is that
    // conversation's payload.
    if (!st.conv.empty()) {
        const auto entry = st.conv.find(st.conv_active);
        if (entry != st.conv.end()) {
            entry->second.stored = 0;
        }
        ++st.conv_counts.resets;
    }
}

// Pick the conversation this request continues and attach its host store.
// Called once per request, under the inference lock, after the prompt is
// parsed and validated and before anything reads or writes KVMem state.
//
// With --kvmem-conversations absent this returns before touching anything:
// no table, no bookkeeping, no store switch, no policy, no trace. Default
// behavior is then identical by construction rather than by argument.
void llama_driver_begin_request(LlamaEngineState & st, const kvmem_prompt & prompt,
                                       const std::string & client_id) {
    if (st.conv_limits.max_stores <= 1) {
        return;
    }
    if (st.conv.find(st.conv_active) == st.conv.end()) {
        return; // never armed, or arming failed at startup
    }
    const int eval_end = (int) prompt.tokens.size() - (st.spec.ok ? 1 : 0);
    std::vector<kvmem_store_match> matches;
    matches.reserve(st.conv_table.entries().size());
    for (const auto & held : st.conv_table.entries()) {
        matches.push_back(conversation_match(st, held.id, prompt));
    }
    // cache_reset is deliberately not a separate action: the policy still maps
    // the request to a conversation, and the fork path below clears that one
    // store. One client resetting its own history must not wipe another
    // client's store.
    kvmem_store_limits limits = st.conv_limits;
    limits.attached = st.conv_active;
    const kvmem_store_plan plan = kvmem_store_select(matches, eval_end, st.spec.ok,
            client_id, limits);
    const bool allocating = plan.action == kvmem_store_action::fresh && plan.id < 0;
    // Two of the conditions the switch needs are already knowable: the
    // previous request's rows must be committed, and conv_active must name the
    // store the adapter actually has attached. Check them before the plan is
    // executed, because a refusal after the fact would have destroyed an LRU
    // victim and created a store for a conversation the request then does not
    // move to. resolve() is pure and plan.evict never names plan.id, so asking
    // it here gives the same answer it gives below.
    const kvmem_store_table::entry * attached = st.conv_table.find(st.conv_active);
    const int32_t attached_store = attached ? attached->store_id : -1;
    const bool would_switch = allocating || st.conv_table.resolve(plan) != st.conv_active;
    const bool refused = would_switch &&
            (!st.mm_committed || attached_store != llama_kvmem_store_current());
    if (refused) {
        LOG_WRN("srv    KVMEM store switch refused action=%s conv=%d committed=%d parked=%d active=%d\n",
                kvmem_store_action_name(plan.action), st.conv_table.resolve(plan),
                (int) st.mm_committed, (int) attached_store, (int) llama_kvmem_store_current());
    }
    if (!refused) {
        const int room = (int) matches.size() - st.conv_limits.max_stores + (allocating ? 1 : 0);
        int evicted = 0;
        for (int id : plan.evict) {
            if (!conversation_evict(st, id, evicted < room ? "lru" : "bytes")) {
                // The policy excludes both the target and the attached store, so
                // this is unreachable. Never fall back to dropping the row: the
                // adapter handle would survive with nothing naming it.
                LOG_WRN("srv    KVMEM eviction plan named conv=%d, which cannot be released; skipped\n", id);
                continue;
            }
            ++evicted;
        }
    }
    // Eviction is executed above, one entry at a time, so resolve() only names
    // the target and rejects a stale id from an earlier plan. A refused plan
    // stays on the attached conversation and neither evicts nor allocates; the
    // caps it declined to enforce are enforced again at the next turn that
    // commits. That is not necessarily this one: conversation_commit() runs
    // only from llama_driver_commit(), so a turn that fails after the mapping never
    // re-measures, and the byte overage stays until some later turn commits.
    int target = refused ? st.conv_active : st.conv_table.resolve(plan);
    // Every way this request can end up on the attached conversation after
    // planning another one. None of them forks, parks or creates anything, so
    // none of them may be counted as a fork.
    bool fell_back = refused;
    bool force_reset = false;
    if (!refused && allocating) {
        const int32_t store_id = llama_kvmem_store_create();
        if (store_id >= 0) {
            target = st.conv_table.add(store_id);
            st.conv.emplace(target, kvmem_conversation{});
        } else {
            // No further host store available. Reuse the least recently used
            // one, cleared below once the switch has actually attached it.
            target = conversation_lru_victim(st);
            force_reset = target >= 0;
        }
    }
    if (target < 0 || st.conv.find(target) == st.conv.end()) {
        target = st.conv_active; // stale plan id: fall back to today's path
        force_reset = false;
        fell_back = true;
    }
    bool switched = false;
    bool restaged = false;
    if (target != st.conv_active) {
        const int parked_id = st.conv_active;
        const kvmem_store_table::entry * parked = st.conv_table.find(parked_id);
        const int32_t parked_store = parked ? parked->store_id : -1;
        const kvmem_store_table::entry * held = st.conv_table.find(target);
        const auto outgoing = st.conv.find(parked_id);
        const auto incoming = st.conv.find(target);
        // st.mm_committed and the conv_active/adapter-active agreement were
        // checked above, before the plan was executed, and nothing since can
        // have changed the adapter's active store: store_create appends and
        // store_destroy refuses the active id. What is left to check is a
        // table entry or a payload row this switch needs and cannot find. On
        // any of those the switch would report success without moving anything
        // (llama_kvmem_store_switch short-circuits a switch to the active
        // store) and both failure detectors below would read clean while this
        // conversation decoded against another one's KV.
        if (!held || outgoing == st.conv.end() || incoming == st.conv.end()) {
            // A handle the adapter does not know, or a row the table and the
            // payload map disagree about. Stay where we are rather than switch
            // on ambiguous state.
            LOG_WRN("srv    KVMEM store switch refused conv=%d held=%d outgoing=%d incoming=%d\n",
                    target, (int) (held != nullptr), (int) (outgoing != st.conv.end()),
                    (int) (incoming != st.conv.end()));
            target = st.conv_active;
            force_reset = false;
            fell_back = true;
        } else {
            llama_driver_swap_conversation(st, outgoing->second);
            restaged = llama_kvmem_store_switch(held->store_id);
            if (llama_kvmem_store_current() == held->store_id) {
                llama_driver_swap_conversation(st, incoming->second);
                st.conv_active = target;
                // The context's recurrent state still belongs to the previous
                // conversation, so multimodal_restore's live short-circuit
                // must not skip the restore (llama-kvmem-driver.cpp, checkpoint restore).
                st.mm_live_checkpoint.reset();
                switched = true;
                ++st.conv_counts.switches;
                if (!restaged && (st.mm_live_row > 0 || !st.cached_tokens.empty())) {
                    // Attached but holding no rows: either this store was
                    // already empty or the adapter could not rebuild its
                    // working set from host RAM. The payload would claim rows
                    // the KV no longer has, so drop it and let prefill take
                    // its ordinary cache-miss path.
                    kvmem_diag("KVMEM_TRACE store_stale id=%d rows=%d reason=working_set_not_rebuilt\n",
                            target, st.mm_live_row);
                    llama_driver_clear(st);
                    force_reset = false; // already empty, and resets count once
                }
                // The switch clears the outgoing store when the adapter cannot
                // drain it safely, and the bool above describes the incoming
                // store only. Cross-check what the parked handle still holds
                // instead of trusting it: a payload claiming rows its store no
                // longer has would resume a checkpoint against KV that was
                // never restored, and nothing later reconciles the two.
                if (parked_store >= 0 && llama_kvmem_store_rows(parked_store) == 0 &&
                    !outgoing->second.cached_tokens.empty()) {
                    kvmem_diag("KVMEM_TRACE store_wiped id=%d rows=%d reason=outgoing_not_drainable\n",
                            parked_id, (int) outgoing->second.cached_tokens.size());
                    llama_driver_drop_conversation(outgoing->second);
                }
            } else {
                // Nothing moved. Put the previous conversation back and let
                // prefill decide against it, as a single-store server would.
                llama_driver_swap_conversation(st, outgoing->second);
                LOG_WRN("srv    KVMEM store switch failed conv=%d store=%d\n",
                        target, (int) held->store_id);
                // The same cross-check the success branch runs, for the same
                // reason: swap_conv() clears the outgoing store before the
                // attach whenever it cannot be drained, so a throw out of the
                // attach leaves this branch restoring a payload that claims
                // rows the store no longer has. The outgoing conversation is
                // the attached one again here, so its payload lives in st.*.
                if (parked_store >= 0 && llama_kvmem_store_rows(parked_store) == 0 &&
                    !st.cached_tokens.empty()) {
                    kvmem_diag("KVMEM_TRACE store_wiped id=%d rows=%d reason=switch_failed\n",
                            parked_id, (int) st.cached_tokens.size());
                    llama_driver_clear(st);
                }
                target = st.conv_active;
                force_reset = false;
                fell_back = true;
            }
        }
    }
    if (force_reset) {
        // Clear the reused store and its payload here so eviction accounting
        // is complete before the unrelated prompt begins prefill.
        kvmem_diag("KVMEM_TRACE store_reuse id=%d reason=no_store_available\n", target);
        llama_driver_clear(st);
    }
    st.conv_table.touch(target, ++st.conv_clock);
    if (fell_back) {
        ++st.conv_counts.refusals;
    } else if (plan.action == kvmem_store_action::extend && target == plan.id) {
        ++st.conv_counts.extends;
    } else {
        ++st.conv_counts.forks;
    }
    llama_driver_publish_conversations(st);
    kvmem_diag("KVMEM_TRACE store_select action=%s id=%d lcp=%d keep=%d stores=%zu "
            "bytes=%llu reason=%s switched=%d restaged=%d\n",
            kvmem_store_action_name(plan.action), target, plan.lcp, plan.keep,
            st.conv_table.entries().size(), (unsigned long long) st.conv_table.bytes_total(),
            plan.reason, (int) switched, (int) restaged);
}

void llama_driver_commit(LlamaEngineState & st, const std::vector<llama_token> & prompt,
                          const std::vector<llama_token> & gen) {
    llama_kvmem_attention_finish(true);
    const auto history = llama_kvmem_get_history();
    if (st.kparams.lrfu && history.committed) {
        const auto input_end = st.history_input_end ? st.history_input_end : uint32_t(prompt.size());
        const auto prefix = st.history_input_prompt
            ? st.active_prompt->common_prefix(*st.history_input_prompt) : 0;
        st.history_points.resize(kvmem::lrfu_prefix_points(st.history_points, prefix, input_end));
        st.history_points.push_back(kvmem::lrfu_input_point(input_end, st.kparams.block_tokens, history));
        st.history_input_prompt = st.active_prompt->prefix(input_end)->cache_index();
    }
    st.cached_tokens = prompt;
    st.cached_tokens.insert(st.cached_tokens.end(), gen.begin(), gen.end());
    st.last_n_gen = (int) gen.size();
    multimodal_commit(st, gen);
    const uint32_t stored = llama_kvmem_store_n_tokens();
    conversation_commit(st, stored);
    kvmem_diag("KVMEM_TRACE cache_commit n_prompt=%d n_gen=%d n_cached=%d stored=%u\n",
            (int) prompt.size(), (int) gen.size(), (int) st.cached_tokens.size(),
            stored);
}

void llama_driver_validate_capacity(const LlamaEngineState & st, const kvmem_prompt & prompt, int end) {
    if (!st.kparams.enabled || !st.kparams.budget || !prompt.has_media()) return;
    const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
    const uint32_t budget = st.kparams.budget / block;
    std::vector<std::pair<uint32_t, uint32_t>> groups;
    for (const auto & range : prompt.media_ranges()) {
        const uint32_t lo = (range.first ? range.first - 1 : 0) / block;
        const uint32_t hi = (std::min<uint32_t>(end, range.second + 1) + block - 1) / block;
        if (!groups.empty() && lo < groups.back().second) groups.back().second = hi;
        else groups.emplace_back(lo, hi);
    }
    const uint32_t sink = std::max(1u, st.kparams.sink_tokens / block);
    for (const auto & group : groups) {
        if (group.second - group.first + std::min(group.first, sink) > budget)
            throw std::invalid_argument("image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget");
    }
    // Text suffixes may be trimmed after the first pass. Only an image group
    // that cannot fit intact with the sink is a hard capacity error.
}

// Included after the single-slot server state and stream helpers.
static bool checkpoint_execution_held(const LlamaEngineState & st, const MultimodalCheckpointData * data) {
    if (st.mm_live_checkpoint.get() == data || (st.mm_rollback && st.mm_rollback->data.get() == data)) return true;
    for (const auto & cp : st.mm_checkpoints) if (cp.data.get() == data) return true;
    return false;
}

static void checkpoint_tier(LlamaEngineState & st, bool drain, bool force) {
    std::vector<kvmem::CheckpointInfo> fixed;
    for (const auto & cp : st.mm_checkpoints) fixed.push_back(cp.checkpoint_info());
    st.mm_archive.prune(fixed);
    if (!st.mm_archive.policy().enabled) return;
    if (st.mm_checkpoint_write) {
        auto & job = *st.mm_checkpoint_write;
        if (drain) { try { job.ticket.wait(); } catch (...) {} }
        if (!job.ticket.ready()) return;
        bool owned = false;
        for (const auto & e : st.mm_archive) owned |= e.info.identity == job.source.get() &&
            llama_kvmem_tail_mean_valid(e.record.row, e.record.data->tail_mean);
        if (job.ticket.status() == kvmem::StorageTaskStatus::Complete && owned &&
                !checkpoint_execution_held(st, job.source.get()) && !job.source->shared.load()) {
            auto & data = *const_cast<MultimodalCheckpointData *>(job.source.get());
            data.cold_recurrent = data.recurrent.size(); data.cold_carry = data.draft_carry.size();
            data.cold = std::move(job.destination);
            data.cold_identity = 0;
            if (data.accounting) data.accounting->live_bytes -= data.cold_recurrent + data.cold_carry;
            std::vector<uint8_t>().swap(data.recurrent); std::vector<uint8_t>().swap(data.draft_carry);
            kvmem_diag("KVMEM_CHECKPOINT_TIER write_bytes=%zu cold=1\n", data.cold_recurrent + data.cold_carry);
        }
        st.mm_checkpoint_write.reset();
    }
    if (drain) return;
    const auto spill = kvmem_execution_spill_file(st.execution.get());
    if (!spill) return;
    std::set<const MultimodalCheckpointData *> hot;
    for (const auto & cp : st.mm_checkpoints) if (cp.data && !cp.data->cold) hot.insert(cp.data.get());
    const kvmem::CheckpointArchive<MultimodalCheckpoint>::Entry * victim = nullptr;
    for (const auto & e : st.mm_archive) if (e.info.identity && !e.record.data->cold) {
        hot.insert(e.record.data.get());
        if (!e.record.data->shared.load() && !checkpoint_execution_held(st, e.record.data.get()) && (!victim || e.info.used < victim->info.used)) victim = &e;
    }
    if (!victim || (!force && hot.size() <= st.mm_archive.policy().hot_count)) return;
    const auto source = victim->record.data;
    try {
        const auto n = source->recurrent.size() + source->draft_carry.size();
        auto destination = spill->reserve(kvmem::checkpoint_image_size(n, 2));
        if (!st.mm_checkpoint_io) st.mm_checkpoint_io = std::make_unique<kvmem::StorageIoQueue>(1, st.mm_archive.policy().host_bytes);
        auto ticket = st.mm_checkpoint_io->submit(n, [source, destination] {
            kvmem::write_checkpoint_image(destination, uint64_t(reinterpret_cast<uintptr_t>(source.get())),
                {{source->recurrent.data(), source->recurrent.size()}, {source->draft_carry.data(), source->draft_carry.size()}});
        });
        if (ticket) st.mm_checkpoint_write = std::make_unique<LlamaEngineState::CheckpointWrite>(LlamaEngineState::CheckpointWrite{source, destination, ticket});
    } catch (const std::exception & error) {
        kvmem_diag("KVMEM_CHECKPOINT_TIER skipped=%s\n", error.what());
    }
}

static void checkpoint_make_room(LlamaEngineState & st, size_t bytes, bool optional = false,
                                std::shared_ptr<MultimodalCheckpointAccounting> account = {}) {
    if (!st.mm_archive.policy().enabled) return;
    if (!account) account=st.mm_checkpoint_accounting;
    const auto budget = st.mm_archive.policy().host_bytes;
    const auto fits = [&] { const auto used = account->live_bytes.load();
        return used <= budget && bytes <= budget-used; };
    if (fits()) return;
    // Only required captures/loads may wait for a single in-flight write.
    if (!optional) {
        checkpoint_tier(st, true);
        while (!fits()) {
            checkpoint_tier(st, false, true);
            if (!st.mm_checkpoint_write) break;
            const auto before = account->live_bytes.load();
            checkpoint_tier(st, true);
            if (account->live_bytes.load() >= before) break;
        }
        // RAM-only or full disk: release optional records, never native leases.
        for (auto & entry : st.mm_archive) if (!fits() && entry.info.identity &&
                !checkpoint_execution_held(st, entry.record.data.get())) entry = {};
        while (!fits()) {
            const int victim = conversation_lru_victim(st);
            if (victim < 0 || !conversation_evict(st, victim, "checkpoint_host_capacity")) break;
        }
        if (!fits() && st.mm_checkpoint_reclaim) st.mm_checkpoint_reclaim(account, bytes);
    }
    if (!fits()) throw std::runtime_error("checkpoint Host budget exhausted");
}

struct CheckpointHostReservation {
    std::shared_ptr<MultimodalCheckpointAccounting> account;
    size_t bytes;
    CheckpointHostReservation(std::shared_ptr<MultimodalCheckpointAccounting> owner, size_t n, size_t limit)
        : account(std::move(owner)), bytes(n) {
        if (!account->reserve(bytes,limit)) throw std::runtime_error("checkpoint Host reservation exceeded budget");
    }
    ~CheckpointHostReservation() { if (account) account->live_bytes-=bytes; }
    void publish(size_t actual) {
        if(actual>bytes) throw std::logic_error("checkpoint grew beyond reservation");
        account->live_bytes-=bytes-actual; account.reset();
    }
};

template<class Source> static kvmem_conversation fork_conversation(const Source& source, int keep) {
    const MultimodalCheckpoint * selected = nullptr;
    for (const auto& cp : source.mm_checkpoints) if (cp.row == keep) selected = &cp;
    for (const auto& entry : source.mm_archive) if (entry.info.identity && entry.record.row == keep) selected = &entry.record;
    if (!selected || !selected->data) throw std::runtime_error("native fork checkpoint disappeared");
    selected->data->shared.store(true);
    kvmem_conversation result;
    result.cached_tokens = source.cached_tokens; result.cached_prompt = source.cached_prompt;
    result.history_input_prompt = source.history_input_prompt; result.history_points = source.history_points;
    auto cp = *selected;
    result.mm_checkpoints.remember(cp);
    result.mm_archive.remember(cp, kvmem::CoveragePurpose::Input);
    result.mm_live_row = keep; result.last_n_gen = source.last_n_gen;
    return result;
}
kvmem_conversation llama_driver_fork_conversation(const kvmem_conversation& source, int keep) { return fork_conversation(source, keep); }
kvmem_conversation llama_driver_fork_conversation(const LlamaEngineState& source, int keep) { return fork_conversation(source, keep); }

std::shared_ptr<const MultimodalCheckpointData> llama_driver_materialize_checkpoint(
        LlamaEngineState & st, const std::shared_ptr<const MultimodalCheckpointData> & saved) {
    if (!saved->cold) return saved;
    const auto bytes = saved->cold_recurrent + saved->cold_carry;
    const bool shared = saved->shared.load();
    const auto charge = bytes + (shared ? saved->resident_bytes() : 0);
    checkpoint_make_room(st, charge, false, saved->accounting);
    CheckpointHostReservation reservation(saved->accounting,charge,st.mm_archive.policy().host_bytes);
    std::shared_ptr<MultimodalCheckpointData> copy;
    if (shared) {
        copy = std::make_shared<MultimodalCheckpointData>();
        copy->tail_mean = saved->tail_mean;
        // Assign accounting only after successful publication. On any failed
        // allocation/read the reservation alone releases the temporary charge.
    }
    std::vector<uint8_t> recurrent(saved->cold_recurrent), carry(saved->cold_carry);
    kvmem::read_checkpoint_image(saved->cold, saved->cold_identity ? saved->cold_identity : uint64_t(reinterpret_cast<uintptr_t>(saved.get())),
        {{recurrent.data(), recurrent.size()}, {carry.data(), carry.size()}});
    auto & data = shared ? *copy : *const_cast<MultimodalCheckpointData *>(saved.get());
    data.recurrent.swap(recurrent); data.draft_carry.swap(carry); data.cold.reset();
    reservation.publish(shared ? copy->resident_bytes() : bytes);
    if (shared) copy->accounting = saved->accounting;
    kvmem_diag("KVMEM_CHECKPOINT_TIER read_bytes=%zu cold=0\n", bytes);
    return shared ? std::shared_ptr<const MultimodalCheckpointData>(std::move(copy)) : saved;
}

static MultimodalCheckpoint multimodal_checkpoint(LlamaEngineState & st, int row, bool optional = false) {
    MultimodalCheckpoint result;
    result.row = row;
    if (st.mm_live_checkpoint && st.mm_live_row == row) {
        result.data = st.mm_live_checkpoint;
        ++st.mm_perf.shared;
        return result;
    }
    kvmem_scoped_ms timer(st.mm_perf.save_ms);
    ++st.mm_perf.saves;
    auto data = std::make_shared<MultimodalCheckpointData>();
    llama_synchronize(st.ctx);
    {
        kvmem_scoped_ms mean_timer(st.mm_perf.mean_ms);
        if (!llama_kvmem_get_tail_mean(row, data->tail_mean))
            throw std::runtime_error("mean-K checkpoint is not an accepted prefix");
    }
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t size = llama_state_seq_get_size_ext(st.ctx, 0, flags);
    const auto carry_bytes = st.spec.ok ? common_speculative_state_size(st.spec.spec, 0) : 0;
    const auto reserve = size + data->resident_bytes() + carry_bytes;
    checkpoint_make_room(st, optional ? 2*reserve : reserve, optional);
    CheckpointHostReservation reservation(st.mm_checkpoint_accounting,reserve,
        st.mm_archive.policy().enabled ? st.mm_archive.policy().host_bytes : SIZE_MAX);
    data->recurrent.resize(size);
    if (llama_state_seq_get_data_ext(st.ctx, data->recurrent.data(), size, 0, flags) != size) {
        throw std::runtime_error("multimodal recurrent checkpoint failed");
    }
    {
        kvmem_scoped_ms carry_timer(st.mm_perf.carry_ms);
        if (st.spec.ok && !common_speculative_get_state(st.spec.spec, 0, data->draft_carry, carry_bytes)) {
            throw std::runtime_error("MTP carry checkpoint failed");
        }
    }
    reservation.publish(data->bytes());
    data->accounting = st.mm_checkpoint_accounting;
    result.data = std::move(data);
    st.mm_live_checkpoint = result.data;
    return result;
}

static bool checkpoint_nearby(LlamaEngineState & st, int row) {
    const auto * earlier = st.mm_archive.nearby(row, [](const auto & cp) {
        return llama_kvmem_tail_mean_valid(cp.row, cp.data->tail_mean);
    });
    if (!earlier) return false;
    kvmem_diag("KVMEM_CHECKPOINT_NEARBY frontier=%d earlier=%d\n",row,earlier->row);
    return true;
}

static void multimodal_remember(LlamaEngineState & st, MultimodalCheckpoint checkpoint) {
    if (st.mm_archive.policy().enabled && checkpoint.row > 0) {
        if (kvmem::has_purpose(checkpoint.purposes, kvmem::CheckpointPurpose::ReuseAnchor))
            st.mm_archive.remember(checkpoint, kvmem::CoveragePurpose::Anchor);
        else if (checkpoint.purposes == kvmem::CheckpointPurpose::Endpoint)
            st.mm_archive.remember(checkpoint, kvmem::CoveragePurpose::Endpoint);
    }
    st.mm_checkpoints.remember(std::move(checkpoint));
    checkpoint_tier(st);
}

static void multimodal_restore(LlamaEngineState & st, const MultimodalCheckpoint & checkpoint, bool truncate,
                               bool restore_history = false) {
    kvmem_scoped_ms timer(st.mm_perf.restore_ms);
    const bool live = st.mm_live_checkpoint == checkpoint.data && st.mm_live_row == checkpoint.row;
    if (!checkpoint.data) throw std::runtime_error("missing multimodal checkpoint data");
    if (!llama_kvmem_tail_mean_valid(checkpoint.row, checkpoint.data->tail_mean))
        throw std::runtime_error("stale or foreign mean-K/recurrent checkpoint");
    const auto restore_data = live ? checkpoint.data : llama_driver_materialize_checkpoint(st, checkpoint.data);
    if (live) ++st.mm_perf.restore_skips;
    else ++st.mm_perf.restores;
    llama_synchronize(st.ctx);
    if (st.spec.ctx_dft) llama_synchronize(st.spec.ctx_dft);
    llama_kvmem_decode_mean_flush();
    llama_kvmem_decode_mean_discard();
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const auto & data = restore_data->recurrent;
    if (!live && llama_state_seq_set_data_ext(st.ctx, data.data(), data.size(), 0, flags) != data.size()) {
        throw std::runtime_error("multimodal recurrent restore failed");
    }
    if (!llama_kvmem_remove_logical(st.ctx, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted target rows");
    }
    if (st.spec.ctx_dft && !llama_kvmem_remove_logical(st.spec.ctx_dft, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted MTP rows");
    }
    if (st.spec.ok && !live) {
        kvmem_scoped_ms carry_timer(st.mm_perf.carry_ms);
        common_speculative_set_state(st.spec.spec, 0, restore_data->draft_carry);
    }
    if (truncate) {
        llama_kvmem_truncate_cached(checkpoint.row);
        if (!llama_kvmem_set_tail_mean(checkpoint.row, checkpoint.data->tail_mean))
            throw std::runtime_error("mean-K checkpoint restore rejected");
        if (!llama_kvmem_restore_checkpoint_tail())
            throw std::runtime_error("checkpoint KV coverage is incomplete");
    }
    st.mm_archive.truncate(checkpoint.row);
    st.mm_archive.touch(checkpoint.data.get());
    st.mm_live_row = checkpoint.row;
    st.mm_live_checkpoint = restore_data;
    if (restore_history) llama_kvmem_restore_history(checkpoint.data->tail_mean.history);
}

void llama_driver_finish(LlamaEngineState & st) {
    if (st.mm_committed || !st.mm_rollback) return;
    try {
        llama_kvmem_set_replay(false);
        multimodal_restore(st, *st.mm_rollback, true, true);
        if (st.kparams.lrfu) llama_kvmem_restore_history(st.rollback_history);
        llama_kvmem_begin_cached_turn();
        st.mm_query.reset();
        st.mm_pending_query.reset();
        st.cached_prompt = st.mm_rollback_prompt;
        st.cached_tokens = st.cached_prompt ? st.cached_prompt->tokens : std::vector<llama_token>{};
        st.cached_tokens.resize(std::min(st.cached_tokens.size(), (size_t) st.mm_live_row));
        multimodal_remember(st, *st.mm_rollback);
        kvmem_diag("KVMEM_TRACE multimodal_rollback context=%p row=%d\n", (void *) st.ctx, st.mm_live_row);
        kvmem_diag("KVMEM_CHECKPOINT_ROLLBACK live_bytes=%zu peak_bytes=%zu\n",
                st.mm_checkpoint_accounting->live_bytes.load(), st.mm_checkpoint_accounting->peak_bytes.load());
        st.mm_committed = true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        LOG_ERR("srv    KVMEM_TRACE multimodal_rollback_failed error=%s\n", e.what());
        // A failed recurrent restore cannot remain a reusable prefix.
        try { llama_driver_clear(st); }
        catch (const std::exception & clear_error) {
            LOG_ERR("srv    KVMEM cache clear failed: %s\n", clear_error.what());
            st.cached_tokens.clear();
            st.cached_prompt.reset();
            st.mm_checkpoints.clear();
            st.mm_archive.clear();
            st.mm_live_checkpoint.reset();
            st.mm_live_row = 0;
        }
        st.mm_committed = true;
    }
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
}

static int multimodal_decode_span(LlamaEngineState & st, int begin, int end, bool replay, kvmem::RequestControl * io) {
    const auto & prompt = *st.active_prompt;
    auto dispatch = [&](llama_batch batch) -> int {
        if (!kvmem::poll_request(io)) return KVMEM_DECODE_ABORT;
        kvmem_diag("KVMEM_TRACE multimodal_decode context=%p rows=[%d,%d) model_pos=%d image=%d replay=%d\n",
                (void *) st.ctx, batch.logical_pos[0], batch.logical_pos[batch.n_tokens - 1] + 1,
                batch.pos[0], batch.token == nullptr, replay);
        const auto start = std::chrono::steady_clock::now();
        const bool diagnostic = llama_kvmem_get_transfer_stats().enabled;
        st.mm_live_checkpoint.reset();
        int rc = llama_decode(st.ctx, batch);
        if (diagnostic) {
            llama_synchronize(st.ctx);
            st.mm_perf.target_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        const auto draft_start = std::chrono::steady_clock::now();
        if (rc == 0 && st.spec.ok && !kvmem_spec_process(st.spec.spec, st.ctx, batch)) rc = -1;
        if (diagnostic && st.spec.ok) {
            llama_synchronize(st.spec.ctx_dft);
            st.mm_perf.draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - draft_start).count();
        }
        llama_synchronize(st.ctx);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        (replay ? st.mm_perf.replay_ms : st.mm_perf.first_ms) += elapsed;
        kvmem_diag("KVMEM_TRACE multimodal_compute rows=%d elapsed_ms=%.3f image=%d replay=%d\n",
                batch.n_tokens, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                batch.token == nullptr, replay);
        if (rc == 0) {
            st.mm_live_row = batch.logical_pos[batch.n_tokens - 1] + 1;
            if (replay) st.mm_replayed += batch.n_tokens;
            else {
                const int tail = std::clamp(st.mm_lcp - batch.logical_pos[0], 0, batch.n_tokens);
                st.mm_tail_replayed += tail;
                if (batch.token) st.mm_new_text += batch.n_tokens - tail;
                else st.mm_new_image += batch.n_tokens - tail;
                // Count actual first-pass work, including reconstruction after the
                // reused checkpoint; exclude the later retrieval query replay.
                if (io && io->on_prefill) io->on_prefill(batch.n_tokens, (int) prompt.tokens.size() - batch.logical_pos[0]);
            }
        }
        return rc;
    };
    int row = begin;
    while (row < end) {
        if (!kvmem::poll_request(io)) return KVMEM_DECODE_ABORT;
        if (prompt.tokens[row] == LLAMA_TOKEN_NULL) {
            const int next = (int) prompt.media_end(row);
            if (next > end) throw std::runtime_error("prefill boundary splits an image");
            if (!replay) {
                auto checkpoint = multimodal_checkpoint(st, row);
                checkpoint.purposes = kvmem::CheckpointPurpose::MediaBoundary;
                multimodal_remember(st, std::move(checkpoint));
            }
            const int rc = st.vision->decode(st.ctx, prompt, row, st.n_batch, dispatch);
            if (rc != 0) return rc;
            row = next;
            continue;
        }
        const int limit = !replay && st.mm_archive.policy().enabled
            ? int(st.mm_capture_plan.cut(row, std::min(end, row + st.n_batch))) : std::min(end, row + st.n_batch);
        int next = row;
        while (next < limit && prompt.tokens[next] != LLAMA_TOKEN_NULL) ++next;
        std::vector<llama_pos> pos(next - row), logical(next - row);
        const auto pos0 = prompt.model_pos(row);
        for (int i = row; i < next; ++i) {
            pos[i - row] = pos0 + i - row;
            logical[i - row] = i;
        }
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.tokens.data()) + row, next - row);
        batch.pos = pos.data();
        batch.logical_pos = logical.data();
        std::vector<int32_t> n_seq(next - row, 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> seq_ids(next - row, &seq);
        std::vector<int8_t> outputs(next - row, 0);
        outputs.back() = !st.spec.ok;
        batch.n_seq_id = n_seq.data();
        batch.seq_id = seq_ids.data();
        batch.logits = outputs.data();
        const int rc = dispatch(batch);
        if (rc != 0) return rc;
        row = next;
        if (!replay && st.mm_archive.policy().enabled) {
            checkpoint_tier(st);
            if (const auto purpose = st.mm_capture_plan.due(row)) {
                try {
                    if (!checkpoint_nearby(st,row)) {
                        auto cp = multimodal_checkpoint(st, row, true);
                        const bool retained = st.mm_archive.remember(cp, *purpose);
                        kvmem_diag("KVMEM_CHECKPOINT_COVERAGE frontier=%d purpose=%d retained=%d count=%zu bytes=%llu\n",
                            row, int(*purpose), int(retained), st.mm_archive.size(), (unsigned long long) st.mm_archive.bytes());
                    }
                } catch (const std::exception & error) {
                    kvmem_diag("KVMEM_CHECKPOINT_COVERAGE skipped=%s frontier=%d\n", error.what(), row);
                }
                st.mm_capture_plan.saved(row);
                checkpoint_tier(st);
            }
        }
    }
    return 0;
}

static bool run_prefill_multimodal(LlamaEngineState & st, kvmem::RequestControl * io, int * n_cache_hit) {
    const auto started = std::chrono::steady_clock::now();
    st.mm_perf = {};
    const auto copies_before = llama_kvmem_get_transfer_stats();
    st.mm_checkpoint_accounting->peak_bytes.store(st.mm_checkpoint_accounting->live_bytes.load());
    st.mm_pending_query.reset();
    try {
        st.mm_error.clear();
        st.mm_error_status = 500;
        if (st.mm_reset_requested) {
            kvmem_diag("KVMEM_TRACE multimodal_reset context=%p reason=explicit_cache_reset\n", (void *) st.ctx);
            llama_driver_clear(st);
            st.mm_reset_requested = false;
        }
        st.mm_new_text = st.mm_new_image = st.mm_replayed = st.mm_tail_replayed = 0;
        const auto & prompt = *st.active_prompt;
        const int eval_end = (int) prompt.tokens.size() - (st.spec.ok ? 1 : 0);
        const int lcp = st.cached_prompt ? (int) prompt.common_prefix(*st.cached_prompt) : 0;
        const int observed_lcp = std::max(lcp, std::exchange(st.mm_observed_lcp, 0));
        st.mm_lcp = lcp;
        // Sequence checkpoints do not restore logits. Ordinary decoding must evaluate
        // at least one token; MTP evaluates the pending prompt token in spec_generate.
        const int keep = int(kvmem::prefix_restore_limit(lcp, st.mm_live_row, eval_end, !st.spec.ok));
        MultimodalCheckpoint base;
        bool found = false;
        {
            kvmem_scoped_ms timer(st.mm_perf.select_checkpoint_ms);
            auto checkpoint = st.mm_checkpoints.select(keep, [](const auto & candidate) {
                return llama_kvmem_tail_mean_valid(candidate.row, candidate.data->tail_mean);
            });
            const auto archived = st.mm_archive.select(keep, [](const auto & candidate) {
                return llama_kvmem_tail_mean_valid(candidate.row, candidate.data->tail_mean);
            });
            if (archived && (!checkpoint || archived->row > checkpoint->row)) checkpoint = archived;
            while (checkpoint) {
                base = *checkpoint;
                try { base.data = llama_driver_materialize_checkpoint(st, base.data); found = true; break; }
                catch (const std::exception & error) {
                    kvmem_diag("KVMEM_CHECKPOINT_READ_FALLBACK row=%d reason=%s\n", base.row, error.what());
                    const int earlier = base.row-1;
                    st.mm_archive.erase(base.data.get());
                    base = {};
                    checkpoint = st.mm_checkpoints.select(earlier, [](const auto & cp) {
                        return llama_kvmem_tail_mean_valid(cp.row, cp.data->tail_mean);
                    });
                    const auto previous = st.mm_archive.select(earlier, [](const auto & cp) {
                        return llama_kvmem_tail_mean_valid(cp.row, cp.data->tail_mean);
                    });
                    if (previous && (!checkpoint || previous->row > checkpoint->row)) checkpoint = previous;
                }
            }
        }
        if (!found) {
            // Shared template tokens do not identify a conversation. Like llama-server,
            // treat a missing recurrent checkpoint as a cache miss and evaluate the supplied prompt.
            kvmem_diag("KVMEM_TRACE multimodal_reset context=%p reason=%s lcp=%d keep=%d cached_rows=%zu live_rows=%d oldest_checkpoint=%d checkpoint_count=%zu\n",
                    (void *) st.ctx, st.cached_prompt ? "no_recurrent_checkpoint" : "new_conversation",
                    lcp, keep, st.cached_tokens.size(), st.mm_live_row,
                    st.mm_checkpoints.empty() ? -1 : st.mm_checkpoints.front().row, st.mm_checkpoints.size());
            llama_driver_clear(st);
            st.mm_lcp = 0; // No old rows survived the reset; count all evaluated rows as new.
            base = multimodal_checkpoint(st, 0);
        }
        st.mm_rollback = std::make_shared<MultimodalCheckpoint>(base);
        st.mm_rollback_prompt = st.cached_prompt ? st.cached_prompt->prefix(base.row) : nullptr;
        st.mm_committed = false;
        try {
            multimodal_restore(st, base, true, true);
        } catch (const std::exception & error) {
            kvmem_diag("KVMEM_TRACE checkpoint_restore_fallback row=%d reason=%s\n", base.row, error.what());
            llama_driver_clear(st);
            st.mm_lcp = 0;
            base = multimodal_checkpoint(st, 0);
            st.mm_rollback = std::make_shared<MultimodalCheckpoint>(base);
            st.mm_rollback_prompt.reset();
            multimodal_restore(st, base, true, true);
        }
        begin_history_round(st);
        llama_kvmem_begin_cached_turn();
        std::vector<uint32_t> starts, ends;
        const auto ranges = prompt.media_ranges();
        for (const auto & range : ranges) {
            starts.push_back(range.first > 0 ? range.first - 1 : 0);
            ends.push_back(std::min<uint32_t>(prompt.tokens.size(), range.second + 1));
        }
        llama_kvmem_set_media_ranges(starts.data(), ends.data(), starts.size());
        const int user_begin = std::clamp(st.kparams.query_begin, 0, eval_end);
        const int user_end = std::clamp(st.kparams.query_end, user_begin, eval_end);
        const auto media = prompt.media_identity();
        const auto cached_query = st.mm_query;
        const bool same_query = st.turn_query_exact && cached_query &&
            cached_query->begin == user_begin && cached_query->end == user_end &&
            cached_query->force == st.kparams.force_pos && cached_query->user == st.turn_last_user &&
            base.row >= user_end && cached_query->media == media &&
            prompt.common_prefix(*cached_query->prefix) >= (size_t) user_end;
        const bool capture_user = st.turn_query_exact &&
            user_begin >= base.row && user_end > user_begin;
        int query = std::max(base.row, std::min(st.kparams.query_begin, eval_end));
        if (!ranges.empty()) query = std::max(query, (int) ranges.back().second);
        query = std::min(query, eval_end);
        const bool evaluated_user = st.turn_query_exact && user_end>user_begin && base.row>=user_end;
        const bool continuation = st.turn_tool_continuation || evaluated_user;
        const bool retrieve = !continuation && st.kparams.enabled && st.kparams.method == 1 && query < eval_end;
        llama_kvmem_set_request_span(query, eval_end, st.kparams.force_pos);
        llama_kvmem_turn_spans spans;
        spans.mandatory = {{query, eval_end}};
        spans.replay_begin = query;
        if (!continuation) spans.query = capture_user
            ? std::vector<llama_kvmem_row_range>{{user_begin, user_end}}
            : std::vector<llama_kvmem_row_range>{{std::max(query, eval_end - st.query_max_tokens), eval_end}};
        llama_kvmem_set_turn_spans(spans);
        std::string path = continuation ? "append_pressure" : "first_pass";
        std::string reason = continuation ? "ordinary_continuation" : "no_query_suffix";
        const std::string reuse_fallback = "none";
        const bool reused_query = false;
        // Continuations never import old Q, reselect cold history or probe tools.
        if (continuation) llama_kvmem_freeze_query(true);
        bool all_resident = false;
        {
            kvmem_scoped_ms timer(st.mm_perf.decision_ms);
            all_resident = retrieve && st.query_replay_auto &&
                llama_kvmem_can_append(eval_end, st.spec.ok ? st.spec_n_max + 1 : 0, true, reason);
        }
        if (all_resident) llama_kvmem_keep_selected();
        const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
        const auto admission = kvmem::query_replay_plan(
            {uint64_t(query), uint64_t(eval_end)}, query, eval_end, st.query_replay_max_tokens,
            block, std::min<uint64_t>((uint64_t(query)+block-1)/block, st.kparams.budget/block),
            (uint64_t(st.kparams.budget)+st.kparams.gen_reserve)/block,
            st.spec.ok ? st.spec_n_max + 1 : 0);
        const bool capture_query=retrieve && admission.admitted && !all_resident;
        if (!capture_query) {
            spans.query.clear();
            llama_kvmem_set_turn_spans(spans);
            llama_kvmem_freeze_query(true);
        }
        if (n_cache_hit) *n_cache_hit = base.row;
        std::vector<kvmem::PrefixRange> input_groups;
        for (const auto & range : ranges) input_groups.push_back({range.first, range.second});
        const int anchor = int(kvmem::prefix_anchor(base.row, observed_lcp, query, input_groups));
        const auto legal_cut = [&](uint64_t cut) {
            if (!cut || cut > uint64_t(eval_end)) return uint64_t(0);
            for (const auto & group : input_groups) if (group.begin < cut && cut < group.end) return uint64_t(0);
            return cut;
        };
        st.mm_capture_plan = kvmem::CheckpointCapturePlan(base.row, st.mm_archive.policy().interval,
            legal_cut(st.mm_root_boundary), legal_cut(st.mm_tail_boundary));
        if (multimodal_decode_span(st, base.row, anchor, false, io) != 0)
            throw std::runtime_error("multimodal prefix prefill failed or cancelled");
        if (anchor > 0 && !checkpoint_nearby(st,anchor)) {
            try {
                auto checkpoint = multimodal_checkpoint(st, anchor);
                checkpoint.purposes = kvmem::CheckpointPurpose::ReuseAnchor;
                multimodal_remember(st, std::move(checkpoint));
            } catch (const std::exception & error) {
                kvmem_diag("KVMEM_TRACE reuse_anchor_skipped row=%d reason=%s\n", anchor, error.what());
            }
        }
        if (multimodal_decode_span(st, anchor, query, false, io) != 0) throw std::runtime_error("multimodal prefill failed or cancelled");
        auto query_checkpoint = multimodal_checkpoint(st, query);
        query_checkpoint.purposes = kvmem::CheckpointPurpose::QueryReplay;
        multimodal_remember(st, query_checkpoint);
        const auto probe_view = llama_kvmem_get_attention_view();
        if (multimodal_decode_span(st, query, eval_end, false, io) != 0) throw std::runtime_error("multimodal query prefill failed or cancelled");
        if (retrieve) {
            bool replay = true;
            bool replay_fits = true;
            {
                kvmem_scoped_ms timer(st.mm_perf.retrieval_ms);
                if (all_resident) {
                    if (!llama_kvmem_commit_resident())
                        throw std::runtime_error("resident Query proof invalidated during prefill");
                    path = "all_resident";
                    replay = false;
                } else if (!admission.admitted) {
                    path = "query_replay_skipped";
                    reason = admission.reason;
                    replay = false;
                } else {
                    const auto selection = llama_kvmem_preview_retrieval(&query_checkpoint.data->tail_mean);
                    if (st.query_replay_auto && llama_kvmem_commit_unchanged(probe_view, selection)) {
                        path = "unchanged_selection";
                        reason = "same_attention_view";
                        replay = false;
                    } else {
                        path = "query_replay";
                        if (st.query_replay_auto) reason = "selection_or_attention_view_changed";
                        // Check the actual selection: complete image groups also
                        // consume slots, so a text-only budget estimate is insufficient.
                        const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
                        for (uint32_t id = query / block; id < ((uint32_t) eval_end + block - 1) / block; ++id) {
                            if (!std::binary_search(selection.blocks.begin(), selection.blocks.end(), id)) {
                                replay_fits = false;
                                break;
                            }
                        }
                        replay_fits = replay_fits && llama_kvmem_selection_fits(selection, eval_end,
                            st.spec.ok ? st.spec_n_max + 1 : 0);
                        if (replay_fits) llama_kvmem_apply_selection(selection);
                    }
                }
            }
            if (replay && !replay_fits) {
                // Admission failure preserves the complete first-pass view.
                path = "query_replay_skipped";
                reason = "replay_exceeds_budget";
                replay = false;
                kvmem_diag("KVMEM_TRACE replay_skipped reason=over_budget query=[%d,%d) replay_rows=%d budget_tokens=%u\n",
                        query, eval_end, eval_end - query, st.kparams.budget);
            }
            if (replay) {
                multimodal_restore(st, query_checkpoint, false);
                if (!llama_kvmem_set_tail_mean(query,query_checkpoint.data->tail_mean))
                    throw std::runtime_error("Query replay statistics checkpoint rejected");
                llama_kvmem_set_replay(true);
                const int rc = multimodal_decode_span(st, query, eval_end, true, io);
                llama_kvmem_set_replay(false);
                if (rc != 0) throw std::runtime_error("multimodal query replay failed or cancelled");
            }
        }
        if (continuation && same_query) st.mm_pending_query = cached_query;
        if (capture_query && capture_user && st.kparams.enabled && st.kparams.method == 1) {
            auto saved = std::make_shared<MultimodalQuery>();
            if (llama_kvmem_get_query(saved->state) &&
                    *std::max_element(saved->state.count.begin(), saved->state.count.end()) == (uint32_t) (user_end - user_begin)) {
                saved->begin = user_begin;
                saved->end = user_end;
                saved->force = st.kparams.force_pos;
                saved->user = st.turn_last_user;
                saved->media = media;
                saved->prefix = prompt.prefix(user_end);
                st.mm_pending_query = std::move(saved);
            }
        }
        const auto & counts = st.mm_pending_query ? st.mm_pending_query->state.count : std::vector<uint32_t>{};
        const uint32_t q_rows = continuation || counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());
        const char * source = continuation ? "none" :
            capture_user ? "user" : "bootstrap_suffix";
        if (!retrieve && !continuation) reason = "no_query_suffix";
        kvmem_diag("KVMEM_PREFILL_DECISION path=%s reason=%s reuse_fallback=%s append=[%d,%d) query=[%d,%d) feature=[%d,%d) query_source=%s query_reused=%d q_rows=%u replay_rows=%u decision_ms=%.3f\n",
                path.c_str(), reason.c_str(), reuse_fallback.c_str(), base.row, eval_end, query, eval_end,
                spans.query.empty() ? 0 : spans.query.front().begin, spans.query.empty() ? 0 : spans.query.back().end, source, reused_query, q_rows, st.mm_replayed, st.mm_perf.decision_ms);
        llama_kvmem_accept_retrieval_round();
        llama_kvmem_pin_working_set();
        // History changed even when the native view did not.
        if (st.kparams.lrfu) st.mm_live_checkpoint.reset();
        multimodal_remember(st, multimodal_checkpoint(st, eval_end));
        std::vector<uint8_t> carry;
        llama_pos synced = 0;
        if (st.spec.ok) {
            common_speculative_get_state(st.spec.spec, 0, carry);
            if (carry.size() < sizeof(synced)) throw std::runtime_error("MTP carry missing");
            std::memcpy(&synced, carry.data(), sizeof(synced));
            if (synced != eval_end) throw std::runtime_error("MTP has unsynchronized visual rows");
        }
        kvmem_diag("KVMEM_TRACE multimodal_prefill context=%p prefix_hit_rows=%d lcp=%d new_text_rows=%u new_image_rows=%u replayed_rows=%u vision_encode_calls=%u encoder_ms=%.2f logical_cursor=%d model_cursor=%d mtp_synced_rows=%d cached_tail_rows=%u replay_reason=%s embedding_cache_bytes=%zu checkpoint_bytes=%zu\n",
                (void *) st.ctx, base.row, lcp, st.mm_new_text, st.mm_new_image, st.mm_replayed,
                prompt.encode_calls, prompt.encode_ms,
                eval_end, prompt.model_pos(eval_end), synced, st.mm_tail_replayed, st.mm_replayed ? reason.c_str() : "none",
                st.vision ? st.vision->cache_bytes() : 0,
                std::accumulate(st.mm_checkpoints.begin(), st.mm_checkpoints.end(), size_t(0),
                    [](size_t n, const auto & c) { return n + c.data->bytes(); }));
        std::set<const MultimodalCheckpointData *> unique;
        size_t unique_bytes = 0;
        size_t ref_bytes = 0;
        auto count = [&](const MultimodalCheckpoint & cp) {
            if (cp.data && unique.insert(cp.data.get()).second) {
                unique_bytes += cp.data->bytes();
                ref_bytes += cp.data->bytes() * cp.data.use_count();
            }
        };
        for (const auto & cp : st.mm_checkpoints) count(cp);
        for (const auto & entry : st.mm_archive) if (entry.info.identity) count(entry.record);
        if (st.mm_rollback) count(*st.mm_rollback);
        count(base);
        count(query_checkpoint);
        const auto & p = st.mm_perf;
        kvmem_diag("KVMEM_PREFILL_PERF total_ms=%.3f first_ms=%.3f replay_ms=%.3f retrieval_ms=%.3f checkpoint_select_ms=%.3f checkpoint_save_ms=%.3f checkpoint_restore_ms=%.3f mean_nested_ms=%.3f carry_nested_ms=%.3f saves=%u restores=%u shared=%u restore_skips=%u checkpoint_unique_bytes=%zu\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count(),
                p.first_ms, p.replay_ms, p.retrieval_ms, p.select_checkpoint_ms, p.save_ms, p.restore_ms,
                p.mean_ms, p.carry_ms, p.saves, p.restores, p.shared, p.restore_skips, unique_bytes);
        kvmem_diag("KVMEM_CHECKPOINT_MEMORY ref_bytes=%zu unique_bytes=%zu live_bytes=%zu peak_bytes=%zu\n",
                ref_bytes, unique_bytes, st.mm_checkpoint_accounting->live_bytes.load(), st.mm_checkpoint_accounting->peak_bytes.load());
        if (copies_before.enabled) {
            const auto copies = llama_kvmem_get_transfer_stats();
            fprintf(stderr, "KVMEM_PREFILL_PHASE path=%s first_ms=%.3f replay_ms=%.3f retrieval_ms=%.3f replay_rows=%u query=[%d,%d)\n",
                    path.c_str(), p.first_ms, p.replay_ms, p.retrieval_ms,
                    st.mm_replayed, query, eval_end);
            fprintf(stderr, "KVMEM_PREFILL_DIAGNOSTIC target_ms=%.3f draft_ms=%.3f extra_sync=1 adapter_h2d_bytes=%llu adapter_d2h_bytes=%llu adapter_d2d_bytes=%llu h2d_calls=%llu d2h_calls=%llu d2d_calls=%llu\n",
                    p.target_ms, p.draft_ms,
                    (unsigned long long) (copies.bytes[0] - copies_before.bytes[0]),
                    (unsigned long long) (copies.bytes[1] - copies_before.bytes[1]),
                    (unsigned long long) (copies.bytes[2] - copies_before.bytes[2]),
                    (unsigned long long) (copies.calls[0] - copies_before.calls[0]),
                    (unsigned long long) (copies.calls[1] - copies_before.calls[1]),
                    (unsigned long long) (copies.calls[2] - copies_before.calls[2]));
        }
        return true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        if (dynamic_cast<const std::invalid_argument *>(&e)) st.mm_error_status = 400;
        LOG_ERR("srv    KVMEM_TRACE multimodal_error error=%s\n", e.what());
        llama_driver_finish(st);
        return false;
    }
}

static void multimodal_commit(LlamaEngineState & st, const std::vector<llama_token> & gen) {
    st.cached_prompt = st.active_prompt->with_generated(gen);
    st.mm_live_row = st.kparams.enabled ? (int) llama_kvmem_store_n_tokens()
        : st.mm_live_row;
    multimodal_remember(st, multimodal_checkpoint(st, st.mm_live_row));
    st.mm_query = st.mm_pending_query;
    st.mm_pending_query.reset();
    st.mm_committed = true;
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
    if (st.mm_archive.policy().enabled) {
        // Execution slots are leases for this turn. Once committed their payloads
        // compete in the bounded historical directory, including tiny budgets.
        for (const auto& cp:st.mm_checkpoints) st.mm_archive.remember(cp,
            kvmem::has_purpose(cp.purposes,kvmem::CheckpointPurpose::ReuseAnchor) ? kvmem::CoveragePurpose::Anchor :
            kvmem::has_purpose(cp.purposes,kvmem::CheckpointPurpose::Endpoint) ? kvmem::CoveragePurpose::Endpoint : kvmem::CoveragePurpose::Periodic);
        st.mm_checkpoints.clear();
        checkpoint_tier(st);
        std::set<const void*> seen;
        const auto report = [&](const MultimodalCheckpoint& cp) {
            if (cp.data && seen.insert(cp.data.get()).second)
                kvmem_diag("KVMEM_CHECKPOINT_DIRECTORY frontier=%d cold=%d bytes=%zu identity=%p\n",
                    cp.row, int(bool(cp.data->cold)), cp.data->bytes(), (const void*)cp.data.get());
        };
        for (const auto& cp:st.mm_checkpoints) report(cp);
        for (const auto& entry:st.mm_archive) if(entry.info.identity) report(entry.record);
    }
    kvmem_diag("KVMEM_CHECKPOINT_COMMIT live_bytes=%zu peak_bytes=%zu\n",
            st.mm_checkpoint_accounting->live_bytes.load(), st.mm_checkpoint_accounting->peak_bytes.load());
}

int llama_driver_decode_generated(LlamaEngineState & st, llama_token id, int row) {
    st.mm_live_checkpoint.reset();
    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_pos logical = row;
    llama_pos pos = st.active_prompt->model_pos(row);
    batch.logical_pos = &logical;
    batch.pos = &pos;
    const int rc = llama_decode(st.ctx, batch);
    if (rc == 0) st.mm_live_row = row + 1;
    return rc;
}



bool llama_driver_prepare(LlamaEngineState & st, const std::vector<llama_token> &,
                          kvmem::RequestControl * io, int * n_cache_hit) {
    return run_prefill_multimodal(st, io, n_cache_hit);
}

static std::string trim_copy(const std::string & s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\n' || s[a] == '\r' || s[a] == '\t')) {
        ++a;
    }
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\n' || s[b - 1] == '\r' || s[b - 1] == '\t')) {
        --b;
    }
    return s.substr(a, b - a);
}

// Last ChatML user *role block* in the rendered prompt, not rfind(content).
// Thinking/tool text can quote last_user; only <|im_start|>user ... <|im_end|>
// counts. If last_user is set, pick the last block whose content equals it
// (two identical user turns → the later block).
static bool find_last_user_role_block(const std::string & prompt, const std::string & last_user,
                                      size_t & content0, size_t & content1, int & n_blocks, int & pick) {
    static const char * hdrs[] = {
        "<|im_start|>user\n",
        "<|im_start|>user\r\n",
        "<|im_start|>user",
    };
    struct Blk {
        size_t c0;
        size_t c1;
    };
    std::vector<Blk> blks;
    size_t search = 0;
    while (search < prompt.size()) {
        size_t best = std::string::npos;
        size_t best_len = 0;
        for (const char * h : hdrs) {
            const size_t n = std::strlen(h);
            const size_t p = prompt.find(h, search);
            if (p == std::string::npos) {
                continue;
            }
            if (best == std::string::npos || p < best || (p == best && n > best_len)) {
                best = p;
                best_len = n;
            }
        }
        if (best == std::string::npos) {
            break;
        }
        const size_t c0 = best + best_len;
        const size_t end = prompt.find("<|im_end|>", c0);
        if (end == std::string::npos) {
            break;
        }
        blks.push_back(Blk{c0, end});
        search = best + 1;
    }
    n_blocks = (int) blks.size();
    pick = -1;
    if (blks.empty()) {
        return false;
    }
    const std::string want = trim_copy(last_user);
    if (!want.empty()) {
        for (int i = n_blocks - 1; i >= 0; --i) {
            const std::string got = trim_copy(prompt.substr(blks[(size_t) i].c0,
                    blks[(size_t) i].c1 - blks[(size_t) i].c0));
            if (got == want) {
                pick = i;
                break;
            }
        }
    }
    if (pick < 0) {
        // No exact content match (template wrapping). Do not fall back to
        // the last user-role header: Qwen tools are often rendered as user
        // + <tool_response>. Leave the caller to query-last fallback.
        if (!want.empty()) {
            return false;
        }
        pick = n_blocks - 1;
    }
    content0 = blks[(size_t) pick].c0;
    content1 = blks[(size_t) pick].c1;
    return content0 < content1;
}

void llama_driver_query_span(LlamaEngineState & st, const std::string & prompt, const std::string & last_user,
                              const std::vector<llama_token> & toks, int & qbegin, int & qend) {
    qbegin = -1;
    qend = (int) toks.size();
    size_t c0 = 0;
    size_t c1 = 0;
    int n_blocks = 0;
    int pick = -1;
    if (find_last_user_role_block(prompt, last_user, c0, c1, n_blocks, pick)) {
        const std::string prefix = prompt.substr(0, c0);
        const std::string through = prompt.substr(0, c1);
        qbegin = (int) llama_driver_tokenize(st.vocab, prefix, true).size();
        qend = (int) llama_driver_tokenize(st.vocab, through, true).size();
        kvmem_diag("KVMEM_TRACE query_loc method=role_block n_user_blocks=%d pick=%d "
                "span=[%zu,%zu) tokens=[%d,%d)\n",
                n_blocks, pick, c0, c1, qbegin, qend);
    }
    if (qend > (int) toks.size()) {
        qend = (int) toks.size();
    }
    if (qbegin < 0 || qend <= qbegin) {
        qend = (int) toks.size();
        const int last = std::min(st.query_last_fallback, qend);
        qbegin = qend > last ? qend - last : 0;
        kvmem_diag("KVMEM_TRACE query_loc method=query_last tokens=[%d,%d)\n",
                qbegin, qend);
    }
    if (qbegin >= qend) {
        qbegin = 0;
    }
}

std::vector<kvmem::GraphMessageSpan> llama_driver_graph_messages(const LlamaEngineState& st,
    const common_chat_templates_inputs& inputs,const common_chat_params& formatted,const kvmem_prompt& prompt) {
    // Recover ChatML headers only from template-origin text, never from user
    // content. Unsupported templates/media retain the specified unknown weight.
    if(prompt.has_media()) return {};
    const auto text=formatted.prompt.str();
    const std::string marker="<|im_start|>";
    std::vector<size_t> boundaries;
    size_t base=0;
    for(const auto& part:formatted.prompt.parts) {
        if(!part.is_input) for(size_t at=part.text.find(marker);at!=std::string::npos;
            at=part.text.find(marker,at+marker.size())) boundaries.push_back(base+at);
        base+=part.text.size();
    }
    if(boundaries.size()!=inputs.messages.size()+1) return {};
    for(size_t i=0;i<boundaries.size();++i) {
        const auto begin=boundaries[i]+marker.size();
        const auto end=text.find_first_of(" \n\r",begin);
        const auto role=text.substr(begin,end-begin);
        if(role!=(i<inputs.messages.size()?inputs.messages[i].role:"assistant")) return {};
    }
    std::vector<uint64_t> rows(boundaries.size(),UINT64_MAX);
    size_t cursor=0,next=0;
    for(size_t row=0;row<prompt.tokens.size();++row) {
        while(next<boundaries.size() && boundaries[next]==cursor) rows[next++]=row;
        const auto piece=common_token_to_piece(st.vocab,prompt.tokens[row],true);
        if(text.compare(cursor,piece.size(),piece)!=0) {
            if(row==0 && prompt.tokens[row]==llama_vocab_bos(st.vocab)) continue;
            return {};
        }
        cursor+=piece.size();
        if(next<boundaries.size() && boundaries[next]<cursor) return {};
    }
    if(cursor!=text.size() || next!=boundaries.size()) return {};
    std::vector<kvmem::GraphMessageSpan> result;
    for(size_t i=0;i+1<rows.size();++i) if(rows[i]<rows[i+1]) result.push_back({rows[i],rows[i+1],true});
    result.push_back({rows.back(),UINT64_MAX,false});
    return result;
}

bool llama_driver_native_query_span(const LlamaEngineState & st, const std::string & formatted,
                                      const common_chat_templates_inputs & inputs, const kvmem_prompt & prompt,
                                      int & begin, int & end) {
    // Render the structured prefix ending at the real last user. This retains
    // native vision wrappers and does not confuse tool_response user-style
    // blocks with an actual user message. Require an exact causal prefix match.
    auto prefix_inputs = inputs;
    while (!prefix_inputs.messages.empty() && prefix_inputs.messages.back().role != "user")
        prefix_inputs.messages.pop_back();
    if (prefix_inputs.messages.empty()) return false;
    prefix_inputs.add_generation_prompt = false;
    std::string prefix;
    try {
        prefix = common_chat_templates_apply(st.tmpls.get(), prefix_inputs).prompt.str();
    } catch (const std::exception &) {
        return false; // A template may require the full trailing tool sequence.
    }
    size_t c0 = 0, c1 = 0;
    int count = 0, pick = -1;
    if (!find_last_user_role_block(prefix, "", c0, c1, count, pick) ||
            formatted.compare(0, c1, prefix, 0, c1) != 0) return false;
    int full_count = 0, unused = -1;
    if (!find_last_user_role_block(formatted, "", c0, c1, full_count, unused)) return false;
    common_chat_msg_delimiters delimiters;
    delimiters.add(COMMON_CHAT_ROLE_USER, "<|im_start|>user");
    delimiters.add(COMMON_CHAT_ROLE_UNKNOWN, "<|im_end|>");
    delimiters.tokenize(st.vocab);
    const auto spans = prompt.message_spans(delimiters);
    std::vector<common_chat_msg_span> users;
    for (const auto & span : spans.spans) if (span.role == COMMON_CHAT_ROLE_USER) users.push_back(span);
    if ((int) users.size() != full_count || pick < 0 || pick >= (int) users.size()) return false;
    begin = users[pick].pos + delimiters.delimiters.front().tokens.size();
    end = users[pick].pos + users[pick].len;
    for (const auto & image : prompt.media_ranges()) {
        if ((int) image.first >= begin && (int) image.second <= end) begin = image.second;
    }
    if (begin >= end) return false;
    std::string text;
    for (int row = begin; row < end; ++row)
        text += common_token_to_piece(st.vocab, prompt.tokens[row], false);
    if (trim_copy(text).empty()) return false;
    kvmem_diag("KVMEM_TRACE query_loc method=native_role pick=%d tokens=[%d,%d)\n", pick, begin, end);
    return true;
}

void llama_driver_clamp_query(const LlamaEngineState & st, int & qbegin, int & qend) {
    const int cap = st.query_max_tokens;
    if (cap > 0 && qend > qbegin && (qend - qbegin) > cap) {
        const auto bounded=kvmem::bounded_query(std::max(0,qbegin),qend-std::max(0,qbegin),qend,cap);
        kvmem_diag("KVMEM_TRACE query_clamp span=[%d,%d) tokens=%d cap=%d -> [%d,%d)\n",
                qbegin, qend, qend - qbegin, cap, qend - cap, qend);
        qbegin = int(bounded.begin);
    }
}
