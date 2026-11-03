#pragma once

// Included after the server's conversation payload and checkpoint helpers.
class kvmem_lane_conversations {
    friend struct kvmem_lane_conversations_test_access;
    using bundle_ptr = std::unique_ptr<llama_kvmem_store_bundle, decltype(&llama_kvmem_store_bundle_free)>;
    struct snapshot {
        kvmem_store_match match;
        std::shared_ptr<const kvmem_prompt> prompt;
    };
    struct entry {
        int lane = -1;
        bool busy = false, switching = false;
        size_t refs = 0;
        uint64_t used = 0, bytes = 0;
        uint64_t native_bytes = 0, reserved_bytes = 0, attachment_bytes = 0;
        std::string client_id;
        bundle_ptr parked {nullptr, llama_kvmem_store_bundle_free};
        kvmem_conversation payload;
        snapshot committed;
        std::unique_ptr<kvmem_session_files> files;
    };
public:
    struct operation {
        int lane = -1, id = -1, outgoing = -1, source = -1;
        bool fresh = false, added = false, attached = false, ready = false, finished = false, source_reserved = false;
        bool reuse_observed = false;
        size_t prepare_begin = 0;
        int observed_lcp = 0;
    };
    class reference {
    public:
        ~reference() {
            std::lock_guard<std::mutex> lock(owner_.mu_);
            if (registered_ && !key_.empty()) {
                auto it = owner_.waiting_keys_.find(key_);
                if (it != owner_.waiting_keys_.end() && --it->second == 0) owner_.waiting_keys_.erase(it);
            }
            auto it = owner_.entries_.find(id_);
            if (it != owner_.entries_.end() && id_ >= 0) --it->second.refs;
        }
    private:
        friend class kvmem_lane_conversations;
        reference(kvmem_lane_conversations & owner, std::string key) : owner_(owner), key_(std::move(key)) {}
        kvmem_lane_conversations & owner_;
        std::string key_;
        int id_ = -1;
        bool registered_ = false;
    };
    kvmem_lane_conversations(const std::vector<ServerState *> & lanes, int count, uint64_t bytes,
                            bool inactive_limit = false)
        : lanes_(lanes), resident_(lanes.size()), max_(count), inactive_limit_(inactive_limit), bytes_max_(bytes) {
        for (size_t lane = 0; lane < lanes.size(); ++lane) {
            const int id = next_id_++;
            auto & held = entries_.emplace(id, entry{}).first->second;
            held.lane = (int)lane;
            kvmem_execution_scope execution(lanes[lane]->execution.get());
            held.bytes = llama_kvmem_store_active_bytes();
            if (lane == 0) {
                disk_ = kvmem_execution_spill_file(lanes[lane]->execution.get());
                if (disk_) {
                    payload_budget_ = llama_kvmem_payload_budget();
                    disk_path_ = std::filesystem::u8path(lanes[lane]->kparams.payload_disk_dir);
                }
            }
            held.native_bytes = llama_kvmem_store_payload_bytes(llama_kvmem_store_current(), false);
            resident_[lane] = id;
        }
        for(auto* lane:lanes_) lane->mm_checkpoint_reclaim = [this](const auto& account,size_t bytes) {
            reclaim_checkpoint_host(account,bytes);
        };
    }
    ~kvmem_lane_conversations() {
        for(auto* lane:lanes_) lane->mm_checkpoint_reclaim = {};
    }
    std::shared_ptr<reference> pin(const std::string & key) {
        auto result = std::shared_ptr<reference>(new reference(*this, key));
        std::lock_guard<std::mutex> lock(mu_);
        if (!key.empty()) ++waiting_keys_[key];
        result->registered_ = true;
        return result;
    }
    size_t preview(const kvmem_prompt & prompt, const std::string & key, bool reset,
                   const std::shared_ptr<reference> & ref, bool discover = true) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto plan = plan_locked(prompt, key, discover);
        if (!reset && ref && ref->id_ < 0 && plan.id >= 0) {
            ref->id_ = plan.id;
            ++entries_.at(plan.id).refs;
        }
        return reset ? 0 : (size_t) std::max(0, plan.keep);
    }
    int select(const kvmem_prompt & prompt, const std::string & key, bool reset,
               const std::vector<bool> & busy, bool reserve, operation & op, bool discover = true) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto plan = plan_locked(prompt, key, discover);
        bool fork = !reset && plan.action == kvmem_store_action::fork;
        if (fork && !eligible(entries_.at(plan.source), busy)) return -1;
        int id = reset || fork ? -1 : plan.id;
        bool fresh = reset || plan.action == kvmem_store_action::fresh;
        if (id < 0 && !fork && !key.empty()) {
            for (const auto & item : entries_) if (item.second.client_id == key) { id = item.first; break; }
        }
        if (id < 0) {
            for (const auto & item : entries_) {
                const auto & held = item.second;
                if (held.client_id.empty() && held.committed.match.rows == 0 && eligible(held, busy)) {
                    // A resident target cannot be rebound on the source lane.
                    // When capacity permits, allocate a detached branch instead.
                    if (fork && held.lane >= 0 &&
                            (inactive_limit_ || entries_.size() < (size_t)max_)) continue;
                    id = item.first; break;
                }
            }
        }
        if (id < 0 && !inactive_limit_ && entries_.size() >= (size_t)max_) {
            // Queued previews are hints, not capacity reservations. select()
            // replans and media_ready() rechecks any evicted prefix. Prefer the
            // request's old slot, including when it is also the proposed source.
            if (!key.empty()) {
                for (const auto& item : entries_) {
                    if (item.second.client_id == key && eligible(item.second, busy)) {
                        id = item.first;
                        break;
                    }
                }
            }
            if (id < 0) for (const auto& item : entries_) {
                if (!eligible(item.second, busy) || (fork && item.first == plan.source)) continue;
                if (id < 0 || item.second.used < entries_.at(id).used) id = item.first;
            }
            // No other idle slot exists: recompute on the source's own slot
            // rather than wait forever for a queued request to release a hint.
            if (id < 0 && fork) id = plan.source;
            if (id < 0) return -1;
        }
        if (id >= 0 && !eligible(entries_.at(id), busy)) return -1;
        if (fork && id >= 0 && (id == plan.source || entries_.at(id).lane >= 0)) {
            fork = false;
            fresh = true;
        }
        const size_t begin = fresh && !fork ? 0 : (size_t) std::max(0, plan.keep);
        op.prepare_begin = begin;
        op.observed_lcp = reset ? 0 : plan.observed_lcp;
        if (!prompt.media_ready(begin)) return -2;
        int lane = fork ? entries_.at(plan.source).lane : id >= 0 ? entries_.at(id).lane : -1;
        if (lane < 0) {
            for (size_t candidate = 0; candidate < busy.size(); ++candidate) {
                const auto & outgoing = entries_.at(resident_[candidate]);
                if (!busy[candidate] && !outgoing.busy && !outgoing.switching) { lane = (int)candidate; break; }
            }
        }
        if (lane < 0) return -1;
        if (id >= 0 && entries_.at(id).lane >= 0 && entries_.at(id).lane != lane)
            throw std::logic_error("cache target is resident on another lane");
        if (!reserve) return lane;
        op.added = id < 0;
        if (op.added) {
            id = next_id_;
            entries_.emplace(id, entry{});
            ++next_id_;
        }
        op.lane = lane;
        op.id = id;
        op.outgoing = resident_[lane];
        op.fresh = fresh;
        op.source = fork ? plan.source : -1;
        if (fork) { entries_.at(op.source).switching = true; op.source_reserved = true; }
        entries_.at(id).busy = true;
        if (id != op.outgoing) entries_.at(op.outgoing).switching = true;
        return lane;
    }
    void attach(operation & op, ServerState & st, const std::string & key) {
        std::string client_id = key; // Allocate before changing either owner.
        entry * incoming;
        entry * outgoing;
        entry * fork_source = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu_);
            incoming = &entries_.at(op.id);
            outgoing = &entries_.at(op.outgoing);
            if (op.source >= 0) fork_source = &entries_.at(op.source);
        }
        // source_reserved prevents erasure while fork/I/O runs without mu_.
        // All map lookups above must remain under mu_, including the source.
        if (fork_source) {
            auto& source = *fork_source;
            if (source.lane < 0) restore_attachments(source);
            auto payload = source.lane < 0 ? llama_driver_fork_conversation(source.payload, int(op.prepare_begin))
                                          : llama_driver_fork_conversation(st, int(op.prepare_begin));
            bundle_ptr branch {llama_kvmem_store_bundle_fork(source.lane < 0 ? source.parked.get() : nullptr), llama_kvmem_store_bundle_free};
            incoming->payload = std::move(payload); incoming->parked = std::move(branch);
            op.fresh = false;
        }
        bool restaged = true;
        if (op.id == op.outgoing) {
            if (op.fresh) llama_driver_clear(st);
        } else {
            if (!op.fresh) restore_attachments(*incoming);
            bundle_ptr bundle {nullptr, llama_kvmem_store_bundle_free};
            bundle = std::move(incoming->parked);
            if (!bundle) bundle.reset(llama_kvmem_store_bundle_create());
            if (op.fresh) {
                llama_kvmem_store_bundle_reset(bundle.get());
                llama_driver_drop_conversation(incoming->payload);
                incoming->files.reset();
            }
            llama_driver_swap_conversation(st, outgoing->payload);
            try {
                restaged = llama_kvmem_store_bundle_swap(bundle.get());
            } catch (...) {
                incoming->parked = std::move(bundle);
                llama_driver_swap_conversation(st, outgoing->payload);
                if (!llama_kvmem_store_n_tokens() && !st.cached_tokens.empty()) llama_driver_clear(st);
                throw;
            }
            outgoing->parked = std::move(bundle);
            llama_driver_swap_conversation(st, incoming->payload);
            {
                std::lock_guard<std::mutex> lock(mu_);
                // The handover has happened even if subsequent repair fails.
                incoming->lane = op.lane;
                outgoing->lane = -1;
                resident_[op.lane] = op.id;
                op.attached = true;
            }
            st.mm_live_checkpoint.reset();
            if (disk_) { st.mm_rollback.reset(); st.mm_rollback_prompt.reset(); st.mm_pending_query.reset(); }
            if (!restaged && !st.cached_tokens.empty()) llama_driver_clear(st);
            if (!llama_kvmem_store_bundle_rows(outgoing->parked.get())) llama_driver_drop_conversation(outgoing->payload);
        }
        st.mm_reset_requested = false;
        st.mm_observed_lcp = op.fresh ? op.observed_lcp : 0;
        const auto incoming_native = llama_kvmem_store_payload_bytes(llama_kvmem_store_current(), false);
        std::unique_lock<std::mutex> lock(mu_);
        // Publish the actual owners before any accounting allocation can fail.
        incoming->lane = op.lane;
        resident_[op.lane] = op.id;
        op.attached = true;
        if (op.fresh && !op.added && (!incoming->client_id.empty() || incoming->committed.match.rows > 0)) ++evictions_;
        incoming->client_id.swap(client_id);
        incoming->native_bytes = incoming_native;
        incoming->attachment_bytes = 0;
        if (op.id != op.outgoing) {
            outgoing->lane = -1;
            outgoing->committed = describe(outgoing->payload, llama_kvmem_store_bundle_rows(outgoing->parked.get()));
            outgoing->bytes = payload_bytes(outgoing->payload) + llama_kvmem_store_bundle_bytes(outgoing->parked.get());
            outgoing->native_bytes = llama_kvmem_store_bundle_payload_bytes(outgoing->parked.get());
            if (disk_ && bytes_max_ && bytes_locked() > bytes_max_) {
                // switching protects this owner while its lane thread performs
                // I/O outside the catalog lock and outside other GPU lanes.
                lock.unlock();
                demote_attachments(op.outgoing, *outgoing);
                lock.lock();
                outgoing->bytes = payload_bytes(outgoing->payload) + llama_kvmem_store_bundle_bytes(outgoing->parked.get());
            }
            outgoing->switching = false;
            ++switches_;
        }
        if (op.fresh || op.source >= 0) ++forks_; else ++extends_;
        if(op.source>=0)++shared_forks_;
        if (op.source_reserved) { entries_.at(op.source).switching = false; op.source_reserved = false; }
        op.ready = true;
        kvmem_diag("KVMEM_LANE_STORE lane=%d out=%d in=%d fresh=%d restaged=%d stores=%zu\n",
                op.lane, op.outgoing, op.id, (int)op.fresh, (int)restaged, entries_.size());
    }
    void reserve_payload(const operation & op, ServerState & st, const kvmem_prompt & prompt, int predict) {
        if (!disk_) return;
        const uint64_t rows = uint64_t(prompt.tokens.size()) + uint64_t(std::max(0, predict)) +
            (st.spec.ok ? uint64_t(std::max(0, st.spec_n_max)) + 1 : 0);
        if (rows > UINT32_MAX) throw std::invalid_argument("native KV token count overflow");
        const auto required = llama_kvmem_store_payload_capacity(uint32_t(rows));
        std::lock_guard<std::mutex> lock(mu_);
        auto& owner = entries_.at(op.id);
        const auto reservation = required > owner.native_bytes ? required-owner.native_bytes : 0;
        const auto fits = [&] {
            const auto used = retained_bytes_locked(op.id);
            return used <= payload_budget_ && reservation <= payload_budget_ - used;
        };
        while (!fits()) {
            auto victim = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                const auto& held = it->second;
                if (it->first == op.id || held.lane >= 0 || held.busy || held.switching || protected_entry(held)) continue;
                if (!held.native_bytes && !held.attachment_bytes) continue;
                if (victim == entries_.end() || held.used < victim->second.used) victim = it;
            }
            if (victim == entries_.end())
                throw std::runtime_error("native KV capacity is reserved by other requests or retained lane owners");
            if (victim->first == op.outgoing) {
                // Reclaim the previous owner's content here; session quota
                // enforcement can retire the empty catalog node afterwards.
                auto& held = victim->second;
                llama_kvmem_store_bundle_reset(held.parked.get());
                llama_driver_drop_conversation(held.payload);
                held.files.reset();
                held.committed = snapshot{};
                held.client_id.clear();
                held.native_bytes = held.attachment_bytes = held.bytes = 0;
            } else entries_.erase(victim);
            ++evictions_;
        }
        owner.reserved_bytes = reservation;
    }
    void finish(operation & op, ServerState & st) noexcept {
        if (op.finished) return;
        op.finished = true;
        struct source_guard {
            kvmem_lane_conversations& owner; operation& op;
            ~source_guard() { if(op.source_reserved) { std::lock_guard<std::mutex> lock(owner.mu_); auto it=owner.entries_.find(op.source); if(it!=owner.entries_.end())it->second.switching=false; op.source_reserved=false; } }
        } release_source{*this,op};
        try {
            kvmem_execution_scope execution(st.execution.get());
            llama_driver_finish(st);
            if (st.cached_prompt) st.cached_prompt = st.cached_prompt->cache_index();
            if (st.mm_query && st.mm_query->prefix) {
                auto query = std::make_shared<MultimodalQuery>(*st.mm_query);
                query->prefix = query->prefix->cache_index();
                st.mm_query = std::move(query);
            }
            auto committed = op.attached && !op.ready ? snapshot{} : describe(st, llama_kvmem_store_n_tokens());
            const uint64_t bytes = payload_bytes(st) + llama_kvmem_store_active_bytes();
            const uint64_t native = llama_kvmem_store_payload_bytes(llama_kvmem_store_current(), false);
            st.active_prompt.reset();
            std::lock_guard<std::mutex> lock(mu_);
            const int active = op.attached ? op.id : op.outgoing;
            auto & held = entries_.at(active);
            held.committed = std::move(committed);
            held.bytes = bytes;
            held.native_bytes = native;
            held.reserved_bytes = 0;
            held.used = ++clock_;
            if (!op.attached && op.added) entries_.erase(op.id);
            else entries_.at(op.id).busy = false;
            held.switching = false;
            if (!op.ready) {
                auto target = entries_.find(op.id);
                if (target != entries_.end()) target->second.committed = snapshot{};
                if (op.attached && op.outgoing != op.id && op.source != op.outgoing) entries_.at(op.outgoing).committed = snapshot{};
                entries_.at(op.outgoing).switching = false;
            }
            // Successful attach released outgoing. Another lane may already
            // have evicted it or reserved it for a different handover.
            enforce_sessions(op.lane);
            enforce_bytes();
        } catch (const std::exception & e) {
            LOG_ERR("srv    KVMEM lane commit failed: %s\n", e.what());
            std::lock_guard<std::mutex> lock(mu_);
            if (!op.attached && op.added) entries_.erase(op.id);
            auto it = entries_.find(op.attached ? op.id : op.outgoing);
            if (it != entries_.end()) { it->second.busy = false; it->second.reserved_bytes = 0; it->second.committed = snapshot{}; }
            auto target = entries_.find(op.id);
            if (target != entries_.end()) target->second.busy = false;
            auto out = entries_.find(op.outgoing);
            if (!op.ready && out != entries_.end()) { out->second.switching = false; if(op.source!=op.outgoing)out->second.committed = snapshot{}; }
        }
        st.active_prompt.reset();
    }
    void observe_reuse(operation & op, size_t actual_tokens) {
        std::lock_guard<std::mutex> lock(mu_);
        if (op.reuse_observed) return;
        op.reuse_observed = true;
        planned_reuse_tokens_ += op.prepare_begin;
        actual_reuse_tokens_ += actual_tokens;
        if (!actual_tokens) ++actual_misses_;
        else if (op.source >= 0) ++actual_shared_hits_;
        else ++actual_own_hits_;
    }
    json status(int lane = -1) const {
        std::lock_guard<std::mutex> lock(mu_);
        size_t active = 0, retained = 0;
        for (const auto & item : entries_) {
            active += item.second.busy ? 1 : 0;
            retained += !item.second.busy && item.second.committed.match.rows > 0 ? 1 : 0;
        }
        json result = {{"count", entries_.size()}, {"max", max_}, {"bytes", bytes_locked()},
            {"active_sessions", active}, {"retained_sessions", retained},
            {"retained_sessions_limit", inactive_limit_ ? json(max_) : json(nullptr)},
            {"session_limit_kind", inactive_limit_ ? "inactive" : "legacy_total"},
            {"bytes_max", bytes_max_}, {"switches", switches_}, {"evictions", evictions_},
            {"extends", extends_}, {"forks", forks_}};
        uint64_t logical = 0;
        for (const auto& item : entries_) logical += item.second.native_bytes;
        const auto physical = llama_kvmem_execution_storage_stats(lanes_[0]->execution.get());
        result["native_logical_bytes"] = logical;
        result["host_payload_bytes"] = physical.host_bytes;
        result["disk_payload_bytes"] = physical.disk_bytes;
        result["physical_payload_bytes"] = physical.host_bytes + physical.disk_bytes;
        result["sharing_credit_lower_bound_bytes"] = logical > physical.host_bytes + physical.disk_bytes ? logical-physical.host_bytes-physical.disk_bytes : 0;
        result["shared_forks"] = shared_forks_;
        result["history_hits"] = actual_shared_hits_ + actual_own_hits_;
        result["own_history_hits"] = actual_own_hits_;
        result["shared_history_hits"] = actual_shared_hits_;
        result["history_misses"] = actual_misses_;
        result["planned_reuse_tokens"] = planned_reuse_tokens_;
        result["actual_reuse_tokens"] = actual_reuse_tokens_;
        result["shared_prefix_fork_supported"] = lanes_[0]->kparams.cpu_bytes == 0;
        if (lane >= 0) {
            const int id = resident_.at((size_t)lane);
            result["resident"] = id;
            result["switching"] = entries_.at(id).switching;
        }
        return result;
    }
private:
    mutable std::mutex mu_;
    std::vector<ServerState *> lanes_;
    std::vector<int> resident_;
    std::map<int, entry> entries_;
    std::map<std::string, size_t> waiting_keys_;
    std::shared_ptr<kvmem::SpillFile> disk_;
    std::filesystem::path disk_path_;
    uint64_t payload_budget_ = 0, snapshot_generation_ = 0;
    int max_, next_id_ = 0;
    bool inactive_limit_ = false;
    bool warned_bytes_ = false;
    uint64_t bytes_max_, clock_ = 0, switches_ = 0, evictions_ = 0, extends_ = 0, forks_ = 0, shared_forks_ = 0;
    uint64_t actual_own_hits_ = 0, actual_shared_hits_ = 0, actual_misses_ = 0;
    uint64_t planned_reuse_tokens_ = 0, actual_reuse_tokens_ = 0;
    uint64_t retained_bytes_locked(int exclude = -1) const {
        const auto physical = llama_kvmem_execution_storage_stats(lanes_[0]->execution.get());
        uint64_t bytes = physical.host_bytes + physical.disk_bytes;
        for (const auto& item : entries_) {
            if (item.first == exclude) continue;
            const auto extra = item.second.reserved_bytes;
            if (extra > UINT64_MAX-bytes) throw std::overflow_error("retained KV capacity overflow");
            bytes += extra;
        }
        return bytes;
    }
    void restore_attachments(entry& held) {
        if (!held.payload.payload) return;
        auto& payload = *held.payload.payload;
        for (uint32_t i = 0; i < payload.chunks.size(); ++i) {
            if (held.files) payload.restore(*held.files, i);
            else if (!payload.chunks[i].in_ram) throw std::logic_error("cold conversation has no attachment owner");
        }
        if (!payload.complete()) throw std::runtime_error("incomplete parked conversation restore");
        llama_kvmem_store_bundle_thaw(held.parked.get());
        held.payload.payload.reset();
        held.payload.cold = false;
    }
    void demote_attachments(int id, entry& held) noexcept {
        try {
            auto& conv = held.payload;
            if (!conv.payload) {
                std::vector<kvmem::SnapshotBuffer> buffers;
                llama_driver_conversation_buffers(conv, buffers);
                llama_kvmem_store_bundle_freeze(held.parked.get(), buffers);
                uint64_t generation;
                { std::lock_guard<std::mutex> lock(mu_); generation = ++snapshot_generation_; }
                try { conv.payload = std::make_unique<kvmem_session_payload>(id, generation, buffers); }
                catch (...) { llama_kvmem_store_bundle_thaw(held.parked.get()); throw; }
                conv.cold = true;
            }
            if (!held.files) held.files = std::make_unique<kvmem_session_files>(disk_path_, disk_);
            for (uint32_t i = 0; i < conv.payload->chunks.size(); ++i) {
                const auto& chunk = conv.payload->chunks[i];
                if (!chunk.in_ram) continue;
                const auto bytes = chunk.disk_bytes();
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    const auto used = retained_bytes_locked();
                    if (used > payload_budget_ || bytes > payload_budget_ - used) break;
                    held.attachment_bytes += bytes;
                }
                try { conv.payload->spill(*held.files, i); }
                catch (...) {
                    std::lock_guard<std::mutex> lock(mu_);
                    held.attachment_bytes = held.files->session_bytes(id);
                    throw;
                }
            }
        } catch (const std::exception& e) {
            LOG_WRN("srv    KVMEM parked attachment spill retained valid source: %s\n", e.what());
        }
    }
    kvmem_store_plan plan_locked(const kvmem_prompt & prompt, const std::string & key, bool discover) const {
        std::vector<kvmem_store_match> matches;
        for (const auto & item : entries_) {
            auto match = item.second.committed.match;
            match.id = item.first;
            match.used = item.second.used;
            match.bytes = item.second.bytes;
            match.client_id = item.second.client_id;
            match.lcp = item.second.committed.prompt ? (int)prompt.common_prefix(*item.second.committed.prompt) : 0;
            matches.push_back(std::move(match));
        }
        kvmem_store_limits limits;
        limits.max_stores = std::numeric_limits<int>::max();
        return kvmem_store_select_shared(matches, (int)prompt.tokens.size() - (lanes_[0]->spec.ok ? 1 : 0),
                                    lanes_[0]->spec.ok, key, limits, discover && !lanes_[0]->kparams.cpu_bytes);
    }
    bool eligible(const entry & held, const std::vector<bool> & busy) const {
        return !held.busy && !held.switching && (held.lane < 0 || !busy[(size_t)held.lane]);
    }
    bool protected_entry(const entry & held) const {
        return held.refs > 0 || (!held.client_id.empty() && waiting_keys_.count(held.client_id));
    }
    template<class Source> static snapshot describe(const Source & source, uint32_t stored) {
        snapshot result;
        result.prompt = source.cached_prompt;
        result.match.rows = (int)source.cached_tokens.size();
        result.match.last_n_gen = source.last_n_gen;
        result.match.allow_bound_edit = source.mm_archive.policy().enabled;
        // The committed native store bounds the live reusable prefix.
        result.match.live_row = (int)stored;
        for (const auto & checkpoint : source.mm_checkpoints) result.match.ckpt_rows.push_back(checkpoint.row);
        for (const auto & entry : source.mm_archive)
            if (entry.info.identity) result.match.ckpt_rows.push_back(entry.record.row);
        return result;
    }
    template<class Source> static uint64_t payload_bytes(const Source & source) {
        uint64_t bytes = source.cached_tokens.capacity()*sizeof(llama_token);
        bytes += source.mm_checkpoints.capacity()*sizeof(MultimodalCheckpoint);
        std::set<const MultimodalCheckpointData *> unique;
        for (const auto & checkpoint : source.mm_checkpoints) {
            if (checkpoint.data && unique.insert(checkpoint.data.get()).second) bytes += checkpoint.data->resident_bytes();
        }
        for (const auto & entry : source.mm_archive) if (entry.info.identity && entry.record.data &&
                unique.insert(entry.record.data.get()).second) bytes += entry.record.data->resident_bytes();
        if (source.mm_live_checkpoint && unique.insert(source.mm_live_checkpoint.get()).second) bytes += source.mm_live_checkpoint->resident_bytes();
        if (source.cached_prompt) bytes += source.cached_prompt->index_bytes();
        if (source.mm_query) {
            bytes += sizeof(MultimodalQuery) + source.mm_query->user.capacity();
            bytes += source.mm_query->state.count.capacity()*sizeof(uint32_t);
            for (const auto & sum : source.mm_query->state.rows) bytes += sizeof(sum) + sum.capacity()*sizeof(float);
            if (source.mm_query->prefix) bytes += source.mm_query->prefix->index_bytes();
        }
        return bytes;
    }
    uint64_t bytes_locked() const {
        uint64_t bytes = 0;
        for (const auto & item : entries_) bytes += item.second.bytes;
        return bytes;
    }
    void enforce_sessions(int finishing_lane) {
        if (!inactive_limit_) return;
        // Queued previews are hints, not retained-history reservations: select()
        // replans under mu_ and asks for media preparation again after eviction.
        // Never evict an active or switching owner. Resident idle lanes can be
        // cleared only while owning their execution lock. try_lock avoids the
        // inverse lock order of another lane concurrently entering finish().
        for (;;) {
            size_t retained = 0;
            for (const auto & item : entries_)
                retained += !item.second.busy && item.second.committed.match.rows > 0 ? 1 : 0;
            if (retained <= size_t(max_)) break;
            auto victim = entries_.end();
            std::unique_lock<std::mutex> lane_lock;
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                const auto & held = it->second;
                if (held.busy || held.switching || held.committed.match.rows == 0) continue;
                if (victim != entries_.end() && held.used >= victim->second.used) continue;
                std::unique_lock<std::mutex> candidate_lock;
                if (held.lane >= 0 && held.lane != finishing_lane) {
                    candidate_lock = std::unique_lock<std::mutex>(lanes_[held.lane]->mu, std::try_to_lock);
                    if (!candidate_lock.owns_lock()) continue;
                }
                victim = it;
                lane_lock = std::move(candidate_lock);
            }
            if (victim == entries_.end()) throw std::logic_error("inactive history quota has no idle victim");
            auto & held = victim->second;
            if (held.lane >= 0) {
                auto & state = *lanes_[held.lane];
                kvmem_execution_scope execution(state.execution.get());
                llama_driver_clear(state);
                held.committed = snapshot{};
                held.client_id.clear();
                held.bytes = payload_bytes(state) + llama_kvmem_store_active_bytes();
                held.native_bytes = llama_kvmem_store_payload_bytes(llama_kvmem_store_current(), false);
            } else {
                entries_.erase(victim);
            }
            ++evictions_;
        }
        for (auto it = entries_.begin(); it != entries_.end();) {
            const auto & held = it->second;
            if (held.lane < 0 && !held.busy && !held.switching && !held.committed.match.rows)
                it = entries_.erase(it);
            else ++it;
        }
    }
    void enforce_bytes() {
        while (bytes_max_ && bytes_locked() > bytes_max_) {
            auto victim = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                const auto & held = it->second;
                if (held.lane >= 0 || held.busy || held.switching || protected_entry(held)) continue;
                // Under SSD mode the soft RAM target cannot discard an owner
                // merely because its retained metadata/native working RAM is
                // above that target. Hard H/D admission remains authoritative.
                if (disk_ && held.payload.payload) continue;
                if (victim == entries_.end() || held.used < victim->second.used) victim = it;
            }
            if (victim == entries_.end()) break;
            entries_.erase(victim);
            ++evictions_;
        }
        const bool over = bytes_max_ && bytes_locked() > bytes_max_;
        if (over && !warned_bytes_) LOG_WRN("srv    KVMEM retained lane/queued stores exceed soft host byte cap\n");
        warned_bytes_ = over;
    }
    void reclaim_checkpoint_host(const std::shared_ptr<MultimodalCheckpointAccounting>& account,size_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto fits=[&] { const auto used=account->live_bytes.load();
            return used<=account->limit && bytes<=account->limit-used; };
        const auto owns=[&](const auto& checkpoint) {
            return checkpoint && checkpoint->accounting==account && checkpoint->resident_bytes()!=0;
        };
        while(!fits()) {
            auto victim=entries_.end();
            for(auto it=entries_.begin();it!=entries_.end();++it) {
                const auto& held=it->second;
                if(held.lane>=0 || held.busy || held.switching || protected_entry(held)) continue;
                bool charged=owns(held.payload.mm_live_checkpoint);
                for(const auto& cp:held.payload.mm_checkpoints) charged|=owns(cp.data);
                for(const auto& entry:held.payload.mm_archive) if(entry.info.identity) charged|=owns(entry.record.data);
                if(charged && (victim==entries_.end() || held.used<victim->second.used)) victim=it;
            }
            if(victim==entries_.end()) break;
            kvmem_diag("KVMEM_CHECKPOINT_HOST evict_idle_owner=%d\n",victim->first);
            entries_.erase(victim);
            ++evictions_;
        }
    }
};
