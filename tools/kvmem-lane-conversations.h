#pragma once

// Included after the server's conversation payload and checkpoint helpers.
class kvmem_lane_conversations {
    friend struct kvmem_lane_conversations_test_access;
    using bundle_ptr = std::unique_ptr<llama_kvmem_store_bundle, decltype(&llama_kvmem_store_bundle_free)>;
    struct snapshot {
        kvmem_store_match match;
        std::shared_ptr<const kvmem_prompt> prompt;
        uint32_t input_end = 0; // canonical full prior input, excluding generation header
    };
    struct entry {
        int lane = -1;
        bool busy = false, switching = false;
        bool cpu_arena_parked = false; // mu_-protected commitment, never inspect a moving bundle
        size_t refs = 0;
        uint64_t used = 0, bytes = 0;
        uint64_t native_bytes = 0, reserved_bytes = 0, attachment_bytes = 0;
        uint64_t cache_group = 0;
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
        bool source_pinned = false; // parked parent of an executing private branch
        bool reuse_observed = false;
        bool resident_reuse = false;
        bool content_continuation = false;
        uint64_t source_group = 0;
        size_t prepare_begin = 0;
        int observed_lcp = 0;
    };
    class reference {
    public:
        ~reference() {
            std::lock_guard<std::mutex> lock(owner_.mu_);
            auto it = owner_.entries_.find(id_);
            if (it != owner_.entries_.end() && id_ >= 0) --it->second.refs;
        }
    private:
        friend class kvmem_lane_conversations;
        reference(kvmem_lane_conversations & owner, const std::string &) : owner_(owner) {}
        kvmem_lane_conversations & owner_;
        int id_ = -1;
    };
    kvmem_lane_conversations(const std::vector<ServerState *> & lanes, int count, uint64_t bytes,
                            bool inactive_limit = false)
        : lanes_(lanes), resident_(lanes.size()), max_(count), inactive_limit_(inactive_limit), bytes_max_(bytes) {
        snapshot_slots_ = std::max(lanes.size(), size_t(std::max(1,count))*2);
        if (!lanes.empty()) cpu_arena_per_store_ = lanes.front()->kparams.cpu_bytes;
        const auto arena_stores = uint64_t(snapshot_slots_) + uint64_t(lanes.size());
        if (cpu_arena_per_store_ && arena_stores > UINT64_MAX / cpu_arena_per_store_)
            throw std::overflow_error("legacy CPU arena budget overflow");
        cpu_arena_budget_ = arena_stores * cpu_arena_per_store_;
        for (size_t lane = 0; lane < lanes.size(); ++lane) {
            if (lanes[lane]->kparams.cpu_bytes != cpu_arena_per_store_)
                throw std::invalid_argument("KVMem lanes need identical CPU arena commitments");
            const int id = next_id_++;
            auto & held = entries_.emplace(id, entry{}).first->second;
            held.lane = (int)lane;
            held.cache_group = next_group_++;
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
        return std::shared_ptr<reference>(new reference(*this, key));
    }
    size_t preview(const kvmem_prompt & prompt, const std::string & key, bool reset,
                   const std::shared_ptr<reference> & ref, bool discover = true) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto plan = plan_locked(prompt, key, discover);
        // A queued preview is a rechecked hint. Only an admitted native
        // operation reserves its source, so waiting HTTP requests cannot pin
        // all retained metadata or RAM behind unrelated storage pressure.
        (void) ref;
        return reset ? 0 : (size_t) std::max(0, plan.keep);
    }
    int select(const kvmem_prompt & prompt, const std::string & key, bool reset,
               const std::vector<bool> & busy, bool reserve, operation & op, bool discover = true) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto plan = plan_locked(prompt, key, discover);
        const int source = plan.action == kvmem_store_action::extend ? plan.id : plan.source;
        // Preserving a resident working set also requires an exact completed
        // endpoint in its published restore directory. Coverage limits may
        // retain only an earlier legal point; use the ordinary private fork
        // from that point instead of making the warm optimization fail.
        const auto exact_resident_tip = [&] {
            if(source<0)return false;
            const auto& held=entries_.at(source);
            const auto& match=held.committed.match;
            return held.lane>=0 && match.live_row>0 &&
                std::find(match.ckpt_rows.begin(),match.ckpt_rows.end(),match.live_row)!=match.ckpt_rows.end();
        };
        const bool resident_reuse = !reset && plan.action == kvmem_store_action::extend && exact_resident_tip();
        bool fork = !reset && plan.action != kvmem_store_action::fresh;
        if (fork && !eligible(entries_.at(source), busy)) return -1;
        int id = -1;
        bool fresh = reset || plan.action == kvmem_store_action::fresh;
        if (!resident_reuse) {
            for (const auto & item : entries_) {
                const auto & held = item.second;
                if (held.committed.match.rows == 0 && !protected_entry(held) && eligible(held, busy)) {
                    // A resident target cannot be rebound on the source lane.
                    // When capacity permits, allocate a detached branch instead.
                    if (fork && held.lane >= 0 &&
                            entries_.size() < snapshot_slots_) continue;
                    id = item.first; break;
                }
            }
        }
        if (id < 0 && !resident_reuse && entries_.size() >= snapshot_slots_) {
            // Queued previews are hints, not capacity reservations. select()
            // replans and media_ready() rechecks any evicted prefix. Reclaim
            // idle capacity without privileging any client-supplied label.
            if (id < 0) for (const auto& item : entries_) {
                if (protected_entry(item.second) || !eligible(item.second, busy) || (fork && item.first == source)) continue;
                if (prefer_victim(item.second,id<0?nullptr:&entries_.at(id))) id = item.first;
            }
            // An executing branch may temporarily exceed the idle directory
            // count while its source is leased. Never destroy the source to
            // manufacture room for a fork; normal finish enforces retention.
            if (id < 0 && !fork) return -1;
        }
        if (id >= 0 && !eligible(entries_.at(id), busy)) return -1;
        if (fork && id >= 0 && (id == source || entries_.at(id).lane >= 0)) {
            // The destination's resident binding cannot be moved between
            // lanes. Allocate a branch while the source is leased instead.
            id = -1;
        }
        const size_t begin = fresh && !fork ? 0 : (size_t) std::max(0, plan.keep);
        op.prepare_begin = begin;
        op.observed_lcp = reset ? 0 : plan.observed_lcp;
        if (!prompt.media_ready(begin)) return -2;
        int lane = fork ? entries_.at(source).lane : id >= 0 ? entries_.at(id).lane : -1;
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
        entries_.at(id).cache_group = next_group_++;
        op.lane = lane;
        op.id = id;
        op.outgoing = resident_[lane];
        op.fresh = fresh;
        op.source = fork ? source : -1;
        op.resident_reuse = resident_reuse;
        op.content_continuation = fork && plan.action == kvmem_store_action::extend &&
            entries_.at(source).committed.input_end && uint64_t(plan.lcp)>=entries_.at(source).committed.input_end;
        op.source_group = fork ? entries_.at(source).cache_group : 0;
        if (fork) { entries_.at(op.source).switching = true; op.source_reserved = true; }
        entries_.at(id).busy = true;
        size_t active_branches=0;
        for(const auto& item:entries_)active_branches+=item.second.busy;
        peak_admitted_requests_=std::max(peak_admitted_requests_,active_branches);
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
        // Reserve independently from raw Host/Disk payload quotas, before any
        // source-preserving native fork or fresh runtime constructor allocates.
        cpu_arena_reservation arena_reservation(*this, op);
        const auto fork_started = std::chrono::steady_clock::now();
        uint64_t fork_host_bytes = 0;
        if (op.resident_reuse) {
            // Keep the GPU working set in place, but seal an independent host
            // source before this request can truncate/append/replay anything.
            auto payload = llama_driver_snapshot_conversation(st, int(llama_kvmem_store_n_tokens()));
            bundle_ptr branch {llama_kvmem_store_bundle_fork(nullptr), llama_kvmem_store_bundle_free};
            fork_host_bytes = llama_kvmem_store_bundle_bytes(branch.get());
            outgoing->payload = std::move(payload);
            outgoing->parked = std::move(branch);
        } else if (fork_source) {
            auto& source = *fork_source;
            if (source.lane < 0) restore_attachments(source);
            auto payload = source.lane < 0 ? llama_driver_fork_conversation(source.payload, int(op.prepare_begin))
                                          : llama_driver_fork_conversation(st, int(op.prepare_begin));
            bundle_ptr branch {llama_kvmem_store_bundle_fork(source.lane < 0 ? source.parked.get() : nullptr), llama_kvmem_store_bundle_free};
            fork_host_bytes = llama_kvmem_store_bundle_bytes(branch.get());
            incoming->payload = std::move(payload); incoming->parked = std::move(branch);
            op.fresh = false;
        }
        bool restaged = true;
        if (op.resident_reuse) {
            // The complete GPU state already belongs to the new private branch.
        } else if (op.id == op.outgoing) {
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
        if (fork_source) {
            snapshot_fork_us_ += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()-fork_started).count());
            snapshot_fork_host_bytes_ += fork_host_bytes;
            if (op.resident_reuse) ++resident_reuses_;
        }
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
            const auto input_end=outgoing->committed.input_end;
            outgoing->committed = describe(outgoing->payload, llama_kvmem_store_bundle_rows(outgoing->parked.get()));
            outgoing->committed.input_end=input_end;
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
        if (op.source_reserved) {
            // Handover no longer needs exclusive access. Sibling requests may
            // fork this source, but an executing branch must never evict its
            // parked parent to pay for its own checkpoint or restore staging.
            auto & source = entries_.at(op.source);
            ++source.refs;
            op.source_pinned = true;
            source.switching = false;
            op.source_reserved = false;
        }
        arena_reservation.finish_locked();
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
                if (prefer_victim(held,victim==entries_.end()?nullptr:&victim->second)) victim = it;
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
            void release_locked() {
                if (!op.source_reserved && !op.source_pinned) return;
                const auto source = owner.entries_.find(op.source);
                if (source != owner.entries_.end()) {
                    if (op.source_reserved) source->second.switching = false;
                    if (op.source_pinned) --source->second.refs;
                }
                op.source_reserved = op.source_pinned = false;
            }
            ~source_guard() {
                if (!op.source_reserved && !op.source_pinned) return;
                std::lock_guard<std::mutex> lock(owner.mu_);
                release_locked();
            }
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
            if(st.history_input_end && st.history_input_end<=committed.match.live_row)
                committed.input_end=st.history_input_end;
            const uint64_t bytes = payload_bytes(st) + llama_kvmem_store_active_bytes();
            const uint64_t native = llama_kvmem_store_payload_bytes(llama_kvmem_store_current(), false);
            st.active_prompt.reset();
            std::lock_guard<std::mutex> lock(mu_);
            const int active = op.attached ? op.id : op.outgoing;
            auto & held = entries_.at(active);
            if(!held.cache_group)held.cache_group=next_group_++;
            held.committed = std::move(committed);
            held.bytes = bytes;
            held.native_bytes = native;
            held.reserved_bytes = 0;
            held.used = ++clock_;
            classify_content_group_locked(active,held);
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
            // Native completion no longer needs the source lease. Release it
            // before publication so a shared template prefix cannot pin the
            // only retained group and reject an otherwise eligible new history.
            release_source.release_locked();
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
        if (actual_tokens && op.content_continuation && op.source_group) {
            auto branch = entries_.find(op.id);
            if (branch != entries_.end()) branch->second.cache_group = op.source_group;
        }
        planned_reuse_tokens_ += op.prepare_begin;
        actual_reuse_tokens_ += actual_tokens;
        if (!actual_tokens) ++actual_misses_;
        else if (op.source >= 0 && !op.resident_reuse) ++actual_shared_hits_;
        else ++actual_own_hits_;
    }
    static json reuse_observation(const operation& op,std::optional<size_t> actual=std::nullopt) {
        const bool snapshot=op.source>=0 && !op.resident_reuse;
        return {{"cache_semantics_revision",2},
            {"planned_cached_tokens",op.prepare_begin},
            {"planned_cache_source",!op.prepare_begin?"miss":snapshot?"shared":"own"},
            {"actual_cached_tokens",actual?json(*actual):json(nullptr)},
            {"cache_source",!actual?json(nullptr):!*actual?json("miss"):snapshot?json("shared"):json("own")},
            {"restore_path",!actual?json(nullptr):!*actual?json("miss"):snapshot?json("snapshot"):json("continuation")}};
    }
    json status(int lane = -1) const {
        std::lock_guard<std::mutex> lock(mu_);
        size_t active = 0, retained = 0;
        std::set<uint64_t> groups;
        for (const auto & item : entries_) {
            active += item.second.busy ? 1 : 0;
            retained += !item.second.busy && item.second.committed.match.rows > 0 ? 1 : 0;
            if(item.second.committed.match.rows>0)groups.insert(item.second.cache_group);
        }
        json result = {{"count", entries_.size()}, {"max", max_}, {"bytes", bytes_locked()},
            {"active_sessions", active}, {"retained_sessions", groups.size()},
            {"retained_snapshot_count", retained}, {"retained_groups", groups.size()},
            {"retained_sessions_limit", max_}, {"retained_snapshot_limit", snapshot_slots_},
            {"session_limit_kind", "content_groups"},
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
        result["shared_prefix_fork_supported"] = true;
        result["content_cache_revision"] = 1;
        result["peak_admitted_requests"] = peak_admitted_requests_;
        result["cache_semantics_revision"] = 2;
        result["snapshot_fork_host_bytes_scope"] = "logical_host_including_shared_payload";
        result["resident_reuses"] = resident_reuses_;
        result["snapshot_fork_us"] = snapshot_fork_us_;
        result["snapshot_fork_logical_host_bytes"] = snapshot_fork_host_bytes_;
        const auto arena_live = cpu_arena_live_stores_locked() * cpu_arena_per_store_;
        result["cpu_arena_budget_bytes"] = cpu_arena_budget_;
        result["cpu_arena_per_store_bytes"] = cpu_arena_per_store_;
        result["cpu_arena_live_charge_bytes"] = arena_live;
        result["cpu_arena_reserved_bytes"] = cpu_arena_reserved_stores_ * cpu_arena_per_store_;
        result["cpu_arena_charged_bytes"] = arena_live + cpu_arena_reserved_stores_ * cpu_arena_per_store_;
        result["cpu_arena_reservation_failures"] = cpu_arena_reservation_failures_;
        result["cpu_arena_accounting_scope"] = "configured_per_store_commitments; active+parked+inflight; separate_from_raw_H_D_and_RSS";
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
    std::shared_ptr<kvmem::SpillFile> disk_;
    std::filesystem::path disk_path_;
    uint64_t payload_budget_ = 0, snapshot_generation_ = 0;
    int max_, next_id_ = 0;
    bool inactive_limit_ = false;
    bool warned_bytes_ = false;
    uint64_t bytes_max_, clock_ = 0, switches_ = 0, evictions_ = 0, extends_ = 0, forks_ = 0, shared_forks_ = 0;
    uint64_t actual_own_hits_ = 0, actual_shared_hits_ = 0, actual_misses_ = 0;
    uint64_t planned_reuse_tokens_ = 0, actual_reuse_tokens_ = 0;
    uint64_t resident_reuses_ = 0, snapshot_fork_us_ = 0, snapshot_fork_host_bytes_ = 0;
    size_t peak_admitted_requests_ = 0;
    size_t snapshot_slots_ = 0;
    uint64_t cpu_arena_per_store_ = 0, cpu_arena_budget_ = 0;
    uint64_t cpu_arena_reserved_stores_ = 0, cpu_arena_reservation_failures_ = 0;
    uint64_t next_group_ = 1;
    uint64_t cpu_arena_live_stores_locked() const noexcept {
        uint64_t count = lanes_.size(); // Every lane keeps one configured runtime.
        for (const auto& item : entries_) count += item.second.cpu_arena_parked;
        return count;
    }
    class cpu_arena_reservation {
    public:
        cpu_arena_reservation(kvmem_lane_conversations& owner, const operation& op)
            : owner_(owner), incoming_(op.id), outgoing_(op.outgoing) {
            if (!owner_.cpu_arena_per_store_) return;
            std::lock_guard<std::mutex> lock(owner_.mu_);
            const bool allocate = op.resident_reuse || op.source >= 0 ||
                (op.id != op.outgoing && !owner_.entries_.at(op.id).cpu_arena_parked);
            if (!allocate) return;
            const auto capacity = owner_.cpu_arena_budget_ / owner_.cpu_arena_per_store_;
            const auto live = owner_.cpu_arena_live_stores_locked();
            if (live >= capacity || owner_.cpu_arena_reserved_stores_ >= capacity - live) {
                ++owner_.cpu_arena_reservation_failures_;
                throw std::runtime_error("legacy CPU arena capacity reserved by retained sources or in-flight forks");
            }
            ++owner_.cpu_arena_reserved_stores_;
            charged_ = true;
        }
        cpu_arena_reservation(const cpu_arena_reservation&) = delete;
        cpu_arena_reservation& operator=(const cpu_arena_reservation&) = delete;
        ~cpu_arena_reservation() {
            if (!owner_.cpu_arena_per_store_ || finished_) return;
            std::lock_guard<std::mutex> lock(owner_.mu_);
            finish_locked();
        }
        void finish_locked() noexcept {
            if (!owner_.cpu_arena_per_store_ || finished_) return;
            // attach owns these switching entries. After success OR exception,
            // account any installed native handle before releasing its token.
            // Other threads only read this marker, not the moving unique_ptr.
            for (const auto id : {incoming_, outgoing_}) {
                const auto it = owner_.entries_.find(id);
                if (it != owner_.entries_.end()) it->second.cpu_arena_parked = bool(it->second.parked);
            }
            if (charged_) --owner_.cpu_arena_reserved_stores_;
            finished_ = true;
        }
    private:
        kvmem_lane_conversations& owner_;
        int incoming_, outgoing_;
        bool charged_ = false, finished_ = false;
    };
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
            // Handover moves the native payload, not this committed metadata.
            // Keep its proven prefix in the plan: select() waits for eligible
            // instead of prematurely choosing a much shorter unrelated source.
            if (item.second.busy) continue;
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
                                    lanes_[0]->spec.ok, key, limits, discover);
    }
    bool eligible(const entry & held, const std::vector<bool> & busy) const {
        return !held.busy && !held.switching && (held.lane < 0 || !busy[(size_t)held.lane]);
    }
    bool protected_entry(const entry & held) const {
        return held.refs > 0;
    }
    bool prefer_victim(const entry& candidate,const entry* current) const {
        if(!current)return true;
        std::vector<kvmem::ContentCacheSlot> slots;
        size_t candidate_slot=0;
        for(const auto& item:entries_) {
            const auto& held=item.second;
            if(&held==&candidate)candidate_slot=slots.size();
            slots.push_back({held.committed.match.rows>0 || &held==&candidate || &held==current,
                &held!=&candidate && &held!=current,held.used,nullptr,held.cache_group});
        }
        const auto selected=kvmem::select_content_eviction(slots);
        return selected && *selected==candidate_slot;
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
        bytes += source.mm_archive.capacity()*sizeof(kvmem::CheckpointArchive<MultimodalCheckpoint>::Entry);
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
    void classify_content_group_locked(int incoming_id,entry& incoming) {
        const auto& input=incoming.committed;
        if(!input.prompt || !input.input_end || input.input_end>input.prompt->tokens.size()) {
            incoming.cache_group=next_group_++;return;
        }
        struct Group { bool compatible=true,exact=false;size_t common=0; };
        std::map<uint64_t,Group> groups;
        for(const auto& item:entries_) {
            if(item.first==incoming_id || !item.second.committed.match.rows || !item.second.cache_group)continue;
            const auto& prior=item.second.committed;
            auto& group=groups[item.second.cache_group];
            if(!prior.prompt || !prior.input_end || prior.input_end>prior.prompt->tokens.size()) {
                group.compatible=false;continue;
            }
            const auto required=std::min(input.input_end,prior.input_end);
            const auto lcp=input.prompt->common_prefix(*prior.prompt);
            if(lcp<required) {group.compatible=false;continue;}
            group.common=std::max(group.common,size_t(required));
            group.exact|=input.input_end==prior.input_end;
        }
        uint64_t selected=0;Group best;
        for(const auto& item:groups) {
            const auto& group=item.second;
            if(!group.compatible)continue;
            if(!selected || (group.exact && !best.exact) ||
                    (group.exact==best.exact && group.common>best.common)) {
                selected=item.first;best=group;
            }
        }
        incoming.cache_group=selected?selected:next_group_++;
    }
    void enforce_sessions(int finishing_lane) {
        const auto incoming_id=resident_.at(size_t(finishing_lane));
        auto& incoming=entries_.at(incoming_id);
        if(incoming.committed.match.rows>0) {
            // The incoming version is already attached, but is excluded from
            // the directory view so the common planner reserves exactly one
            // slot for it. Groups express capacity fairness, never identity.
            std::vector<int> ids;
            std::vector<kvmem::ContentCacheSlot> slots;
            std::vector<std::unique_lock<std::mutex>> lane_locks;
            for(const auto& item:entries_) {
                const auto& held=item.second;
                bool pinned=held.busy || held.switching || protected_entry(held);
                if(item.first!=incoming_id && held.lane>=0 && held.lane!=finishing_lane && !pinned) {
                    std::unique_lock<std::mutex> candidate(lanes_[held.lane]->mu,std::try_to_lock);
                    if(!candidate.owns_lock())pinned=true;
                    lane_locks.push_back(std::move(candidate));
                }
                ids.push_back(item.first);
                slots.push_back({item.first!=incoming_id && held.committed.match.rows>0,
                    pinned,held.used,nullptr,held.cache_group});
            }
            const auto plan=kvmem::plan_content_group_publication(slots,snapshot_slots_,size_t(max_),incoming.cache_group);
            if(plan) {
                for(auto slot:plan.retired_slots) {
                    auto victim=entries_.find(ids.at(slot));
                    if(victim==entries_.end())continue;
                    auto& held=victim->second;
                    if(held.lane>=0) {
                        auto& state=*lanes_[held.lane];
                        kvmem_execution_scope execution(state.execution.get());
                        llama_driver_clear(state);
                        held.committed=snapshot{};held.client_id.clear();
                        held.bytes=payload_bytes(state)+llama_kvmem_store_active_bytes();
                        held.native_bytes=llama_kvmem_store_payload_bytes(llama_kvmem_store_current(),false);
                    } else entries_.erase(victim);
                    ++evictions_;
                }
            } else {
                // Publication is optional. Keep this request's valid native
                // binding while declining new directory visibility.
                incoming.committed=snapshot{};
            }
        }
        for (auto it = entries_.begin(); it != entries_.end();) {
            const auto & held = it->second;
            if (held.lane < 0 && !held.busy && !held.switching && !protected_entry(held) && !held.committed.match.rows)
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
                if (prefer_victim(held,victim==entries_.end()?nullptr:&victim->second)) victim = it;
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
        const auto fits=[&] { const auto used=account->used();
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
                if(charged && prefer_victim(held,victim==entries_.end()?nullptr:&victim->second)) victim=it;
            }
            if(victim==entries_.end()) break;
            kvmem_diag("KVMEM_CHECKPOINT_HOST evict_idle_owner=%d\n",victim->first);
            entries_.erase(victim);
            ++evictions_;
        }
    }
};
