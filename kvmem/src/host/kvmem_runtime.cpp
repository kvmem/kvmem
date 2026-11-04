#include "kvmem/kvmem_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace kvmem {

KvMemRuntime::KvMemRuntime(KvMemRuntimeConfig cfg, KvMemBackend *backend)
    : cfg_(std::move(cfg)),
      store_(cfg_.store),
      backend_(backend ? backend : &null_backend_) {
    trace_ = std::getenv("KVMEM_TRACE") != nullptr;
    slot_bytes_ = cfg_.store.estimated_block_bytes;
    if (cfg_.cpu_bytes > 0 && slot_bytes_ > 0) {
        PinnedKvTierConfig pcfg;
        pcfg.total_bytes = cfg_.cpu_bytes;
        pcfg.slot_bytes = slot_bytes_;
        cpu_tier_ = std::make_unique<PinnedKvTier>(pcfg);
        if (cpu_tier_->enabled()) {
            cpu_arena_.assign(
                static_cast<size_t>(cpu_tier_->slot_count()) * slot_bytes_, 0);
        }
    }
    const bool gpu_fmt_spill = cpu_tier_ && cpu_tier_->enabled();
    if (slot_bytes_ > 0 && gpu_fmt_spill) {
        scratch_.assign(static_cast<size_t>(slot_bytes_), 0);
    }
    if (cpu_tier_) {
        fprintf(stderr, "KVMEM_TIERS cpu_bytes=%llu cpu_slots=%u slot_bytes=%llu\n",
                (unsigned long long)cfg_.cpu_bytes, cpu_tier_->slot_count(),
                (unsigned long long)slot_bytes_);
    }
}

void KvMemRuntime::rebind_backend(KvMemBackend * backend) {
    if (pending_ || !pending_gpu_frees_.empty()) {
        throw std::logic_error("cannot rebind a runtime with a pending plan");
    }
    for (const auto & block : store_.blocks()) {
        if (block.gpu_slot >= 0 || block.in_flight) {
            throw std::logic_error("cannot rebind a GPU-resident runtime");
        }
    }
    backend_ = backend ? backend : &null_backend_;
}

void KvMemRuntime::clone_cpu_tier_from(const KvMemRuntime & source) {
    if (this == &source) throw std::invalid_argument("cannot clone a CPU tier onto itself");
    if (pending_ || source.pending_ || !pending_gpu_frees_.empty() || !source.pending_gpu_frees_.empty())
        throw std::logic_error("cannot clone a CPU tier with pending transfers");
    if (cfg_.cpu_bytes != source.cfg_.cpu_bytes || slot_bytes_ != source.slot_bytes_ ||
        cpu_arena_.size() != source.cpu_arena_.size())
        throw std::invalid_argument("incompatible CPU tier geometry");
    auto tier = source.cpu_tier_ ? std::make_unique<PinnedKvTier>(*source.cpu_tier_) : nullptr;
    // Prepare all throwing metadata first. The matching destination arena was
    // already allocated by its runtime; copying into it needs no third arena.
    if (!cpu_arena_.empty())
        std::memcpy(cpu_arena_.data(), source.cpu_arena_.data(), cpu_arena_.size());
    cpu_tier_ = std::move(tier);
}

uint8_t *KvMemRuntime::cpu_ptr(int32_t slot) {
    if (slot < 0 || cpu_arena_.empty()) {
        return nullptr;
    }
    return cpu_arena_.data() + static_cast<size_t>(slot) * slot_bytes_;
}

const uint8_t *KvMemRuntime::cpu_ptr(int32_t slot) const {
    if (slot < 0 || cpu_arena_.empty()) {
        return nullptr;
    }
    return cpu_arena_.data() + static_cast<size_t>(slot) * slot_bytes_;
}

void KvMemRuntime::trace_tier(const char *tag, uint32_t block_id, int32_t slot) const {
    if (!trace_) {
        return;
    }
    fprintf(stderr, "KVMEM_TRACE %s block=%u slot=%d\n", tag, block_id, slot);
}

void KvMemRuntime::register_append(uint32_t n_tokens) {
    store_.register_append(n_tokens);
}

std::vector<KvMemDroppedBlock> KvMemRuntime::truncate_to(uint32_t token_pos) {
    auto dropped = store_.truncate_to(token_pos);
    for (const auto &d : dropped) {
        if (cpu_tier_ && d.cpu_slot >= 0) {
            cpu_tier_->release_block(d.block_id);
        }

        if (d.gpu_slot >= 0) {
            backend_->free_gpu_slot(d.gpu_slot);
        }
    }
    return dropped;
}

KvMemPlan KvMemRuntime::prepare_reselect(
        const std::vector<uint32_t> &mandatory, bool force_raw_refresh) {
    return prepare_selection(preview_reselect(mandatory), force_raw_refresh);
}

std::vector<uint32_t> KvMemRuntime::preview_reselect(const std::vector<uint32_t> & mandatory) const {
    return store_.pick_topk_blocks(mandatory);
}

KvMemPlan KvMemRuntime::prepare_selection(const std::vector<uint32_t> & selected, bool force_raw_refresh) {
    if (pending_) throw std::logic_error("cannot replace a pending residency plan");
    native_plan_.reset();
    last_plan_ = store_.set_selection(selected, force_raw_refresh);
    pending_ = true;
    return last_plan_;
}

bool KvMemRuntime::commit_resident_selection(const std::vector<uint32_t> & selected) {
    if (pending_ || !store_.commit_resident_selection(selected)) return false;
    native_plan_.reset();
    last_plan_ = {};
    for (uint32_t id : selected) last_plan_.total_window_tokens += store_.blocks()[id].n_tokens;
    return true;
}

KvMemPlan KvMemRuntime::prepare_prefill_pressure(
        const std::vector<uint32_t> &mandatory) {
    return prepare_selection(store_.pick_prefill_pressure_blocks(mandatory));
}

NativeResidencyPlan KvMemRuntime::prepare_native_selection(const std::vector<uint32_t> & selected) {
    if (pending_) throw std::logic_error("cannot replace a pending residency plan");
    auto plan = store_.plan_native_selection(selected);
    // Allocate the return copy before installing pending state.
    native_plan_ = plan;
    last_plan_ = {};
    pending_ = true;
    return plan;
}

bool KvMemRuntime::maybe_offload_native_during_prefill(uint32_t incoming_tokens,
        uint32_t resident_tokens, uint32_t pool_tokens,
        const std::vector<uint32_t> & mandatory) {
    if (!store_.prefill_needs_offload(resident_tokens, incoming_tokens, pool_tokens)) return false;
    prepare_native_selection(store_.pick_prefill_pressure_blocks(mandatory));
    return true;
}

bool KvMemRuntime::maybe_offload_during_prefill(
        uint32_t incoming_tokens,
        uint32_t resident_tokens,
        uint32_t pool_tokens,
        const std::vector<uint32_t> &mandatory) {
    if (!store_.prefill_needs_offload(resident_tokens, incoming_tokens, pool_tokens)) {
        return false;
    }
    prepare_prefill_pressure(mandatory);
    return true;
}

void KvMemRuntime::spill_outgoing() {
    if (!pending_) {
        return;
    }
    pending_gpu_frees_.clear();
    for (uint32_t id : stage_out_blocks()) {
        stage_out(id);
    }
}

void KvMemRuntime::admit_incoming() {
    if (!pending_) {
        return;
    }
    for (int32_t slot : pending_gpu_frees_) {
        backend_->free_gpu_slot(slot);
    }
    pending_gpu_frees_.clear();
    for (uint32_t id : stage_in_blocks()) {
        stage_in(id);
    }
    if (native_plan_ && !store_.commit_resident_selection(native_plan_->selected))
        throw std::runtime_error("native slot admission did not produce the selected residency");
    pending_ = false;
}

void KvMemRuntime::finish_reselect() {
    spill_outgoing();
    admit_incoming();
}

void KvMemRuntime::stage_out(uint32_t block_id) {
    if (block_id >= store_.block_count()) return;
    const int32_t gpu_slot = store_.blocks()[block_id].gpu_slot;

    const bool persist = cfg_.store.select_method == KvMemMethod::Retrieval;
    // Packed K/V already live in RawKvStore. GPU-format D2H is only for the
    // CPU block arena; skip it when those tiers are off.
    if (persist && gpu_slot >= 0 && slot_bytes_ > 0 && !scratch_.empty()) {
        backend_->copy_block_to_host(block_id, gpu_slot, scratch_.data(), slot_bytes_);
    }

    if (cpu_tier_ && cpu_tier_->enabled() && slot_bytes_ > 0 && !cpu_arena_.empty()) {
        auto p = cpu_tier_->place_block_evicting(block_id);
        if (p.slot >= 0) {
            if (p.evicted_block >= 0) {
                const uint32_t vic = static_cast<uint32_t>(p.evicted_block);
                store_.set_block_cpu_copy(vic, -1);
                if (store_.blocks()[vic].tier == KvTier::CPU) {
                    store_.set_block_tier(vic, KvTier::CPU, -1);
                }
            }
            if (!scratch_.empty()) {
                std::memcpy(cpu_ptr(p.slot), scratch_.data(), static_cast<size_t>(slot_bytes_));
            }
            store_.set_block_tier(block_id, KvTier::CPU, p.slot);
            store_.set_block_cpu_copy(block_id, p.slot);
            if (gpu_slot >= 0) {
                pending_gpu_frees_.push_back(gpu_slot);
            }
            trace_tier("stage_out_cpu", block_id, p.slot);
            return;
        }
    }

    // No lower-tier capacity: drop GPU residency.
    store_.set_block_tier(block_id, KvTier::CPU, -1);
    if (gpu_slot >= 0) {
        pending_gpu_frees_.push_back(gpu_slot);
    }
}

void KvMemRuntime::stage_in(uint32_t block_id) {
    if (block_id >= store_.block_count()) return;
    const KvMemBlock b = store_.blocks()[block_id];
    if (b.tier == KvTier::GPU && b.gpu_slot >= 0) {
        return;
    }
    const int32_t gpu = backend_->alloc_gpu_slot();
    if (native_plan_ && gpu < 0)
        throw std::runtime_error("no GPU slot for native admission");
    const uint8_t *src = nullptr;
    try {
        if (b.cpu_slot >= 0) src = cpu_ptr(b.cpu_slot);
        if (src && gpu >= 0 && slot_bytes_ > 0) {
            backend_->copy_block_from_host(block_id, gpu, src, slot_bytes_);
        }
    } catch (...) {
        // The source copy still owns these bytes. Return only the unused
        // destination; no block metadata has been changed yet.
        if (gpu >= 0) backend_->free_gpu_slot(gpu);
        throw;
    }
    if (cpu_tier_ && b.cpu_slot >= 0) {
        cpu_tier_->release_block(block_id);
        store_.set_block_cpu_copy(block_id, -1);
    }
    store_.set_block_tier(block_id, KvTier::GPU, -1);
    store_.set_block_gpu_slot(block_id, gpu);

}

} // namespace kvmem
