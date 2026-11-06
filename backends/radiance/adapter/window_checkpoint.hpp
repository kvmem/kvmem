#pragma once
#include "state_checkpoint.hpp"

namespace kvmem::radiance {
// Immutable native window bytes at an accepted frontier. Windows are copied
// immediately: their rolling pages may be overwritten by the next draft pass.
class WindowCheckpoint {
    rad::KVManager* owner_ = nullptr;
    std::vector<rad::KVManager::WindowTable> tables_;
    int64_t frontier_ = 0, floor_ = 0;
    std::vector<uint8_t> bytes_;
    std::unique_ptr<TieredBuffer> parked_;
    std::shared_ptr<StorageIoQueue> io_;
    bool valid_ = false;

    struct Copies {
        std::vector<GroupPool> pools;
        std::vector<CopySpan> spans;
        uint64_t bytes = 0;
    };
    Copies copies(const std::vector<rad::KVManager::WindowTable>& tables) const {
        Copies result;
        for (const auto& table : tables) {
            const auto* plan = owner_->plan(table.group);
            if (!plan || plan->kind != RAD_KV_WINDOW || plan->runtime_win ||
                !plan->base || plan->pool_bytes <= 0 || plan->layer_bytes_per_block() <= 0)
                throw std::runtime_error("window checkpoint geometry changed");
            result.pools.push_back({table.group, plan->base, uint64_t(plan->pool_bytes)});
            for (int32_t layer = 0; layer < plan->n_layers; ++layer) {
                for (auto page : table.blocks) {
                    const auto count = uint64_t(plan->layer_bytes_per_block());
                    if (page < 0 || page >= owner_->live_blocks(table.group) ||
                        owner_->block_refcount(table.group, page) < 1 ||
                        count > uint64_t(INT64_MAX) - result.bytes)
                        throw std::runtime_error("window checkpoint page is not backed");
                    result.spans.push_back({table.group,
                        uint64_t(layer) * plan->layer_stride() + uint64_t(page) * count,
                        result.bytes, count});
                    result.bytes += count;
                }
            }
        }
        return result;
    }
    void prepare(rad::KVManager& kv, uint64_t sequence, int64_t frontier) {
        if (valid_ || frontier < 0 || !kv.has_sequence(sequence) || frontier > kv.seq_tokens(sequence))
            throw std::runtime_error("invalid window checkpoint frontier");
        owner_ = &kv; frontier_ = frontier; floor_ = kv.window_floor(sequence);
        if (floor_ < 0 || floor_ > frontier_) throw std::runtime_error("invalid window checkpoint floor");
        for (const auto& plan : kv.plans()) {
            if (plan.kind != RAD_KV_WINDOW || plan.zero) continue;
            if (!plan.drafter || plan.runtime_win || plan.block_size <= 0)
                throw std::runtime_error("unsupported window checkpoint group");
            const auto original_first = kv.first_block_pos(sequence, plan.index);
            if (original_first < 0 || original_first > frontier || original_first % plan.block_size ||
                plan.window_blocks <= 0) throw std::runtime_error("window starts beyond checkpoint");
            // Match native trim_window at the accepted frontier. Older pages
            // retained for a previous prefill chunk are no longer conditioning;
            // duplicating those for forks can exceed the native window pool.
            const auto first = std::max(original_first,
                std::max<int64_t>(0, frontier / plan.block_size - (plan.window_blocks - 1)) * plan.block_size);
            auto blocks = kv.block_table(sequence, plan.index);
            const auto skip = size_t((first - original_first) / plan.block_size);
            const auto count = size_t((frontier - first + plan.block_size - 1) / plan.block_size);
            if (skip > blocks.size() || blocks.size() - skip < count)
                throw std::runtime_error("window checkpoint has missing pages");
            blocks.erase(blocks.begin(), blocks.begin() + skip);
            blocks.resize(count); // Exclude unevaluated reserved lookahead pages.
            tables_.push_back({plan.index, first, std::move(blocks)});
            floor_ = std::max(floor_, first);
        }
    }
public:
    static uint64_t required_bytes(rad::KVManager& kv, uint64_t sequence, int64_t frontier) {
        WindowCheckpoint geometry;
        geometry.prepare(kv, sequence, frontier);
        return geometry.copies(geometry.tables_).bytes;
    }
    WindowCheckpoint fork() const {
        if (!valid_) throw std::runtime_error("cannot fork an incomplete window checkpoint");
        WindowCheckpoint next;
        next.owner_ = owner_; next.tables_ = tables_;
        next.frontier_ = frontier_; next.floor_ = floor_; next.bytes_ = bytes_;
        if (parked_) next.parked_ = parked_->fork();
        next.io_ = io_; next.valid_ = true;
        return next;
    }
    void save(rad::KVManager& kv, uint64_t sequence, int64_t frontier,
              int device, RadStream stream) {
        prepare(kv, sequence, frontier);
        const auto batch = copies(tables_);
        if (batch.bytes) {
            PageTransfer transfer(device, batch.bytes, stream);
            transfer.spill(batch.spans, batch.pools, stream);
            if (!transfer.drain()) throw std::runtime_error("window checkpoint copy failed");
            bytes_ = transfer.payload();
        }
        valid_ = true;
    }
    std::vector<uint8_t> load_cpu() const {
        if (!valid_) throw std::runtime_error("incomplete window checkpoint");
        return parked_ ? parked_->read() : bytes_;
    }
    size_t bytes() const noexcept { return parked_ ? parked_->size() : bytes_.size(); }
    void account(SharedPayloadAccounting& accounting) const {
        if (parked_) parked_->account(accounting);
    }
    void park(const std::shared_ptr<HostKvStorage>& storage, const std::shared_ptr<StorageIoQueue>& io) {
        if (!valid_) throw std::runtime_error("incomplete window checkpoint");
        if (bytes()) park_buffer(bytes_, parked_, storage, io);
        io_ = io;
    }
    void restore(rad::KVManager& kv, uint64_t sequence, int device, RadStream stream,
                 const std::vector<uint8_t>* prepared = nullptr) const {
        if (!valid_ || owner_ != &kv || kv.seq_tokens(sequence) != frontier_ ||
            (prepared && prepared->size() != bytes()))
            throw std::runtime_error("window checkpoint destination mismatch");
        if (tables_.empty()) return;
        const auto loaded = !prepared ? load_cpu() : std::vector<uint8_t>{};
        const auto& payload = prepared ? *prepared : loaded;
        auto tables = tables_;
        std::vector<std::pair<int32_t, int32_t>> reserved;
        size_t count = 0;
        for (const auto& table : tables) {
            count += table.blocks.size();
            if (kv.reserve_release_space(table.group,
                    table.blocks.size() + kv.block_table(sequence, table.group).size()) < 0)
                throw std::runtime_error("window restore release reservation failed");
        }
        reserved.reserve(count);
        auto release = [&] {
            for (const auto& item : reserved) kv.release_block(item.first, item.second);
            reserved.clear();
        };
        try {
            for (auto& table : tables) for (auto& page : table.blocks) {
                int32_t allocated = -1;
                if (kv.reserve_block(table.group, &allocated) < 0)
                    throw MemoryError(MemoryErrorCode::BudgetExceeded, "window restore allocation failed: group=" +
                        std::to_string(table.group) + " need=" + std::to_string(table.blocks.size()) +
                        " reserved=" + std::to_string(reserved.size()) + " free=" +
                        std::to_string(kv.free_blocks(table.group)) + " frontier=" + std::to_string(frontier_));
                reserved.emplace_back(table.group, allocated); page = allocated;
            }
            const auto batch = copies(tables);
            if (batch.bytes != payload.size()) throw std::runtime_error("window restore size changed");
            if (batch.bytes) {
                PageTransfer transfer(device, batch.bytes, stream);
                try {
                    transfer.restore(batch.spans, batch.pools, stream, payload);
                    if (!transfer.drain()) throw std::runtime_error("window restore copy failed");
                } catch (...) {
                    if (!transfer.drain()) reserved.clear();
                    throw;
                }
            }
            if (kv.restore_window_tables(sequence, frontier_, floor_, tables) < 0)
                throw std::runtime_error("window restore publication failed");
            release();
        } catch (...) { release(); throw; }
    }
};
} // namespace kvmem::radiance
