#pragma once

// Host spill for attention KV, indexed by logical block then layer.
//
// Product path: ordered K sum (F32) at first write, mean-K computed on read;
// packed GPU-format K/V (q8_0 etc)
// copied at stage-out. Restore is memcpy; orig pos does not need unrotated K.
// `write_layer_k_rows` (unrotated token-major K) remains for tests only.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <atomic>
#include "kvmem/host_kv_storage.hpp"

namespace kvmem {

class SnapshotWriter;
class SnapshotReader;
struct SnapshotBuffer;

struct RawKvStoreConfig {
    uint32_t n_layer = 0;
    uint32_t n_embd_k = 0;
    uint32_t n_embd_v = 0;
    uint32_t block_tokens = 128;
    uint64_t k_row_bytes = 0;      // 0 = FP16 (n_embd_k * 2); else opaque K row
    uint64_t k_gpu_row_bytes = 0;  // 0 = no packed GPU K; else bytes per token row
    uint64_t v_gpu_row_bytes = 0;  // 0 = no packed GPU V; else bytes per token row
    std::shared_ptr<HostKvStorage> payload_storage;
};

// Immutable packed-record exchange. Native layout/frontier proof remains with
// the caller; this API proves stored row coverage and account ownership only.
struct PackedKvSelection {
    uint32_t block = 0, layer = 0, valid_tokens = 0;
};
struct PackedKvOptions {
    // Caller-supplied metadata admission allowance, not a separate payload pool.
    size_t metadata_limit = SIZE_MAX;
    const std::atomic<bool> * cancelled = nullptr;
};
class PackedKvSnapshot {
public:
    struct Entry {
        PackedKvSelection selection;
        HostKvRecord k, v;
    };
    PackedKvSnapshot(PackedKvSnapshot &&) noexcept = default;
    PackedKvSnapshot & operator=(PackedKvSnapshot &&) noexcept = default;
    const std::vector<Entry> & entries() const noexcept { return entries_; }
    const RawKvStoreConfig & config() const noexcept { return config_; }
    size_t metadata_bytes() const noexcept {
        return sizeof(*this) + entries_.capacity() * sizeof(Entry);
    }
private:
    friend class RawKvStore;
    PackedKvSnapshot() = default;
    RawKvStoreConfig config_;
    std::vector<Entry> entries_;
};

class RawKvStore {
public:
    explicit RawKvStore(RawKvStoreConfig cfg);
    ~RawKvStore();

    RawKvStore(const RawKvStore &) = delete;
    RawKvStore &operator=(const RawKvStore &) = delete;

    const RawKvStoreConfig &config() const { return cfg_; }

    void ensure_blocks(uint32_t block_count);
    // Reserve packed destinations without marking any row valid. Caller has
    // drained capture/I/O writers before preparing a native transfer.
    void reserve_packed(uint32_t block_id, const std::vector<uint32_t> & layers);

    void write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                            const float * k, const float * v);
    void write_layer_tokens_f16(uint32_t pos0, uint32_t n, uint32_t il,
                                const uint16_t * k, const uint16_t * v);
    // Opaque unrotated K rows (`k_row_bytes`). Optional F32 is mean only.
    void write_layer_k_rows(uint32_t pos0, uint32_t n, uint32_t il,
                            const uint8_t * k, const float * k_f32);
    void write_layer_v_gpu(uint32_t pos0, uint32_t n, uint32_t il, const uint8_t * v);
    // Packed GPU-format K (RoPE+Hadamard+quant already applied). RAM only.
    void write_layer_k_gpu(uint32_t pos0, uint32_t n, uint32_t il, const uint8_t * k);
    // Prefill mean-K only (no raw-K rows). k is token-major F32, n_embd_k per token.
    void write_layer_mean_k(uint32_t pos0, uint32_t n, uint32_t il, const float * k);
    // Decode running-sum: `sum` is already reduced over `n` tokens in one block.
    void write_layer_mean_sum(uint32_t pos0, uint32_t n, uint32_t il, const float * sum);

    bool has_block(uint32_t block_id) const;
    bool has_k(uint32_t block_id, uint32_t il) const;
    bool has_k_gpu(uint32_t block_id, uint32_t il, uint32_t n = 1) const;
    bool has_v(uint32_t block_id, uint32_t il) const;
    bool has_v_gpu(uint32_t block_id, uint32_t il, uint32_t n = 1) const;
    uint32_t n_tokens(uint32_t block_id) const;

    bool copy_k(uint32_t block_id, uint32_t il, float * out) const;
    bool copy_v(uint32_t block_id, uint32_t il, float * out) const;
    bool copy_k_rows(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;
    bool copy_k_gpu(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;
    bool copy_v_gpu(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;

    // Normalize the stored sum by mean_tokens; absent statistics return zeros.
    void mean_k(uint32_t block_id, uint32_t il, float * out) const;

    size_t bytes_k() const;
    size_t bytes_v() const;
    size_t allocated_bytes() const;
    uint64_t payload_bytes(bool disk_only = false) const;
    uint64_t payload_capacity_bytes(uint32_t tokens, uint32_t populated_layers = UINT32_MAX) const;
    uint64_t capacity_bytes(uint32_t tokens, uint32_t populated_layers = UINT32_MAX) const;
    void snapshot_write(SnapshotWriter & out);
    void snapshot_read(SnapshotReader & in, uint32_t max_blocks);
    // Caller freezes the detached store until all bindings have been restored.
    void snapshot_buffers(std::vector<SnapshotBuffer> & buffers);


    void clear();
    // Preserve only the valid prefix, including a partial last block.
    void truncate_to(uint32_t token_pos);
    void invalidate_packed_from(uint32_t token_pos);
    // In-process tail checkpoint: per layer, valid count followed by the F32 sum.
    std::vector<float> mean_checkpoint(uint32_t token_pos) const;
    void restore_mean_checkpoint(uint32_t token_pos, const std::vector<float> & state);
    // Copy the mean-K index, never raw/packed KV. Used by a native branch whose
    // payload lives in its separately shared page archive. The caller restores
    // the checkpoint's partial-tail sum after truncation.
    std::unique_ptr<RawKvStore> fork_mean_prefix(uint32_t token_pos) const;
    // Caller has drained native writers. Packed records become sealed shared
    // sources; raw rows and mutable statistics remain private to the branch.
    std::unique_ptr<RawKvStore> fork_prefix(uint32_t token_pos);

    // PROTOTYPE: caller drains native writers and serializes store lifecycle.
    // Every selected layer requires both planes. Full records share storage;
    // partial records are private capacity-sized copies with zero padding.
    // Sealing/inspection is under mu_; payload allocation/read is outside mu_.
    PackedKvSnapshot export_packed(const std::vector<PackedKvSelection> & selection,
                                  PackedKvOptions options = {});
    // Atomically replace ALL packed bindings. Unselected packed planes vanish;
    // existing raw rows/MeanK statistics remain private and unchanged. Caller
    // must restore their checkpoint version and native layout separately.
    // Import rejects a foreign storage account. No payload allocation or I/O.
    void import_packed(const PackedKvSnapshot & snapshot, PackedKvOptions options = {});

private:
    struct LayerBlk {
        uint32_t n_tokens = 0;
        uint32_t k_gpu_tokens = 0;
        uint32_t v_gpu_tokens = 0;
        uint32_t mean_tokens = 0;
        std::vector<uint8_t> k;
        std::vector<uint16_t> v;
        HostKvRecord k_gpu;
        HostKvRecord v_gpu;
        // Export proof only. Legacy serialized records intentionally have none.
        std::vector<uint64_t> k_coverage, v_coverage;
        std::vector<float> k_sum;
        bool k_gpu_fmt = false;
        bool v_gpu_fmt = false;
    };
    struct BlockRaw {
        std::vector<LayerBlk> layers;
    };

    uint64_t k_row_bytes() const;
    bool k_is_f16() const;
    void capture_mean_f16(LayerBlk & lb) const;
    void add_mean_f32(LayerBlk & lb, uint32_t off, uint32_t take, const float * k);
    RawKvStoreConfig cfg_;
    std::shared_ptr<HostKvStorage> payload_storage_;
    std::vector<BlockRaw> blocks_;
    mutable std::mutex mu_;

};

} // namespace kvmem
