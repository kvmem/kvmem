#pragma once

#include "host_kv_storage.hpp"
#include "layerwise_query.hpp"
#include "spill_io.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kvmem {
namespace detail { struct QueryProjectionDescriptor; }

// An engine-owned format/version identifies the native Q/K/V, residual, gate,
// etc. byte representations. Planes are opaque; row_stride includes any native
// row padding. A chunk stores each plane contiguously, in this declared order.
// No tensor type, dtype conversion, layer computation or GPU access is involved.
struct QueryProjectionPlane {
    uint32_t id = 0;
    uint64_t row_stride = 0;
};
struct QueryProjectionLayout {
    std::string format;
    uint32_t version = 0;
    std::vector<QueryProjectionPlane> planes;
};
struct QueryProjectionPlaneBytes {
    uint32_t id = 0;
    const void* data = nullptr; // caller-owned CPU-readable rows; consumed synchronously
    size_t bytes = 0;
};

// A sealed immutable projected chunk. Its byte source and identity survive the
// tape owner and the HostKvStorage facade; a deferred CPU reader can retain this
// object without pinning RAM. Reads acquire only a short source read lease and
// never promote an entire cold record into RAM.
class QueryProjectionChunkSource {
public:
    const LayerwiseQueryIdentity& identity() const;
    uint32_t model_layer() const;
    QueryRange rows() const noexcept { return rows_; }
    const QueryProjectionLayout& layout() const;
    size_t size() const noexcept { return bytes_; }
    explicit operator bool() const noexcept { return bool(source_) || bool(spill_); }
    void read_rows(const LayerwiseQueryIdentity&, uint32_t model_layer,
                   uint32_t plane, uint64_t begin, uint64_t count,
                   void* destination, size_t destination_bytes) const;
private:
    friend class QueryProjectionTape;
    QueryProjectionChunkSource(std::shared_ptr<const detail::QueryProjectionDescriptor>,
                               QueryRange rows, HostKvSource);
    QueryProjectionChunkSource(std::shared_ptr<const detail::QueryProjectionDescriptor>,
                               QueryRange rows, SpillExtent, uint64_t offset, size_t bytes);
    std::shared_ptr<const detail::QueryProjectionDescriptor> descriptor_;
    QueryRange rows_;
    HostKvSource source_;
    SpillExtent spill_;
    uint64_t offset_ = 0;
    size_t bytes_ = 0;
};

// One fixed model layer and one exact Query evaluation span. The caller must
// supply a separate activation account explicitly; this class has no access to
// a canonical KV account. It reserves the complete layer byte count before any
// projected rows are admitted. Existing HostKvReservation admits against RAM,
// so total tape bytes must initially fit that account's RAM limit even if the
// account also has SSD backing. Sealed chunks can subsequently be evicted.
// The explicit SpillFile overload instead reserves the entire disk span and
// writes caller-supplied chunks directly, with no tape-owned RAM allocation.
// It permits a layer larger than the activation RAM account. A surviving source
// retains that complete disk span until its last reader releases it.
// Native owners decide execution order/fences and release sources after use.
class QueryProjectionTape {
public:
    QueryProjectionTape(HostKvStorage& activation_account, LayerwiseQueryIdentity,
                        uint32_t model_layer, QueryProjectionLayout);
    QueryProjectionTape(SpillFile& activation_spill, LayerwiseQueryIdentity,
                        uint32_t model_layer, QueryProjectionLayout);
    // Explicit finite tape metadata admission. Reserve the entire conservative
    // descriptor/layout/chunk-table bound before allocating that representation
    // or reserving/writing payload. The metadata account must be independent of
    // canonical KV. Payload bytes and caller-owned handles/buffers are separate.
    QueryProjectionTape(HostKvStorage& activation_account,HostKvStorage& metadata_account,
                        uint64_t max_chunks,LayerwiseQueryIdentity,
                        uint32_t model_layer,QueryProjectionLayout);
    QueryProjectionTape(SpillFile& activation_spill,HostKvStorage& metadata_account,
                        uint64_t max_chunks,LayerwiseQueryIdentity,
                        uint32_t model_layer,QueryProjectionLayout);
    static uint64_t metadata_bound(const QueryProjectionLayout&,uint64_t max_chunks);
    // Zero means the original, caller-qualified constructor did not install a
    // separate metadata limit. Exported sources retain the descriptor's full
    // conservative charge after abort/destruction, until their last owner dies.
    uint64_t metadata_bytes() const noexcept;
    uint64_t max_chunks() const noexcept;

    ~QueryProjectionTape();
    QueryProjectionTape(const QueryProjectionTape&) = delete;
    QueryProjectionTape& operator=(const QueryProjectionTape&) = delete;

    const LayerwiseQueryIdentity& identity() const;
    uint32_t model_layer() const;
    const QueryProjectionLayout& layout() const;
    uint64_t total_bytes() const noexcept;
    uint64_t reserved_bytes() const noexcept;
    uint64_t covered_end() const noexcept { return next_; }
    size_t chunk_count() const noexcept { return chunks_.size(); }
    bool complete() const noexcept { return state_ == State::Complete; }
    bool aborted() const noexcept { return state_ == State::Aborted; }

    QueryProjectionChunkSource append(const LayerwiseQueryIdentity&, uint32_t model_layer,
        uint64_t begin, uint64_t count, const std::vector<QueryProjectionPlaneBytes>&);
    // Complete only after exact contiguous coverage of the evaluation span.
    void finish(const LayerwiseQueryIdentity&, uint32_t model_layer);
    QueryProjectionChunkSource chunk(const LayerwiseQueryIdentity&, uint32_t model_layer,
                                     size_t index) const;
    // A bounded range may span sealed chunks. It must already be covered; full
    // layer completion is not required to read a previously sealed chunk.
    void read_rows(const LayerwiseQueryIdentity&, uint32_t model_layer,
                   uint32_t plane, uint64_t begin, uint64_t count,
                   void* destination, size_t destination_bytes) const;
    // No device work is cancelled here. Exported immutable sources stay valid;
    // unused reservation and tape-owned chunk references are released.
    void abort() noexcept;
private:
    enum class State { Recording, Complete, Aborted };
    struct Chunk {
        QueryRange rows;
        HostKvRecord record;
        uint64_t offset = 0;
    };
    QueryProjectionTape(LayerwiseQueryIdentity, uint32_t model_layer, QueryProjectionLayout);
    QueryProjectionTape(HostKvStorage* metadata_account,uint64_t max_chunks,
                        LayerwiseQueryIdentity,uint32_t model_layer,QueryProjectionLayout);

    void check(const LayerwiseQueryIdentity&, uint32_t model_layer) const;
    std::shared_ptr<const detail::QueryProjectionDescriptor> descriptor_;
    std::unique_ptr<HostKvReservation> reservation_;
    SpillExtent spill_;
    uint64_t written_bytes_ = 0;
    std::vector<Chunk> chunks_;
    uint64_t next_ = 0;
    State state_ = State::Recording;
};
} // namespace kvmem
