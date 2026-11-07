#pragma once
#include "kvmem/memory_contract.hpp"
#include <cstddef>
#include <map>
#include <optional>
#include <vector>

namespace kvmem {
class RawKvStore;
struct RawKvStoreConfig;
class HostKvSource;
class HostKvLease;

struct NativeKvRegion {
    PayloadPlane plane;
    uint64_t packed_offset = 0;
};

// Regions retain the original plane encoding, scales and head-shard identity.
// Packed staging preserves plane alignment and zero-fills interplane padding.
class NativeKvLayerRegions {
public:
    NativeKvLayerRegions(const PayloadLayout &, uint32_t layer,
                         std::optional<uint32_t> rank = std::nullopt);
    const std::vector<NativeKvRegion> & regions() const noexcept { return regions_; }
    uint64_t packed_bytes() const noexcept { return packed_bytes_; }
    uint64_t plane_bytes() const noexcept { return plane_bytes_; }
    uint64_t record_bytes() const noexcept { return record_bytes_; }
    // Only a complete layer is a standalone layout. Rank-filtered regions keep
    // model-global head coordinates and must not pretend to be a full layer.
    const PayloadLayout & packed_layout() const;
    // Read any slice into caller-owned, disjoint staging. Each I/O is bounded;
    // cold sources stay cold. The caller verifies catalog identity/version and
    // publishes only after all reads succeed. Failed I/O may partially fill staging.
    void read(const HostKvSource &, uint64_t packed_offset, void * destination,
              size_t bytes, size_t max_read_bytes = 64u << 20) const;
    void read(const HostKvLease &, uint64_t packed_offset, void * destination,
              size_t bytes, size_t max_read_bytes = 64u << 20) const;
private:
    std::vector<NativeKvRegion> regions_;
    PayloadLayout packed_layout_;
    std::optional<uint32_t> rank_;
    uint64_t packed_bytes_ = 0, plane_bytes_ = 0, record_bytes_ = 0;
};

// Opaque native bytes, including all declared scale planes and padding. This type
// has no mean-K/Q statistics and never performs quantization or RoPE transforms.
class NativeKvPayload {
public:
    NativeKvPayload(BackendIdentity backend, PayloadLayout layout, BlockDescriptor block,
                    std::vector<uint8_t> bytes);
    const BackendIdentity & backend() const noexcept { return backend_; }
    const PayloadLayout & layout() const noexcept { return layout_; }
    const BlockDescriptor & block() const noexcept { return block_; }
    const std::vector<uint8_t> & bytes() const noexcept { return bytes_; }
private:
    BackendIdentity backend_;
    PayloadLayout layout_;
    BlockDescriptor block_;
    std::vector<uint8_t> bytes_;
};

// Portable bounded host implementation. Engine adapters may instead use their
// native pinned stores, with the same identity/version/completion contract.
// References returned by get are invalidated by put/erase; not async transfer leases.
class NativeKvArchive {
public:
    NativeKvArchive(BackendIdentity backend, SessionIdentity session, PayloadLayout layout,
                    uint64_t capacity_bytes);
    void put(NativeKvPayload payload); // atomic replacement after validation, rejects stale writes
    const NativeKvPayload & get(const BlockDescriptor & expected) const;
    void erase(uint64_t block_id);
    uint64_t bytes() const noexcept { return bytes_; }
    const PayloadLayout & layout() const noexcept { return layout_; }
private:
    BackendIdentity backend_;
    SessionIdentity session_;
    PayloadLayout layout_;
    uint64_t capacity_;
    uint64_t bytes_ = 0;
    std::map<uint64_t, NativeKvPayload> records_;
};

// Compatibility with the existing llama host mirror: K/V are packed token rows;
// quantization scales already embedded in those rows remain opaque. Only supplied
// attention layers participate; GDN layers must not be invented as zero-filled KV.
PayloadLayout raw_kv_payload_layout(const RawKvStoreConfig &, const std::string & format,
                                    const std::vector<uint32_t> & attention_layers);
NativeKvPayload export_raw_kv_payload(const RawKvStore &, const BackendIdentity &, const PayloadLayout &,
                                      const BlockDescriptor &);
// Caller holds the engine-safe boundary. Statistics and checkpoint state are untouched.
// Allocation/I/O failure during import invalidates the caller's attempted restore.
void import_raw_kv_payload(RawKvStore &, const BackendIdentity &, const PayloadLayout &,
                           const NativeKvPayload &);
} // namespace kvmem
