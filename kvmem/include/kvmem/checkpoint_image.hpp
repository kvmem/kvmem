#pragma once
#include "spill_io.hpp"
#include "host_kv_storage.hpp"
#include "snapshot.hpp"
#include <limits>

namespace kvmem {
struct CheckpointWriteSpan { const void * data; size_t bytes; };
struct CheckpointReadSpan { void * data; size_t bytes; };
// Source/target lifetimes and publication belong to the native owner. The shared
// SpillFile accounts this extent with native KV; no private disk quota is created.
inline uint64_t checkpoint_image_size(uint64_t payload, size_t segments) {
    const uint64_t header = 32 + uint64_t(segments)*8;
    if (payload > UINT64_MAX-header) throw std::overflow_error("checkpoint image size overflow");
    return header+payload;
}
namespace detail {
template<class Storage> inline void write_checkpoint_image_to(Storage & extent, uint64_t identity,
                                  const std::vector<CheckpointWriteSpan> & spans) {
    uint64_t offset = 0;
    SnapshotWriter out([&](const void * p, size_t n) { extent.write(offset, p, n); offset += n; });
    out.scalar(uint64_t(0x3154504b434d564bull)); // KVMCKPT1
    out.scalar(identity); out.scalar(uint64_t(spans.size()));
    for (const auto & span : spans) { out.scalar(uint64_t(span.bytes)); out.write(span.data, span.bytes); }
    if (offset+8 != extent.size()) throw std::runtime_error("checkpoint image length mismatch");
    const auto hash = out.hash(); extent.write(offset, &hash, 8);
}
template<class Storage> inline void read_checkpoint_image_from(const Storage & extent, uint64_t identity,
                                 const std::vector<CheckpointReadSpan> & spans) {
    uint64_t offset = 0;
    if (extent.size() < 32) throw SnapshotCorrupt("truncated checkpoint image");
    SnapshotReader in([&](void * p, size_t n) { extent.read(offset, p, n); offset += n; }, extent.size()-8);
    in.expect(uint64_t(0x3154504b434d564bull)); in.expect(identity); in.expect(uint64_t(spans.size()));
    for (const auto & span : spans) { in.expect(uint64_t(span.bytes)); in.read(span.data, span.bytes); }
    if (offset+8 != extent.size()) throw SnapshotCorrupt("checkpoint image length mismatch");
    uint64_t hash = 0; extent.read(offset, &hash, 8);
    if (in.hash() != hash) throw SnapshotCorrupt("checkpoint image checksum mismatch");
}
} // namespace detail
// The destination is private until writing succeeds. Record writes do not seal
// or publish automatically; the caller completes capture, seals, then publishes.
inline void write_checkpoint_image(const SpillExtent & extent, uint64_t identity,
                                  const std::vector<CheckpointWriteSpan> & spans) {
    detail::write_checkpoint_image_to(extent, identity, spans);
}
inline void write_checkpoint_image(HostKvRecord & record, uint64_t identity,
                                  const std::vector<CheckpointWriteSpan> & spans) {
    detail::write_checkpoint_image_to(record, identity, spans);
}
// Read spans are private staging: no native restore/publication before this
// function returns successfully. Failure may have overwritten some staging.
inline void read_checkpoint_image(const SpillExtent & extent, uint64_t identity,
                                 const std::vector<CheckpointReadSpan> & spans) {
    detail::read_checkpoint_image_from(extent, identity, spans);
}
inline void read_checkpoint_image(const HostKvLease & lease, uint64_t identity,
                                 const std::vector<CheckpointReadSpan> & spans) {
    detail::read_checkpoint_image_from(lease, identity, spans);
}
inline void read_checkpoint_image(const HostKvRecord & record, uint64_t identity,
                                 const std::vector<CheckpointReadSpan> & spans) {
    read_checkpoint_image(record.lease(), identity, spans);
}
} // namespace kvmem
