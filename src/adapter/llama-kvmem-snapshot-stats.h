#pragma once
#include <cstdint>

// Process-wide successful host captures, including target and draft attention.
// Counts actual bytes written, not logical references or estimated block sizes.
// Failed transfers and other D2H traffic remain in the transfer diagnostics.
enum class llama_kvmem_capture_plane { packed_k, packed_v, native_state };
struct llama_kvmem_snapshot_stats {
    uint64_t packed_k_bytes = 0, packed_v_bytes = 0, native_state_bytes = 0;
};
void llama_kvmem_record_snapshot_capture(llama_kvmem_capture_plane plane, uint64_t bytes);
llama_kvmem_snapshot_stats llama_kvmem_get_snapshot_stats();
