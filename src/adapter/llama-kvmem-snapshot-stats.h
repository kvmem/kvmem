#pragma once
#include <cstdint>

// Process-wide successful host captures, including target and draft attention.
// Counts actual bytes written, not logical references or estimated block sizes.
// Failed transfers and other D2H traffic remain in the transfer diagnostics.
enum class llama_kvmem_capture_plane { packed_k, packed_v, native_state };
enum class llama_kvmem_snapshot_phase { attention_capture, native_capture, image_allocate, image_write, image_read, attention_restore };
void llama_kvmem_record_snapshot_time(llama_kvmem_snapshot_phase phase, uint64_t microseconds);
struct llama_kvmem_snapshot_stats {
    uint64_t packed_k_bytes = 0, packed_v_bytes = 0, native_state_bytes = 0;
    uint64_t attention_capture_us = 0, native_capture_us = 0, image_allocate_us = 0;
    uint64_t image_write_us = 0, image_read_us = 0, attention_restore_us = 0;
};
void llama_kvmem_record_snapshot_capture(llama_kvmem_capture_plane plane, uint64_t bytes);
llama_kvmem_snapshot_stats llama_kvmem_get_snapshot_stats();
