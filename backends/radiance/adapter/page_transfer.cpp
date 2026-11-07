#include "page_transfer.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>

namespace kvmem::radiance {
namespace {
void require_ok(int rc, const char* operation) {
    if (rc < 0) {
        const char* detail = rad_dev_last_error();
        throw MemoryError(MemoryErrorCode::TransferFailed,
                          std::string(operation) + (detail ? ": " + std::string(detail) : ""));
    }
}
}
struct PageTransfer::State {
    int device;
    uint64_t bytes;
    void* host = nullptr;
    RadStream stream = nullptr, compute = nullptr;
    RadEvent ready = nullptr, done = nullptr;
    bool started = false, out = false, failed = false, owns_stream = false;
    bool complete = false, event_armed = false;
    bool drain() noexcept {
        if (complete) return true;
        if (rad_dev_set(device) < 0) return false;
        if (event_armed && rad_event_sync(done) >= 0) { complete = true; return true; }
        // An enqueue failure may have left only the source fence or part of the
        // copies queued. Drain both producers before releasing their resources.
        if ((compute && rad_stream_sync(compute) < 0) ||
            (stream && stream != compute && rad_stream_sync(stream) < 0)) return false;
        complete = true;
        return true;
    }
    ~State() {
        // On an unrecoverable device error it is unsafe to release memory still
        // referenced by a queue. Retain it until process teardown; the owning
        // engine must treat drain()==false as fatal, not continue the request.
        if (!drain() || rad_dev_set(device) < 0) return;
        if (ready) rad_event_destroy(ready);
        if (done) rad_event_destroy(done);
        if (stream && owns_stream) rad_stream_destroy(stream);
        if (host) rad_dev_free(host, RAD_MEM_HOST_PINNED);
    }
};

PageTransfer::PageTransfer(int device, uint64_t bytes, RadStream borrowed)
    : state_(std::make_unique<State>()) {
    state_->device = device; state_->bytes = bytes;
    if (device < 0 || !bytes || bytes > uint64_t(std::numeric_limits<int64_t>::max()) ||
        bytes > std::numeric_limits<size_t>::max())
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid Radiance transfer allocation");
    require_ok(rad_dev_set(device), "select transfer device");
    state_->host = rad_dev_alloc(int64_t(bytes), RAD_MEM_HOST_PINNED);
    if (!state_->host) throw MemoryError(MemoryErrorCode::BudgetExceeded, "transfer staging allocation");
    state_->stream = borrowed;
    if (!borrowed) {
        require_ok(rad_stream_create(&state_->stream, 0), "create transfer stream");
        state_->owns_stream = true;
    }
    require_ok(rad_event_create(&state_->ready), "create source-ready event");
    require_ok(rad_event_create(&state_->done), "create transfer-complete event");
}
PageTransfer::~PageTransfer() = default;

void PageTransfer::begin(bool out, const std::vector<CopySpan>& spans,
                         const std::vector<GroupPool>& pools, RadStream compute,
                         std::span<const uint8_t> payload) {
    auto& s = *state_;
    if (s.failed || (s.started && (out || s.out || !s.complete)))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "transfer is not reusable");
    const uint64_t bytes=out ? s.bytes : payload.size();
    if (!compute || !bytes || bytes>s.bytes || spans.empty())
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid transfer input");
    // Validate the whole batch before a fence or copy can be enqueued.
    std::map<int32_t, GroupPool> by_group;
    for (auto p : pools) {
        if (p.group < 0 || !p.base || !p.bytes ||
            p.bytes > uint64_t(std::numeric_limits<int64_t>::max()) ||
            !by_group.emplace(p.group, p).second)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid group pool");
    }
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    std::map<int32_t, std::vector<std::pair<uint64_t, uint64_t>>> device_ranges;
    for (auto span : spans) {
        auto p = by_group.find(span.group);
        if (p == by_group.end() || !span.bytes || span.record_offset > bytes ||
            span.bytes > bytes - span.record_offset || span.pool_offset > p->second.bytes ||
            span.bytes > p->second.bytes - span.pool_offset)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "transfer span outside pool or record");
        ranges.emplace_back(span.record_offset, span.record_offset + span.bytes);
        device_ranges[span.group].emplace_back(span.pool_offset, span.pool_offset + span.bytes);
    }
    std::sort(ranges.begin(), ranges.end());
    uint64_t end = 0;
    for (auto r : ranges) {
        if (r.first != end)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "record has overlapping or missing bytes");
        end = r.second;
    }
    if (end != bytes) throw MemoryError(MemoryErrorCode::InvalidPlan, "incomplete native record");
    for (auto& item : device_ranges) {
        auto& rs = item.second;
        std::sort(rs.begin(), rs.end());
        for (size_t i = 1; i < rs.size(); ++i) if (rs[i].first < rs[i - 1].second)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "overlapping physical transfer spans");
    }
    require_ok(rad_dev_set(s.device), "select transfer device");
    if (!out) std::memcpy(s.host, payload.data(), size_t(bytes));
    s.started = true; s.complete = false; s.out = out; s.compute = compute;
    s.event_armed = false; // a prior chunk's completed event cannot fence this one
    try {
        if (s.stream != compute) {
            require_ok(rad_event_record(s.ready, compute), "record compute fence");
            require_ok(rad_event_wait(s.stream, s.ready), "order transfer after compute");
        }
        for (auto span : spans) {
            auto* dev = static_cast<uint8_t*>(by_group.at(span.group).base) + span.pool_offset;
            auto* host = static_cast<uint8_t*>(s.host) + span.record_offset;
            require_ok(rad_memcpy_async(out ? host : dev, out ? dev : host,
                                       int64_t(span.bytes), s.stream), "native page copy");
        }
        require_ok(rad_event_record(s.done, s.stream), "record transfer completion");
        s.event_armed = true;
        // The transaction publishes only after poll() is Complete. Do not queue
        // a wait in compute here: event lifetime would then outlive this handle.
    } catch (...) { s.failed = true; throw; }
}
void PageTransfer::spill(const std::vector<CopySpan>& spans, const std::vector<GroupPool>& pools,
                         RadStream compute) { begin(true, spans, pools, compute, {}); }
void PageTransfer::restore(const std::vector<CopySpan>& spans, const std::vector<GroupPool>& pools,
                           RadStream compute, std::span<const uint8_t> payload) {
    begin(false, spans, pools, compute, payload);
}
TransferPoll PageTransfer::poll() {
    auto& s = *state_;
    if (!s.started) throw MemoryError(MemoryErrorCode::InvalidPlan, "transfer has not started");
    if (s.failed) return {TransferStatus::Failed, "Radiance transfer failed"};
    if (rad_dev_set(s.device) < 0) {
        s.failed = true; return {TransferStatus::Failed, "select transfer device"};
    }
    const int rc = rad_event_query(s.done);
    if (rc < 0) { s.failed = true; return {TransferStatus::Failed, "query transfer completion"}; }
    if (rc) s.complete = true;
    return {rc ? TransferStatus::Complete : TransferStatus::Pending, {}};
}
std::span<const uint8_t> PageTransfer::payload_view() {
    if (!state_->out || poll().status != TransferStatus::Complete)
        throw MemoryError(MemoryErrorCode::Pending, "spill has not completed");
    return {static_cast<const uint8_t*>(state_->host), size_t(state_->bytes)};
}
std::vector<uint8_t> PageTransfer::payload() {
    const auto view=payload_view();
    return {view.begin(), view.end()};
}
bool PageTransfer::drain() noexcept {
    auto& s = *state_;
    if (!s.drain()) {
        s.failed = true; return false;
    }
    return true;
}
} // namespace kvmem::radiance
