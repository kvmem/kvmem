#include "statistics_capture.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "rad_device.h"
#include "rad_plugin.h"
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>

namespace kvmem::radiance {
namespace {
void check(int rc, const char* message) {
    if (rc < 0) throw MemoryError(MemoryErrorCode::TransferFailed, message);
}
[[noreturn]] void bad(const char* message) { throw MemoryError(MemoryErrorCode::InvalidPlan, message); }
uint64_t bytes(uint64_t a, uint64_t b) {
    if (a && b > uint64_t(std::numeric_limits<int64_t>::max()) / a) bad("statistics size overflow");
    return a * b;
}
int width(const RadTensor& t, uint32_t rows, uint64_t columns) {
    if (!t.data || t.rank < 2 || t.rank > RAD_MAX_RANK || t.shape[0] != rows) bad("Q/K row shape mismatch");
    uint64_t dense = 1;
    for (int d = int(t.rank) - 1; d >= 1; --d) {
        if (t.shape[d] <= 0 || uint64_t(t.stride[d]) != dense) bad("Q/K inner axes must be dense");
        dense = bytes(dense, uint64_t(t.shape[d]));
    }
    if (dense != columns || t.stride[0] < int64_t(columns)) bad("Q/K head geometry mismatch");
    if (t.dtype == RAD_F32) return 4;
    if (t.dtype == RAD_F16 || t.dtype == RAD_BF16) return 2;
    bad("Q/K statistics require floating-point activations");
}
float value(const void* p, uint32_t dtype, size_t at) {
    if (dtype == RAD_F32) return static_cast<const float*>(p)[at];
    const auto v = static_cast<const uint16_t*>(p)[at];
    return dtype == RAD_BF16 ? rad_bf16_to_f32(v) : rad_f16_to_f32(v);
}
}
struct StatisticsCapture::State {
    struct Layer {
        StatisticsLayer geometry;
        void* q = nullptr;
        void* k = nullptr;
        uint32_t qdtype = 0, kdtype = 0;
        bool seen = false;
    };
    int device;
    uint32_t max_rows, first = 0, rows = 0, qfirst = 0, qrows = 0;
    SessionIdentity session;
    int32_t* positions = nullptr;
    std::vector<Layer> layers;
    RadStream stream = nullptr;
    RadEvent done = nullptr;
    int step = -1;
    bool active = false, sealed = false, failed = false, event_armed = false;
    bool early_event = false;
    ~State() {
        if (rad_dev_set(device) < 0 || (stream && rad_stream_sync(stream) < 0)) return;
        if (done) rad_event_destroy(done);
        for (auto& l : layers) {
            if (l.q) rad_dev_free(l.q, RAD_MEM_HOST_PINNED);
            if (l.k) rad_dev_free(l.k, RAD_MEM_HOST_PINNED);
        }
        if (positions) rad_dev_free(positions, RAD_MEM_HOST_PINNED);
    }
};
StatisticsCapture::StatisticsCapture(int device, std::vector<StatisticsLayer> layers, uint32_t max_rows)
    : state_(std::make_unique<State>()) {
    auto& s = *state_;
    s.device = device; s.max_rows = max_rows;
    const char* early = std::getenv("KVMEM_CAPTURE_EARLY_SEAL");
    s.early_event = early && std::strcmp(early, "1") == 0;
    if (s.early_event) std::fprintf(stderr, "KVMEM_CAPTURE_EARLY_SEAL enabled=1 same_producer_stream=1\n");
    if (device < 0 || !max_rows || layers.empty()) bad("missing statistics geometry");
    std::set<int> ids;
    std::set<uint32_t> ordinals;
    for (const auto& g : layers) {
        if (g.layer < 0 || !g.group || !g.head_dim || !g.kv_heads || !g.query_heads ||
            g.query_heads % g.kv_heads || !ids.insert(g.layer).second ||
            g.ordinal >= layers.size() || !ordinals.insert(g.ordinal).second)
            bad("ambiguous statistics layer/head identity");
        // Shared mean-K storage has one width; refuse a heterogeneous model until
        // its statistics representation is explicitly supported by that core.
        if (g.head_dim != layers[0].head_dim || g.kv_heads != layers[0].kv_heads ||
            g.query_heads != layers[0].query_heads) bad("heterogeneous attention statistics widths unsupported");
    }
    check(rad_dev_set(device), "select statistics device");
    s.layers.reserve(layers.size());
    for (const auto& g : layers) {
        s.layers.push_back({g});
        auto& l = s.layers.back();
        l.q = rad_dev_alloc(int64_t(bytes(bytes(max_rows, g.head_dim), bytes(g.query_heads, 4))), RAD_MEM_HOST_PINNED);
        l.k = rad_dev_alloc(int64_t(bytes(bytes(max_rows, g.head_dim), bytes(g.kv_heads, 4))), RAD_MEM_HOST_PINNED);
        if (!l.q || !l.k) throw MemoryError(MemoryErrorCode::BudgetExceeded, "statistics staging allocation failed");
    }
    s.positions = static_cast<int32_t*>(rad_dev_alloc(int64_t(bytes(max_rows, 4)), RAD_MEM_HOST_PINNED));
    if (!s.positions) throw MemoryError(MemoryErrorCode::BudgetExceeded, "statistics positions allocation failed");
    check(rad_event_create(&s.done), "create statistics completion event");
}
StatisticsCapture::~StatisticsCapture() = default;
void StatisticsCapture::begin(SessionIdentity session, uint32_t first, uint32_t rows,
                               uint32_t query_start, uint32_t query_end) {
    auto& s = *state_;
    if (s.active || s.failed || !session.id || !session.incarnation || !rows || rows > s.max_rows ||
        first > uint32_t(INT32_MAX) || rows > uint32_t(INT32_MAX) - first || query_end < query_start)
        bad("invalid or overlapping statistics frame");
    s.session = session; s.first = first; s.rows = rows;
    const uint32_t begin = std::max(first, query_start), end = std::min(first + rows, query_end);
    s.qfirst = begin < end ? begin - first : 0;
    s.qrows = begin < end ? end - begin : 0;
    s.stream = nullptr; s.step = -1; s.active = true; s.sealed = false; s.event_armed = false;
    for (auto& l : s.layers) l.seen = false;
}
int StatisticsCapture::observe(void* opaque, int layer, rad_kvgroup group, int64_t hd,
                               const RadTensor& q, const RadTensor& k, const RadBatch& batch, RadStream stream) {
    if (!opaque) return RAD_E_INVAL;
    auto& s = *static_cast<StatisticsCapture*>(opaque)->state_;
    try {
        if (!s.active || s.sealed || s.failed || !stream || batch.n_seq != 1 ||
            batch.draft_pass != 0 || batch.n_tok != s.rows || !batch.positions)
            bad("statistics frame cannot accept this batch");
        auto found = std::find_if(s.layers.begin(), s.layers.end(), [=](const auto& l) { return l.geometry.layer == layer; });
        if (found == s.layers.end() || found->seen || found->geometry.group != group || found->geometry.head_dim != hd)
            bad("duplicate or undeclared statistics layer");
        auto& l = *found;
        const uint64_t qw = bytes(l.geometry.head_dim, l.geometry.query_heads);
        const uint64_t kw = bytes(l.geometry.head_dim, l.geometry.kv_heads);
        const int qb = width(q, s.rows, qw), kb = width(k, s.rows, kw);
        if (s.stream && (s.stream != stream || s.step != batch.step)) bad("statistics frame spans multiple producer passes");
        check(rad_dev_set(s.device), "select statistics device");
        if (!s.stream) {
            s.stream = stream; s.step = batch.step;
            check(rad_memcpy_async(s.positions, batch.positions, int64_t(s.rows) * 4, stream), "copy original positions");
        }
        if (s.qrows) {
            const auto* source = static_cast<const unsigned char*>(q.data) + bytes(s.qfirst, bytes(q.stride[0], qb));
            check(rad_memcpy_2d_async(l.q, int64_t(bytes(qw, qb)), source, int64_t(bytes(q.stride[0], qb)),
                                      int64_t(bytes(qw, qb)), s.qrows, stream), "capture pre-RoPE Q");
        }
        check(rad_memcpy_2d_async(l.k, int64_t(bytes(kw, kb)), k.data, int64_t(bytes(k.stride[0], kb)),
                                  int64_t(bytes(kw, kb)), s.rows, stream), "capture pre-RoPE K");
        l.qdtype = q.dtype; l.kdtype = k.dtype; l.seen = true;
        // Fence the last captured bytes while the remaining model/drafter work
        // can cover completion. seal() still validates the frame and is required
        // before poll/commit can publish it; never fence an incomplete frame.
        if (s.early_event && std::all_of(s.layers.begin(), s.layers.end(), [](const auto& item) { return item.seen; })) {
            check(rad_event_record(s.done, stream), "arm final statistics copy completion");
            s.event_armed = true;
        }
        return RAD_OK;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "KVMEM_CAPTURE_ERROR %s layer=%d group=%d rows=%u batch_rows=%lld sequences=%lld draft_pass=%d active=%d sealed=%d\n",
            e.what(), layer, int(group), s.rows, (long long)batch.n_tok, (long long)batch.n_seq,
            batch.draft_pass, int(s.active), int(s.sealed));
        s.failed = true; return RAD_E_STATE;
    } catch (...) { s.failed = true; return RAD_E_STATE; }
}
void StatisticsCapture::seal() {
    auto& s = *state_;
    if (!s.active || s.sealed || s.failed || !s.stream ||
        std::any_of(s.layers.begin(), s.layers.end(), [](const auto& l) { return !l.seen; }))
        bad("incomplete statistics frame: every attention layer is required");
    check(rad_dev_set(s.device), "select statistics device");
    if (!s.event_armed) check(rad_event_record(s.done, s.stream), "seal statistics frame");
    s.event_armed = true;
    s.sealed = true;
}
TransferPoll StatisticsCapture::poll() {
    auto& s = *state_;
    if (s.failed) return {TransferStatus::Failed, "statistics capture failed"};
    if (!s.active || !s.sealed) return {TransferStatus::Pending, {}};
    check(rad_dev_set(s.device), "select statistics device");
    const auto rc = rad_event_query(s.done);
    if (rc < 0) { s.failed = true; return {TransferStatus::Failed, "statistics completion query failed"}; }
    return {rc ? TransferStatus::Complete : TransferStatus::Pending, {}};
}
QueryStatistics StatisticsCapture::commit(RawKvStore& store, SessionIdentity session,
                                         uint32_t accepted, uint32_t query_start, uint32_t query_end) {
    auto& s = *state_;
    if (poll().status != TransferStatus::Complete) throw MemoryError(MemoryErrorCode::Pending, "statistics frame not complete");
    if (!(session == s.session) || accepted > s.rows) bad("statistics acceptance identity/count mismatch");
    if (query_end < query_start) bad("invalid query acceptance span");
    const uint32_t begin = std::max(s.first, query_start), end = std::min(s.first + accepted, query_end);
    if (begin < end && (begin - s.first < s.qfirst || end - s.first > s.qfirst + s.qrows))
        bad("query acceptance extends outside captured Q rows");
    const auto& cfg = store.config();
    const uint64_t kw = bytes(s.layers[0].geometry.head_dim, s.layers[0].geometry.kv_heads);
    const uint64_t qw = bytes(s.layers[0].geometry.head_dim, s.layers[0].geometry.query_heads);
    if (cfg.n_layer != s.layers.size() || cfg.n_embd_k != kw) bad("statistics store geometry mismatch");
    for (uint32_t row = 0; row < s.rows; ++row)
        if (s.positions[row] != int64_t(s.first) + row) bad("statistics positions differ from the accepted original range");
    QueryStatistics result;
    result.sums.assign(size_t(bytes(s.layers.size(), qw)), 0.0f);
    result.token_counts.assign(s.layers.size(), 0);
    // Allocate/convert before writing the shared store. A store failure means
    // the caller must discard this logical session; do not retry partial credit.
    std::vector<std::vector<float>> keys(s.layers.size());
    for (const auto& l : s.layers) {
        auto& key = keys[l.geometry.ordinal];
        key.resize(size_t(bytes(accepted, kw)));
        for (size_t i = 0; i < key.size(); ++i) key[i] = value(l.k, l.kdtype, i);
        for (uint32_t row = 0; row < accepted; ++row) if (s.first + row >= query_start && s.first + row < query_end) {
            ++result.token_counts[l.geometry.ordinal];
            for (uint64_t col = 0; col < qw; ++col)
                result.sums[l.geometry.ordinal * qw + col] += value(l.q, l.qdtype, (row - s.qfirst) * qw + col);
        }
    }
    try {
        for (const auto& l : s.layers)
            store.write_layer_mean_k(s.first, accepted, l.geometry.ordinal, keys[l.geometry.ordinal].data());
    } catch (...) { s.failed = true; throw; }
    s.active = false;
    s.stream = nullptr; // completion was observed; never retain a retired Ctx's stream
    return result;
}
bool StatisticsCapture::discard() noexcept {
    auto& s = *state_;
    if (rad_dev_set(s.device) < 0 || (s.stream && rad_stream_sync(s.stream) < 0)) { s.failed = true; return false; }
    s.active = false; s.sealed = false; s.event_armed = false; s.stream = nullptr;
    // Invalid geometry/order poisons an instance; a clean cancellation does not.
    return true;
}
} // namespace kvmem::radiance
