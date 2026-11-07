#include "memory_test_support.hpp"
#include "kvmem/layerwise_request.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>

namespace {
constexpr uint32_t block_tokens = 4;
constexpr uint32_t evaluation_rows = 4;

MemoryDescriptor request_descriptor() {
    MemoryDescriptor d;
    d.identity = {"scalar-request-reference", "native-f32-v1", "two-layer-fixture", "state-v1"};
    d.layout = {"two-native-layer-record", 1, block_tokens, 64, 4, {
        {2, PayloadPlaneKind::Key, 0, 16, 4, 4},
        {2, PayloadPlaneKind::Value, 16, 16, 4, 4},
        {7, PayloadPlaneKind::Key, 32, 16, 4, 4},
        {7, PayloadPlaneKind::Value, 48, 16, 4, 4}}};
    d.capabilities = {block_tokens, 1024, true, true, true, true, true, false};
    d.validate();
    return d;
}
KvMemStoreConfig policy_config() {
    KvMemStoreConfig c;
    c.block_tokens = block_tokens; c.select_budget = block_tokens;
    c.sink_blocks = c.recent_blocks = 0;
    return c;
}

struct Fixture {
    MemoryDescriptor descriptor = request_descriptor();
    LayerwiseQueryIdentity identity{{true, {42, 3}, 5, 6, 8, 9}, 77, {8, 12}, {9, 11}};
    MemoryStamp revision = identity.history;
    KvMemStore policy{policy_config()};
    RawKvStore keys{{2, 1, 0, block_tokens}};
    HostKvStorage archive{128};
    std::map<uint64_t, HostKvRecord> records;
    std::vector<LayerMemorySnapshot> initial;
    std::vector<LayerwiseRequestLayer> specs;
    Fixture() {
        policy.register_append(8);
        MemorySnapshot state;
        state.stamp = identity.history;
        // Native IDs deliberately differ from dense policy/statistics ordinals.
        state.blocks = {{{{42, 3}, 100, 0, 4, 1, 2}, 1}, {{{42, 3}, 200, 4, 4, 2, 2}, 2}};
        state.resident = {100}; initial.push_back({2, state});
        state.stamp.view_version = 11; state.resident = {200}; initial.push_back({7, state});
        const MemoryBudget budget{4, 4, 8, 192};
        specs = {{2, {0, 1, 1, 1, 1.f}, budget, {}}, {7, {1, 1, 1, 1, 1.f}, budget, {}}};
        for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
            const float negative[]{-4.f}, positive[]{4.f};
            keys.write_layer_mean_sum(0, 4, ordinal, negative);
            keys.write_layer_mean_sum(4, 4, ordinal, positive);
        }
        for (const auto& copy : initial[0].memory.blocks) {
            std::array<float, 16> bytes{};
            const float key = copy.block.id == 100 ? -1.f : 1.f;
            const float value = 3.f * key;
            for (size_t layer = 0; layer < 2; ++layer) {
                std::fill_n(bytes.begin() + layer * 8, 4, key);
                std::fill_n(bytes.begin() + layer * 8 + 4, 4, value);
            }
            auto record = archive.allocate(sizeof(bytes), bytes.data()); record.seal();
            records.emplace(copy.block.id, std::move(record));
        }
    }
    std::unique_ptr<LayerwiseQueryRequest> request(std::vector<LayerMemorySnapshot> states = {},
                                                std::vector<LayerwiseRequestLayer> layers = {}) {
        if (states.empty()) states = initial;
        if (layers.empty()) layers = specs;
        return std::make_unique<LayerwiseQueryRequest>(identity,
            LayerwiseMemoryCatalog(descriptor, std::move(states)), policy, keys, std::move(layers),
            block_tokens, [this] { return revision; });
    }
};

struct NativeLayer {
    ExecutionView view;
    std::array<float, 8> bytes{}; // four native K rows, then four V rows
    std::array<float, 8> suffix{};
    uint64_t block = 0;
    bool fenced = false;
};
struct NativeReference {
    Fixture& fixture;
    std::array<NativeLayer, 2> layers;
    // Two hidden features: Q1 reads feature zero, Q2 reads feature0+1000*feature1.
    // Headers/trailers produce strong logits opposed to the actual user Query
    // in both layers, so accidentally scoring them reverses the correct choice.
    std::array<std::array<float, 2>, evaluation_rows> hidden{{
        {{-1000.f, 10.f}}, {{1.f, 0.f}}, {{1.f, 0.f}}, {{-1000.f, 10.f}}}};
    std::array<std::array<uint32_t, evaluation_rows>, 2> projected{}, attended{};
    uint64_t restore_bytes = 0;
    uint32_t fences = 0;
    explicit NativeReference(Fixture& f) : fixture(f) {
        for (size_t i = 0; i < layers.size(); ++i) {
            auto& native = layers[i];
            native.block = f.initial[i].memory.resident[0];
            NativeKvLayerRegions region(f.descriptor.layout, f.initial[i].model_layer);
            region.read(f.records.at(native.block).source(), 0, native.bytes.data(), sizeof(native.bytes));
        }
    }
    std::array<float, evaluation_rows> projection(size_t ordinal) const {
        std::array<float, evaluation_rows> q{};
        for (size_t r = 0; r < q.size(); ++r)
            q[r] = hidden[r][0] + (ordinal ? 1000.f * hidden[r][1] : 0.f);
        return q;
    }
    ExecutionView restore(size_t ordinal, const LayerWorkingSetPlan& plan,
                          const std::function<void()>& during_io = {}) {
        auto& native = layers[ordinal];
        CHECK(plan.plan.restore.size() == 1 && plan.plan.spill.empty());
        const auto& block = plan.plan.restore.front();
        auto source = fixture.records.at(block.id).source();
        auto* destination = reinterpret_cast<uint8_t*>(native.bytes.data());
        constexpr size_t split = 13; // deliberately crosses a float boundary
        plan.payload.read(source, 0, destination, split, 7);
        if (during_io) during_io();
        plan.payload.read(source, split, destination + split, sizeof(native.bytes) - split, 7);
        restore_bytes += sizeof(native.bytes);
        native.block = block.id;
        // Native byte validation/fence precedes publishing the observed view.
        const float sign = native.block == 100 ? -1.f : 1.f;
        for (size_t r = 0; r < block_tokens; ++r) {
            CHECK(native.bytes[r] == sign);
            CHECK(native.bytes[r + block_tokens] == 3.f * sign);
        }
        native.fenced = true; ++fences;
        native.view.stamp = fixture.initial[ordinal].memory.stamp;
        ++native.view.stamp.view_version;
        native.view.blocks = {{block, 0}};
        native.view.visible_tokens = block.valid_tokens;
        return native.view;
    }
    void attention(size_t ordinal, const std::array<float, evaluation_rows>& q,
                   uint32_t first, uint32_t count) {
        auto& native = layers[ordinal];
        CHECK(native.fenced && native.view.blocks[0].block.id == native.block);
        for (uint32_t row = first; row < first + count; ++row) {
            CHECK(attended[ordinal][row]++ == 0);
            std::array<double, block_tokens> weights{};
            double maximum = -INFINITY, denominator = 0, output = 0;
            for (size_t k = 0; k < block_tokens; ++k)
                maximum = std::max(maximum, weights[k] = double(q[row]) * native.bytes[k]);
            for (auto& weight : weights) denominator += weight = std::exp(weight - maximum);
            for (size_t k = 0; k < block_tokens; ++k)
                output += weights[k] / denominator * native.bytes[k + block_tokens];
            // A scalar output projection of -1 and an actual residual update.
            // The next layer's Query is computed from this changed native state.
            hidden[row][0] -= float(output);
            native.suffix[row] = q[row];
            native.suffix[row + block_tokens] = hidden[row][0];
        }
    }
    std::vector<LayerMemorySnapshot> accepted() const {
        auto result = fixture.initial;
        for (size_t ordinal = 0; ordinal < result.size(); ++ordinal) {
            for (auto n : attended[ordinal]) CHECK(n == 1);
            auto& state = result[ordinal].memory;
            state.stamp = layers[ordinal].view.stamp;
            ++state.stamp.history_revision; ++state.stamp.view_version; ++state.stamp.execution_history;
            state.stamp.frontier = fixture.identity.evaluation.end;
            state.blocks.push_back({{{42, 3}, 300, 8, 4, 3, 1}, 0}); // evaluated, resident suffix
            state.resident = {layers[ordinal].block, 300};
            CHECK(layers[ordinal].suffix[1] != 0.f && layers[ordinal].suffix[5] != 0.f);
        }
        return result;
    }
};

void project(Fixture& f, LayerwiseQueryRequest& request, NativeReference& native,
             size_t ordinal, const std::array<float, evaluation_rows>& q, uint32_t chunk) {
    for (uint32_t first = 0; first < evaluation_rows;) {
        const auto count = std::min(chunk, evaluation_rows - first);
        for (uint32_t row = first; row < first + count; ++row) CHECK(native.projected[ordinal][row]++ == 0);
        request.projected(f.identity, 8 + first, {q.data() + first, uint32_t(ordinal), count, 1, 1, 1, 1.f});
        first += count;
        if (first != evaluation_rows)
            rejects(MemoryErrorCode::Pending, [&] { request.prepare_restore(f.identity); });
    }
}

std::vector<std::vector<uint64_t>> execute(Fixture& f, LayerwiseQueryRequest& request,
                                         NativeReference& native, uint32_t chunk) {
    std::vector<std::vector<uint64_t>> selected;
    for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
        const auto model = ordinal ? 7u : 2u;
        request.begin_layer(f.identity, model);
        const auto q = native.projection(ordinal);
        if (ordinal == 1) {
            CHECK(q[1] == -2.f && q[2] == -2.f); // caused by actual restored V and preceding residual
            CHECK(q[0] > 0.f && q[3] > 0.f); // outside-Query logits would choose the wrong block
        }
        rejects(MemoryErrorCode::InvalidPlan, [&] { request.begin_layer(f.identity, 7); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { request.attended(f.identity, 8, 1); });
        project(f, request, native, ordinal, q, chunk);
        const auto& plan = request.prepare_restore(f.identity);
        CHECK(plan.model_layer == model && plan.plan.next.blocks.size() == 1);
        CHECK(request.selection().selected == std::vector<uint32_t>{ordinal ? 0u : 1u});
        CHECK(request.selection().fixed.empty() && request.selection().historical_tokens == 4);
        CHECK(request.selection().scores.size() == 2);
        // Independent scalar logistic oracle: exactly TWO true Query rows.
        const double probability = 1. / (1. + std::exp(-2. * q[1]));
        CHECK(std::abs(request.selection().scores[1] - 2. * probability) < 1e-6);
        CHECK(std::abs(request.selection().scores[0] - 2. * (1. - probability)) < 1e-6);
        selected.push_back({plan.plan.next.blocks[0].block.id});
        CHECK(selected.back()[0] == (ordinal ? 100u : 200u));
        CHECK(plan.restore_bytes == 32 && plan.payload.plane_bytes() == 32);
        rejects(MemoryErrorCode::InvalidPlan, [&] { request.attended(f.identity, 8, 1); });
        const auto view = native.restore(ordinal, plan);
        request.restored(f.identity, view);
        rejects(MemoryErrorCode::Pending, [&] { request.accepted_layers(); });
        for (uint32_t first = 0; first < evaluation_rows;) {
            const auto count = std::min(chunk, evaluation_rows - first);
            native.attention(ordinal, q, first, count);
            request.attended(f.identity, 8 + first, count);
            first += count;
        }
        request.complete_layer(f.identity);
    }
    CHECK(request.phase() == LayerwiseRequestPhase::AwaitingAcceptance);
    CHECK(request.work().projected_rows == 8 && request.work().attended_rows == 8 && request.work().completed_layers == 2);
    CHECK(native.restore_bytes == 64 && native.fences == 2);
    for (const auto& rows : native.projected) for (auto n : rows) CHECK(n == 1);
    for (const auto& rows : native.attended) for (auto n : rows) CHECK(n == 1);
    return selected;
}

void current_forward_and_acceptance() {
    for (uint32_t chunk : {1u, 2u, 3u, 4u}) {
        Fixture f; auto request = f.request(); NativeReference native(f);
        const float frozen_positive[]{1.f, 1.f};
        const auto frozen = select_layerwise_history(f.policy, f.keys, {frozen_positive, 1, 2, 1, 1, 1, 1.f});
        CHECK(frozen.selected == std::vector<uint32_t>{1}); // stale Q2 would still choose native ID 200
        rejects(MemoryErrorCode::Pending, [&] { request->selection(); });
        rejects(MemoryErrorCode::Pending, [&] { request->accepted_layers(); });
        auto foreign = f.identity; ++foreign.request;
        rejects(MemoryErrorCode::Stale, [&] { request->begin_layer(foreign, 2); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { request->begin_layer(f.identity, 7); });
        const auto selected = execute(f, *request, native, chunk);
        CHECK(selected == std::vector<std::vector<uint64_t>>({{200}, {100}}));
        auto accepted = native.accepted();
        // Acceptance errors must not expose a partially committed result.
        auto bad = accepted; bad.pop_back();
        rejects(MemoryErrorCode::InvalidPlan, [&] { request->accepted(f.identity, bad); });
        rejects(MemoryErrorCode::Pending, [&] { request->accepted_layers(); });
        bad = accepted;
        for (auto& layer : bad) {
            ++layer.memory.stamp.session.incarnation;
            for (auto& copy : layer.memory.blocks) copy.block.session = layer.memory.stamp.session;
        }
        rejects(MemoryErrorCode::Stale, [&] { request->accepted(f.identity, bad); });
        bad = accepted;
        for (auto& layer : bad) { ++layer.memory.blocks[0].block.content_version; ++layer.memory.blocks[0].host_version; }
        rejects(MemoryErrorCode::Stale, [&] { request->accepted(f.identity, bad); });
        bad = accepted;
        for (auto& layer : bad) layer.memory.stamp.history_revision = f.identity.history.history_revision;
        rejects(MemoryErrorCode::Stale, [&] { request->accepted(f.identity, bad); });
        bad = accepted;
        for (auto& layer : bad) --layer.memory.stamp.frontier;
        rejects(MemoryErrorCode::Stale, [&] { request->accepted(f.identity, bad); });
        bad = accepted;
        for (auto& layer : bad) layer.memory.stamp.valid = false;
        rejects(MemoryErrorCode::InvalidSession, [&] { request->accepted(f.identity, bad); });
        bad = accepted; ++bad[0].memory.stamp.execution_history;
        rejects(MemoryErrorCode::Stale, [&] { request->accepted(f.identity, bad); });
        bad = accepted;
        bad[0].memory.resident = {100, 300}; // cold100 was never restored by model layer2
        rejects(MemoryErrorCode::InvalidPlan, [&] { request->accepted(f.identity, bad); });
        bad = accepted;
        for (auto& layer : bad) layer.memory.resident = {100, 200, 300}; // twelve rows exceed native eight-row device budget
        rejects(MemoryErrorCode::BudgetExceeded, [&] { request->accepted(f.identity, bad); });
        CHECK(request->phase() == LayerwiseRequestPhase::AwaitingAcceptance);
        rejects(MemoryErrorCode::Pending, [&] { request->accepted_layers(); });
        // The index lease may end now. Final native acceptance does not re-read
        // the pre-Query source after the complete model already consumed it.
        ++f.revision.history_revision;
        request->accepted(f.identity, accepted);
        CHECK(request->phase() == LayerwiseRequestPhase::Accepted);
        const auto& actual = request->accepted_layers();
        CHECK(actual.size() == 2 && actual[0].model_layer == 2 && actual[1].model_layer == 7);
        CHECK(actual[0].memory.resident == std::vector<uint64_t>({200, 300}));
        CHECK(actual[1].memory.resident == std::vector<uint64_t>({100, 300}));
        for (const auto& layer : actual) {
            CHECK(layer.memory.stamp.frontier == 12 && layer.memory.blocks.size() == 3);
            CHECK(layer.memory.blocks[2].block.original_start == 8 && layer.memory.blocks[2].block.valid_tokens == 4);
        }
        rejects(MemoryErrorCode::InvalidPlan, [&] { request->begin_layer(f.identity, 2); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { request->accepted(f.identity, accepted); });
    }
    // Final resident-only Pressure can discard either layer's old selected
    // history while retaining accepted suffix rows; it cannot recall cold IDs.
    {
        Fixture f; auto request = f.request(); NativeReference native(f);
        execute(f,*request,native,2);
        auto accepted = native.accepted();
        for (auto& layer : accepted) layer.memory.resident = {300};
        request->accepted(f.identity,std::move(accepted));
        CHECK(request->accepted_layers()[0].memory.resident == std::vector<uint64_t>({300}));
    }
}

void admission_failures() {
    Fixture f;
    auto no_host = f.initial;
    for (auto& layer : no_host) { layer.memory.resident = {100}; layer.memory.blocks[0].host_version = 0; }
    rejects(MemoryErrorCode::Pending, [&] { f.request(no_host); });
    auto specs = f.specs; specs[0].budget.selected_tokens = 8;
    rejects(MemoryErrorCode::InvalidPlan, [&] { f.request({}, specs); });
    specs = f.specs; specs[1].query.layer = 0;
    rejects(MemoryErrorCode::InvalidPlan, [&] { f.request({}, specs); });
    auto mismatch = f.initial; ++mismatch[0].memory.stamp.execution_history;
    rejects(MemoryErrorCode::Stale, [&] { f.request(mismatch); });
    specs = f.specs; specs[0].budget.device_tokens = 4;
    auto request = f.request({}, specs); NativeReference native(f);
    request->begin_layer(f.identity, 2);
    project(f, *request, native, 0, native.projection(0), 2);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { request->prepare_restore(f.identity); });
    CHECK(request->phase() == LayerwiseRequestPhase::Aborted);
}

void native_failure_boundaries() {
    for (bool stale_source : {false, true}) {
        Fixture f; auto request = f.request(); NativeReference native(f);
        request->begin_layer(f.identity, 2);
        project(f, *request, native, 0, native.projection(0), 1);
        const auto& plan = request->prepare_restore(f.identity);
        auto view = native.restore(0, plan, [&] {
            if (stale_source) {
                const float changed[]{8.f}; f.keys.write_layer_mean_sum(0, 4, 0, changed);
                ++f.revision.history_revision;
            }
        }); // source mutation occurs between bounded native transfer reads
        if (!stale_source) {
            view.blocks[0].block = f.initial[0].memory.blocks[0].block; // actual post-fence view names wrong KV
        }
        rejects(MemoryErrorCode::Stale, [&] { request->restored(f.identity, view); });
        CHECK(request->phase() == LayerwiseRequestPhase::Aborted);
        CHECK(request->work().attended_rows == 0 && native.fences == 1);
        rejects(MemoryErrorCode::Pending, [&] { request->accepted_layers(); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { request->attended(f.identity, 8, 4); });
    }
    {
        Fixture f; auto request = f.request(); NativeReference native(f);
        request->begin_layer(f.identity, 2);
        const auto q = native.projection(0);
        request->projected(f.identity, 8, {q.data(), 0, 1, 1, 1, 1, 1.f});
        rejects(MemoryErrorCode::InvalidPlan, [&] { request->projected(f.identity, 8, {q.data(), 0, 1, 1, 1, 1, 1.f}); });
        CHECK(request->phase() == LayerwiseRequestPhase::Aborted);
    }
    {
        Fixture f; auto request = f.request(); NativeReference native(f);
        request->begin_layer(f.identity, 2);
        project(f, *request, native, 0, native.projection(0), 4);
        const auto& plan = request->prepare_restore(f.identity);
        request->restored(f.identity, native.restore(0, plan));
        request->attended(f.identity, 8, 1);
        rejects(MemoryErrorCode::InvalidPlan, [&] { request->complete_layer(f.identity); });
        CHECK(request->phase() == LayerwiseRequestPhase::Aborted);
    }
}

void native_query_scoring_boundary() {
    for (int failure = 0; failure < 4; ++failure) {
        Fixture f; auto request = f.request(); NativeReference native(f);
        request->begin_layer(f.identity,2);
        const auto q = native.projection(0);
        NativeCurrentLayerQuery view{q.data(),sizeof(q),0,4,1,1,1,1.f,
            NativeQueryScalar::Float32,sizeof(float),sizeof(float),sizeof(float)};
        uint32_t calls = 0;
        auto score = [&](const RawKvStore& keys, NativeCurrentLayerQuery tile, const QueryScoreFrame& frame) {
            ++calls;
            CHECK(tile.rows == 2 && tile.data == q.data()+1); // exact Query intersection only
            CHECK(frame.accepted_through == 8 && frame.starts.size() == 2);
            auto result = layerwise_mean_key_scores(keys,
                {static_cast<const float*>(tile.data),tile.layer,tile.rows,tile.query_heads,
                 tile.kv_heads,tile.head_dim,tile.attention_scale},frame);
            if (failure == 1) result.pop_back();
            if (failure == 2) result[0] = -1;
            if (failure == 3) ++f.revision.history_revision;
            return result;
        };
        if (failure) {
            rejects(failure == 3 ? MemoryErrorCode::Stale : MemoryErrorCode::InvalidPlan,[&] {
                request->projected_native(f.identity,8,view,score);
            });
            CHECK(request->phase() == LayerwiseRequestPhase::Aborted && request->work().attended_rows == 0);
        } else {
            request->projected_native(f.identity,8,view,score);
            CHECK(request->prepare_restore(f.identity).plan.next.blocks[0].block.id == 200);
        }
        CHECK(calls == 1);
    }
    // Native Q may have a token-minor storage permutation, with head padding.
    std::array<float,32> padded{};
    NativeCurrentLayerQuery permuted{padded.data(),sizeof(padded),0,4,2,1,3,1.f,
        NativeQueryScalar::Float32,4,64,16};
    validate_native_layer_query(permuted);
    auto sliced = native_layer_query_rows(permuted,1,2);
    CHECK(sliced.data == padded.data()+1 && sliced.bytes == sizeof(padded)-4 && sliced.rows == 2);
    validate_native_layer_query(sliced);
    // These checks are tensor transport bounds, independent of score math.
    for (int failure = 0; failure < 3; ++failure) {
        auto invalid = permuted;
        if (failure == 0) invalid.bytes = 100;
        if (failure == 1) invalid.head_stride = 16;
        if (failure == 2) invalid.row_stride = std::numeric_limits<size_t>::max();
        bool rejected = false;
        try { validate_native_layer_query(invalid); } catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
    }
}
} // namespace

int main() {
    current_forward_and_acceptance();
    admission_failures();
    native_failure_boundaries();
    native_query_scoring_boundary();
    std::puts("layerwise request: current-Q scalar reference, native byte restore, fences and acceptance passed");
}
