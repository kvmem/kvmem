#include "memory_test_support.hpp"
#include <algorithm>
#include <map>
#include <set>

// Elastic generation: one request's committed logical history grows far past the
// device window (B selected + R reserve). Decode appends a block per step, the
// working-set plan evicts the oldest evictable pages to host storage, and later
// steps restore early generation pages by selection, exactly as query-conditioned
// retrieval would. The contract admits every transaction; the only frontier
// ceiling is MemoryCapabilities::max_position, and the only hard budgets are the
// visible selection (B), the device residency (B+R), and the host payload (H).
namespace {

constexpr uint64_t kBlock = 64;
constexpr uint64_t kB = 8192;        // selected tokens
constexpr uint64_t kR = 1024;        // reserve tokens
constexpr uint64_t kWindow = kB + kR;
constexpr SessionIdentity kSession{7, 1};

BlockDescriptor make_block(const MemoryDescriptor & d, uint64_t id, uint32_t tokens) {
    return {kSession, id, id * d.layout.block_tokens,
            tokens ? tokens : d.layout.block_tokens, 1, 7};
}

struct ElasticBackend : MemoryBackend {
    MemoryDescriptor desc;
    MemorySnapshot state;
    NativeKvArchive host;
    std::map<uint64_t, NativeKvPayload> device;
    uint64_t next_id = 0;
    bool complete = false, busy = false;

    ElasticBackend(uint64_t initial_blocks, uint64_t host_quota)
        : desc(::descriptor(2)), host(desc.identity, kSession, desc.layout, host_quota) {
        state.stamp = {true, kSession, 1, 0, 0, 1};
        for (uint64_t i = 0; i < initial_blocks; ++i) append_block(0);
    }

    void append_block(uint32_t tokens) {
        const auto b = make_block(desc, next_id++, tokens);
        state.blocks.push_back({b, 0});
        device.emplace(b.id, payload(desc, b));
        state.resident.push_back(b.id);
        state.stamp.frontier += b.valid_tokens;
        ++state.stamp.history_revision;
    }

    MemoryDescriptor descriptor() const override { return desc; }
    MemorySnapshot snapshot() const override { return state; }
    MemoryStamp stamp() const noexcept override { return state.stamp; }
    void invalidate() noexcept override { state.stamp.valid = false; }

    struct Transfer : MemoryTransfer {
        ElasticBackend & be;
        WorkingSetPlan plan;
        bool spilling = false, restoring = false, spill_done = false, done = false;
        explicit Transfer(ElasticBackend & backend, const WorkingSetPlan & p) : be(backend), plan(p) {}
        ~Transfer() override { CHECK(done); }
        void start_spill() override { CHECK(!spilling && !restoring); spilling = true; }
        TransferPoll poll() override {
            CHECK(spilling || restoring);
            if (!be.complete) return {TransferStatus::Pending, {}};
            be.complete = false;
            if (spilling) {
                for (const auto & b : plan.spill) be.host.put(be.device.at(b.id));
                spill_done = true;
            } else {
                for (const auto & b : plan.restore) be.device.emplace(b.id, be.host.get(b));
            }
            return {TransferStatus::Complete, {}};
        }
        void start_restore() override {
            CHECK(spill_done);
            spilling = false;
            restoring = true;
            for (const auto & b : plan.evict) be.device.erase(b.id);
        }
        void publish(const ExecutionView & view) override {
            be.state.stamp = view.stamp;
            be.state.resident.clear();
            for (const auto & b : view.blocks) be.state.resident.push_back(b.block.id);
            for (auto & entry : be.state.blocks) {
                try { be.host.get(entry.block); entry.host_version = entry.block.content_version; }
                catch (const MemoryError &) { entry.host_version = 0; }
            }
            done = true;
            be.busy = false;
        }
        AbortResult abort() noexcept override { done = true; be.busy = false; return AbortResult::SessionInvalidated; }
    };

    std::unique_ptr<MemoryTransfer> prepare(const WorkingSetPlan & p) override {
        if (busy) throw MemoryError(MemoryErrorCode::Pending, "engine already has a transfer owner");
        busy = true;
        return std::make_unique<Transfer>(*this, p);
    }
};

ExecutionView run(MemorySession & session, ElasticBackend & be) {
    for (;;) {
        const auto phase = session.advance();
        if (phase == MemoryPhase::Ready) break;
        CHECK(phase == MemoryPhase::Spilling || phase == MemoryPhase::Restoring);
        be.complete = true;
    }
    return session.publish();
}

// The engine's per-step selection: sink pages and the trailing pages stay
// mandatory; the rest of the B budget is filled with rotating early picks so
// restores of evicted generation pages are exercised every step.
std::vector<uint64_t> step_selection(const ElasticBackend & be, uint64_t step,
                                     std::vector<uint64_t> & mandatory) {
    constexpr uint64_t sink = 4, recent = 8;
    mandatory.clear();
    for (uint64_t i = 0; i < sink && i < be.next_id; ++i) mandatory.push_back(i);
    const uint64_t last = be.next_id - 1;
    for (uint64_t i = 0; i < recent && last >= i; ++i) mandatory.push_back(last - i);
    std::sort(mandatory.begin(), mandatory.end());
    mandatory.erase(std::unique(mandatory.begin(), mandatory.end()), mandatory.end());
    std::set<uint64_t> chosen(mandatory.begin(), mandatory.end());
    const uint64_t fill = kB / kBlock - mandatory.size();
    const uint64_t span = be.next_id > sink ? be.next_id - sink : 0;
    for (uint64_t k = 0; k < fill && k < span; ++k) {
        chosen.insert(sink + (step * 37 + k * 101) % span);
    }
    return std::vector<uint64_t>(chosen.begin(), chosen.end());
}

void elastic_generation() {
    const uint64_t initial = kWindow / kBlock;   // history exactly fills the device window
    const uint64_t steps = 3 * initial;          // grow the history to 4x the window
    const auto proto = std::make_shared<ElasticBackend>(1, 1); // descriptor geometry only
    const uint64_t record = proto->desc.layout.record_bytes;
    const uint64_t total = initial + steps;
    auto be = std::make_shared<ElasticBackend>(initial, record * total);
    MemorySession session(be);
    const MemoryBudget budget{kB, kR, kWindow, record * total};
    std::vector<uint64_t> mandatory;
    std::set<uint64_t> spilled; // every block ever evicted keeps a host copy
    uint64_t restores = 0, spills = 0;
    for (uint64_t step = 0; step < steps; ++step) {
        be->append_block(step + 1 == steps ? uint32_t(kBlock / 2) : 0); // final block is partial
        const auto selected = step_selection(*be, step, mandatory);
        const auto & plan = session.prepare(be->stamp(), selected, mandatory, budget);
        CHECK(plan.next.stamp.frontier == be->stamp().frontier);
        CHECK(plan.next.visible_tokens <= kB);
        CHECK(plan.device_tokens_required <= kWindow);
        for (const auto & b : plan.spill) spilled.insert(b.id);
        CHECK(plan.host_payload_bytes_required == spilled.size() * record);
        restores += plan.restore.size();
        spills += plan.spill.size();
        // A restored generation page must come back with its original bytes.
        if (!plan.restore.empty()) {
            const auto id = plan.restore.front().id;
            const auto & catalog = be->state.blocks;
            CHECK(id < catalog.size());
            CHECK(be->state.resident.end() ==
                  std::find(be->state.resident.begin(), be->state.resident.end(), id));
        }
        const auto view = run(session, *be);
        CHECK(be->state.resident.size() <= kWindow / kBlock);
        CHECK(view.blocks.front().compact_start == 0);
        CHECK(view.blocks.back().block.id == be->next_id - 1); // trailing page stays selected
        uint64_t address = 0;
        for (const auto & v : view.blocks) {
            CHECK(v.compact_start == address);
            address += v.block.valid_tokens;
        }
        CHECK(address == view.visible_tokens);
        if (!plan.restore.empty()) {
            const auto id = plan.restore.front().id;
            CHECK(be->device.at(id).bytes() == payload(be->desc, be->state.blocks[id].block).bytes());
        }
    }
    CHECK(be->stamp().frontier == 4 * kWindow - kBlock / 2);
    CHECK(be->stamp().frontier < be->desc.capabilities.max_position);
    CHECK(restores > 0 && spills > 0);
    CHECK(be->device.size() <= kWindow / kBlock);

    // The host payload budget is the admission bound for elastic history: one
    // record short of the retained history must be rejected, never silently dropped.
    auto small = budget;
    small.host_payload_bytes = spilled.size() * record - 1;
    const auto selection = step_selection(*be, steps, mandatory);
    rejects(MemoryErrorCode::BudgetExceeded,
            [&] { session.prepare(be->stamp(), selection, mandatory, small); });

    // max_position is the only logical frontier ceiling.
    const auto saved = be->stamp();
    be->state.stamp.frontier = be->desc.capabilities.max_position + 1;
    rejects(MemoryErrorCode::InvalidPlan,
            [&] { session.prepare(be->stamp(), {be->next_id - 1}, {}, budget); });
    be->state.stamp = saved;
}

} // namespace

int main() {
    try {
        elastic_generation();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "elastic decode test failed: %s\n", e.what());
        return 1;
    }
    std::printf("elastic decode window test passed\n");
    return 0;
}
