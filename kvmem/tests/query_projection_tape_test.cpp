#include "memory_test_support.hpp"
#include "kvmem/query_projection_tape.hpp"
#include "kvmem/spill_io.hpp"
#include <array>
#include <cstring>
#include <filesystem>
#include <memory>

namespace {
LayerwiseQueryIdentity identity() {
    return {{true, {7, 3}, 5, 6, 32, 9}, 41, {32, 40}, {34, 39}};
}
QueryProjectionLayout layout() { return {"backend-native-q-residual-fixture", 1, {{2, 3}, {9, 5}}}; }
std::vector<uint8_t> rows(uint32_t plane, uint64_t begin, uint64_t count, size_t stride) {
    std::vector<uint8_t> out(size_t(count) * stride);
    for (uint64_t r = 0; r < count; ++r) for (size_t b = 0; b < stride; ++b)
        out[size_t(r) * stride + b] = uint8_t(plane * 17 + (begin + r) * 11 + b * 3);
    return out;
}
QueryProjectionChunkSource append(QueryProjectionTape& tape, uint64_t begin, uint64_t count) {
    const auto q = rows(2, begin, count, 3), residual = rows(9, begin, count, 5);
    return tape.append(identity(), 7, begin, count, {{2, q.data(), q.size()}, {9, residual.data(), residual.size()}});
}
void check_rows(const QueryProjectionTape& tape, uint32_t plane, uint64_t begin, uint64_t count, size_t stride) {
    std::vector<uint8_t> output(size_t(count) * stride);
    tape.read_rows(identity(), 7, plane, begin, count, output.data(), output.size());
    CHECK(output == rows(plane, begin, count, stride));
}
void memory_tests() {
    HostKvStorage canonical(8); // projection payload never touches this independent KV quota
    auto kv = canonical.allocate(8);
    HostKvStorage activation(64);
    for (int field = 0; field < 4; ++field) {
        auto zero = identity();
        if (field == 0) zero.history.session.id = 0;
        if (field == 1) zero.history.session.incarnation = 0;
        if (field == 2) zero.history.history_revision = 0;
        if (field == 3) zero.history.execution_history = 0;
        rejects(MemoryErrorCode::InvalidPlan, [&] {
            QueryProjectionTape invalid(activation, zero, 7, layout());
        });
    }
    {
        QueryProjectionTape tape(activation, identity(), 7, layout());
        CHECK(tape.total_bytes() == 64 && tape.reserved_bytes() == 64);
        CHECK(activation.charged_bytes() == 64 && canonical.charged_bytes() == 8);
        CHECK(!tape.complete() && tape.covered_end() == 32 && tape.chunk_count() == 0);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { activation.reserve(1); });
        auto first = append(tape, 32, 2);
        CHECK(first.size() == 16 && first.rows().begin == 32 && first.rows().end == 34);
        CHECK(first.model_layer() == 7 && first.identity().request == 41);
        CHECK(first.layout().planes[1].row_stride == 5);
        CHECK(tape.reserved_bytes() == 48 && tape.covered_end() == 34);
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.finish(identity(), 7); });
        CHECK(!tape.complete() && tape.reserved_bytes() == 48);
        check_rows(tape, 9, 32, 2, 5);

        std::array<uint8_t, 10> output{};
        auto foreign = identity(); ++foreign.request;
        rejects(MemoryErrorCode::Stale, [&] { first.read_rows(foreign, 7, 9, 32, 2, output.data(), output.size()); });
        foreign = identity(); ++foreign.history.view_version;
        rejects(MemoryErrorCode::Stale, [&] { tape.finish(foreign, 7); });
        foreign = identity(); --foreign.query.begin;
        rejects(MemoryErrorCode::Stale, [&] { tape.chunk(foreign, 7, 0); });
        rejects(MemoryErrorCode::Stale, [&] { tape.chunk(identity(), 3, 0); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.chunk(identity(), 7, 1); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { first.read_rows(identity(), 7, 9, 33, 2, output.data(), output.size()); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { first.read_rows(identity(), 7, 1, 32, 2, output.data(), output.size()); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { first.read_rows(identity(), 7, 9, 32, 2, nullptr, output.size()); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { first.read_rows(identity(), 7, 9, 32, 2, output.data(), output.size() - 1); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { append(tape, 32, 1); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { append(tape, 35, 1); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.append(identity(), 7, 34, 0, {}); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.append(identity(), 7, 34, UINT64_MAX, {}); });
        const auto q = rows(2, 34, 1, 3), r = rows(9, 34, 1, 5);
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.append(identity(), 7, 34, 1, {{2, q.data(), q.size()}}); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.append(identity(), 7, 34, 1, {{2, q.data(), q.size()}, {2, r.data(), r.size()}}); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.append(identity(), 7, 34, 1, {{2, q.data(), q.size()}, {9, nullptr, r.size()}}); });
        CHECK(tape.covered_end() == 34 && tape.reserved_bytes() == 48 && tape.chunk_count() == 1);
        append(tape, 34, 3);
        append(tape, 37, 3);
        CHECK(!tape.complete() && tape.reserved_bytes() == 0 && tape.covered_end() == 40);
        tape.finish(identity(), 7);
        CHECK(tape.complete());
        check_rows(tape, 2, 32, 8, 3);
        check_rows(tape, 9, 33, 5, 5); // starts/ends inside chunks and crosses two boundaries
        tape.read_rows(identity(), 7, 9, 40, 0, nullptr, 0);
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.finish(identity(), 7); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { append(tape, 40, 1); });
    }
    CHECK(activation.charged_bytes() == 0 && canonical.charged_bytes() == 8);
    // Partitioning changes record boundaries, never the opaque native row data.
    {
        QueryProjectionTape tape(activation, identity(), 7, layout());
        append(tape, 32, 8); tape.finish(identity(), 7);
        check_rows(tape, 2, 32, 8, 3); check_rows(tape, 9, 33, 5, 5);
    }
    // Aborting releases the unused reservation and tape-owned chunks. An
    // exported sealed source remains immutable and charged until its last owner.
    {
        QueryProjectionTape tape(activation, identity(), 7, layout());
        auto first = append(tape, 32, 2);
        append(tape, 34, 2);
        tape.abort();
        CHECK(tape.aborted() && !tape.complete() && tape.reserved_bytes() == 0 && tape.chunk_count() == 0);
        CHECK(activation.charged_bytes() == 16);
        std::array<uint8_t, 5> output{};
        first.read_rows(identity(), 7, 9, 33, 1, output.data(), output.size());
        CHECK(std::memcmp(output.data(), rows(9, 33, 1, 5).data(), output.size()) == 0);
        rejects(MemoryErrorCode::InvalidPlan, [&] { tape.chunk(identity(), 7, 0); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { append(tape, 36, 4); });
    }
    CHECK(activation.charged_bytes() == 0);
    // Destruction has the same reservation/source lifetime guarantees as abort.
    auto survivor = [] {
        HostKvStorage temporary(64);
        QueryProjectionTape tape(temporary, identity(), 7, layout());
        return append(tape, 32, 2);
    }();
    std::array<uint8_t, 6> output{};
    survivor.read_rows(identity(), 7, 2, 32, 2, output.data(), output.size());
    CHECK(std::memcmp(output.data(), rows(2, 32, 2, 3).data(), output.size()) == 0);

    HostKvStorage tight(63);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { QueryProjectionTape tape(tight, identity(), 7, layout()); });
    CHECK(tight.charged_bytes() == 0);
    auto bad_layout = layout(); bad_layout.planes[1].id = 2;
    rejects(MemoryErrorCode::InvalidPlan, [&] { QueryProjectionTape tape(activation, identity(), 7, bad_layout); });
    bad_layout = layout(); bad_layout.planes[0].row_stride = 0;
    rejects(MemoryErrorCode::InvalidPlan, [&] { QueryProjectionTape tape(activation, identity(), 7, bad_layout); });
    bad_layout = layout(); bad_layout.planes[0].row_stride = UINT64_MAX;
    rejects(MemoryErrorCode::InvalidPlan, [&] { QueryProjectionTape tape(activation, identity(), 7, bad_layout); });
    bad_layout = {"overflow", 1, {{0, UINT64_MAX / 8 + 1}}};
    rejects(MemoryErrorCode::InvalidPlan, [&] { QueryProjectionTape tape(activation, identity(), 7, bad_layout); });
    auto bad_identity = identity(); bad_identity.evaluation.end = 32;
    rejects(MemoryErrorCode::InvalidPlan, [&] { QueryProjectionTape tape(activation, bad_identity, 7, layout()); });
    bad_identity = identity(); bad_identity.query.end = 41;
    rejects(MemoryErrorCode::InvalidPlan, [&] { QueryProjectionTape tape(activation, bad_identity, 7, layout()); });
    CHECK(activation.charged_bytes() == 0);
}

void disk_tests(const std::filesystem::path& directory) {
    auto spill = std::make_shared<SpillFile>(directory, 128);
    HostKvStorage activation(64, spill);
    {
        QueryProjectionTape tape(activation, identity(), 7, layout());
        auto first = append(tape, 32, 2);
        append(tape, 34, 3); append(tape, 37, 3);
        tape.finish(identity(), 7);
        activation.trim(0);
        CHECK(activation.charged_bytes() == 0 && activation.disk_bytes() == 64);
        const auto before = activation.io_stats();
        check_rows(tape, 9, 33, 5, 5);
        const auto after = activation.io_stats();
        CHECK(after.read_bytes - before.read_bytes == 25); // requested rows only
        CHECK(activation.charged_bytes() == 0); // no whole-record promotion
        tape.abort();
        CHECK(activation.disk_bytes() == 16); // only exported first chunk remains
        std::array<uint8_t, 3> output{};
        first.read_rows(identity(), 7, 2, 33, 1, output.data(), output.size());
        CHECK(std::memcmp(output.data(), rows(2, 33, 1, 3).data(), output.size()) == 0);
        CHECK(activation.charged_bytes() == 0);
    }
    CHECK(activation.disk_bytes() == 0 && activation.charged_bytes() == 0);
    // Disk capacity failure must not invalidate a previously sealed chunk or
    // turn incomplete tape coverage into completion.
    auto small_spill = std::make_shared<SpillFile>(directory, 8);
    HostKvStorage tight(64, small_spill);
    {
        QueryProjectionTape tape(tight, identity(), 7, layout());
        auto first = append(tape, 32, 2);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { tight.trim(0); });
        CHECK(!tape.complete() && tape.covered_end() == 34 && tape.reserved_bytes() == 48);
        CHECK(tight.charged_bytes() == 64 && tight.disk_bytes() == 0);
        check_rows(tape, 2, 32, 2, 3);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { QueryProjectionTape other(tight, identity(), 11, layout()); });
        check_rows(tape, 9, 32, 2, 5);
    }
    CHECK(tight.charged_bytes() == 0 && tight.disk_bytes() == 0);
}

void direct_disk_tests(const std::filesystem::path& directory) {
    // Full projected layer is eight times this activation RAM limit. The
    // native caller supplies bounded two-row chunks; the tape owns no RAM copy.
    HostKvStorage activation_ram(8), canonical(8);
    auto kv = canonical.allocate(8);
    SpillFile activation_spill(directory, 64);
    {
        QueryProjectionTape tape(activation_spill, identity(), 7, layout());
        CHECK(tape.total_bytes() == 64 && tape.reserved_bytes() == 64);
        CHECK(activation_spill.charged_bytes() == 64 && activation_ram.charged_bytes() == 0);
        const auto before = activation_spill.stats();
        auto first = append(tape, 32, 2);
        CHECK(first.size() == 16 && tape.reserved_bytes() == 48);
        std::array<uint8_t, 6> output{};
        first.read_rows(identity(), 7, 2, 32, 2, output.data(), output.size());
        CHECK(std::memcmp(output.data(), rows(2, 32, 2, 3).data(), output.size()) == 0);
        rejects(MemoryErrorCode::BudgetExceeded, [&] {
            QueryProjectionTape other(activation_spill, identity(), 8, layout());
        });
        CHECK(tape.covered_end() == 34 && !tape.complete());
        for (uint64_t begin = 34; begin < 40; begin += 2) append(tape, begin, 2);
        tape.finish(identity(), 7);
        CHECK(tape.complete() && tape.reserved_bytes() == 0 && tape.chunk_count() == 4);
        CHECK(activation_spill.stats().written_bytes - before.written_bytes == 64);
        const auto reads = activation_spill.stats().read_bytes;
        check_rows(tape, 9, 33, 5, 5);
        CHECK(activation_spill.stats().read_bytes - reads == 25);
        CHECK(activation_ram.charged_bytes() == 0 && canonical.charged_bytes() == 8);
        tape.abort();
        CHECK(tape.aborted() && tape.reserved_bytes() == 0 && tape.chunk_count() == 0);
        // A source pins the declared complete extent, not just its own rows.
        CHECK(activation_spill.charged_bytes() == 64);
        first.read_rows(identity(), 7, 2, 32, 2, output.data(), output.size());
        CHECK(std::memcmp(output.data(), rows(2, 32, 2, 3).data(), output.size()) == 0);
    }
    CHECK(activation_spill.charged_bytes() == 0);
    SpillFile tight(directory, 63);
    rejects(MemoryErrorCode::BudgetExceeded, [&] {
        QueryProjectionTape tape(tight, identity(), 7, layout());
    });
    CHECK(tight.charged_bytes() == 0);
    // Chunk content and file lifetime outlive both tape and account facade.
    auto source = [&] {
        SpillFile temporary(directory, 64);
        QueryProjectionTape tape(temporary, identity(), 7, layout());
        return append(tape, 32, 2);
    }();
    std::array<uint8_t, 5> output{};
    source.read_rows(identity(), 7, 9, 33, 1, output.data(), output.size());
    CHECK(std::memcmp(output.data(), rows(9, 33, 1, 5).data(), output.size()) == 0);
}
void metadata_tests(const std::filesystem::path& directory) {
    const auto bound=QueryProjectionTape::metadata_bound(layout(),4);
    CHECK(bound>4*sizeof(QueryProjectionChunkSource));
    HostKvStorage payload(64),metadata(bound),small(bound-1);
    rejects(MemoryErrorCode::InvalidPlan,[&]{QueryProjectionTape::metadata_bound(layout(),0);});
    rejects(MemoryErrorCode::InvalidPlan,[&]{QueryProjectionTape::metadata_bound(layout(),UINT64_MAX);});
    rejects(MemoryErrorCode::BudgetExceeded,[&]{QueryProjectionTape tape(payload,small,4,identity(),7,layout());});
    CHECK(payload.charged_bytes()==0 && small.charged_bytes()==0); // no payload reservation before metadata
    std::unique_ptr<QueryProjectionChunkSource> lease;
    {
        QueryProjectionTape tape(payload,metadata,4,identity(),7,layout());
        CHECK(tape.max_chunks()==4 && tape.metadata_bytes()==bound && metadata.charged_bytes()==bound);
        CHECK(payload.charged_bytes()==64);
        auto first=append(tape,32,1);lease=std::make_unique<QueryProjectionChunkSource>(first);
        append(tape,33,1);append(tape,34,1);append(tape,35,1);
        const auto old_payload=payload.charged_bytes();
        rejects(MemoryErrorCode::InvalidPlan,[&]{append(tape,36,1);});
        CHECK(tape.covered_end()==36 && tape.chunk_count()==4 && !tape.complete());
        CHECK(payload.charged_bytes()==old_payload && metadata.charged_bytes()==bound);
        check_rows(tape,9,32,4,5); // cap failure preserves sealed rows and reservations
        rejects(MemoryErrorCode::InvalidPlan,[&]{tape.finish(identity(),7);});
        tape.abort();CHECK(tape.aborted() && payload.charged_bytes()==8 && metadata.charged_bytes()==bound);
    }
    CHECK(payload.charged_bytes()==8 && metadata.charged_bytes()==bound);
    std::array<uint8_t,5> output{};lease->read_rows(identity(),7,9,32,1,output.data(),output.size());
    CHECK(std::memcmp(output.data(),rows(9,32,1,5).data(),output.size())==0);
    lease.reset();CHECK(payload.charged_bytes()==0 && metadata.charged_bytes()==0);
    // Complete four-chunk coverage keeps exactly one pre-admitted descriptor
    // charge; complete/finish never allocates a second chunk table.
    {
        QueryProjectionTape tape(payload,metadata,4,identity(),7,layout());
        for(uint64_t begin=32;begin<40;begin+=2)append(tape,begin,2);
        tape.finish(identity(),7);CHECK(tape.complete() && tape.metadata_bytes()==bound);
        check_rows(tape,2,33,6,3);CHECK(metadata.charged_bytes()==bound);
    }
    CHECK(payload.charged_bytes()==0 && metadata.charged_bytes()==0);
    SpillFile disk(directory,64);
    const auto before=disk.stats();
    rejects(MemoryErrorCode::BudgetExceeded,[&]{QueryProjectionTape tape(disk,small,4,identity(),7,layout());});
    CHECK(disk.charged_bytes()==0 && disk.stats().writes==before.writes && small.charged_bytes()==0);
    {
        QueryProjectionTape tape(disk,metadata,4,identity(),7,layout());
        lease=std::make_unique<QueryProjectionChunkSource>(append(tape,32,2));
        for(uint64_t begin=34;begin<40;begin+=2)append(tape,begin,2);
        tape.finish(identity(),7);CHECK(tape.complete() && tape.metadata_bytes()==bound);
        tape.abort();CHECK(disk.charged_bytes()==64 && metadata.charged_bytes()==bound);
    }
    CHECK(disk.charged_bytes()==64 && metadata.charged_bytes()==bound);
    lease->read_rows(identity(),7,9,32,1,output.data(),output.size());
    CHECK(std::memcmp(output.data(),rows(9,32,1,5).data(),output.size())==0);
    lease.reset();CHECK(disk.charged_bytes()==0 && metadata.charged_bytes()==0);
    // Payload admission failure also returns its prior metadata reservation.
    HostKvStorage no_payload(63);SpillFile no_disk(directory,63);
    rejects(MemoryErrorCode::BudgetExceeded,[&]{QueryProjectionTape tape(no_payload,metadata,4,identity(),7,layout());});
    CHECK(no_payload.charged_bytes()==0 && metadata.charged_bytes()==0);
    rejects(MemoryErrorCode::BudgetExceeded,[&]{QueryProjectionTape tape(no_disk,metadata,4,identity(),7,layout());});
    CHECK(no_disk.charged_bytes()==0 && metadata.charged_bytes()==0);
    // Bounded representation ignores unrelated spare capacity in caller-owned
    // input layouts, so adapters can use the core bound without private sizeof.
    auto spare=layout();spare.format.reserve(4096);spare.planes.reserve(100);
    CHECK(QueryProjectionTape::metadata_bound(spare,4)==bound);
    {
        QueryProjectionTape tape(payload,metadata,4,identity(),7,std::move(spare));
        CHECK(tape.metadata_bytes()==bound && metadata.charged_bytes()==bound);
    }
    CHECK(payload.charged_bytes()==0 && metadata.charged_bytes()==0);
}

}

int main(int argc, char** argv) {
    memory_tests();
    const auto directory = argc > 1 ? std::filesystem::path(argv[1])
        : std::filesystem::temp_directory_path();
    disk_tests(directory);
    direct_disk_tests(directory);
    metadata_tests(directory);
    std::puts("Query projection tape: opaque chunk partitions, quota, identity and cold reads passed");
}
