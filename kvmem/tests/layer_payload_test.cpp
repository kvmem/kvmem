#include "memory_test_support.hpp"
#include "kvmem/host_kv_storage.hpp"
#include "kvmem/spill_io.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>

static PayloadLayout sharded_layout() {
    PayloadLayout layout{"layer-shard-fixture", 1, 4, 256, 8, {}};
    layout.planes = {
        {2, PayloadPlaneKind::Key, 8, 16, 8, 4, 0, 0, 2},
        {2, PayloadPlaneKind::KeyScale, 32, 8, 8, 0, 0, 0, 2},
        {2, PayloadPlaneKind::Value, 40, 16, 8, 4, 0, 0, 2},
        {2, PayloadPlaneKind::ValueScale, 64, 8, 8, 0, 0, 0, 2},
        {7, PayloadPlaneKind::PackedKeyValue, 80, 32, 8, 0, 0, 0, 1},
        {2, PayloadPlaneKind::Key, 128, 16, 8, 4, 1, 2, 2},
        {2, PayloadPlaneKind::KeyScale, 144, 8, 8, 0, 1, 2, 2},
        {2, PayloadPlaneKind::Value, 160, 16, 8, 4, 1, 2, 2},
        {2, PayloadPlaneKind::ValueScale, 184, 8, 8, 0, 1, 2, 2},
    };
    layout.validate();
    return layout;
}

static std::vector<uint8_t> bytes(size_t count) {
    std::vector<uint8_t> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = uint8_t(i * 37 + 19);
    return result;
}

static std::vector<uint8_t> expected(const PayloadLayout & layout, uint32_t layer,
                                     std::optional<uint32_t> rank, const std::vector<uint8_t> & input) {
    std::vector<uint8_t> result;
    for (const auto & plane : layout.planes)
        if (plane.layer == layer && (!rank || plane.rank == *rank))
            result.insert(result.end(), input.begin() + size_t(plane.offset),
                          input.begin() + size_t(plane.offset + plane.bytes));
    return result;
}

static void exact_regions() {
    const auto layout = sharded_layout();
    const auto input = bytes(size_t(layout.record_bytes));
    HostKvStorage storage(layout.record_bytes);
    auto record = storage.allocate(input.size(), input.data());
    record.seal();
    for (const auto rank : {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{1}}) {
        const NativeKvLayerRegions regions(layout, 2, rank);
        const auto wanted = expected(layout, 2, rank, input);
        CHECK(regions.record_bytes() == layout.record_bytes);
        CHECK(regions.packed_bytes() == wanted.size());
        CHECK(regions.regions().size() == (rank ? 4 : 8));
        uint64_t offset = 0;
        for (const auto & region : regions.regions()) {
            CHECK(region.plane.layer == 2 && (!rank || region.plane.rank == *rank));
            CHECK(region.packed_offset == offset);
            offset += region.plane.bytes;
        }
        if (rank) rejects(MemoryErrorCode::Unsupported, [&] { regions.packed_layout(); });
        else regions.packed_layout().validate();
        std::vector<uint8_t> output(wanted.size() + 2, 0xa5);
        regions.read(record.source(), 0, output.data() + 1, wanted.size(), 3);
        CHECK(output.front() == 0xa5 && output.back() == 0xa5);
        CHECK(std::equal(wanted.begin(), wanted.end(), output.begin() + 1));
        // Reads cross plane boundaries and may be smaller than one native plane.
        std::vector<uint8_t> slice(wanted.size() - 5);
        regions.read(record.lease(), 3, slice.data(), slice.size(), 5);
        CHECK(std::equal(slice.begin(), slice.end(), wanted.begin() + 3));
        regions.read(record.lease(), regions.packed_bytes(), nullptr, 0);
    }
    const NativeKvLayerRegions packed(layout, 7);
    std::vector<uint8_t> output(size_t(packed.packed_bytes()));
    packed.read(record.source(), 0, output.data(), output.size(), 1);
    CHECK(output == expected(layout, 7, {}, input));
    CHECK(packed.regions().front().plane.kind == PayloadPlaneKind::PackedKeyValue);
    CHECK(storage.charged_bytes() == input.size());
}

static void aligned_padding(const std::filesystem::path & directory) {
    const PayloadLayout layout{"mixed-plane-alignment", 1, 4, 64, 16, {
        {9, PayloadPlaneKind::Key, 8, 10, 8, 0},
        {9, PayloadPlaneKind::Value, 24, 9, 8, 0},
        {9, PayloadPlaneKind::KeyScale, 40, 4, 4, 0},
        {9, PayloadPlaneKind::ValueScale, 48, 4, 4, 0},
    }};
    const NativeKvLayerRegions regions(layout, 9);
    const auto & packed = regions.packed_layout();
    packed.validate();
    CHECK(regions.packed_bytes() == 48 && regions.plane_bytes() == 27);
    CHECK(packed.planes[0].offset == 0 && packed.planes[1].offset == 16);
    CHECK(packed.planes[2].offset == 28 && packed.planes[3].offset == 32);
    const auto input = bytes(size_t(layout.record_bytes));
    auto spill = std::make_shared<SpillFile>(directory, layout.record_bytes);
    HostKvStorage storage(layout.record_bytes, spill);
    auto record = storage.allocate(input.size(), input.data()); record.seal();
    storage.trim(0);
    auto source = record.source();
    std::vector<uint8_t> wanted(size_t(regions.packed_bytes()), 0);
    for (size_t i = 0; i < packed.planes.size(); ++i)
        std::memcpy(wanted.data() + size_t(packed.planes[i].offset),
                    input.data() + size_t(layout.planes[i].offset), size_t(layout.planes[i].bytes));
    std::vector<uint8_t> output(wanted.size(), 0xa5);
    const auto before = storage.io_stats();
    regions.read(source, 0, output.data(), output.size(), 3);
    CHECK(output == wanted && storage.charged_bytes() == 0);
    const auto after = storage.io_stats();
    CHECK(after.read_bytes - before.read_bytes == 27 && after.reads - before.reads == 11);
    std::vector<uint8_t> padding(6, 0xa5);
    regions.read(source, 10, padding.data(), padding.size(), 1);
    CHECK(std::all_of(padding.begin(), padding.end(), [](uint8_t x) { return x == 0; }));
    CHECK(storage.io_stats().read_bytes == after.read_bytes);
    std::vector<uint8_t> slice(27, 0xa5);
    regions.read(source, 7, slice.data(), slice.size(), 3);
    CHECK(std::equal(slice.begin(), slice.end(), wanted.begin() + 7));
}

static void rejects_before_writing() {
    const auto layout = sharded_layout();
    const auto input = bytes(size_t(layout.record_bytes));
    HostKvStorage storage(2 * layout.record_bytes);
    auto record = storage.allocate(input.size(), input.data());
    auto short_record = storage.allocate(input.size() - 1, input.data());
    const NativeKvLayerRegions regions(layout, 2, 1);
    std::vector<uint8_t> output(size_t(regions.packed_bytes()), 0xa5);
    rejects(MemoryErrorCode::InvalidPlan, [&] { NativeKvLayerRegions missing(layout, 3); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { NativeKvLayerRegions missing(layout, 2, 2); });
    auto bad = layout; bad.planes[1].offset = 8;
    rejects(MemoryErrorCode::InvalidPlan, [&] { NativeKvLayerRegions invalid(bad, 2); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { regions.read(HostKvSource{}, 0, output.data(), 1); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { regions.read(short_record.source(), 0, output.data(), 1); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { regions.read(record.source(), UINT64_MAX, output.data(), 1); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { regions.read(record.source(), 1, output.data(), output.size()); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { regions.read(record.source(), 0, nullptr, 1); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { regions.read(record.source(), 0, output.data(), 1, 0); });
    auto * alias = record.writable_data();
    rejects(MemoryErrorCode::InvalidPlan, [&] { regions.read(record.lease(), 0, alias, 1); });
    CHECK(std::all_of(output.begin(), output.end(), [](uint8_t x) { return x == 0xa5; }));
    CHECK(std::memcmp(record.data(), input.data(), input.size()) == 0);
}

static void cold_reads(const std::filesystem::path & directory) {
    const auto layout = sharded_layout();
    const auto input = bytes(size_t(layout.record_bytes));
    auto spill = std::make_shared<SpillFile>(directory, layout.record_bytes);
    HostKvStorage storage(layout.record_bytes, spill);
    auto record = storage.allocate(input.size(), input.data());
    record.seal();
    auto source = record.source();
    storage.trim(0);
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == input.size());
    const NativeKvLayerRegions regions(layout, 2, 1);
    const auto wanted = expected(layout, 2, 1, input);
    std::vector<uint8_t> output(wanted.size());
    const auto before = storage.io_stats();
    // The deferred source owns immutable bytes after the catalog owner disappears.
    record.reset();
    regions.read(source, 0, output.data(), output.size(), 3);
    CHECK(output == wanted);
    const auto after = storage.io_stats();
    CHECK(after.read_bytes - before.read_bytes == wanted.size());
    CHECK(after.reads - before.reads == 18); // ceil(16/3) + ceil(8/3), twice
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == input.size());
    source = {};
    CHECK(storage.disk_bytes() == 0);
}

int main() {
    exact_regions();
    rejects_before_writing();
    const auto directory = std::filesystem::temp_directory_path() /
        ("kvmem-layer-payload-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    cold_reads(directory);
    aligned_padding(directory);
    CHECK(std::filesystem::is_empty(directory));
    std::filesystem::remove(directory);
    std::puts("layer payload: exact native regions, rank selection and bounded cold reads passed");
}
