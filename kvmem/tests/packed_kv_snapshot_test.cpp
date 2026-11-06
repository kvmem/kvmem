// PROTOTYPE contract evidence, pending coordinator review.
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/memory_contract.hpp"
#include "kvmem/snapshot.hpp"
#include "kvmem/spill_io.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x); std::abort(); } } while (0)
using namespace kvmem;
template<class F> void rejects(F fn) {
    bool failed = false;
    try { fn(); } catch (const MemoryError &) { failed = true; }
    CHECK(failed);
}
static RawKvStoreConfig config(uint64_t capacity = 4096) {
    RawKvStoreConfig c;
    c.n_layer = 2; c.n_embd_k = c.n_embd_v = 2; c.block_tokens = 4;
    c.k_gpu_row_bytes = c.v_gpu_row_bytes = 4;
    c.payload_storage = std::make_shared<HostKvStorage>(capacity);
    return c;
}
static void write(RawKvStore & raw, uint32_t pos, uint32_t n, uint32_t layer, uint8_t value = 7) {
    std::vector<uint8_t> bytes(n * 4, value);
    raw.write_layer_k_gpu(pos,n,layer,bytes.data()); raw.write_layer_v_gpu(pos,n,layer,bytes.data());
}
static std::vector<uint8_t> read(const HostKvRecord & r) {
    std::vector<uint8_t> out(r.size()); r.read(0,out.data(),out.size()); return out;
}
static void coverage() {
    auto c = config(); RawKvStore raw(c);
    write(raw,2,1,0);
    CHECK(raw.has_k_gpu(0,0,3) && raw.has_v_gpu(0,0,3)); // inherited API unchanged
    rejects([&] { raw.export_packed({{0,0,3}}); });
    raw.invalidate_packed_from(1);
    CHECK(raw.has_k_gpu(0,0,1));
    rejects([&] { raw.export_packed({{0,0,1}}); });
    write(raw,3,1,0); write(raw,0,2,0);
    rejects([&] { raw.export_packed({{0,0,4}}); }); // row 2 was invalidated
    write(raw,2,1,0);
    auto full = raw.export_packed({{0,0,4}}); // out-of-order complete coverage
    CHECK(full.entries().size() == 1);
    rejects([&] { raw.export_packed({{0,0,4},{0,0,4}}); });
    rejects([&] { raw.export_packed({{0,1,4}}); }); // missing layer
    uint8_t bytes[16] = {};
    raw.write_layer_k_gpu(0,4,1,bytes);
    rejects([&] { raw.export_packed({{0,1,4}}); }); // missing V
    rejects([&] { raw.export_packed({{0,0,5}}); });
    auto child = raw.fork_prefix(3);
    CHECK(child->export_packed({{0,0,3}}).entries().size() == 1);
    rejects([&] { child->export_packed({{0,0,4}}); });
    write(*child,3,1,0,9);
    CHECK(child->export_packed({{0,0,4}}).entries().size() == 1);
    CHECK(read(full.entries()[0].k) == std::vector<uint8_t>(16,7));
    raw.truncate_to(2);
    rejects([&] { raw.export_packed({{0,0,3}}); });
    auto wide_c=config(); wide_c.block_tokens=67; RawKvStore wide(wide_c);
    write(wide,64,3,0); write(wide,0,64,0);
    CHECK(wide.export_packed({{0,0,67}}).entries().size()==1);
    wide.truncate_to(64);
    rejects([&] { wide.export_packed({{0,0,65}}); });
    CHECK(wide.export_packed({{0,0,64}}).entries().size()==1);
    write(wide,64,3,0);
    CHECK(wide.export_packed({{0,0,67}}).entries().size()==1);
    const float legacy[]={1,2}; wide.write_layer_tokens(0,1,0,legacy,nullptr);
    rejects([&] { wide.export_packed({{0,0,1}}); });
    write(wide,0,67,0);
    wide.write_layer_tokens(0,1,0,nullptr,legacy);
    write(wide,66,1,0);
    rejects([&] { wide.export_packed({{0,0,67}}); });
}
static void sharing_and_import() {
    auto c = config(); RawKvStore raw(c), dest(c);
    write(raw,0,4,0); write(raw,8,4,0,8);
    auto a = raw.export_packed({{2,0,4},{0,0,4}});
    auto b = raw.export_packed({{0,0,4},{2,0,4}});
    CHECK(a.entries()[0].k.storage_identity() == b.entries()[0].k.storage_identity());
    const auto charged = c.payload_storage->charged_bytes();
    CHECK(charged == 64);
    const float mean[] = {2,4,6,8}; dest.write_layer_mean_k(0,2,0,mean);
    auto tail_mean = dest.mean_checkpoint(2);
    dest.import_packed(a);
    CHECK(dest.mean_checkpoint(2) == tail_mean);
    CHECK(c.payload_storage->charged_bytes() == charged);
    CHECK(!dest.has_k_gpu(1,0,1));
    auto imported = dest.export_packed({{0,0,4},{2,0,4}});
    CHECK(imported.entries()[1].v.storage_identity() == a.entries()[1].v.storage_identity());
    rejects([&] { dest.import_packed(a,{1}); });
    auto different = c; different.k_gpu_row_bytes = 8;
    RawKvStore mismatch(different);
    rejects([&] { mismatch.import_packed(a); });
    auto foreign_c = config(); RawKvStore foreign(foreign_c);
    write(foreign,0,4,0,42);
    rejects([&] { foreign.import_packed(a); });
    CHECK(read(foreign.export_packed({{0,0,4}}).entries()[0].k) == std::vector<uint8_t>(16,42));
    std::atomic<bool> cancelled{true};
    rejects([&] { dest.import_packed(a,{SIZE_MAX,&cancelled}); });
    rejects([&] { raw.export_packed({{0,0,4}},{SIZE_MAX,&cancelled}); });
    rejects([&] { raw.export_packed({{0,0,4}},{1}); });
    CHECK(read(dest.export_packed({{0,0,4}}).entries()[0].k) == std::vector<uint8_t>(16,7));
    auto private_tail = raw.export_packed({{0,0,3}});
    CHECK(private_tail.entries()[0].k.storage_identity() != a.entries()[0].k.storage_identity());
    auto bytes = read(private_tail.entries()[0].k);
    CHECK(std::all_of(bytes.begin(),bytes.begin()+12,[](uint8_t v){return v==7;}));
    CHECK(std::all_of(bytes.begin()+12,bytes.end(),[](uint8_t v){return v==0;}));
    write(raw,0,4,0,99); raw.clear();
    CHECK(read(a.entries()[0].k) == std::vector<uint8_t>(16,7));
    CHECK(read(private_tail.entries()[0].k) == bytes);
    dest.import_packed(private_tail);
    CHECK(!dest.has_k_gpu(2,0)); // import replaces rather than merges packed bindings
    CHECK(!dest.has_k_gpu(0,0,4));
    write(dest,3,1,0,9);
    CHECK(read(private_tail.entries()[0].k) == bytes);
}
static void failures_and_legacy() {
    {
        auto implicit=config(); implicit.payload_storage.reset(); RawKvStore source(implicit);
        write(source,0,4,0);
        auto child=source.fork_prefix(4);
        auto shared=child->export_packed({{0,0,4}});
        source.import_packed(shared);
        CHECK(source.config().payload_storage == child->config().payload_storage);
    }
    auto c = config(48); RawKvStore raw(c); write(raw,0,4,0);
    rejects([&] { raw.export_packed({{0,0,3}}); }); // K tail fits, V fails: unwind both
    CHECK(c.payload_storage->charged_bytes() == 32);
    auto proof = raw.export_packed({{0,0,4}});
    CHECK(read(proof.entries()[0].k) == std::vector<uint8_t>(16,7));
    std::vector<uint8_t> image;
    SnapshotWriter writer([&](const void * p,size_t n) {
        const auto b=static_cast<const uint8_t *>(p); image.insert(image.end(),b,b+n);
    });
    raw.snapshot_write(writer);
    auto restored_c = config(); RawKvStore restored(restored_c);
    size_t offset=0;
    SnapshotReader reader([&](void * p,size_t n) { std::memcpy(p,image.data()+offset,n); offset+=n; },image.size());
    restored.snapshot_read(reader,1);
    CHECK(restored.has_k_gpu(0,0,4));
    rejects([&] { restored.export_packed({{0,0,4}}); });
    write(restored,0,4,0);
    CHECK(restored.export_packed({{0,0,4}}).entries().size()==1);
    auto storage=restored_c.payload_storage;
    { auto abandoned=restored.export_packed({{0,0,3}}); CHECK(storage->charged_bytes()==64); }
    CHECK(storage->charged_bytes()==32);
    restored.clear(); CHECK(storage->charged_bytes()==0);
}
static void ssd() {
    const auto root=std::filesystem::temp_directory_path()/ ("packed-snapshot-"+std::to_string(std::random_device{}()));
    {
        auto c=config(); auto spill=std::make_shared<SpillFile>(root,4096);
        c.payload_storage=std::make_shared<HostKvStorage>(64,spill);
        RawKvStore raw(c),dest(c); write(raw,0,4,0);
        auto image=raw.export_packed({{0,0,4}});
        c.payload_storage->trim(0);
        CHECK(c.payload_storage->charged_bytes()==0);
        const auto before=spill->stats();
        auto again=raw.export_packed({{0,0,4}});
        dest.import_packed(again);
        CHECK(spill->stats().reads==before.reads && spill->stats().writes==before.writes);
        CHECK(c.payload_storage->charged_bytes()==0);
        raw.clear();
        CHECK(read(image.entries()[0].k)==std::vector<uint8_t>(16,7));
        CHECK(c.payload_storage->charged_bytes()==0); // cold read never promotes
        auto tail=dest.export_packed({{0,0,3}});
        CHECK(read(tail.entries()[0].v)[12]==0);
        CHECK(c.payload_storage->disk_bytes()>0);
    }
    CHECK(std::filesystem::is_empty(root)); std::filesystem::remove(root);
}
int main() {
    coverage(); sharing_and_import(); failures_and_legacy(); ssd();
    std::puts("PASS pending packed snapshot prototype");
}
