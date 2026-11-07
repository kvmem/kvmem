// Build with the host kvmem library. This diagnostic observes the public RKV1
// stream and legacy prefix behavior without modifying any common implementation.
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/memory_contract.hpp"
#include "kvmem/snapshot.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace kvmem;
static std::vector<uint8_t> image(RawKvStore & store) {
    std::vector<uint8_t> bytes;
    SnapshotWriter out([&](const void * p, size_t n) {
        auto first = static_cast<const uint8_t *>(p);
        bytes.insert(bytes.end(), first, first+n);
    });
    store.snapshot_write(out);
    return bytes;
}
static bool can_share(RawKvStore & store) {
    try { (void)store.export_packed({{0,0,4}}); return true; }
    catch (const MemoryError &) { return false; }
}
int main() {
    RawKvStoreConfig cfg{};
    cfg.n_layer=cfg.n_embd_k=cfg.n_embd_v=cfg.k_gpu_row_bytes=cfg.v_gpu_row_bytes=1;
    cfg.block_tokens=4;
    cfg.payload_storage=std::make_shared<HostKvStorage>(1u<<20);
    const uint8_t zeros[4]={};
    RawKvStore complete(cfg), missing(cfg), reloaded(cfg);
    for (auto p : {&complete,&missing}) {
        p->write_layer_k_gpu(0,4,0,zeros);
        if (p==&complete) p->write_layer_v_gpu(0,4,0,zeros);
        else {
            p->write_layer_v_gpu(0,2,0,zeros);
            p->write_layer_v_gpu(3,1,0,zeros);
        }
    }
    auto saved=image(complete);
    size_t offset=0;
    SnapshotReader reader([&](void * p, size_t n) {
        std::memcpy(p,saved.data()+offset,n);offset+=n;
    },saved.size());
    reloaded.snapshot_read(reader,1);
    std::printf("rkv1_identical_bytes=%d full_share_before=%d gap_share_before=%d full_share_after=%d\n",
        int(saved==image(missing)),can_share(complete),can_share(missing),can_share(reloaded));
    RawKvStore means(cfg);
    const float rows[]={2,3,41,42};
    means.write_layer_mean_k(0,2,0,rows);
    const auto prefix=means.mean_checkpoint(2);
    means.write_layer_mean_k(2,2,0,rows+2);
    auto short_mean=means.fork_mean_prefix(2);
    auto inherited=short_mean->mean_checkpoint(2);
    std::printf("saved_count=%.0f saved_sum=%.0f short_fork_count=%.0f short_fork_sum=%.0f\n",
        prefix[0],prefix[1],inherited[0],inherited[1]);
    std::vector<uint8_t> children[2];
    for (int i=0;i<2;++i) {
        RawKvStore parent(cfg);
        const float kv[]={2,3,float(41+i*50),float(42+i*50)};
        parent.write_layer_tokens(0,4,0,kv,kv);
        auto child=parent.fork_prefix(2);
        children[i]=image(*child);
        child->write_layer_tokens(2,1,0,kv,nullptr);
        float v[4]={};
        bool read=child->copy_v(0,0,v);
        std::printf("legacy_parent_%d_child_v_read=%d third_v=%.0f\n",i,read,v[2]);
    }
    std::printf("legacy_serialized_children_equal=%d\n",int(children[0]==children[1]));
}
