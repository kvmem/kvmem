#include "kvmem-conversation-store.h"
#include "kvmem/kv_content.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/spill_io.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <atomic>
#include <thread>

static void check(bool value, const char * why) { if (!value) throw std::runtime_error(why); }
int main() {
    using namespace kvmem;
    KvContent parent(4);
    parent.reserve(8); parent.complete(0, 8);
    auto proof = parent.prefix(5);
    auto child = parent.fork();
    check(!(child->stamp().session == parent.stamp().session), "fork kept mutable owner identity");
    check(child->contains(proof), "fork lost accepted prefix proof");
    parent.truncate(3);
    check(!parent.contains(proof) && child->contains(proof), "parent edit revoked child proof");
    auto grandchild = child->fork();
    child->truncate(2);
    check(!child->contains(proof) && grandchild->contains(proof), "nested branch proof was shared mutably");
    KvContent pending(4); pending.reserve(8); pending.complete(0, 8, false);
    bool refused = false;
    try { pending.fork(); } catch (const MemoryError&) { refused = true; }
    check(refused, "fork accepted speculative-only rows");

    // Different owners execute under different lane locks. Parent revocation
    // must not be read by child/grandchild validation, even concurrently.
    {
        std::vector<std::unique_ptr<KvContent>> parents,children;
        std::vector<std::shared_ptr<const ContentPrefix>> proofs;
        for(int i=0;i<256;++i) {
            auto owner=std::make_unique<KvContent>(4);
            owner->reserve(8);owner->complete(0,8);
            proofs.push_back(owner->prefix(5));
            children.push_back(owner->fork());parents.push_back(std::move(owner));
        }
        std::atomic<bool> start{false},valid{true};
        std::thread revoke([&] {
            while(!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for(size_t i=0;i<parents.size();++i) {
                if(i%3==0) parents[i]->truncate(3);
                else if(i%3==1) parents[i]->reset();
                else parents[i]->invalidate();
            }
        });
        std::thread verify([&] {
            start.store(true,std::memory_order_release);
            for(int repeat=0;repeat<100;++repeat) for(size_t i=0;i<children.size();++i)
                if(!children[i]->contains(proofs[i])) valid.store(false);
        });
        revoke.join();verify.join();
        check(valid.load(),"parent revocation changed concurrently validated child proof");
        for(size_t i=0;i<children.size();++i) {
            auto descendant=children[i]->fork();
            children[i]->truncate(3);children[i]->reserve(8);children[i]->complete(3,8);
            check(!parents[i]->contains(proofs[i]) && !children[i]->contains(proofs[i]),"regrowth resurrected revoked proof");
            check(descendant->contains(proofs[i]),"child revocation changed grandchild proof");
        }
    }

    const auto folder = std::filesystem::temp_directory_path()/"kvmem-shared-cache-test";
    auto spill = std::make_shared<SpillFile>(folder, 4096);
    auto storage = std::make_shared<HostKvStorage>(32, spill);
    RawKvStoreConfig cfg; cfg.n_layer=1; cfg.n_embd_k=1; cfg.n_embd_v=1;
    cfg.block_tokens=4; cfg.k_gpu_row_bytes=2; cfg.v_gpu_row_bytes=2; cfg.payload_storage=storage;
    RawKvStore original(cfg);
    uint8_t bytes[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float keys[8] = {1,2,3,4,5,6,7,8};
    original.write_layer_k_gpu(0,8,0,bytes); original.write_layer_v_gpu(0,8,0,bytes);
    original.write_layer_mean_k(0,5,0,keys);
    const auto tail = original.mean_checkpoint(5);
    original.write_layer_mean_k(5,3,0,keys+5);
    auto branch = original.fork_prefix(5); branch->restore_mean_checkpoint(5,tail);
    check(storage->charged_bytes()+storage->disk_bytes()==32,"fork duplicated physical packed payload");
    float mean=0; branch->mean_k(1,0,&mean); check(mean==5,"partial tail mean came from rejected suffix");
    original.mean_k(1,0,&mean); check(mean==6.5f,"branch changed source statistics");
    storage->trim(0);
    const auto reads=storage->io_stats().reads;
    auto second=original.fork_prefix(8);
    check(storage->io_stats().reads==reads,"fork read cold KV during metadata lookup");
    uint8_t changed[2]={77,88}, out[8]{};
    branch->write_layer_k_gpu(5,1,0,changed);
    check(original.copy_k_gpu(1,0,out,4) && out[2]==11,"child write changed parent packed KV");
    original.clear(); branch.reset();
    check(second->copy_k_gpu(1,0,out,4) && out[2]==11,"source release invalidated reader");
    second.reset();
    check(storage->charged_bytes()==0 && storage->disk_bytes()==0,"last reference leaked physical quota");

    kvmem_store_match a; a.id=1; a.rows=a.live_row=800; a.lcp=320; a.last_n_gen=10;
    a.ckpt_rows={0,256,512,768}; a.client_id="A";
    kvmem_store_limits limits; limits.max_stores=3;
    auto selection=kvmem_store_select_shared({a},600,false,"B",limits,true);
    check(selection.action==kvmem_store_action::fork && selection.source==1 && selection.keep==256,"foreign prefix consumed parent");
    auto tool=kvmem_store_select_shared({a},600,false,"B",limits,false);
    check(tool.action==kvmem_store_action::fresh,"tool continuation discovered foreign history");
    a.allow_bound_edit=true;
    auto edit=kvmem_store_select_shared({a},600,false,"A",limits,true);
    check(edit.action==kvmem_store_action::extend && edit.id==1,"explicit owner edit lost its lineage");
    std::cout << "shared-cache accepted boundaries, independent proofs, sealed KV COW, cold references and owner policy PASS\n";
}
