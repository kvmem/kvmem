#include "tiered_archive.hpp"
#include "tiered_buffer.hpp"
#include "memory_test_support.hpp"
#include "kvmem/spill_io.hpp"
#include "storage_fault_fixture.hpp"
#include <filesystem>
#include <future>
#include <random>
#ifdef __linux__
#include <csignal>
#include <sys/resource.h>
#endif
using namespace kvmem::radiance;

static void check_archive(const std::filesystem::path& directory,int profile) {
    const auto d=descriptor(profile);
    const auto n=d.layout.record_bytes;
    auto disk=std::make_shared<SpillFile>(directory,8*n);
    auto storage=std::make_shared<HostKvStorage>(2*n,disk);
    TieredArchive a(d.identity,{42,3},d.layout,6*n,storage);
    TieredArchive b(d.identity,{42,3},d.layout,6*n,storage);
    for(uint64_t i=0;i<4;++i)a.put(payload(d,block(d,i)));
    b.put(payload(d,block(d,9)));
    CHECK(storage->charged_bytes()<=2*n && storage->disk_bytes()>0);
    storage->trim(0);
    CHECK(storage->charged_bytes()==0 && storage->disk_bytes()==5*n);
    for(uint64_t i=0;i<4;++i) {
        std::vector<uint8_t> bytes(n);
        a.read(block(d,i),bytes.data(),bytes.size());
        CHECK(bytes==payload(d,block(d,i)).bytes());
    }
    auto wrong=block(d,0);++wrong.content_version;
    rejects(MemoryErrorCode::Stale,[&]{a.source(wrong);});
    wrong=block(d,0);++wrong.session.incarnation;
    rejects(MemoryErrorCode::Stale,[&]{a.source(wrong);});
    auto other=d;other.identity.model="other";
    rejects(MemoryErrorCode::InvalidPlan,[&]{a.put(payload(other,block(d,0)));});
    auto changed=payload(d,block(d,0)).bytes();changed.back()^=1;
    rejects(MemoryErrorCode::Stale,[&]{a.put(NativeKvPayload(d.identity,d.layout,block(d,0),changed));});
    auto same=TieredArchive::stage(payload(d,block(d,1)),storage,a.previous(block(d,1)));
    a.publish(std::move(same));
    CHECK(storage->charged_bytes()==0); // immutable retry adds no RAM record
    rejects(MemoryErrorCode::Stale,[&]{
        TieredArchive::stage(NativeKvPayload(d.identity,d.layout,block(d,0),changed),storage,a.previous(block(d,0)));
    });
    // Deferred sources survive owner erase, but do not promote cold bytes to H.
    auto source=a.source(block(d,0));a.erase(0);
    CHECK(a.bytes()==3*n && storage->disk_bytes()==5*n);
    StorageIoQueue io(2,2*n);
    std::promise<void> release,started;
    auto gate=release.get_future().share();
    auto first=io.submit(n,[&]{started.set_value();gate.wait();});
    started.get_future().wait();
    std::vector<uint8_t> read(n);
    auto second=io.submit(n,[source,&read]{source.read(0,read.data(),read.size());});
    source={};
    CHECK(!second.ready() && io.pinned_bytes()==2*n);
    release.set_value();first.wait();second.wait();
    CHECK(read==payload(d,block(d,0)).bytes());
    CHECK(storage->charged_bytes()==0 && storage->disk_bytes()==4*n);
    // Replacing a version preserves a separately held immutable old source.
    auto old=a.source(block(d,1));auto next=block(d,1);++next.content_version;
    auto stale=TieredArchive::stage(payload(d,block(d,1)),storage,a.previous(block(d,1)));
    a.put(payload(d,next));
    rejects(MemoryErrorCode::Stale,[&]{a.publish(std::move(stale));});
    old.read(0,read.data(),read.size());CHECK(read==payload(d,block(d,1)).bytes());
    a.read(next,read.data(),read.size());CHECK(read==payload(d,next).bytes());
    old={};
    storage->trim(0);
#ifdef __linux__
    // Real filesystem read failure stays an exception; no zero/stale-byte success.
    truncate_fixture_spill(directory);
    rejects(MemoryErrorCode::TransferFailed,[&]{a.read(next,read.data(),read.size());});
#endif
}
static void failed_write(const std::filesystem::path& directory) {
#ifdef __linux__
    const auto d=descriptor(0);const auto n=d.layout.record_bytes;
    auto storage=std::make_shared<HostKvStorage>(n,std::make_shared<SpillFile>(directory,2*n));
    TieredArchive a(d.identity,{42,3},d.layout,4*n,storage);
    a.put(payload(d,block(d,0)));
    rlimit previous{};CHECK(getrlimit(RLIMIT_FSIZE,&previous)==0);
    const auto handler=std::signal(SIGXFSZ,SIG_IGN);
    auto limit=previous;limit.rlim_cur=0;CHECK(setrlimit(RLIMIT_FSIZE,&limit)==0);
    bool failed=false;
    try{a.put(payload(d,block(d,1)));}
    catch(const MemoryError& e){failed=e.code()==MemoryErrorCode::TransferFailed;}
    CHECK(setrlimit(RLIMIT_FSIZE,&previous)==0);std::signal(SIGXFSZ,handler);
    CHECK(failed && a.bytes()==n && storage->disk_bytes()==0 && storage->charged_bytes()==n);
    std::vector<uint8_t> bytes(n);a.read(block(d,0),bytes.data(),bytes.size());
    CHECK(bytes==payload(d,block(d,0)).bytes());
    rejects(MemoryErrorCode::Stale,[&]{a.source(block(d,1));});
#endif
}
static void checkpoint_alias_accounting() {
    const auto d=descriptor(0);const auto n=d.layout.record_bytes;
    auto storage=std::make_shared<HostKvStorage>(8*n);
    TieredArchive active(d.identity,{42,3},d.layout,8*n,storage);
    MemorySnapshot view;
    for(uint64_t i=0;i<3;++i) {
        auto item=block(d,i);active.put(payload(d,item));
        view.blocks.push_back({item,item.content_version});
    }
    auto input=active.fork({42,4},view);
    CHECK(input.bytes()==3*n && active.shared_bytes(input)==3*n && storage->charged_bytes()==3*n);
    // A newer payload at the same logical ID no longer shares that allocation.
    auto changed=block(d,1);++changed.content_version;active.put(payload(d,changed));
    CHECK(active.shared_bytes(input)==2*n && storage->charged_bytes()==4*n);
    CHECK(active.bytes()+input.bytes()-active.shared_bytes(input)==4*n);
    auto sibling=input.fork({43,1},[&]{
        auto next=view;for(auto& b:next.blocks)b.block.session={42,4};return next;
    }());
    SharedPayloadAccounting account;
    active.account(account);input.account(account);sibling.account(account);
    CHECK(account.credit()==5*n);
    CHECK(active.bytes()+input.bytes()+sibling.bytes()-account.credit()==storage->charged_bytes());
    // Even identical bytes in an independent allocation must be charged.
    TieredArchive independent(d.identity,{42,3},d.layout,8*n,storage);
    independent.put(payload(d,block(d,0)));
    CHECK(active.shared_bytes(independent)==0);
}
int main() {
    checkpoint_alias_accounting();
    const auto directory=std::filesystem::temp_directory_path()/
        ("kvmem-radiance-archive-"+std::to_string(std::random_device{}()));
    CHECK(std::filesystem::create_directory(directory));
    for(int profile=0;profile<3;++profile)check_archive(directory,profile);
    failed_write(directory);
    {
        auto storage=std::make_shared<HostKvStorage>(1024,std::make_shared<SpillFile>(directory,16*1024));
        auto io=std::make_shared<StorageIoQueue>(2,32*1024);
        std::vector<uint8_t> bytes(7001);
        for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(37*i+11);
        const auto expected=bytes;
        std::unique_ptr<TieredBuffer> parked;
        park_buffer(bytes,parked,storage,io);
        CHECK(bytes.empty() && parked->size()==expected.size());
        CHECK(storage->charged_bytes()<=1024 && storage->disk_bytes()>0);
        CHECK(read_buffer(*parked,io)==expected);
        auto branch=parked->fork();
        SharedPayloadAccounting shared;
        parked->account(shared);branch->account(shared);
        CHECK(shared.credit()==expected.size());
        CHECK(storage->charged_bytes()+storage->disk_bytes()==expected.size());
        parked.reset();CHECK(branch->read()==expected);
        branch.reset();
        parked.reset();CHECK(storage->charged_bytes()==0 && storage->disk_bytes()==0);
        auto small=std::make_shared<HostKvStorage>(1024,std::make_shared<SpillFile>(directory,1024));
        bytes=expected;
        rejects(MemoryErrorCode::BudgetExceeded,[&]{park_buffer(bytes,parked,small,io);});
        CHECK(bytes==expected && !parked); // failed parking retains the original complete state
        CHECK(small->charged_bytes()==0 && small->disk_bytes()==0);
    }
    CHECK(std::filesystem::is_empty(directory));std::filesystem::remove(directory);
    std::puts("PASS native archive: shared H/D, exact planes, version isolation, deferred read lifetime and real I/O errors");
}
