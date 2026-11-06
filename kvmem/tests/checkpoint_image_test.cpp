#include "kvmem/checkpoint_image.hpp"
#include "kvmem/snapshot_buffer.hpp"
#include <array>
#include <cstring>
#include <future>
#include <iostream>
#include <filesystem>
using namespace kvmem;
static void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f) { bool failed=false; try { f(); } catch(const std::exception&) { failed=true; } require(failed,"bad checkpoint accepted"); }
int main() {
    const auto path=std::filesystem::temp_directory_path();
    SpillFile file(path,4096);
    const std::array<uint8_t,5> first{0,255,7,4,19};
    const std::array<uint8_t,3> second{2,1,17};
    std::array<uint8_t,5> a{}; std::array<uint8_t,3> b{};
    {
        auto disk=std::make_shared<SpillFile>(path,4096);
        HostKvStorage storage(256,disk);
        auto image=storage.allocate(checkpoint_image_size(8,2));
        write_checkpoint_image(image,97,{{first.data(),first.size()},{second.data(),second.size()}});
        image.seal();
        auto saved=image.fork();
        { auto lease=image.lease();
          read_checkpoint_image(lease,97,{{a.data(),a.size()},{b.data(),b.size()}});
          require(a==first && b==second,"hot record image changed bits"); }
        require(storage.demote(image),"checkpoint record did not migrate");
        image.reset();
        read_checkpoint_image(saved,97,{{a.data(),a.size()},{b.data(),b.size()}});
        require(a==first && b==second && storage.charged_bytes()==0,"cold read promoted or changed image");
        rejects([&]{read_checkpoint_image(saved,98,{{a.data(),a.size()},{b.data(),b.size()}});});
        rejects([&]{read_checkpoint_image(saved,97,{{a.data(),4},{b.data(),b.size()}});});
        auto corrupt_image=saved.fork();
        const uint8_t corrupt_byte=42; corrupt_image.write(36,&corrupt_byte,1); corrupt_image.seal();
        require(storage.demote(corrupt_image),"corrupt test image did not migrate");
        rejects([&]{read_checkpoint_image(corrupt_image,97,{{a.data(),a.size()},{b.data(),b.size()}});});
        read_checkpoint_image(saved,97,{{a.data(),a.size()},{b.data(),b.size()}});
        require(a==first && b==second,"failed branch damaged saved checkpoint");
        auto short_record=storage.allocate(16); short_record.seal();
        rejects([&]{read_checkpoint_image(short_record,97,{{a.data(),a.size()}});});
    }
    auto extent=file.reserve(checkpoint_image_size(8,2));
    write_checkpoint_image(extent,97,{{first.data(),first.size()},{second.data(),second.size()}});
    read_checkpoint_image(extent,97,{{a.data(),a.size()},{b.data(),b.size()}});
    require(a==first && b==second,"native image changed bits");
    rejects([&] { read_checkpoint_image(extent,98,{{a.data(),a.size()},{b.data(),b.size()}}); });
    rejects([&] { read_checkpoint_image(extent,97,{{a.data(),4},{b.data(),b.size()}}); });
    uint8_t corrupt=0;
    extent.write(36,&corrupt,1);
    rejects([&] { read_checkpoint_image(extent,97,{{a.data(),a.size()},{b.data(),b.size()}}); });
    auto short_file=file.reserve(checkpoint_image_size(8,2));
    rejects([&] { read_checkpoint_image(short_file,97,{{a.data(),a.size()},{b.data(),b.size()}}); });
    rejects([&] { file.reserve(4096); });
    StorageIoQueue queue(1,1024);
    std::promise<void> release,entered;
    auto barrier=release.get_future().share();
    auto payload=std::make_shared<std::array<uint8_t,5>>(first);
    std::weak_ptr<std::array<uint8_t,5>> weak=payload;
    auto pending=file.reserve(checkpoint_image_size(5,1));
    auto ticket=queue.submit(5,[source=payload,pending,barrier,&entered] {
        entered.set_value(); barrier.wait();
        write_checkpoint_image(pending,123,{{source->data(),source->size()}});
    });
    entered.get_future().wait();
    payload.reset();
    require(!weak.expired(),"in-flight write lost source lease");
    rejects([&] { queue.submit(5,[]{}); });
    require(!ticket.cancel(),"started I/O cancelled without drain");
    release.set_value();ticket.wait();
    require(weak.expired(),"completed I/O retained source lease");
    read_checkpoint_image(pending,123,{{a.data(),a.size()}});
    require(a==first,"leased asynchronous image corrupted");
    auto failure=queue.submit(1,[]{throw std::runtime_error("injected write failure");});
    rejects([&]{failure.wait();});
    require(failure.status()==StorageTaskStatus::Failed,"failed write published as complete");
    struct Account { size_t used=5, limit=5; } account;
    std::vector<uint8_t> values(first.begin(),first.end()), encoded;
    auto binding=SnapshotBuffer::bind(values);
    binding.accounting=&account;
    binding.account=[](void* p,int64_t delta) { auto& a=*static_cast<Account*>(p); a.used=size_t(int64_t(a.used)+delta); };
    binding.reserve=[](void* p,uint64_t n) { auto& a=*static_cast<Account*>(p);
        if(a.used>a.limit || n>a.limit-a.used) return false; a.used+=size_t(n); return true; };
    SnapshotWriter writer([&](const void* p,size_t n) { const auto* b=static_cast<const uint8_t*>(p);encoded.insert(encoded.end(),b,b+n); });
    binding.write(binding.object,writer);binding.release();
    auto restore=[&] { size_t offset=0; SnapshotReader reader([&](void* p,size_t n) {
        std::memcpy(p,encoded.data()+offset,n);offset+=n; },encoded.size()); binding.restore(reader); };
    account.limit=4;rejects(restore);
    require(values.empty() && account.used==0,"whole-session restore allocated before admission");
    account.limit=5;encoded[0]++;rejects(restore);
    require(values.empty() && account.used==0,"failed session restore leaked reservation");
    encoded[0]--;restore();
    require(values==std::vector<uint8_t>(first.begin(),first.end()) && account.used==5,"session restore did not transfer reservation");
    binding.release();require(account.used==0,"session release lost accounting");
    std::cout<<"checkpoint images: exact bytes, identity, checksum, short read, disk limit, in-flight lease and failure passed\n";
}
