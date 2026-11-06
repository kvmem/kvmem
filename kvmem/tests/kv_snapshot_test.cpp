#include "kvmem/kv_snapshot.hpp"
#include "kvmem/spill_io.hpp"
#include <filesystem>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace kvmem;
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(0)
template<class F> void refuses(F&& f,MemoryErrorCode code) {
    try{f();}catch(const MemoryError& e){CHECK(e.code()==code);return;}
    throw std::runtime_error("expected refusal");
}
int main() {
    const auto path=std::filesystem::temp_directory_path()/("kvmem-snapshot-"+std::to_string(std::random_device{}()));
    std::filesystem::create_directory(path);
    {
        auto spill=std::make_shared<SpillFile>(path,4096);
        HostKvStorage storage(128,spill);
        auto full=std::make_shared<SealedKvBlock>(64);
        auto tail=std::make_shared<SealedKvBlock>(32);
        const std::vector<uint8_t> a(64,11),b(32,22),c(64,33);
        KvSnapshot first({full,tail},{{0,32,0,32},{1,0,32,32},{0,0,64,32}},96);
        CHECK(first.missing_bytes()==96);
        refuses([&]{first.read();},MemoryErrorCode::Pending);
        CHECK(full->seal(storage,a.data(),a.size()));CHECK(tail->seal(storage,b.data(),b.size()));
        CHECK(first.ready() && first.missing_bytes()==0);
        std::vector<uint8_t> expected(96,11);std::fill(expected.begin()+32,expected.begin()+64,22);
        CHECK(first.read()==expected);
        auto grown=std::make_shared<SealedKvBlock>(64);
        KvSnapshot next({full,grown},{{0,0,0,64},{1,0,64,64}},128);
        CHECK(next.missing_bytes()==64);
        CHECK(grown->seal(storage,c.data(),c.size()));
        CHECK(!full->seal(storage,a.data(),a.size()));
        CHECK(storage.charged_bytes()+storage.disk_bytes()==160);
        CHECK(first.read()==expected);CHECK(next.read().back()==33);
        const auto alias=full->record();
        auto imported=std::make_shared<SealedKvBlock>(alias);
        CHECK(imported->record().storage_identity()==alias.storage_identity());
        CHECK(storage.charged_bytes()+storage.disk_bytes()==160);
        storage.trim(0);CHECK(storage.charged_bytes()==0);
        CHECK(first.read()==expected && next.read().front()==11);
        CHECK(storage.charged_bytes()==0); // cold read does not promote
        refuses([&]{KvSnapshot bad({full},{{0,0,0,32},{0,0,32,32}},64);},MemoryErrorCode::InvalidPlan);
        refuses([&]{KvSnapshot bad({full},{{0,0,1,64}},65);},MemoryErrorCode::InvalidPlan);
        refuses([&]{full->seal(storage,a.data(),63);},MemoryErrorCode::InvalidPlan);
        auto pending=std::make_shared<SealedKvBlock>(129);
        refuses([&]{pending->seal(storage,std::vector<uint8_t>(129,7).data(),129);},MemoryErrorCode::BudgetExceeded);
        CHECK(!pending->ready() && first.read()==expected);
    }
    {
        HostKvStorage storage(4096);
        auto origin=std::make_shared<SealedKvBlock>(64);
        std::vector<uint8_t> data(64,9);
        auto a=std::async(std::launch::async,[&]{origin->seal(storage,data.data(),data.size());});
        auto b=std::async(std::launch::async,[&]{origin->seal(storage,data.data(),data.size());});
        a.get();b.get();CHECK(storage.charged_bytes()==64);
        KvSnapshot source({origin},{{0,0,0,64}},64);
        auto branch=source;origin.reset();CHECK(branch.read()==data);
    }
    CHECK(std::filesystem::is_empty(path));std::filesystem::remove(path);
    std::cout<<"PASS shared snapshot: incremental records, native ranges, cold reads, quota failure and concurrent publication\n";
}
