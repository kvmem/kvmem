#include "platform/file.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
int main(int argc, char** argv) {
    if(argc!=2) return 1;
    using rad::platform::File;
    File f; std::string error;
    // Only a new caller-designated fixture may be created; never clobber a file.
    if(std::filesystem::exists(argv[1])) return 2;
    if(!f.open(argv[1],File::Read|File::Write|File::Create,&error)) { std::puts(error.c_str()); return 3; }
    constexpr int64_t n=65536;
    if(!f.resize(n)) return 4;
    std::array<unsigned char,256> expected{};
    for(size_t i=0;i<expected.size();++i) expected[i]=static_cast<unsigned char>(i*19+5);
    for(int64_t off=0;off<n;off+=expected.size())
        if(f.write_at(expected.data(),expected.size(),off)!=expected.size()) return 5;
    std::atomic<bool> ok{true}; std::vector<std::thread> readers;
    for(int t=0;t<4;++t) readers.emplace_back([&]{
        std::array<unsigned char,256> got{};
        for(int i=0;i<64;++i) if(f.read_at(got.data(),got.size(),i*256)!=got.size() || got!=expected) ok=false;
    });
    for(auto& t:readers)t.join();
    if(!ok || f.size()!=n || !f.flush()) return 6;
    rad::platform::FileView view;
    if(!view.open(f,3,513)) return 7;
    for(int i=0;i<513;++i) if(view.data()[i]!=expected[(i+3)%256]) return 8;
    view.close();
    unsigned char byte{};
    if(f.read_at(&byte,1,n)!=0 || f.read_at(&byte,1,-1)!=-1 ||
       f.read_at(&byte,2,INT64_MAX)!=-1 || view.open(f,n-1,2)) return 9;
    f.close();
    if(f.read_at(&byte,1,0)!=-1) return 10;
    // The test owns this one explicit fixture; no recursive cleanup.
    std::filesystem::remove(argv[1]);
    std::puts("PASS: concurrent positional I/O, mapping with unaligned offset, EOF, bounds, closed handle");
    return 0;
}
