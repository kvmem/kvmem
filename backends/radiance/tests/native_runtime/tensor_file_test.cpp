#include "format/safetensors.h"
#include "text/gguf.h"
#include "format/imatrix.h"
#include "platform/file.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
int main(int argc,char**argv) {
    if(argc!=2 || std::filesystem::exists(argv[1])) return 1;
    const std::string header=R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    const uint64_t size=header.size();
    const std::array<float,2> data{1.25f,-2.5f};
    rad::platform::File f; std::string error;
    if(!f.open(argv[1],rad::platform::File::Read|rad::platform::File::Write|rad::platform::File::Create,&error)) return 2;
    if(f.write_at(&size,8,0)!=8 || f.write_at(header.data(),size,8)!=static_cast<int64_t>(size) ||
       f.write_at(data.data(),8,size+8)!=8 || !f.flush()) return 3;
    f.close();
    rad::SafeTensorsFile tensors;
    if(tensors.open(argv[1])!=RAD_OK) return 4;
    const auto* tensor=tensors.find("x");
    if(!tensor || tensor->dtype!=RAD_F32 || tensor->shape!=std::vector<int64_t>{2} ||
       tensor->bytes!=8 || std::memcmp(tensor->data,data.data(),8)!=0) return 5;
    tensors.close();
    if(!f.open(argv[1],rad::platform::File::Read|rad::platform::File::Write,&error) || !f.resize(9)) return 6;
    f.close();
    if(tensors.open(argv[1])!=RAD_E_FORMAT) return 7;
    tensors.close();
    // A complete minimal GGUF v3 with one F32[2] tensor, at offset64.
    std::array<unsigned char,72> gguf{};
    std::memcpy(gguf.data(),"GGUF",4);
    auto put=[&](size_t off, auto value) { std::memcpy(gguf.data()+off,&value,sizeof(value)); };
    put(4,uint32_t{3}); put(8,uint64_t{1}); put(16,uint64_t{0});
    put(24,uint64_t{1}); gguf[32]='x'; put(33,uint32_t{1});
    put(37,uint64_t{2}); put(45,uint32_t{0}); put(49,uint64_t{0});
    std::memcpy(gguf.data()+64,data.data(),8);
    if(!f.open(argv[1],rad::platform::File::Read|rad::platform::File::Write|rad::platform::File::Truncate,&error) ||
       f.write_at(gguf.data(),gguf.size(),0)!=static_cast<int64_t>(gguf.size())) return 8;
    f.close();
    rad::GgufFile g;
    if(g.open(argv[1])!=RAD_OK || !g.tensor("x") ||
       std::memcmp(g.tensor_data(*g.tensor("x")),data.data(),8)!=0) return 9;
    g.close();
    // Legacy imatrix with one named two-column entry and two calls.
    std::array<unsigned char,25> im{};
    auto iput=[&](size_t off, auto value) { std::memcpy(im.data()+off,&value,sizeof(value)); };
    iput(0,int32_t{1}); iput(4,int32_t{1}); im[8]='x';
    iput(9,int32_t{2}); iput(13,int32_t{2}); iput(17,float{2}); iput(21,float{6});
    if(!f.open(argv[1],rad::platform::File::Read|rad::platform::File::Write|rad::platform::File::Truncate,&error) ||
       f.write_at(im.data(),im.size(),0)!=static_cast<int64_t>(im.size())) return 10;
    f.close();
    rad::ImatrixFile matrix; std::vector<float> importance;
    if(matrix.open(argv[1])!=RAD_OK || !matrix.importance("x",0,&importance) ||
       importance!=std::vector<float>{0.5f,1.5f}) return 11;
    matrix.close();
    std::filesystem::remove(argv[1]);
    std::puts("PASS: safetensors, GGUF and imatrix data; truncated input refused");
    return 0;
}
