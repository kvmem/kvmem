#include "device/directio.h"
#include "rad_types.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
int main(int argc, char** argv) {
    if (argc != 2 || std::filesystem::exists(argv[1])) return 1;
    constexpr size_t N=8192;
    struct alignas(4096) Block { std::array<unsigned char,N> bytes; };
    auto input=std::make_unique<Block>(), output=std::make_unique<Block>();
    for (size_t i=0;i<N;++i) input->bytes[i]=static_cast<unsigned char>(i*17+3);
    rad::DirectFile f;
    // Require actual unbuffered I/O; no silent buffered success on this target.
    if(f.open(argv[1],rad::RAD_DIO_WRITE|rad::RAD_DIO_CREATE|rad::RAD_DIO_TRUNC)!=RAD_OK || !f.direct()) return 2;
    if(f.write_at(input->bytes.data(),0,N)!=RAD_OK || f.size()!=N) return 3;
    if(f.read_at(output->bytes.data(),0,N)!=RAD_OK || output->bytes!=input->bytes) return 4;
    std::array<unsigned char,513> patch{}; patch.fill(91);
    if(f.write_at(patch.data(),4093,patch.size())!=RAD_OK) return 5;
    std::memcpy(input->bytes.data()+4093,patch.data(),patch.size());
    if(f.read_at(output->bytes.data(),0,N)!=RAD_OK || output->bytes!=input->bytes) return 6;
    std::array<unsigned char,29> tail{}; tail.fill(71);
    if(f.write_at(tail.data(),N,tail.size())!=RAD_OK || f.size()!=N+tail.size()) return 7;
    std::array<unsigned char,29> read_tail{};
    if(f.read_at(read_tail.data(),N,read_tail.size())!=RAD_OK || read_tail!=tail) return 8;
    if(f.read_at(read_tail.data(),N+1,read_tail.size())!=RAD_E_IO) return 9;
    f.close();
    if(f.read_at(read_tail.data(),0,1)!=RAD_E_STATE) return 10;
    std::filesystem::remove(argv[1]);
    std::puts("PASS: unbuffered aligned I/O, cross-block patch, short tail, EOF and closed file");
    return 0;
}
