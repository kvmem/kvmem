// Real libr4d accepted-state export against an independent CPU rank-one recurrence.
#include "state_materialization.hpp"
#include "platform/module.h"
#include "rad_abi_layout.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

using kvmem::radiance::GdnMaterialize;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)

namespace {
constexpr int heads = 2, dim = 128, factors_per_token = 4 * dim + dim + 8;
constexpr int slot_elements = heads * dim * dim, marker = 16 * factors_per_token;
constexpr size_t guard_bytes = 64;

struct Buffer {
    void* data;
    int kind;
    Buffer(size_t bytes, int allocation_kind) : data(rad_dev_alloc(int64_t(bytes),allocation_kind)), kind(allocation_kind) {
        CHECK(data);
    }
    ~Buffer() { rad_dev_free(data,kind); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Stream {
    RadStream value = nullptr;
    Stream() { CHECK(rad_stream_create(&value,0)==RAD_OK); }
    ~Stream() { if(value) { (void)rad_stream_sync(value); rad_stream_destroy(value); } }
};

struct Module {
    void* handle;
    explicit Module(const char* path) {
        std::string error;
        handle=rad::platform::module_open(std::filesystem::absolute(path).string(),&error);
        if(!handle)throw std::runtime_error(error);
    }
    ~Module() { rad::platform::module_close(handle); }
};

void write_value(std::vector<uint8_t>& bytes,size_t index,bool fp16,float value) {
    if(fp16) {
        const uint16_t half=rad_f32_to_f16(value);
        std::memcpy(bytes.data()+index*2,&half,2);
    } else std::memcpy(bytes.data()+index*4,&value,4);
}

double read_value(const uint8_t* bytes,size_t index,bool fp16) {
    if(fp16) {
        uint16_t half;
        std::memcpy(&half,bytes+index*2,2);
        return rad_f16_to_f32(half);
    }
    float value;
    std::memcpy(&value,bytes+index*4,4);
    return value;
}

std::vector<uint8_t> initial_state(bool fp16) {
    const size_t element=fp16 ? 2 : 4;
    std::vector<uint8_t> bytes(size_t(2*slot_elements)*element,0);
    for(int head=0;head<heads;++head) {
        const size_t anchor=size_t(head)*dim*dim, factors=slot_elements+anchor;
        for(int row=0;row<dim;++row)
            for(int column=0;column<dim;++column)
                write_value(bytes,anchor+row*dim+column,fp16,
                            .02f*std::sin(float(13*head+3*row+column)*.07f));
        for(int token=0;token<16;++token) {
            const size_t at=factors+size_t(token)*factors_per_token;
            for(int group=0;group<4;++group) {
                for(int column=0;column<dim;++column)
                    write_value(bytes,at+group*dim+column,fp16,
                                .04f*std::sin(float(11*head+5*token+column)*.13f));
                write_value(bytes,at+5*dim+2*group,fp16,-.01f*(token+1));
                write_value(bytes,at+5*dim+2*group+1,fp16,.2f+.01f*token);
            }
            for(int row=0;row<dim;++row)
                write_value(bytes,at+4*dim+row,fp16,.15f*std::cos(float(head+token+row)*.09f));
        }
        write_value(bytes,factors+marker,fp16,2.0f);
    }
    return bytes;
}

std::vector<double> reference(const std::vector<uint8_t>& bytes,bool fp16,int accepted) {
    std::vector<double> output(slot_elements);
    for(int head=0;head<heads;++head) {
        const size_t anchor=size_t(head)*dim*dim, factors=slot_elements+anchor;
        for(int row=0;row<dim;++row) {
            double state[dim];
            for(int column=0;column<dim;++column)
                state[column]=read_value(bytes.data(),anchor+row*dim+column,fp16);
            for(int token=0;token<accepted;++token) {
                const size_t at=factors+size_t(token)*factors_per_token;
                const double decay=std::exp(read_value(bytes.data(),at+5*dim,fp16));
                const double beta=read_value(bytes.data(),at+5*dim+1,fp16);
                double product=0;
                for(int column=0;column<dim;++column) {
                    state[column]*=decay;
                    product+=state[column]*read_value(bytes.data(),at+column,fp16);
                }
                const double update=(read_value(bytes.data(),at+4*dim+row,fp16)-product)*beta;
                for(int column=0;column<dim;++column)
                    state[column]+=update*read_value(bytes.data(),at+column,fp16);
            }
            for(int column=0;column<dim;++column)output[anchor+row*dim+column]=state[column];
        }
    }
    return output;
}

void check_case(GdnMaterialize materialize,bool fp16,int accepted) {
    const auto original=initial_state(fp16);
    const auto expected=reference(original,fp16,accepted);
    const size_t bytes=original.size(),element=fp16 ? 2 : 4;
    Buffer live(bytes,RAD_MEM_DEVICE),clone(bytes+2*guard_bytes,RAD_MEM_DEVICE);
    Buffer upload(bytes+2*guard_bytes,RAD_MEM_HOST_PINNED),readback(bytes+2*guard_bytes,RAD_MEM_HOST_PINNED);
    Stream stream;
    std::memset(upload.data,0xA5,bytes+2*guard_bytes);
    std::memcpy(static_cast<uint8_t*>(upload.data)+guard_bytes,original.data(),bytes);
    CHECK(rad_memcpy_async(live.data,static_cast<uint8_t*>(upload.data)+guard_bytes,int64_t(bytes),stream.value)==RAD_OK);
    CHECK(rad_memcpy_async(clone.data,upload.data,int64_t(bytes+2*guard_bytes),stream.value)==RAD_OK);
    void* payload=static_cast<uint8_t*>(clone.data)+guard_bytes;
    CHECK(rad_memcpy_async(payload,live.data,int64_t(bytes),stream.value)==RAD_OK);
    CHECK(rad_stream_sync(stream.value)==RAD_OK);
    // The export performs a zero-token pass with null projection inputs internally.
    CHECK(materialize(payload,heads,fp16,accepted)==0);
    CHECK(rad_memcpy_async(readback.data,clone.data,int64_t(bytes+2*guard_bytes),stream.value)==RAD_OK);
    CHECK(rad_stream_sync(stream.value)==RAD_OK);
    const auto* result=static_cast<const uint8_t*>(readback.data);
    CHECK(std::all_of(result,result+guard_bytes,[](uint8_t value){return value==0xA5;}));
    CHECK(std::all_of(result+guard_bytes+bytes,result+2*guard_bytes+bytes,[](uint8_t value){return value==0xA5;}));
    const uint8_t* state=result+guard_bytes;
    double maximum_error=0;
    for(size_t i=0;i<expected.size();++i) {
        const double actual=read_value(state,i,fp16);
        CHECK(std::isfinite(actual));
        maximum_error=std::max(maximum_error,std::abs(actual-expected[i]));
        CHECK(std::abs(actual-expected[i])<=(fp16 ? 4e-5 : 2e-6));
    }
    for(int head=0;head<heads;++head)
        for(int i=0;i<dim*dim;++i)if(i!=marker) {
            const size_t at=size_t(slot_elements+head*dim*dim+i)*element;
            CHECK(std::memcmp(state+at,original.data()+at,element)==0);
        }
    std::vector<uint8_t> before_invalid(state,state+bytes);
    CHECK(materialize(nullptr,heads,fp16,accepted)<0);
    CHECK(materialize(payload,0,fp16,accepted)<0);
    CHECK(materialize(payload,65536,fp16,accepted)<0);
    CHECK(materialize(payload,heads,2,accepted)<0);
    CHECK(materialize(payload,heads,fp16,-1)<0);
    CHECK(materialize(payload,heads,fp16,17)<0);
    CHECK(rad_memcpy_async(readback.data,payload,int64_t(bytes),stream.value)==RAD_OK);
    CHECK(rad_stream_sync(stream.value)==RAD_OK);
    CHECK(std::memcmp(readback.data,before_invalid.data(),bytes)==0);
    CHECK(rad_memcpy_async(readback.data,live.data,int64_t(bytes),stream.value)==RAD_OK);
    CHECK(rad_stream_sync(stream.value)==RAD_OK);
    CHECK(std::memcmp(readback.data,original.data(),bytes)==0);
    std::cout << "PASS materializer fp16=" << fp16 << " accepted=" << accepted
              << " max_absolute_error=" << maximum_error << '\n';
}

void check_conv_window() {
    // Qwen's BF16 convolution history has width 4 and 10 entries at draft depth 7.
    constexpr size_t channels=3,slots=10,element=2,history=3,bytes=channels*slots*element;
    std::vector<uint8_t> original(bytes+2*guard_bytes,0x5A);
    for(size_t i=0;i<bytes;++i)original[guard_bytes+i]=uint8_t(i+1);
    for(uint32_t accepted:{1u,8u}) {
        auto result=original;
        kvmem::radiance::canonical_conv_window(result.data()+guard_bytes,bytes,slots,4,element,accepted);
        CHECK(std::equal(result.begin(),result.begin()+guard_bytes,original.begin()));
        CHECK(std::equal(result.end()-guard_bytes,result.end(),original.end()-guard_bytes));
        for(size_t channel=0;channel<channels;++channel) {
            const size_t at=guard_bytes+channel*slots*element;
            CHECK(std::memcmp(result.data()+at,original.data()+at+(accepted-1)*element,history*element)==0);
            CHECK(std::all_of(result.begin()+at+history*element,result.begin()+at+slots*element,
                              [](uint8_t value){return value==0;}));
        }
    }
    for(uint32_t accepted:{0u,9u,16u}) {
        auto result=original;
        bool refused=false;
        try { kvmem::radiance::canonical_conv_window(result.data()+guard_bytes,bytes,slots,4,element,accepted); }
        catch(const std::runtime_error&) { refused=true; }
        CHECK(refused && result==original);
    }
    std::cout << "PASS convolution accepted-window bounds and guards\n";
}
}

int main(int argc,char** argv) {
    try {
        CHECK(argc==2 || argc==3);
        const int device=argc==3 ? std::stoi(argv[2]) : 0;
        RadDeviceProps props{};
        CHECK(rad_dev_props(device,&props)==RAD_OK && !props.is_host_backend);
        CHECK(rad_dev_set(device)==RAD_OK);
        std::cout << "GPU " << props.name << " arch=" << props.arch << '\n';
        Module module(argv[1]);
        const auto abi=reinterpret_cast<uint32_t (*)()>(rad::platform::module_symbol(module.handle,"rad_plugin_abi_version"));
        const auto layout=reinterpret_cast<uint64_t (*)()>(rad::platform::module_symbol(module.handle,"rad_plugin_abi_layout"));
        CHECK(abi && abi()==RAD_ABI_VERSION && layout && layout()==rad_abi_layout());
        const auto materialize=reinterpret_cast<GdnMaterialize>(rad::platform::module_symbol(module.handle,"rad_gdn_materialize_v1"));
        CHECK(materialize);
        for(bool fp16:{false,true})for(int accepted:{0,1,8,16})check_case(materialize,fp16,accepted);
        check_conv_window();
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n';return 1; }
}
