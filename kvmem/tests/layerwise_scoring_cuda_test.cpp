#include "kvmem/page_softmax_cuda.hpp"
#include "kvmem/raw_kv_store.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
void cuda_check(cudaError_t e) { if (e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct DeviceBytes {
    void* data=nullptr;
    explicit DeviceBytes(size_t bytes) { cuda_check(cudaMalloc(&data,bytes)); }
    ~DeviceBytes() { cudaFree(data); }
};
float represented(float value,kvmem::NativeQueryScalar type,void* destination) {
    if (type==kvmem::NativeQueryScalar::Float32) std::memcpy(destination,&value,4);
    else if (type==kvmem::NativeQueryScalar::Float16) {
        const auto half=__float2half_rn(value);
        std::memcpy(destination,&half,2); value=__half2float(half);
    } else {
        uint32_t bits; std::memcpy(&bits,&value,4);
        bits+=0x7fffu+((bits>>16)&1u);
        const auto bf16=uint16_t(bits>>16); std::memcpy(destination,&bf16,2);
        bits=uint32_t(bf16)<<16; std::memcpy(&value,&bits,4);
    }
    // Explicit recall operand boundary, independent from device packing.
    return __half2float(__float2half_rn(value));
}

void qualify(kvmem::NativeQueryScalar type,bool token_minor,uint32_t rows,uint32_t blocks) {
    constexpr uint32_t heads=16,kvheads=2,dim=256,layer=1;
    const auto scalar=kvmem::native_query_scalar_bytes(type);
    kvmem::RawKvStore keys({2,kvheads*dim,0,4});
    std::vector<float> means(size_t(blocks)*kvheads*dim),sum(kvheads*dim);
    for (uint32_t b=0;b<blocks;++b) {
        for (size_t d=0;d<sum.size();++d) {
            const float mean=float(int((b*13+d*7)%101)-50)/16.f;
            means[size_t(b)*sum.size()+d]=__half2float(__float2half_rn(mean));
            sum[d]=4.f*mean;
        }
        keys.write_layer_mean_sum(b*4,4,layer,sum.data());
        std::fill(sum.begin(),sum.end(),800.f); // unrelated layer is an adversary
        keys.write_layer_mean_sum(b*4,4,0,sum.data());
    }
    kvmem::NativeCurrentLayerQuery view;
    view.layer=layer;view.rows=rows;view.query_heads=heads;view.kv_heads=kvheads;view.head_dim=dim;
    view.attention_scale=0.0625f;view.scalar=type;
    view.row_stride=token_minor?scalar:(size_t(heads)*dim+8)*scalar;
    view.dimension_stride=token_minor?size_t(rows)*scalar:scalar;
    view.head_stride=token_minor?(size_t(rows)*dim+8)*scalar:size_t(dim)*scalar;
    view.bytes=size_t(rows-1)*view.row_stride+size_t(heads-1)*view.head_stride+
        size_t(dim-1)*view.dimension_stride+scalar;
    std::vector<unsigned char> native(view.bytes,0xff);
    std::vector<float> query(size_t(rows)*heads*dim);
    for (uint32_t r=0;r<rows;++r) for (uint32_t h=0;h<heads;++h) for (uint32_t d=0;d<dim;++d) {
        const auto index=(size_t(r)*heads+h)*dim+d;
        query[index]=represented(float(int((index*17)%127)-63)/31.f,type,
            native.data()+size_t(r)*view.row_stride+size_t(h)*view.head_stride+size_t(d)*view.dimension_stride);
    }
    auto frame=kvmem::query_score_frame(uint64_t(blocks)*4,4,3,1,1,{0,3,blocks-1});
    check(frame.score_mask[3] && !frame.transport_mask[3],"mandatory transport unexpectedly changed score domain");
    std::vector<double> expected(blocks,0),logits(blocks);
    for (uint32_t r=0;r<rows;++r) for (uint32_t h=0;h<heads;++h) {
        double maximum=-INFINITY,total=0;
        for (uint32_t b=0;b<blocks;++b) if (frame.score_mask[b]) {
            double dot=0;
            for (uint32_t d=0;d<dim;++d)
                dot+=double(query[(size_t(r)*heads+h)*dim+d])*
                    means[(size_t(b)*kvheads+h/(heads/kvheads))*dim+d];
            logits[b]=dot*view.attention_scale;maximum=std::max(maximum,logits[b]);
        }
        for (uint32_t b=0;b<blocks;++b) if (frame.score_mask[b]) total+=std::exp(logits[b]-maximum);
        for (uint32_t b=0;b<blocks;++b) if (frame.score_mask[b])
            expected[b]+=std::exp(logits[b]-maximum)/total/heads;
    }
    cudaStream_t stream;cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    struct Stream { cudaStream_t value;~Stream(){cudaStreamSynchronize(value);cudaStreamDestroy(value);} } owner{stream};
    DeviceBytes device(view.bytes);view.data=device.data;
    cuda_check(cudaMemcpyAsync(device.data,native.data(),native.size(),cudaMemcpyHostToDevice,stream));
    const auto bytes=kvmem::PageSoftmaxCuda::workspace_bytes(blocks,heads,kvheads,dim);
    DeviceBytes workspace(bytes);
    kvmem::PageSoftmaxCuda scorer(0,workspace.data,bytes,blocks,heads,kvheads,dim);
    auto actual=scorer.score_current_layer(keys,view,frame,stream);
    double worst=0;
    for (size_t b=0;b<actual.size();++b) {
        const auto error=std::abs(actual[b]-expected[b]);worst=std::max(worst,error);
        check(std::isfinite(actual[b]) && error<=2e-6+expected[b]*3e-5,"native Q score differs from represented-input FP64 oracle");
        if (!frame.score_mask[b]) check(actual[b]==0,"masked page acquired native mass");
    }
    check(std::abs(std::accumulate(actual.begin(),actual.end(),0.)-rows)<rows*2e-5,"native Q score lost raw mass");
    // Repartition native row views. Each call retains exactly the same page
    // denominator; no per-tile top-k or Query averaging is introduced.
    std::vector<double> tiled(blocks,0);
    for (uint32_t first=0;first<rows;first+=7) {
        const auto scores=scorer.score_current_layer(keys,kvmem::native_layer_query_rows(view,first,std::min(7u,rows-first)),frame,stream);
        for (size_t b=0;b<tiled.size();++b) tiled[b]+=scores[b];
    }
    for (size_t b=0;b<tiled.size();++b)
        check(std::abs(tiled[b]-expected[b])<=2e-6+expected[b]*3e-5,"native tiled Query score changed formula");
    // A failed operand must not poison the borrowed workspace for a retry.
    auto bad=frame;++bad.counts[0];
    bool rejected=false;
    try { scorer.score_current_layer(keys,view,bad,stream); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"stale target-layer coverage was accepted");
    const auto retry=scorer.score_current_layer(keys,view,frame,stream);
    check(retry==actual,"failed native scoring changed the next valid result");
    // Reuse the native buffer with an invalid first operand, including an empty
    // score domain. Padding remains adversarial but is never read as Q.
    std::array<unsigned char,4> saved{};std::memcpy(saved.data(),native.data(),scalar);
    uint32_t bad_bits=type==kvmem::NativeQueryScalar::Float32?0x7fc00000u:
        (type==kvmem::NativeQueryScalar::Float16?0x7e00u:0x7fc0u);
    cuda_check(cudaMemcpyAsync(device.data,&bad_bits,scalar,cudaMemcpyHostToDevice,stream));
    for (bool empty:{false,true}) {
        auto domain=frame;if (empty) std::fill(domain.score_mask.begin(),domain.score_mask.end(),0);
        rejected=false;
        try { scorer.score_current_layer(keys,view,domain,stream); } catch (const std::invalid_argument&) { rejected=true; }
        check(rejected,"nonfinite native Query operand was accepted");
    }
    cuda_check(cudaMemcpyAsync(device.data,saved.data(),scalar,cudaMemcpyHostToDevice,stream));
    check(scorer.score_current_layer(keys,view,frame,stream)==actual,"operand failure poisoned scorer retry");
    cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    rejected=false;
    try { scorer.score_current_layer(keys,view,frame,stream); } catch (const std::invalid_argument&) { rejected=true; }
    cudaGraph_t graph=nullptr;cuda_check(cudaStreamEndCapture(stream,&graph));cuda_check(cudaGraphDestroy(graph));
    check(rejected,"captured native Query scoring was accepted");
    std::cout<<"scalar="<<int(type)<<" token_minor="<<token_minor<<" rows="<<rows<<" blocks="<<blocks
        <<" worst_abs="<<worst<<" workspace="<<scorer.device_bytes()<<'\n';
}
}

int main() {
    try {
        cudaDeviceProp properties{};cuda_check(cudaGetDeviceProperties(&properties,0));
        std::cout<<"native current-Q qualification: "<<properties.name<<'\n';
        for (auto type:{kvmem::NativeQueryScalar::Float32,kvmem::NativeQueryScalar::Float16,kvmem::NativeQueryScalar::BFloat16})
            for (bool token_minor:{false,true}) qualify(type,token_minor,129,7);
        qualify(kvmem::NativeQueryScalar::BFloat16,false,2,8193);
        std::cout<<"native current-Q represented-input FP64 oracle passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
