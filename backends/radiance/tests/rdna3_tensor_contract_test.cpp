#include "r3di_args.h"
#include <cstdio>

int main() {
    RadTensor t{};
    t.rank=3;t.shape[0]=2;t.shape[1]=3;t.shape[2]=128;
    rad_tensor_pack(&t);
    long long rows=0;
    auto check=[&](long long pitch, long long count) {
        if(r3di_rows_pitch(&t,128,&rows)!=pitch || (pitch>=0 && rows!=count)) {
            std::fprintf(stderr,"incorrect flattened tensor pitch\n");return false;
        }
        return true;
    };
    if(!check(128,6))return 1;
    t.stride[0]=512;
    if(!check(-1,0))return 1; // padding between batches cannot be flattened
    t.stride[0]=768;t.stride[1]=256;
    if(!check(256,6))return 1; // uniformly padded rows can be flattened
    t.shape[1]=1;t.stride[0]=256;t.stride[1]=7;
    if(!check(256,2))return 1; // ignore strides on singleton dimensions
    t.shape[0]=1;t.shape[1]=3;t.stride[0]=1;t.stride[1]=128;
    if(!check(128,3))return 1;
    t.stride[2]=2;
    if(!check(-1,0))return 1;
    t=RadTensor{};t.rank=2;t.shape[0]=1;t.shape[1]=128;rad_tensor_pack(&t);
    t.stride[0]=1;
    if(!check(1,1))return 1; // retain invalid explicit pitch for launch rejection
    std::puts("RDNA3 tensor pitch contracts passed");
}
