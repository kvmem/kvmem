// Independent scalar reference for the packed store, attention and KVMem Pressure.
#include "attention_kernel.hpp"
#include "rad_device.h"
#include "rad_plugin.h"
#include "platform/module.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
using namespace kvmem::radiance;
#define CHECK(x) do { if(!(x))throw std::runtime_error(#x); } while(false)
struct Buffer {
    void* p; size_t bytes;
    Buffer(size_t n):p(rad_dev_alloc(n,RAD_MEM_DEVICE)),bytes(n){CHECK(p);}
    ~Buffer(){rad_dev_free(p,RAD_MEM_DEVICE);}
    void put(const void* src,RadStream s){CHECK(rad_memcpy_async(p,src,bytes,s)>=0);CHECK(rad_stream_sync(s)>=0);}
    void get(void* dst,RadStream s){CHECK(rad_memcpy_async(dst,p,bytes,s)>=0);CHECK(rad_stream_sync(s)>=0);}
};
static RadTensor tensor(void* p,uint32_t dt,std::initializer_list<int64_t> dims) {
    RadTensor t{};t.data=p;t.dtype=dt;t.rank=dims.size();std::copy(dims.begin(),dims.end(),t.shape);rad_tensor_pack(&t);return t;
}
static void run(void* module,KvPacking fmt,unsigned ctx,unsigned nq,RadStream stream) {
    constexpr unsigned hd=256,kh=2,qh=12,bs=16;
    const bool k6=fmt==KvPacking::K6V4;
    const unsigned row=kv_row_elements(hd,fmt),nb=(ctx+bs-1)/bs;
    const char* name=k6?"k6v4":"fp8v4";
    auto symbol=[&](std::string n){auto* f=rad::platform::module_symbol(module,n.c_str());CHECK(f);return f;};
    auto store=reinterpret_cast<RadLaunchFn>(symbol(std::string("r4d_rad_kv_store_")+name));
    std::vector<uint16_t> k(ctx*kh*hd),v(k.size()),q(nq*qh*hd),out(q.size());
    for(size_t i=0;i<k.size();++i) {
        // Whole zero groups and positive/negative values spanning every packed high-bit position.
        k[i]=rad_f32_to_bf16((i/32)%13 ? float(int((i*17+3)%79)-39)*.031f : 0.f);
        v[i]=rad_f32_to_bf16((i/32)%11 ? float(int((i*7+11)%47)-23)*.043f : 0.f);
    }
    for(size_t i=0;i<q.size();++i)q[i]=rad_f32_to_bf16(float(int((i*3+5)%29)-14)*.027f);
    std::vector<int32_t> table(nb),slots(ctx);
    for(unsigned b=0;b<nb;++b)table[b]=nb-1-b;
    for(unsigned t=0;t<ctx;++t)slots[t]=table[t/bs]*bs+t%bs;
    std::vector<uint8_t> got(nb*kh*bs*row,0xA5),expected=got;
    std::vector<float> rk(k.size()),rv(k.size());
    // Reference packs one scalar at a time. No GPU shuffles/vector packing or production decoder.
    for(unsigned t=0;t<ctx;++t)for(unsigned h=0;h<kh;++h)for(unsigned g=0;g<hd/32;++g) {
        const size_t src=(t*kh+h)*hd+g*32;
        auto* dst=expected.data()+((table[t/bs]*kh+h)*bs+t%bs)*row;
        float km=0,vm=0;
        for(unsigned d=0;d<32;++d){km=std::max(km,std::abs(rad_bf16_to_f32(k[src+d])));vm=std::max(vm,std::abs(rad_bf16_to_f32(v[src+d])));}
        const uint16_t ks=rad_f32_to_f16(km/31),vs=rad_f32_to_f16(vm/7);
        const float kscale=rad_f16_to_f32(ks),vscale=rad_f16_to_f32(vs);
        if(k6)std::memcpy(dst+hd*5/4+g*2,&ks,2);
        std::memcpy(dst+(k6?hd*21/16:hd*3/2)+g*2,&vs,2);
        for(unsigned d=0;d<32;++d) {
            const unsigned c=g*32+d;
            if(k6) {
                const int value=kscale?int(std::nearbyint(rad_bf16_to_f32(k[src+d])/kscale)):0;
                const unsigned code=unsigned(std::clamp(value,-31,31))&63;
                const auto lo=c/2,hi=hd/2+(c/16)*4+(c%8)/2;
                const unsigned ls=4*(c%2),hs=(c%16>=8?4:0)+2*(c%2);
                dst[lo]=(dst[lo]&~(15u<<ls))|((code&15)<<ls);
                dst[hi]=(dst[hi]&~(3u<<hs))|((code>>4)<<hs);
                rk[src+d]=float(std::clamp(value,-31,31))*kscale;
            } else {
                dst[c]=rad_f32_to_fp8e4m3(rad_bf16_to_f32(k[src+d]));rk[src+d]=rad_fp8e4m3_to_f32(dst[c]);
            }
            const int vi=vscale?int(std::nearbyint(rad_bf16_to_f32(v[src+d])/vscale)):0;
            const unsigned vo=(k6?hd*3/4:hd)+c/2,shift=4*(c%2);
            dst[vo]=(dst[vo]&~(15u<<shift))|((unsigned(std::clamp(vi,-7,7))&15)<<shift);
            rv[src+d]=float(std::clamp(vi,-7,7))*vscale;
        }
    }
    Buffer kd(k.size()*2),vd(v.size()*2),qd(q.size()*2),cache(got.size()),sd(slots.size()*4),td(table.size()*4),used(4),od(out.size()*2);
    kd.put(k.data(),stream);vd.put(v.data(),stream);qd.put(q.data(),stream);cache.put(got.data(),stream);sd.put(slots.data(),stream);td.put(table.data(),stream);
    const int32_t count=ctx;used.put(&count,stream);
    RadParam params[]={RAD_INT("head_dim",hd),RAD_INT("n_head_kv",kh),RAD_INT("block_size",bs),RAD_INT("gqa",6),
        RAD_INT("n_head",qh),RAD_INT("max_seqs",1),RAD_INT("max_ctx",ctx),RAD_INT("q_len",nq),RAD_INT("causal",1),RAD_INT("window",0)};
    RadTensor ts[]={tensor(kd.p,RAD_BF16,{ctx,kh,hd}),tensor(vd.p,RAD_BF16,{ctx,kh,hd}),tensor(sd.p,RAD_I32,{ctx}),
        tensor(cache.p,RAD_F8E4M3,{nb,kh,bs,row})};
    RadArgs a{};a.t=ts;a.n_t=4;a.p=params;a.n_p=std::size(params);
    CHECK(store(&a,stream)==RAD_OK);cache.get(got.data(),stream);
    if(got!=expected){auto p=std::mismatch(got.begin(),got.end(),expected.begin()).first-got.begin();std::cerr<<name<<" byte "<<p<<" got "<<int(got[p])<<" expected "<<int(expected[p])<<'\n';CHECK(got==expected);}
    RadTensor at[]={tensor(qd.p,RAD_BF16,{nq,qh,hd}),ts[3],tensor(td.p,RAD_I32,{1,nb}),tensor(used.p,RAD_I32,{1}),{},{},{},tensor(od.p,RAD_BF16,{nq,qh,hd})};
    a.t=at;a.n_t=8;
    std::unique_ptr<Buffer> scratch;
    if(nq<=10) {
        auto size=reinterpret_cast<RadScratchFn>(symbol(std::string("r4d_rad_attn_decode_h256_")+name+"_scratch"));
        auto n=size(&a);CHECK(n>0);scratch=std::make_unique<Buffer>(n);a.scratch=scratch->p;a.scratch_bytes=n;
    }
    auto attn=reinterpret_cast<RadLaunchFn>(symbol(std::string(nq<=10?"r4d_rad_attn_decode_h256_":"r4d_rad_attn_prefill_h256_")+name));
    CHECK(attn(&a,stream)==RAD_OK);od.get(out.data(),stream);
    // FP64 softmax over independently dequantized keys/values, including causal tails.
    double max_error=0,rms=0,norm=0;
    for(unsigned r=0;r<nq;++r)for(unsigned h=0;h<qh;++h) {
        const unsigned end=ctx-nq+r+1;std::vector<double> weights(end);double maxlog=-1e300;
        for(unsigned t=0;t<end;++t) {
            double sum=0;for(unsigned d=0;d<hd;++d)sum+=double(rad_bf16_to_f32(q[(r*qh+h)*hd+d]))*rk[(t*kh+h/6)*hd+d];
            weights[t]=sum/16;maxlog=std::max(maxlog,weights[t]);
        }
        double denom=0;for(auto& w:weights){w=std::exp(w-maxlog);denom+=w;}
        for(unsigned d=0;d<hd;++d) {
            double want=0;for(unsigned t=0;t<end;++t)want+=weights[t]*rv[(t*kh+h/6)*hd+d]/denom;
            const double err=rad_bf16_to_f32(out[(r*qh+h)*hd+d])-want;
            CHECK(std::isfinite(err));max_error=std::max(max_error,std::abs(err));rms+=err*err;norm+=want*want;
        }
    }
    CHECK(max_error<.006 && std::sqrt(rms/std::max(norm,1e-20))<.015);
    // KVMem page masses must read the same stored K, with sparse original positions.
    std::vector<AttentionPage> pages;for(unsigned b=0;b<nb;++b)pages.push_back({b*bs,std::min(bs,ctx-b*bs),b*bs});
    const uint32_t sample=0;Buffer pd(pages.size()*sizeof(AttentionPage)),rd(4),lse(qh*nb*4),masses(nb*4);
    pd.put(pages.data(),stream);rd.put(&sample,stream);
    AttentionKernelArgs ka{at[0],at[1],static_cast<const int32_t*>(td.p),static_cast<const uint32_t*>(rd.p),static_cast<const AttentionPage*>(pd.p),
        1,nb,qh,hd,ctx-nq,static_cast<float*>(lse.p),static_cast<float*>(masses.p),fmt};
    CHECK(attention_page_masses(ka,stream)==RAD_OK);std::vector<float> actual(nb);masses.get(actual.data(),stream);
    std::vector<double> page_ref(nb);const unsigned end=ctx-nq+1;
    for(unsigned h=0;h<qh;++h) {
        std::vector<double> w(end);double den=0;
        for(unsigned t=0;t<end;++t){double dot=0;for(unsigned d=0;d<hd;++d)dot+=double(rad_bf16_to_f32(q[h*hd+d]))*rk[(t*kh+h/6)*hd+d];w[t]=std::exp(dot/16);den+=w[t];}
        for(unsigned t=0;t<end;++t)page_ref[t/bs]+=w[t]/den/qh;
    }
    for(unsigned b=0;b<nb;++b)CHECK(std::abs(actual[b]-page_ref[b])<2e-5);
    ka.cache.shape[3]++;CHECK(attention_page_masses(ka,stream)==RAD_E_INVAL);
    std::cout<<name<<" ctx="<<ctx<<" q="<<nq<<" store byte-exact, attention max="<<max_error<<" relative="<<std::sqrt(rms/norm)<<", Pressure reference passed\n";
}
static void fused_store(void* module,KvPacking fmt,unsigned m,RadStream stream) {
    constexpr unsigned hd=256,qh=24,kh=4,bs=16,nb=5,rot=64;
    const bool k6=fmt==KvPacking::K6V4;const char* suffix=k6?"k6v4":"fp8v4";
    auto launch=[&](std::string name){auto f=reinterpret_cast<RadLaunchFn>(rad::platform::module_symbol(module,name.c_str()));CHECK(f);return f;};
    std::vector<uint16_t> qg(m*qh*hd*2),k(m*kh*hd),v(k.size());
    for(size_t i=0;i<qg.size();++i)qg[i]=rad_f32_to_bf16(float(int(i%47)-23)*.031f);
    for(size_t i=0;i<k.size();++i){k[i]=rad_f32_to_bf16(float(int(i%37)-18)*.053f);v[i]=rad_f32_to_bf16(float(int(i%31)-15)*.041f);}
    std::vector<float> w(hd),cs(m*rot);std::vector<int32_t> pos(m),slots(m);
    for(unsigned d=0;d<hd;++d)w[d]=float(int(d%13)-6)*.003f;
    for(unsigned t=0;t<m;++t){pos[t]=t;slots[t]=(t/bs)*bs+(bs-1-t%bs);for(unsigned d=0;d<rot/2;++d){float angle=float(t)*std::pow(10000.f,-2.f*d/rot);cs[t*rot+d]=std::cos(angle);cs[t*rot+rot/2+d]=std::sin(angle);}}
    // Negative slots still produce Q/K, but never change a cache byte.
    slots[m-1]=-1;
    Buffer qgd(qg.size()*2),kd(k.size()*2),vd(v.size()*2),wd(w.size()*4),cd(cs.size()*4),pd(m*4),sd(m*4);
    qgd.put(qg.data(),stream);kd.put(k.data(),stream);vd.put(v.data(),stream);wd.put(w.data(),stream);cd.put(cs.data(),stream);pd.put(pos.data(),stream);sd.put(slots.data(),stream);
    Buffer qo(m*qh*hd*2),ko(k.size()*2),qof(qo.bytes),kof(ko.bytes),cache(nb*kh*bs*kv_row_elements(hd,fmt)),fused(cache.bytes);
    std::vector<uint8_t> initial(cache.bytes,0xA5);cache.put(initial.data(),stream);fused.put(initial.data(),stream);
    RadTensor t[]={tensor(qgd.p,RAD_BF16,{m,2*qh*hd}),tensor(kd.p,RAD_BF16,{m,kh*hd}),tensor(cd.p,RAD_F32,{m,rot}),tensor(pd.p,RAD_I32,{m}),
        tensor(wd.p,RAD_F32,{hd}),tensor(wd.p,RAD_F32,{hd}),tensor(qo.p,RAD_BF16,{m,qh*hd}),tensor(ko.p,RAD_BF16,{m,kh*hd}),{},
        tensor(vd.p,RAD_BF16,{m,kh*hd}),tensor(sd.p,RAD_I32,{m}),tensor(fused.p,RAD_F8E4M3,{nb,kh,bs,kv_row_elements(hd,fmt)})};
    RadParam p[]={RAD_INT("head_dim",hd),RAD_INT("n_head",qh),RAD_INT("n_head_kv",kh),RAD_INT("rot",rot),RAD_INT("block_size",bs),RAD_F64("eps",1e-6),RAD_F64("wadd",1)};
    RadArgs a{};a.t=t;a.n_t=9;a.p=p;a.n_p=std::size(p);
    CHECK(launch("r4d_rad_qk_norm_rope_gate")(&a,stream)==RAD_OK);
    RadTensor st[]={t[7],t[9],t[10],t[11]};st[3].data=cache.p;RadArgs sa=a;sa.t=st;sa.n_t=4;
    CHECK(launch(std::string("r4d_rad_kv_store_")+suffix)(&sa,stream)==RAD_OK);
    t[6].data=qof.p;t[7].data=kof.p;a.n_t=12;
    CHECK(launch(std::string("r4d_rad_qk_norm_rope_gate_kv_store_")+suffix)(&a,stream)==RAD_OK);
    for(auto pair:{std::pair<Buffer*,Buffer*>{&qo,&qof},{&ko,&kof},{&cache,&fused}}) {
        std::vector<uint8_t> x(pair.first->bytes),y(x.size());pair.first->get(x.data(),stream);pair.second->get(y.data(),stream);CHECK(x==y);
    }
    std::cout<<suffix<<" fused Q/K/cache byte identity, M="<<m<<" passed\n";
}
int main(int argc,char** argv) {
    try {
        CHECK(argc==2);CHECK(rad_dev_set(0)==RAD_OK);RadStream stream=nullptr;CHECK(rad_stream_create(&stream,0)==RAD_OK);
        std::string error;auto* module=rad::platform::module_open(argv[1],&error);if(!module)std::cerr<<error<<'\n';CHECK(module);
        for(auto fmt:{KvPacking::Fp8V4,KvPacking::K6V4}) {
            for(auto shape:{std::pair<unsigned,unsigned>{17,1},{137,4},{137,10},{137,11},{137,65},{4097,1}})run(module,fmt,shape.first,shape.second,stream);
            for(unsigned rows:{1u,5u,64u})fused_store(module,fmt,rows,stream);
        }
        rad_stream_destroy(stream);rad::platform::module_close(module);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
