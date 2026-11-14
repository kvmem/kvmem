#include "llama-memory-kvmem.h"
#include "llama-kvmem-capture.h"
#include "llama-kvmem-diag.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef KVMEM_PAGE_SOFTMAX_CUDA
#include "llama-kvmem-attention-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <stdexcept>

namespace {
ggml_tensor* borrowed(ggml_context* ctx,ggml_tensor* source) {
    auto* leaf=ggml_new_tensor(ctx,source->type,4,source->ne);
    std::memcpy(leaf->nb,source->nb,sizeof(leaf->nb));
    leaf->data=source->data; leaf->buffer=source->buffer;
    return leaf;
}
} // namespace

struct llama_memory_kvmem::AttentionSampler {
    struct Device {
        ggml_backend_t backend=nullptr;
        ggml_gallocr_t allocator=nullptr;
        std::vector<uint8_t> metadata;
#ifdef KVMEM_PAGE_SOFTMAX_CUDA
        llama_kvmem_attention_cuda cuda;
#endif
        explicit Device(ggml_backend_dev_t device) {
            backend=ggml_backend_dev_init(device,nullptr);
            if(!backend) throw std::runtime_error("KVMem cannot initialize attention sampling backend");
            allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            metadata.resize(ggml_tensor_overhead()*1024+ggml_graph_overhead_custom(2048,false));
        }
        ~Device() { if(allocator) ggml_gallocr_free(allocator); if(backend) ggml_backend_free(backend); }
    };
    std::map<ggml_backend_dev_t,std::unique_ptr<Device>> devices;
    void qualify(const AttentionCapture& capture,const std::vector<uint32_t>& rows,uint32_t block,
        const std::vector<std::vector<float>>& actual,double tolerance=2e-3) {
        const auto* enabled=std::getenv("KVMEM_ATTENTION_VERIFY");
        if(!enabled || enabled[0]!='1') return;
        if(capture.max_bias!=0) throw std::runtime_error("attention oracle requires zero ALiBi bias");
        auto* q=capture.query;auto* k=capture.key;auto* mask=capture.mask;
        std::vector<uint8_t> query_bytes(ggml_nbytes(q)),key_bytes(ggml_nbytes(k));
        ggml_backend_tensor_get(q,query_bytes.data(),0,query_bytes.size());
        ggml_backend_tensor_get(k,key_bytes.data(),0,key_bytes.size());
        auto decode=[](ggml_type type,const uint8_t* data,float* output,int64_t count) {
            if(type==GGML_TYPE_F32) std::memcpy(output,data,size_t(count)*sizeof(float));
            else ggml_get_type_traits(type)->to_float(data,output,count);
        };
        const auto dim=size_t(q->ne[0]),heads=size_t(q->ne[2]),tokens=size_t(k->ne[1]);
        std::vector<float> query(dim),key(dim),causal(tokens);
        std::vector<uint8_t> mask_bytes(ggml_row_size(mask->type,mask->ne[0]));
        double worst=0;
        for(size_t sample=0;sample<rows.size();++sample) {
            ggml_backend_tensor_get(mask,mask_bytes.data(),size_t(rows[sample])*mask->nb[1],mask_bytes.size());
            std::vector<float> mask_row(size_t(mask->ne[0]));
            decode(mask->type,mask_bytes.data(),mask_row.data(),mask->ne[0]);
            std::vector<double> expected(actual[sample].size());
            for(size_t head=0;head<heads;++head) {
                decode(q->type,query_bytes.data()+size_t(rows[sample])*q->nb[1]+head*q->nb[2],query.data(),dim);
                std::vector<double> logits(tokens);
                double maximum=-INFINITY;
                for(size_t token=0;token<tokens;++token) {
                    if(!std::isfinite(mask_row[token])) {logits[token]=-INFINITY;continue;}
                    decode(k->type,key_bytes.data()+token*k->nb[1]+head/(heads/size_t(k->ne[2]))*k->nb[2],key.data(),dim);
                    double dot=0;
                    for(size_t d=0;d<dim;++d) dot+=double(query[d])*key[d];
                    dot*=capture.scale;
                    if(capture.softcap>0) dot=capture.softcap*std::tanh(dot/capture.softcap);
                    logits[token]=dot+mask_row[token];maximum=std::max(maximum,logits[token]);
                }
                double denominator=0;
                for(auto& value:logits) denominator+=value=std::exp(value-maximum);
                for(size_t token=0;token<tokens;++token) expected[token/block]+=logits[token]/denominator/heads;
            }
            for(size_t page=0;page<expected.size();++page) {
                if(!std::isfinite(expected[page]) || !std::isfinite(actual[sample][page])) {
                    kvmem_diag("KVMEM_ATTENTION_ORACLE_INVALID layer=%d row=%u page=%zu expected=%g actual=%g q=%s k=%s\n",
                        capture.layer,rows[sample],page,expected[page],actual[sample][page],
                        ggml_type_name(q->type),ggml_type_name(k->type));
                    throw std::runtime_error("attention oracle found a non-finite probability");
                }
                worst=std::max(worst,std::abs(expected[page]-actual[sample][page]));
            }
        }
        kvmem_diag("KVMEM_ATTENTION_ORACLE layer=%d rows=%zu max_abs=%.8g\n",capture.layer,rows.size(),worst);
        // Quantized ggml matmul may quantize its transient Q. Qualify that public
        // route against dense represented Q/K rather than another GPU kernel.
        if(worst>tolerance) throw std::runtime_error("attention sampler differs from dense FP64 oracle");
    }
    std::vector<std::vector<float>> run(const AttentionCapture& capture,
                                        const std::vector<uint32_t>& rows,uint32_t block_tokens) {
        if(!capture.query->buffer || !capture.key->buffer || !capture.mask ||
           capture.query->ne[3]!=1 || capture.key->ne[3]!=1 || capture.mask->ne[2]!=1 ||
           capture.mask->ne[3]!=1 || capture.key->ne[1]%block_tokens)
            throw std::runtime_error("KVMem attention sampling requires aligned single-sequence causal KV");
        const auto blocks=size_t(capture.key->ne[1])/block_tokens;
        const auto heads=size_t(capture.query->ne[2]);
        const size_t batch=std::max<size_t>(1,std::min<size_t>(64,(32u<<20)/
            std::max<size_t>(1,size_t(capture.key->ne[1])*heads*sizeof(float))));
        std::vector<std::vector<float>> result(rows.size(),std::vector<float>(blocks));
        // A partially offloaded model can capture Q on CUDA while its persistent
        // K cache remains on the CPU. Compute beside K and stage only the bounded
        // current-forward Q, rather than borrowing an inaccessible buffer or
        // transferring the full history. observe_attention synchronized the
        // producing scheduler before entering this sampler.
        const auto key_type=ggml_backend_buffer_get_type(capture.key->buffer);
        const auto query_type=ggml_backend_buffer_get_type(capture.query->buffer);
        // Plain CPU allocations need not carry a registered device pointer.
        const auto device=ggml_backend_buffer_is_host(capture.key->buffer) ?
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) : ggml_backend_buft_get_device(key_type);
        if(!device) throw std::runtime_error("KVMem attention K cache has no available backend");
        auto& owner=devices[device];
        if(!owner) owner=std::make_unique<Device>(device);
        if(!ggml_backend_supports_buft(owner->backend,key_type))
            throw std::runtime_error("KVMem attention backend cannot access its K cache");
#ifdef KVMEM_PAGE_SOFTMAX_CUDA
        if(capture.max_bias==0 && owner->cuda.score(capture.query,capture.key,capture.mask,
            capture.scale,capture.softcap,block_tokens,rows,result)) {
            qualify(capture,rows,block_tokens,result,2e-6);
            return result;
        }
#endif
        const bool stage_query=!ggml_backend_supports_buft(owner->backend,query_type);
        if(stage_query) kvmem_diag("KVMEM_ATTENTION_STAGE layer=%d bytes=%zu source=%s destination=%s\n",
            capture.layer,ggml_nbytes(capture.query),ggml_backend_buft_name(query_type),
            ggml_backend_name(owner->backend));
        for(size_t first=0;first<rows.size();first+=batch) {
            const auto count=std::min(batch,rows.size()-first);
            ggml_init_params params{owner->metadata.size(),owner->metadata.data(),true};
            ggml_context_ptr ctx{ggml_init(params)};
            if(!ctx) throw std::bad_alloc();
            auto* graph=ggml_new_graph_custom(ctx.get(),2048,false);
            auto* q=stage_query ? ggml_dup_tensor(ctx.get(),capture.query) : borrowed(ctx.get(),capture.query);
            if(stage_query) {
                std::memcpy(q->nb,capture.query->nb,sizeof(q->nb));
                ggml_set_input(q);
            }
            auto* k=borrowed(ctx.get(),capture.key);
            auto* mask=ggml_new_tensor_2d(ctx.get(),capture.mask->type,k->ne[1],int64_t(count));
            ggml_set_input(mask);
            const auto mask_row=ggml_row_size(mask->type,mask->ne[0]);
            std::vector<uint8_t> mask_host(mask_row*count);
            ggml_tensor* probabilities=nullptr;
            for(size_t i=0;i<count;++i) {
                const auto row=rows[first+i];
                if(row>=q->ne[1] || row>=capture.mask->ne[1] || capture.mask->ne[0]<k->ne[1])
                    throw std::runtime_error("KVMem attention sample is outside its captured forward");
                ggml_backend_tensor_get(capture.mask,mask_host.data()+i*mask_row,size_t(row)*capture.mask->nb[1],mask_row);
                auto* query=ggml_view_4d(ctx.get(),q,q->ne[0],1,q->ne[2],1,q->nb[1],q->nb[2],q->nb[3],size_t(row)*q->nb[1]);
                auto* causal=ggml_view_2d(ctx.get(),mask,mask->ne[0],1,mask->nb[1],i*mask->nb[1]);
                auto* logits=ggml_mul_mat(ctx.get(),k,query);
                ggml_prec_set_acc(logits,GGML_PREC_F32);
                float scale=capture.scale;
                if(capture.softcap>0) {
                    logits=ggml_scale(ctx.get(),ggml_tanh(ctx.get(),ggml_scale(ctx.get(),logits,
                        scale/capture.softcap)),capture.softcap);
                    scale=1.f;
                }
                auto* probability=ggml_soft_max_ext(ctx.get(),logits,causal,scale,capture.max_bias);
                probabilities=probabilities ? ggml_concat(ctx.get(),probabilities,probability,1) : probability;
            }
            auto* mass=ggml_sum_rows(ctx.get(),ggml_reshape_3d(ctx.get(),ggml_cont(ctx.get(),probabilities),
                block_tokens,int64_t(blocks),int64_t(count*heads)));
            mass=ggml_cont_3d(ctx.get(),mass,int64_t(blocks),int64_t(count),int64_t(heads));
            mass=ggml_cont(ctx.get(),ggml_permute(ctx.get(),mass,1,2,0,3));
            mass=ggml_scale(ctx.get(),ggml_sum_rows(ctx.get(),mass),1.f/float(heads));
            mass=ggml_cont_2d(ctx.get(),mass,int64_t(blocks),int64_t(count));
            ggml_set_output(mass); ggml_build_forward_expand(graph,mass);
            if(!ggml_gallocr_alloc_graph(owner->allocator,graph)) throw std::bad_alloc();
            if(stage_query) ggml_backend_tensor_copy(capture.query,q);
            ggml_backend_tensor_set(mask,mask_host.data(),0,mask_host.size());
            if(ggml_backend_graph_compute(owner->backend,graph)!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("KVMem attention sampler execution failed");
            for(size_t i=0;i<count;++i)
                ggml_backend_tensor_get(mass,result[first+i].data(),i*blocks*sizeof(float),blocks*sizeof(float));
        }
        qualify(capture,rows,block_tokens,result);
        return result;
    }
};

void llama_memory_kvmem::capture_attention(ggml_context* ctx,ggml_cgraph* graph,
    ggml_tensor* q,ggml_tensor* k,ggml_tensor* mask,float scale,float max_bias,float softcap,int layer) {
    if(layer<0 || uint32_t(layer)>=n_layer_) return; // target layers only; native MTP is the suffix
    const auto& selected=sampled_layers_;
    if(!std::binary_search(selected.begin(),selected.end(),layer)) return;
    auto* captured=ggml_cont(ctx,q);
    // FA may cast an F32 cache into a temporary F16 K. Keep that operand alive
    // through harvest just like sampled Q; persistent cache views need no copy.
    ggml_set_output(k);
    // A cast mask is also a graph temporary. Retain it until harvest; partial
    // offload can otherwise reuse its storage in a later backend split.
    ggml_set_output(mask);
    ggml_set_output(captured); ggml_build_forward_expand(graph,captured);
    attention_captures_.push_back({captured,k,mask,scale,max_bias,softcap,layer});
}

void llama_memory_kvmem::prepare_attention_batches(const std::vector<llama_ubatch>& batches) {
    attention_batches_.clear(); attention_pending_batch_=UINT32_MAX;
    for(const auto& ub:batches) {
        const auto* positions=ub.logical_pos?ub.logical_pos:ub.pos;
        AttentionBatch batch;
        batch.begin=uint64_t(positions[0]); batch.end=batch.begin+ub.n_tokens;
        batch.epoch=attention_epoch_; batch.replay=replay_;
        batch.slots.assign(n_slots_,UINT64_MAX);
        for(const auto& block:store().blocks()) if(block.gpu_slot>=0)
            batch.slots[size_t(block.gpu_slot)]=block.orig_pos_start;
        batch.tokens.reserve(ub.n_tokens);
        for(uint32_t i=0;i<ub.n_tokens;++i) batch.tokens.push_back(ub.token?ub.token[i]:LLAMA_TOKEN_NULL);
        attention_batches_.push_back(std::move(batch));
    }
}

void llama_memory_kvmem::attention_begin(const llama_token* prompt,uint32_t count) {
    graph_.reserve_extra(feedback_.snapshot().bytes());
    feedback_.finish_output();
    graph_.include_feedback(feedback_.snapshot().bytes());
    const auto marker=[&](const char* text) {
        llama_token tokens[8];
        const int n=llama_tokenize(llama_model_get_vocab(&model_),text,int(std::strlen(text)),tokens,8,false,true);
        return n==1 ? int32_t(tokens[0]) : -1;
    };
    attention_think_open_=marker("<think>"); attention_think_close_=marker("</think>");
    attention_scope_={};
    for(uint32_t i=0;i<count;++i) {
        if(prompt[i]==attention_think_open_ && attention_think_open_>=0) attention_scope_.thinking=true;
        if(prompt[i]==attention_think_close_ && attention_think_close_>=0) attention_scope_.thinking=false;
    }
    attention_output_begin_=count;
}

void llama_memory_kvmem::attention_finish(bool commit) {
    graph_.reserve_extra(feedback_.snapshot().bytes());
    if(commit) feedback_.finish_output();
    else {
        auto state=feedback_.snapshot(); state.pending_output.clear(); state.output_samples=0;
        feedback_.restore(std::move(state));
    }
    graph_.include_feedback(feedback_.snapshot().bytes());
    attention_output_begin_=UINT64_MAX;
}

void llama_memory_kvmem::observe_attention(uint32_t index,uint32_t accepted,ggml_backend_sched* sched) {
    if(index>=attention_batches_.size() || !accepted) return;
    const auto& batch=attention_batches_[index];
    if(batch.epoch!=attention_epoch_ || accepted>batch.tokens.size())
        throw std::runtime_error("KVMem rejected stale accepted attention metadata");
    const auto end=batch.begin+accepted;
    if(end<=graph_.accepted_through) return;
    graph_.set_messages(request_graph_messages_);
    std::vector<uint32_t> rows;
    auto next_scope=attention_scope_;
    const auto output_first=attention_output_begin_<=batch.begin ? 0u :
        uint32_t(std::min<uint64_t>(accepted,attention_output_begin_-batch.begin));
    const bool output=!prefill_capture_ && output_first<accepted;
    if(!batch.replay) {
        if(output) {
            auto plan=kvmem::accepted_output_samples(next_scope,batch.tokens.data()+output_first,accepted-output_first,
                attention_think_open_,attention_think_close_);
            rows=std::move(plan.rows); next_scope=plan.state;
            for(auto& row:rows) row+=output_first;
        } else if(accepted==batch.tokens.size()) {
            rows=kvmem::prefill_sample_rows(batch.begin,accepted,feedback_.snapshot().prefill_end,
                resident_tokens(),kv_size_,1024,32,block_tokens_);
        }
    }
    rows.erase(std::remove_if(rows.begin(),rows.end(),[&](auto row) {
        return batch.begin+row<graph_.accepted_through;
    }),rows.end());
    std::vector<kvmem::AcceptedAttentionRow> observations(rows.size());
    for(size_t i=0;i<rows.size();++i) observations[i].source_position=batch.begin+rows[i];
    if(!rows.empty()) {
        if(sched) ggml_backend_sched_synchronize(sched);
        if(attention_captures_.size()!=sampled_layers_.size())
            throw std::runtime_error("KVMem missed a selected attention layer");
        if(!attention_sampler_) attention_sampler_=std::make_shared<AttentionSampler>();
        for(const auto& capture:attention_captures_) {
            const auto mass=attention_sampler_->run(capture,rows,block_tokens_);
            for(size_t i=0;i<mass.size();++i) for(size_t slot=0;slot<mass[i].size();++slot) {
                const auto value=double(mass[i][slot])/attention_captures_.size();
                if(!std::isfinite(value) || value<0) throw std::runtime_error("invalid sampled attention probability");
                if(slot>=batch.slots.size() || batch.slots[slot]==UINT64_MAX) {
                    if(value>1e-7) throw std::runtime_error("attention sampled an unmapped native slot");
                } else if(value>0) observations[i].mass.push_back({batch.slots[slot],value});
            }
        }
    }
    // Unsampled accepts only advance the graph watermark and parser state.
    // They must not copy the complete pressure history on every decode step.
    if(observations.empty()) {
        graph_=kvmem::prepare_graph_accept(graph_,graph_.accepted_through,end,{},32u<<20);
        attention_scope_=next_scope;
        return;
    }
    auto graph=kvmem::prepare_graph_accept(graph_,graph_.accepted_through,end,observations,32u<<20);
    size_t extra=feedback_.snapshot().bytes();
    for(const auto& row:observations) extra+=row.mass.size()*sizeof(std::pair<uint64_t,double>)*3;
    graph.reserve_extra(extra);
    auto feedback=feedback_;
    if(output) for(const auto& sample:observations) feedback.observe_output(sample.mass);
    else if(!observations.empty()) {
        std::vector<uint64_t> exposed;
        for(auto start:batch.slots) if(start!=UINT64_MAX) exposed.push_back(start);
        std::sort(exposed.begin(),exposed.end());
        feedback.observe_prefill(batch.begin,end,accepted,observations,exposed);
    }
    graph.include_feedback(feedback.snapshot().bytes());
    graph_=std::move(graph); feedback_=std::move(feedback); attention_scope_=next_scope;
    kvmem_diag("KVMEM_ATTENTION phase=%s begin=%llu end=%llu samples=%zu\n",output?"output":"prefill",
        (unsigned long long)batch.begin,(unsigned long long)end,rows.size());
}

void llama_kvmem_capture_attention(ggml_context* ctx,ggml_cgraph* graph,ggml_tensor* q,
    ggml_tensor* k,ggml_tensor* mask,float scale,float max_bias,float softcap,int layer) {
    if(auto* memory=kvmem_capture_active()) memory->capture_attention(ctx,graph,q,k,mask,scale,max_bias,softcap,layer);
}
void llama_kvmem_attention_begin(const llama_token* prompt,uint32_t count) {
    if(auto* memory=kvmem_capture_active()) memory->attention_begin(prompt,count);
}
void llama_kvmem_attention_finish(bool commit) {
    if(auto* memory=kvmem_capture_active()) memory->attention_finish(commit);
}
void llama_kvmem_set_graph_messages(const std::vector<kvmem::GraphMessageSpan>& spans) {
    if(auto* memory=kvmem_capture_active()) memory->graph_messages(spans);
}
