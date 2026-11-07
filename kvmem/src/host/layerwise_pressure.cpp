#include "kvmem/layerwise_pressure.hpp"
#include "kvmem/graph_recall.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace kvmem {
namespace {
[[noreturn]] void fail(MemoryErrorCode c, const char* text) { throw MemoryError(c,text); }
uint64_t add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX-a) fail(MemoryErrorCode::InvalidPlan,"prefill coordinate/counter overflow");
    return a+b;
}
// Match upstream native commit's Python multiply/add boundary; no FMA.
double multiply(double a, double b) { volatile double rounded = a*b; return rounded; }
struct Sum {
    double hi=0, lo=0;
    void add(double x) {
        const double next=hi+x;
        lo += std::abs(hi)>=std::abs(x) ? (hi-next)+x : (x-next)+hi;
        hi=next;
    }
    double value() const { return lo!=0 && std::isfinite(lo) ? hi+lo : hi; }
};
void finite(double x) {
    if (!std::isfinite(x) || x<0) fail(MemoryErrorCode::InvalidPlan,"invalid native prefill attention mass");
}
const LayerwisePrefillValue* find(const LayerwisePrefillState& state, uint64_t start) {
    const auto i=std::lower_bound(state.values.begin(),state.values.end(),start,
        [](const auto& x,uint64_t k){return x.original_start<k;});
    return i!=state.values.end() && i->original_start==start ? &*i : nullptr;
}
}

LayerwisePrefillPressure::LayerwisePrefillPressure(SessionIdentity session,
    std::vector<uint32_t> layers, LayerwisePrefillOptions options)
    :session_(session),layers_(std::move(layers)),options_(options) {
    if (!session.id || !session.incarnation || layers_.empty() ||
        !std::is_sorted(layers_.begin(),layers_.end()) ||
        std::adjacent_find(layers_.begin(),layers_.end())!=layers_.end() ||
        !options.block_tokens || !options.select_tokens || options.select_tokens%options.block_tokens || !options.interval ||
        !std::isfinite(options.half_life) || options.half_life<=0)
        fail(MemoryErrorCode::InvalidPlan,"invalid per-layer prefill policy");
    add(options.select_tokens,options.reserve_tokens);
    state_.session=session;
    for (auto layer:layers_) state_.layers.push_back({layer,{},0,0,0});
}

std::vector<uint32_t> LayerwisePrefillPressure::rows(uint64_t begin,uint32_t count,
    uint64_t local,bool replay) const {
    const auto end=add(begin,count);
    const auto previous=state_.layers.front().end;
    if (replay || end<=previous || !count) return {};
    const auto capacity=add(options_.select_tokens,options_.reserve_tokens);
    const auto used=add(local,count), threshold=add(count,options_.block_tokens);
    // Exact signed Python inequality, without an unsigned subtraction wrap.
    const bool near=used>=capacity || capacity-used<=threshold;
    if (previous && end-previous<options_.interval && !near) return {};
    const uint32_t stride=options_.stride ? options_.stride : count;
    std::vector<uint32_t> result;
    for (uint32_t row=count-1;;) {
        result.push_back(row);
        if (row<stride) break;
        row-=stride;
    }
    std::reverse(result.begin(),result.end());
    return result;
}

bool LayerwisePrefillPressure::accept(const LayerwisePrefillAcceptance& packet) {
    const auto end=packet.history.frontier;
    if (!(packet.history.session==session_)) fail(MemoryErrorCode::Stale,"foreign prefill owner");
    if (!packet.history.valid || !packet.history.history_revision || !packet.history.execution_history ||
        end<packet.count || packet.count>UINT32_MAX || packet.accepted>packet.count)
        fail(MemoryErrorCode::InvalidPlan,"invalid accepted prefill boundary");
    if (packet.accepted!=packet.count || end<=state_.layers.front().end || packet.rows.empty()) return false;
    if (packet.rows!=rows(end-packet.count,uint32_t(packet.count),packet.local_begin))
        fail(MemoryErrorCode::InvalidPlan,"prefill rows differ from the admitted native sample schedule");
    if (packet.layers.size()!=layers_.size())
        fail(MemoryErrorCode::InvalidPlan,"missing independent target-layer prefill mass");
    auto next=state_;
    std::set<uint32_t> seen;
    for (const auto& input:packet.layers) {
        const auto where=std::lower_bound(layers_.begin(),layers_.end(),input.model_layer);
        if (where==layers_.end() || *where!=input.model_layer || !seen.insert(input.model_layer).second)
            fail(MemoryErrorCode::InvalidPlan,"foreign/duplicate target prefill layer");
        const size_t ordinal=where-layers_.begin();
        const auto& old=state_.layers[ordinal];
        auto& target=next.layers[ordinal];
        if (input.history.size()>SIZE_MAX/packet.rows.size() ||
            input.masses.size()!=input.history.size()*packet.rows.size())
            fail(MemoryErrorCode::InvalidPlan,"missing native prefill matrix rows/columns");
        const double decay=std::exp2(-double(end-old.end)/options_.half_life);
        std::map<uint64_t,LayerwisePrefillValue> updated;
        for (auto key:input.history) {
            if (key%options_.block_tokens) fail(MemoryErrorCode::InvalidPlan,"unaligned prefill logical key");
            const auto* prior=find(old,key);
            updated.emplace(key,LayerwisePrefillValue{key,multiply(decay,prior?prior->mass:0),
                                                        multiply(decay,prior?prior->exposure:0)});
        }
        const double inv=1./packet.rows.size();
        for (size_t row=0;row<packet.rows.size();++row) {
            Sum sum;
            for (size_t p=0;p<input.history.size();++p) {
                const auto value=input.masses[row*input.history.size()+p]; finite(value); sum.add(value);
            }
            if (std::abs(sum.value()-1.)>.005)
                fail(MemoryErrorCode::InvalidPlan,"invalid native prefill attention mass sum");
            const auto position=add(packet.local_begin,packet.rows[row]);
            std::set<uint64_t> exposed;
            // Preserve upstream's starts/history zip boundary. Valid native
            // segment packets have one start per history entry.
            for (size_t p=0;p<std::min(input.starts.size(),input.history.size());++p) {
                if (input.starts[p]>position) continue;
                auto& value=updated.at(input.history[p]);
                value.mass+=multiply(inv,input.masses[row*input.history.size()+p]);
                if (exposed.insert(input.history[p]).second) value.exposure+=inv;
            }
        }
        target.values.clear();
        for (const auto& value:updated) target.values.push_back(value.second);
        target.end=end; target.samples=add(target.samples,packet.rows.size()); target.chunks=add(target.chunks,1);
    }
    state_=std::move(next);
    return true;
}

void LayerwisePrefillPressure::restore(LayerwisePrefillSnapshot state,uint64_t position) {
    if (!(state.session==session_)) fail(MemoryErrorCode::Stale,"foreign prefill snapshot");
    if (state.layers.size()!=layers_.size()) fail(MemoryErrorCode::InvalidPlan,"incomplete prefill snapshot");
    for (size_t i=0;i<layers_.size();++i) {
        const auto& layer=state.layers[i];
        if (layer.model_layer!=layers_[i] || layer.end!=state.layers[0].end ||
            layer.samples!=state.layers[0].samples || layer.chunks!=state.layers[0].chunks)
            fail(MemoryErrorCode::InvalidPlan,"inconsistent accepted prefill sampling boundary");
        uint64_t prior=0; bool first=true;
        for (const auto& value:layer.values) {
            finite(value.mass); finite(value.exposure);
            if (value.original_start%options_.block_tokens || (!first && value.original_start<=prior))
                fail(MemoryErrorCode::InvalidPlan,"invalid prefill snapshot logical keys");
            first=false; prior=value.original_start;
        }
    }
    if (state.layers[0].end>position)
        for (auto& layer:state.layers) { layer.values.clear(); layer.end=layer.samples=layer.chunks=0; }
    state_=std::move(state);
}

namespace {
std::vector<LayerwisePressureChoice> select_seeds(const LayerwiseMemoryCatalog& catalog,
    const SessionIdentity& session_,const std::vector<uint32_t>& layers_,
    const LayerwisePrefillOptions& options_,const LayerwisePrefillSnapshot& state_,
    const std::vector<uint64_t>& mandatory,uint64_t budget,uint32_t sinks) {
    if (catalog.descriptor().layout.block_tokens!=options_.block_tokens ||
        catalog.layers().size()!=layers_.size() || !budget || budget%options_.block_tokens ||
        budget>options_.select_tokens || budget/options_.block_tokens>UINT32_MAX ||
        sinks>budget/options_.block_tokens)
        fail(MemoryErrorCode::InvalidPlan,"invalid per-layer resident Pressure budget/catalog");
    if (std::set<uint64_t>(mandatory.begin(),mandatory.end()).size()!=mandatory.size())
        fail(MemoryErrorCode::InvalidPlan,"duplicate mandatory Pressure block");
    std::vector<LayerwisePressureChoice> result;
    std::vector<std::pair<uint64_t,uint32_t>> reference_partial;
    uint64_t reference_tokens=0;
    for (size_t l=0;l<layers_.size();++l) {
        const auto& memory=catalog.snapshot(layers_[l]);
        const auto& history=state_.layers[l];
        if (!(memory.stamp.session==session_) || history.end>memory.stamp.frontier)
            fail(MemoryErrorCode::Stale,"Pressure history differs from accepted native owner");
        if (memory.blocks.size()>UINT32_MAX) fail(MemoryErrorCode::InvalidPlan,"Pressure catalog too large");
        std::set<uint64_t> resident(memory.resident.begin(),memory.resident.end());
        std::set<uint64_t> fixed(mandatory.begin(),mandatory.end());
        for (size_t i=0;i<std::min<size_t>(sinks,memory.blocks.size());++i) fixed.insert(memory.blocks[i].block.id);
        for (auto id:fixed) if (!resident.count(id))
            fail(MemoryErrorCode::Stale,"mandatory/sink Pressure block is outside this layer's resident lineage");
        LayerwisePressureChoice choice; choice.model_layer=layers_[l];
        choice.accepted_score_end=history.end; choice.samples=history.samples; choice.chunks=history.chunks;
        std::vector<uint32_t> candidates,required;
        bool positive=false;
        for (size_t i=0;i<memory.blocks.size();++i) {
            const auto& block=memory.blocks[i].block;
            if (!resident.count(block.id)) continue;
            candidates.push_back(uint32_t(i)); choice.candidates.push_back(block.id);
            const auto* value=find(history,block.original_start);
            const double score=value && value->exposure>0 ? value->mass/value->exposure : 0;
            finite(score); choice.scores.push_back(score); positive|=score>0;
            if (fixed.count(block.id)) {required.push_back(uint32_t(i)); choice.fixed.push_back(block.id);}
        }
        if (!positive) fail(MemoryErrorCode::Pending,"layer Pressure lacks positive accepted attention; no recency fallback");
        if (required.size()>budget/options_.block_tokens)
            fail(MemoryErrorCode::BudgetExceeded,"mandatory Pressure blocks exceed Active budget");
        const auto selected=pressure_selection(candidates,choice.scores,required,uint32_t(budget/options_.block_tokens));
        std::vector<std::pair<uint64_t,uint32_t>> partial;
        for (auto ordinal:selected) {
            const auto& b=memory.blocks[ordinal].block;
            choice.selected.push_back(b.id); choice.selected_tokens=add(choice.selected_tokens,b.valid_tokens);
            choice.charged_tokens=add(choice.charged_tokens,options_.block_tokens);
            if (b.valid_tokens!=options_.block_tokens) partial.emplace_back(b.original_start,b.valid_tokens);
        }
        if (l && (choice.selected_tokens!=reference_tokens || partial!=reference_partial))
            fail(MemoryErrorCode::InvalidPlan,"Pressure changes qualified native length/partial geometry between layers");
        if (!l) {reference_tokens=choice.selected_tokens;reference_partial=std::move(partial);}
        result.push_back(std::move(choice));
    }
    return result;
}
double lookup(const std::vector<std::pair<uint64_t,double>>& values,uint64_t key) {
    const auto i=std::lower_bound(values.begin(),values.end(),key,
        [](const auto& a,uint64_t b){return a.first<b;});
    return i!=values.end() && i->first==key ? i->second : 0.;
}
uint64_t prefix(uint32_t n) { return n>=64 ? UINT64_MAX : (uint64_t(1)<<n)-1; }
void valid_stamp(const MemoryStamp& stamp,const SessionIdentity& session) {
    if (!(stamp.session==session)) fail(MemoryErrorCode::Stale,"foreign accepted output owner");
    if (!stamp.valid || !stamp.history_revision || !stamp.execution_history)
        fail(MemoryErrorCode::InvalidPlan,"invalid native output acceptance identity");
}
}

std::vector<LayerwisePressureChoice> LayerwisePrefillPressure::select(const LayerwiseMemoryCatalog& catalog,
    const std::vector<uint64_t>& mandatory,uint64_t budget,uint32_t sinks) const {
    return select_seeds(catalog,session_,layers_,options_,state_,mandatory,budget,sinks);
}

LayerwiseOutputPressure::LayerwiseOutputPressure(SessionIdentity session,std::vector<uint32_t> layers,
    LayerwisePrefillOptions prefill,std::array<int32_t,5> markers,LayerwiseOutputOptions options)
    :LayerwiseOutputPressure(session,std::move(layers),prefill,LayerwiseExternalOutputScope{},options) {
    std::set<int32_t> unique(markers.begin(),markers.end());
    if(unique.size()!=5 || *unique.begin()<0)
        fail(MemoryErrorCode::Unsupported,"output Pressure requires exact model scope markers/policy");
    markers_=markers;
}
LayerwiseOutputPressure::LayerwiseOutputPressure(SessionIdentity session,std::vector<uint32_t> layers,
    LayerwisePrefillOptions prefill,LayerwiseExternalOutputScope,LayerwiseOutputOptions options)
    :session_(session),layers_(std::move(layers)),prefill_options_(prefill),options_(options) {
    // The all-negative private discriminator denotes the explicit external
    // scope constructor. It is never exposed or consumed as token identities.
    markers_.fill(-1);
    // Reuse the target/order/capacity validation of the same Pressure picker.
    LayerwisePrefillPressure(session_,layers_,prefill_options_);
    if (!options.interval || options.interval>4096 ||
        !std::isfinite(options.decay) || options.decay<0 || options.decay>1 || options.commit_max_bytes<4096 ||
        (options.scope!=LayerwiseOutputScope::ToolCall && options.scope!=LayerwiseOutputScope::NonThinking))
        fail(MemoryErrorCode::Unsupported,"output Pressure requires exact model scope markers/policy");
    state_.session=session_;
    for (auto layer:layers_) state_.layers.push_back({layer,{},{},0,0});
}
void LayerwiseOutputPressure::owner(uint64_t request) const {
    if (!state_.active || !request || request!=state_.request)
        fail(MemoryErrorCode::Stale,"foreign/inactive accepted output request");
}
void LayerwiseOutputPressure::begin(const MemoryStamp& history,uint64_t request,const std::vector<uint32_t>& prompt) {
    if(markers_[0]<0)fail(MemoryErrorCode::Unsupported,"native grammar owner requires begin_scope");
    bool thinking=false;
    // Upstream ActionHistory.begin: the last message/think marker alone decides.
    for (auto i=prompt.rbegin();i!=prompt.rend();++i)
        if (*i==uint32_t(markers_[4]) || *i==uint32_t(markers_[2]) || *i==uint32_t(markers_[3])) {
            thinking=*i==uint32_t(markers_[2]); break;
        }
    begin_state(history,request,{thinking,false});
}
void LayerwiseOutputPressure::begin_scope(const MemoryStamp& history,uint64_t request,LayerwiseOutputScopeState state) {
    if(markers_[0]>=0)fail(MemoryErrorCode::Unsupported,"token marker owner cannot consume external grammar scope");
    begin_state(history,request,state);
}
void LayerwiseOutputPressure::begin_state(const MemoryStamp& history,uint64_t request,LayerwiseOutputScopeState scope) {
    if(scope.thinking && scope.tool_call)fail(MemoryErrorCode::InvalidPlan,"thinking native scope cannot open a tool action");
    valid_stamp(history,session_);
    if (!request) fail(MemoryErrorCode::InvalidPlan,"missing output request identity");
    if (state_.request==request) return; // includes the closed same-ID interval
    if (state_.active) {
        fail(MemoryErrorCode::Pending,"previous output window has not finished");
    }
    if (history.frontier<state_.accepted_end) fail(MemoryErrorCode::Stale,"output window precedes accepted feedback");
    state_.parser={scope.thinking,scope.tool_call,0}; state_.request=request; state_.active=true;
    state_.accepted_end=history.frontier;
    for (auto& layer:state_.layers) { layer.pending.clear(); layer.samples=0; }
    candidate_.clear(); mask_=0; accepted_=0; prepared_=false;
}
void LayerwiseOutputPressure::accept_prefill(const MemoryStamp& history,uint64_t request,uint64_t begin) {
    owner(request);
    valid_stamp(history,session_);
    if (prepared_ || begin!=state_.accepted_end || history.frontier<begin)
        fail(MemoryErrorCode::Stale,"prefill acceptance differs from its active output request frontier");
    state_.accepted_end=history.frontier;
}

uint64_t LayerwiseOutputPressure::prepare(uint64_t request,uint64_t begin,const std::vector<uint32_t>& tokens) {
    if(markers_[0]<0)fail(MemoryErrorCode::Unsupported,"native grammar owner requires prepare_scope");
    owner(request);
    if (tokens.size()>64 || begin!=state_.accepted_end)
        fail(MemoryErrorCode::InvalidPlan,"output verifier extent differs from accepted native frontier");
    add(begin,tokens.size());
    auto parser=state_.parser;
    std::vector<LayerwiseOutputParserState> candidates;
    uint64_t mask=0;
    for (size_t row=0;row<tokens.size();++row) {
        const auto token=int32_t(tokens[row]);
        if (token==markers_[2]) {parser.thinking=true;parser.action=false;}
        else if (token==markers_[3]) parser.thinking=false;
        else if (token==markers_[1]) parser.action=false;
        else if (token==markers_[0] && !parser.thinking) parser.action=true;
        else if (!parser.thinking && (options_.scope==LayerwiseOutputScope::NonThinking || parser.action) &&
                 parser.tokens++%options_.interval==0)
            mask|=uint64_t(1)<<row;
        candidates.push_back(parser);
    }
    candidate_=std::move(candidates); mask_=mask; begin_=begin; accepted_=0; prepared_=true;
    return mask_;
}
uint64_t LayerwiseOutputPressure::prepare_scope(uint64_t request,uint64_t begin,
    const std::vector<LayerwiseOutputScopeRow>& rows) {
    if(markers_[0]>=0)fail(MemoryErrorCode::Unsupported,"token marker owner cannot consume external grammar rows");
    owner(request);
    if(rows.size()>64 || begin!=state_.accepted_end)
        fail(MemoryErrorCode::InvalidPlan,"native grammar verifier differs from accepted frontier");
    add(begin,rows.size());
    auto parser=state_.parser;
    std::vector<LayerwiseOutputParserState> candidates;candidates.reserve(rows.size());
    uint64_t mask=0;
    for(size_t row=0;row<rows.size();++row) {
        const auto& input=rows[row];
        if((input.thinking && input.tool_call) ||
           (input.body && (input.thinking!=parser.thinking || input.tool_call!=parser.action)))
            fail(MemoryErrorCode::InvalidPlan,"native scope body changed state without a control row");
        parser.thinking=input.thinking;parser.action=input.tool_call;
        if(input.body && !parser.thinking && (options_.scope==LayerwiseOutputScope::NonThinking || parser.action) &&
           parser.tokens++%options_.interval==0)
            mask|=uint64_t(1)<<row;
        candidates.push_back(parser);
    }
    candidate_=std::move(candidates);mask_=mask;begin_=begin;accepted_=0;prepared_=true;
    return mask_;
}
uint64_t LayerwiseOutputPressure::accept(const LayerwiseOutputAcceptance& packet) {
    owner(packet.request); valid_stamp(packet.history,session_);
    if (!prepared_ || packet.accepted<accepted_ || packet.accepted>candidate_.size() ||
        packet.history.frontier!=add(begin_,packet.accepted))
        fail(MemoryErrorCode::InvalidPlan,"invalid native accepted output prefix");
    const auto newly=mask_ & prefix(packet.accepted) & ~prefix(accepted_);
    std::vector<size_t> indices;
    uint64_t captured=0;
    for (size_t i=0;i<packet.rows.size();++i) {
        const auto row=packet.rows[i];
        if (row>=candidate_.size() || (i && row<=packet.rows[i-1]))
            fail(MemoryErrorCode::InvalidPlan,"invalid/duplicate native captured output row");
        captured|=uint64_t(1)<<row;
        if (newly & (uint64_t(1)<<row)) indices.push_back(i);
    }
    if ((captured & newly)!=newly) fail(MemoryErrorCode::InvalidPlan,"missing accepted sampled native output row");
    auto next=state_;
    if (newly) {
        if (packet.layers.size()!=layers_.size()) fail(MemoryErrorCode::InvalidPlan,"missing target-layer output mass");
        std::set<uint32_t> seen;
        for (const auto& input:packet.layers) {
            const auto where=std::lower_bound(layers_.begin(),layers_.end(),input.model_layer);
            if (where==layers_.end() || *where!=input.model_layer || !seen.insert(input.model_layer).second)
                fail(MemoryErrorCode::InvalidPlan,"foreign/duplicate target output layer");
            auto& layer=next.layers[where-layers_.begin()];
            if (input.history.size()>SIZE_MAX/packet.rows.size() ||
                input.masses.size()!=input.history.size()*packet.rows.size())
                fail(MemoryErrorCode::InvalidPlan,"invalid native output matrix shape");
            // Exact upstream native staging bound, checked before publication.
            const auto limit=options_.commit_max_bytes;
            if (limit<4096 || layer.pending.size()>limit/320 || input.history.size()>limit/320 ||
                add(layer.pending.size(),input.history.size())>(limit-4096)/320)
                fail(MemoryErrorCode::BudgetExceeded,"output attention staging capacity exceeded before acceptance");
            std::map<uint64_t,double> updated(layer.pending.begin(),layer.pending.end());
            for (auto key:input.history) if (key%prefill_options_.block_tokens)
                fail(MemoryErrorCode::InvalidPlan,"unaligned native output logical key");
            for (auto row:indices) {
                Sum sum;
                for (size_t p=0;p<input.history.size();++p) {
                    const auto mass=input.masses[row*input.history.size()+p]; finite(mass); sum.add(mass);
                }
                if (!std::isfinite(sum.value()) || std::abs(sum.value()-1.)>.005)
                    fail(MemoryErrorCode::InvalidPlan,"invalid accepted output attention mass sum");
                for (size_t p=0;p<input.history.size();++p) {
                    const auto mass=input.masses[row*input.history.size()+p];
                    if (mass!=0.) updated[input.history[p]]+=mass;
                }
            }
            layer.pending.assign(updated.begin(),updated.end()); layer.samples=add(layer.samples,indices.size());
        }
    }
    if (packet.accepted) next.parser=candidate_[packet.accepted-1];
    next.accepted_end=packet.history.frontier;
    state_=std::move(next); accepted_=packet.accepted;
    return newly;
}
void LayerwiseOutputPressure::finish(uint64_t request,const LayerwiseMemoryCatalog& catalog) {
    owner(request);
    if (catalog.layers().size()!=layers_.size() || catalog.descriptor().layout.block_tokens!=prefill_options_.block_tokens)
        fail(MemoryErrorCode::InvalidPlan,"incomplete native output finish catalog");
    auto next=state_;
    for (size_t i=0;i<layers_.size();++i) {
        const auto& memory=catalog.snapshot(layers_[i]);
        valid_stamp(memory.stamp,session_);
        if (memory.stamp.frontier<state_.accepted_end) fail(MemoryErrorCode::Stale,"finish history precedes consumed output");
        auto& layer=next.layers[i];
        std::vector<std::pair<uint64_t,double>> scores;
        for (const auto& copy:memory.blocks) {
            const auto key=copy.block.original_start;
            const double old=lookup(layer.scores,key);
            if (layer.samples) scores.emplace_back(key,multiply(options_.decay,old)+lookup(layer.pending,key)/layer.samples);
            else {
                const auto found=std::lower_bound(layer.scores.begin(),layer.scores.end(),key,
                    [](const auto& a,uint64_t b){return a.first<b;});
                if (found!=layer.scores.end() && found->first==key) scores.push_back(*found);
            }
        }
        layer.scores=std::move(scores);
        if (layer.samples) layer.rounds=add(layer.rounds,1);
        // Literal ActionHistory.finish retains N until the next begin().
        // pressure_seed therefore previews decay*S with empty pending mass in
        // this closed-window interval. Do not silently reset or re-decay state.
        layer.pending.clear();
    }
    next.active=false; state_=std::move(next);
    candidate_.clear(); mask_=0; accepted_=0; prepared_=false;
}
void LayerwiseOutputPressure::restore(LayerwiseOutputSnapshot next,uint64_t position) {
    if (!(next.session==session_)) fail(MemoryErrorCode::Stale,"foreign output snapshot owner");
    if (next.accepted_end>position) fail(MemoryErrorCode::Stale,"output feedback exceeds validated native prefix");
    if (next.layers.size()!=layers_.size() || (next.active && !next.request))
        fail(MemoryErrorCode::InvalidPlan,"incomplete output snapshot");
    for (size_t i=0;i<layers_.size();++i) {
        const auto& layer=next.layers[i];
        if (layer.model_layer!=layers_[i] || layer.samples!=next.layers[0].samples || layer.rounds!=next.layers[0].rounds ||
            (!layer.samples && !layer.pending.empty()) || (!next.active && !layer.pending.empty()))
            fail(MemoryErrorCode::InvalidPlan,"inconsistent accepted target output boundary");
        for (const auto* values:{&layer.scores,&layer.pending}) {
            uint64_t previous=0; bool first=true;
            for (const auto& value:*values) {
                finite(value.second);
                if (value.first%prefill_options_.block_tokens || (!first && value.first<=previous))
                    fail(MemoryErrorCode::InvalidPlan,"invalid output snapshot logical keys");
                first=false; previous=value.first;
            }
        }
    }
    state_=std::move(next); candidate_.clear(); mask_=0; accepted_=0; prepared_=false;
}
std::vector<LayerwisePressureChoice> LayerwiseOutputPressure::select(const LayerwiseMemoryCatalog& catalog,
    const LayerwisePrefillPressure& prefill,const std::vector<uint64_t>& mandatory,uint64_t budget,uint32_t sinks) const {
    const auto& fallback=prefill.snapshot();
    const auto& config=prefill.options();
    if (!(fallback.session==session_) || fallback.layers.size()!=layers_.size())
        fail(MemoryErrorCode::Stale,"foreign output Pressure prefill fallback");
    if (std::tie(config.block_tokens,config.select_tokens,config.reserve_tokens,config.interval,config.stride,config.half_life)!=
        std::tie(prefill_options_.block_tokens,prefill_options_.select_tokens,prefill_options_.reserve_tokens,
                 prefill_options_.interval,prefill_options_.stride,prefill_options_.half_life))
        fail(MemoryErrorCode::InvalidPlan,"output Pressure fallback has another sampling policy");
    auto seed=fallback;
    std::vector<bool> output(layers_.size(),false);
    for (size_t i=0;i<layers_.size();++i) {
        if (fallback.layers[i].model_layer!=layers_[i]) fail(MemoryErrorCode::InvalidPlan,"foreign prefill fallback target");
        const auto& memory=catalog.snapshot(layers_[i]);
        if (memory.stamp.frontier<state_.accepted_end) fail(MemoryErrorCode::Stale,"output Pressure precedes accepted feedback");
        const auto& layer=state_.layers[i];
        std::vector<LayerwisePrefillValue> values;
        for (const auto& block:memory.blocks) {
            if (std::find(memory.resident.begin(),memory.resident.end(),block.block.id)==memory.resident.end()) continue;
            const auto key=block.block.original_start;
            const double old=lookup(layer.scores,key);
            const double score=layer.samples ? multiply(options_.decay,old)+lookup(layer.pending,key)/layer.samples : old;
            finite(score); output[i]=output[i] || score>0;
            values.push_back({key,score,1.});
        }
        if (output[i]) { seed.layers[i].values=std::move(values); seed.layers[i].end=state_.accepted_end; }
    }
    auto choices=select_seeds(catalog,session_,layers_,prefill_options_,seed,mandatory,budget,sinks);
    for (size_t i=0;i<choices.size();++i) {
        choices[i].source=output[i]?LayerwisePressureSource::Output:LayerwisePressureSource::Prefill;
        choices[i].output_samples=state_.layers[i].samples; choices[i].output_rounds=state_.layers[i].rounds;
        if (output[i]) choices[i].accepted_score_end=state_.accepted_end;
    }
    return choices;
}

} // namespace kvmem
