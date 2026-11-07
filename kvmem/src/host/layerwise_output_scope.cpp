#include "kvmem/layerwise_output_scope.hpp"
#include "kvmem/memory_contract.hpp"

#include <algorithm>
#include <set>
#include <utility>

namespace kvmem {
namespace {
[[noreturn]] void invalid_scope(const char* reason) {
    throw MemoryError(MemoryErrorCode::InvalidPlan,reason);
}
}
LayerwiseOutputGrammar::LayerwiseOutputGrammar(LayerwiseOutputGrammarConfig config)
    :config_(std::move(config)) {
    const std::set<int32_t> ids{config_.channel_open,config_.channel_close,config_.turn_open,
        config_.turn_close,config_.tool_open,config_.tool_close};
    if(ids.size()!=6 || *ids.begin()<0 || config_.thinking_channel.empty() ||
       config_.thinking_channel.size()>1024 || config_.thinking_channel.find_first_of("\r\n\0",0,3)!=std::string::npos ||
       !config_.max_token_bytes || config_.max_token_bytes>(1u<<20))
        invalid_scope("native output grammar requires exact distinct controls and bounded channel/token bytes");
}
void LayerwiseOutputGrammar::reset() {
    state_={};count_=accepted_=0;prepared_=false;
}
void LayerwiseOutputGrammar::validate(LayerwiseOutputGrammarState state) const {
    if((state.scope.thinking && state.scope.tool_call) ||
       (state.header!=LayerwiseOutputHeader::None && state.header!=LayerwiseOutputHeader::Channel &&
        state.header!=LayerwiseOutputHeader::Turn) || state.matched>config_.thinking_channel.size() ||
       (state.header!=LayerwiseOutputHeader::Channel && (state.matched || state.mismatch)))
        invalid_scope("invalid accepted native output grammar state");
}
LayerwiseOutputScopeRow LayerwiseOutputGrammar::step(LayerwiseOutputGrammarState& state,
    LayerwiseOutputGrammarToken token) const {
    if(token.id>uint32_t(INT32_MAX) || token.bytes.empty() || token.bytes.size()>config_.max_token_bytes)
        invalid_scope("native output token has no bounded tokenizer byte proof");
    const auto id=int32_t(token.id);
    const auto clear_header=[&]{state.header=LayerwiseOutputHeader::None;state.matched=0;state.mismatch=false;};
    const auto row=[&](bool body){return LayerwiseOutputScopeRow{state.scope.thinking,state.scope.tool_call,body};};
    if(id==config_.turn_open) {
        state.scope={};state.header=LayerwiseOutputHeader::Turn;state.matched=0;state.mismatch=false;
        return row(false);
    }
    if(id==config_.channel_open) {
        state.scope.tool_call=false;state.header=LayerwiseOutputHeader::Channel;state.matched=0;state.mismatch=false;
        return row(false);
    }
    if(id==config_.turn_close) {state.scope={};clear_header();return row(false);}
    if(id==config_.channel_close) {state.scope.thinking=false;state.scope.tool_call=false;clear_header();return row(false);}
    if(state.header!=LayerwiseOutputHeader::None) {
        for(const char byte:token.bytes) {
            if(byte=='\n') {
                if(state.header==LayerwiseOutputHeader::Channel) {
                    state.scope.thinking=!state.mismatch && state.matched==config_.thinking_channel.size();
                    state.scope.tool_call=false;
                }
                clear_header();break;
            }
            if(state.header==LayerwiseOutputHeader::Channel && !state.mismatch) {
                if(state.matched<config_.thinking_channel.size() && byte==config_.thinking_channel[state.matched])++state.matched;
                else state.mismatch=true;
            }
        }
        return row(false);
    }
    if(id==config_.tool_open) {if(!state.scope.thinking)state.scope.tool_call=true;return row(false);}
    if(id==config_.tool_close) {state.scope.tool_call=false;return row(false);}
    return row(true);
}
void LayerwiseOutputGrammar::prompt(LayerwiseOutputGrammarToken token) {
    if(prepared_)invalid_scope("cannot rescan a native prompt with an outstanding output preview");
    auto next=state_;step(next,token);state_=next;
}
std::vector<LayerwiseOutputScopeRow> LayerwiseOutputGrammar::prepare(
    const std::vector<LayerwiseOutputGrammarToken>& tokens) {
    if(tokens.size()>candidates_.size())invalid_scope("native output verifier exceeds 64 consumed rows");
    auto next=state_;
    std::array<LayerwiseOutputGrammarState,64> candidates{};
    std::vector<LayerwiseOutputScopeRow> result;result.reserve(tokens.size());
    for(size_t row=0;row<tokens.size();++row) {result.push_back(step(next,tokens[row]));candidates[row]=next;}
    candidates_=std::move(candidates);count_=uint32_t(tokens.size());accepted_=0;prepared_=true;
    return result;
}
void LayerwiseOutputGrammar::accept(uint32_t accepted) {
    if(!prepared_ || accepted<accepted_ || accepted>count_)
        invalid_scope("native grammar acceptance differs from its prepared prefix");
    if(accepted)state_=candidates_[accepted-1];accepted_=accepted;
}
void LayerwiseOutputGrammar::restore(LayerwiseOutputGrammarState state) {
    validate(state);state_=state;count_=accepted_=0;prepared_=false;
}
uint64_t LayerwiseOutputGrammar::metadata_bytes() const noexcept {
    return sizeof(*this)+config_.thinking_channel.capacity();
}
} // namespace kvmem
