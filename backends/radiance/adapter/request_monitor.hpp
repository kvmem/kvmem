#pragma once
#include "kvmem/monitor.hpp"
#include "rad_core.h"
#include "server/sink.h"
#include "mm/processor.h"
#include <deque>
#include <unordered_map>

namespace kvmem::radiance {
class RequestMonitor {
    kvmem::monitor requests_;
    struct Sample { int64_t ns, count; };
    // Scheduler-lock owned, bounded by active requests. Never retains request text/pointers.
    std::unordered_map<uint64_t,std::deque<Sample>> windows_;
public:
    void reuse(uint64_t request,uint64_t planned,uint64_t actual,bool shared) noexcept {
        try {
            const auto id=std::to_string(request);
            requests_.begin("generation",id);
            requests_.update(id,{{"planned_cached_tokens",planned},{"actual_cached_tokens",actual},
                {"planned_cache_source",planned ? (shared ? "shared" : "own") : "miss"},
                {"cache_source",actual ? (shared ? "shared" : "own") : "miss"}});
        } catch(...) { RAD_ERR("KVMem request reuse metadata allocation failed"); }
    }
    void update(const rad::Request& r,int64_t age_us,int64_t first_us) noexcept {
        try {
            const std::string id=std::to_string(r.id);
            requests_.begin("generation",id);
            const bool terminal=r.state==rad::ReqState::Finished || r.state==rad::ReqState::Failed || r.state==rad::ReqState::Cancelled;
            const char* status=r.state==rad::ReqState::Waiting || r.state==rad::ReqState::Preempted ? "queued" :
                (r.output.empty() ? "reading" : "generating");
            const auto now=kvmem::monitor::now_ms();
            nlohmann::json data={{"status",status},{"prompt_tokens",r.prompt.size()},{"cached_tokens",r.n_cached},
                {"prompt_processed_tokens",std::min<size_t>(r.prompt.size(),std::max(0,r.n_computed))},
                {"generated_tokens",r.output.size()},{"limit",r.max_tokens},{"has_media",bool(r.media)},
                {"started_at_ms",now-age_us/1000},{"finish_reason",r.finish_reason},
                {"ttft_ms",first_us>=0 ? nlohmann::json(double(first_us)/1000) : nlohmann::json(nullptr)},
                {"prefill_ms",nullptr},{"prefill_tps",nullptr},{"prefill_elapsed_ms",nullptr},
                {"decode_ms",nullptr},{"decode_tps",nullptr},{"decode_tps_mean",nullptr},
                {"decode_tps_live",nullptr},{"decode_tps_window_ms",nullptr},{"decode_tps_sampled_at_ms",nullptr}};
            // Request::sink has the scheduler's existing server::Sink contract; null for offline requests.
            if(const auto* sink=static_cast<const rad::server::Sink*>(r.sink)) {
                const auto age=sink->age_ns(),admit=sink->t_admitted_ns(),first=sink->t_first_token_ns(),last=sink->t_last_token_ns();
                const auto count=sink->n_produced();
                data["native_request_id"]=sink->response_id;
                data["generated_tokens"]=count;
                data["drafts_offered"]=sink->n_drafted();data["drafts_accepted"]=sink->n_draft_accepted();
                data["prompt_new_tokens"]=std::max<int64_t>(0,int64_t(r.prompt.size())-r.n_cached);
                if(admit>=0)data["queue_ms"]=double(admit)*1e-6;
                if(first>=0) {
                    data["ttft_ms"]=double(first)*1e-6;
                    if(admit>=0) {
                        const auto ms=double(first-admit)*1e-6;
                        data["prefill_ms"]=ms;
                        if(ms>0 && data["prompt_new_tokens"].get<int64_t>()>0)
                            data["prefill_tps"]=data["prompt_new_tokens"].get<int64_t>()*1000.0/ms;
                    }
                    const auto ms=double(last-first)*1e-6;
                    data["decode_ms"]=ms;
                    // Match native timings_json: tokens after the first / first-to-last duration.
                    if(ms>0 && count>1)data["decode_tps"]=data["decode_tps_mean"]=(count-1)*1000.0/ms;
                    auto& window=windows_[r.id];
                    if(window.empty() || count>window.back().count)window.push_back({last,count});
                    while(window.size()>2 && last-window[1].ns>=2000000000ll)window.pop_front();
                    if(!terminal && window.size()>1 && last-window.front().ns>=100000000ll) {
                        const double span=double(last-window.front().ns)*1e-6;
                        data["decode_tps_live"]=(count-window.front().count)*1000.0/span;
                        data["decode_tps_window_ms"]=span;
                        data["decode_tps_sampled_at_ms"]=now-(age-last)/1000000;
                    }
                } else if(admit>=0 && !terminal)data["prefill_elapsed_ms"]=double(age-admit)*1e-6;
            }
            if(r.media) {
                int64_t dec=0,pre=0,queue=0,enc=0,hits=0,misses=0,tokens=0,encoding=0,queued=0,live=0;
                for(const auto& p:r.media->items) {
                    const auto& v=*p.item->stats;
                    dec+=v.decode_ns.load();pre+=v.preprocess_ns.load();queue+=v.queued_ns.load();enc+=v.encode_ns.load();
                    if(!terminal && v.phase.load()==2){++encoding;live+=std::max<int64_t>(0,rad::mm::vision_now_ns()-v.encode_start_ns.load());}
                    if(!terminal && v.phase.load()==1)++queued;
                    hits+=v.cache_hits.load();misses+=v.cache_misses.load();tokens+=p.item->n_rows();
                }
                data["image"]={{"decode_ms",dec/1e6},{"preprocess_ms",pre/1e6},{"queue_ms",queue/1e6},
                    {"encode_ms",enc/1e6},{"cache_hits",hits},{"cache_misses",misses},{"tokens",tokens},
                    {"encoding_items",encoding},{"queued_items",queued},{"encode_elapsed_ms",live/1e6}};
            }
            requests_.update(id,data);
            if(terminal) {
                requests_.finish(id,r.state==rad::ReqState::Finished ? "completed" :
                    r.state==rad::ReqState::Cancelled ? "disconnected" : "error");
                windows_.erase(r.id);
            }
        } catch(...) { RAD_ERR("KVMem request monitor metadata allocation failed"); }
    }
    nlohmann::json snapshot() const { return requests_.snapshot(); }
};
} // namespace kvmem::radiance
