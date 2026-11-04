#ifdef NDEBUG
#undef NDEBUG
#endif
#include "request_monitor.hpp"
#include <cassert>
#include <cmath>
#include <thread>
#include <iostream>
static void test_sink_metrics() {
    kvmem::radiance::RequestMonitor monitor;
    rad::server::Sink sink;
    sink.response_id="chatcmpl-fixture";
    rad::Request r;r.id=7;r.sink=&sink;r.prompt.resize(100);r.n_cached=80;
    r.state=rad::ReqState::Waiting;
    monitor.update(r,0,-1);
    auto row=monitor.snapshot()["active"][0];
    assert(row["native_request_id"]==sink.response_id && row["prefill_ms"].is_null());
    sink.set_prompt_stats(100,80);sink.mark_admitted();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    sink.push(1);r.output.push_back(1);r.n_computed=100;r.state=rad::ReqState::Running;
    monitor.update(r,1000,1);
    row=monitor.snapshot()["active"][0];
    assert(row["prefill_ms"].get<double>()>0 && row["prompt_new_tokens"]==20);
    assert(row["decode_tps_mean"].is_null());
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    sink.add_drafts(3,2);sink.push(2);sink.push(3);r.output.insert(r.output.end(),{2,3});
    monitor.update(r,1000,1);
    row=monitor.snapshot()["active"][0];
    assert(row["drafts_offered"]==3 && row["drafts_accepted"]==2);
    assert(row["generated_tokens"]==3 && row["decode_tps_live"].get<double>()>0);
    const auto mean=row["decode_tps_mean"].get<double>();
    // Finishing/cache work must not lower the frozen native decode rate.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    r.state=rad::ReqState::Finished;r.finish_reason="stop";
    monitor.update(r,1000000,1);sink.finish(rad::server::Finish::Stop);
    row=monitor.snapshot()["recent"][0];
    assert(std::abs(row["decode_tps_mean"].get<double>()-mean)<1e-9 && row["decode_tps_live"].is_null());
    assert(monitor.snapshot()["active"].empty());
    rad::Request offline;offline.id=8;offline.state=rad::ReqState::Failed;
    monitor.update(offline,0,-1);
    row=monitor.snapshot()["recent"][0];
    assert(row["prefill_ms"].is_null() && row["decode_tps"].is_null() && !row.contains("native_request_id"));
    std::cout<<"native monitor timing, identity, MTP, terminal and offline invariants passed\n";
}

#include "request_monitor.hpp"
#include <cstdlib>
#include <iostream>
static void check(bool ok){if(!ok)std::abort();}
int main() {
    test_sink_metrics();
    kvmem::radiance::RequestMonitor monitor;
    rad::Request a,b;a.id=1;b.id=2;a.prompt={12345,54321};b.prompt={99999};
    a.max_tokens=b.max_tokens=10;
    monitor.update(a,0,-1);monitor.update(b,0,-1);
    monitor.reuse(a.id,90,0,true);monitor.reuse(b.id,64,64,true);
    const auto cache_rows=monitor.snapshot()["active"];
    check(cache_rows[0]["actual_cached_tokens"]==0 && cache_rows[0]["planned_cached_tokens"]==90);
    check(cache_rows[0]["cache_source"]=="miss" && cache_rows[1]["cache_source"]=="shared");
    check(cache_rows[0]["cache_semantics_revision"]==2 && cache_rows[0]["restore_path"]=="miss");
    check(cache_rows[1]["restore_path"]=="snapshot");
    check(monitor.snapshot()["active"].size()==2);
    monitor.reuse(a.id,80,80,false);
    const auto own=monitor.snapshot()["active"][0];
    check(own["cache_source"]=="own" && own["planned_cache_source"]=="own");
    check(own["cache_semantics_revision"]==2 && own["restore_path"]=="continuation");
    check(own["actual_cached_tokens"]==80 && own["planned_cached_tokens"]==80);
    a.state=rad::ReqState::Running;a.n_computed=2;a.output={70000,80000};
    monitor.update(a,3000000,1000000);
    const auto row=monitor.snapshot()["active"][0];
    check(row["generated_tokens"]==2 && row["ttft_ms"]==1000 && row["decode_tps"].is_null());
    check(row["prefill_tps"].is_null());
    const auto data=monitor.snapshot().dump();
    check(data.find("54321")==std::string::npos && data.find("80000")==std::string::npos);
    a.state=rad::ReqState::Cancelled;a.finish_reason="cancelled";monitor.update(a,4000000,1000000);
    check(monitor.snapshot()["active"].size()==1 && monitor.snapshot()["recent"][0]["status"]=="disconnected");
    b.state=rad::ReqState::Finished;b.finish_reason="stop";b.output={5};monitor.update(b,4000000,2000000);
    check(monitor.snapshot()["active"].empty() && monitor.snapshot()["recent"].size()==2);
    std::cout << "request metadata isolation, privacy and native timing semantics passed\n";
}
