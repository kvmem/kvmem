#include "batched_observer.hpp"
#include <future>
#include <iostream>
#include <thread>

using namespace kvmem;
using namespace kvmem::radiance;
static void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}

// Exercise the actual controller at an expired deadline without a model/GPU or
// a 30-second sleep. This seam changes no production headers or class layouts.
template<class Tag,auto Member> struct Access {friend auto member(Tag){return Member;}};
struct PendingTag{friend auto member(PendingTag);};
struct SessionTag{friend auto member(SessionTag);};
struct SlotsTag{friend auto member(SlotsTag);};
struct FinishTag{friend auto member(FinishTag);};
struct DrainingTag{friend auto member(DrainingTag);};
struct PrefixTag{friend auto member(PrefixTag);};
template struct Access<PendingTag,&StatisticsObserver::pending_selection_>;
template struct Access<SessionTag,&StatisticsObserver::memory_session_>;
template struct Access<SlotsTag,&BatchedStatisticsObserver::slots_>;
template struct Access<FinishTag,&StatisticsObserver::pending_finish_>;
template struct Access<DrainingTag,&StatisticsObserver::draining_selection_failure_>;
template struct Access<PrefixTag,&StatisticsObserver::pending_prefix_>;

struct CpuBackend final : MemoryBackend {
    MemoryDescriptor desc;MemorySnapshot state;
    std::shared_ptr<StorageIoQueue> io=std::make_shared<StorageIoQueue>(1,64,1);
    std::promise<void> release,started;
    std::shared_future<void> released=release.get_future().share();
    StorageTicket ticket;
    bool storage_failure=false,capacity_failure=false,cpu_stage=true;
    int aborts=0,published=0;
    CpuBackend() {
        desc.identity={"radiance","unit","storage-recovery","cpu-wait"};
        desc.layout={"unit",1,16,32,1,{{3,PayloadPlaneKind::PackedKeyValue,0,32,1,0}}};
        desc.capabilities={16,1024,true,true,true,false,false,true};
        state.stamp={true,{42,1},1,0,64,1};state.resident={0,1,2};
        for(uint64_t i=0;i<4;++i)state.blocks.push_back({{{42,1},i,i*16,16,1,1},i==3 ? 1u:0u});
    }
    MemoryDescriptor descriptor()const override{return desc;}
    MemorySnapshot snapshot()const override{return state;}
    MemoryStamp stamp()const noexcept override{return state.stamp;}
    void invalidate()noexcept override{state.stamp.valid=false;}
    struct Tx final : MemoryTransfer,StorageWaiter {
        CpuBackend& backend;WorkingSetPlan plan;int phase=0;bool aborted=false;
        Tx(CpuBackend& b,WorkingSetPlan p):backend(b),plan(std::move(p)){}
        void start_spill()override {
            phase=1;auto future=backend.released;auto* begun=&backend.started;
            backend.ticket=backend.io->submit(1,[future,begun]{begun->set_value();future.wait();});
        }
        TransferPoll poll()override {
            if(backend.storage_failure)throw StorageFailure("injected storage error");
            if(backend.capacity_failure)throw MemoryError(MemoryErrorCode::BudgetExceeded,"injected storage quota refusal");
            if(phase==1 && !backend.ticket.ready())return {TransferStatus::Pending,{}};
            ++phase;return {TransferStatus::Complete,{}};
        }
        void start_restore()override {check(phase==2,"restore before spill");phase=3;}
        void publish(const ExecutionView& view)override {
            check(phase==4,"publish before restore");backend.state.stamp=view.stamp;
            backend.state.resident.clear();for(const auto& b:view.blocks)backend.state.resident.push_back(b.block.id);
            ++backend.published;
        }
        bool awaiting_storage()const noexcept override{return phase==1 && backend.cpu_stage;}
        bool cpu_io_pending()const noexcept override{return backend.cpu_stage && backend.ticket && !backend.ticket.ready();}
        bool storage_quiescent()override{return !backend.ticket || backend.ticket.ready();}
        void cancel_queued_storage()noexcept override {if(backend.ticket)backend.ticket.cancel();}
        AbortResult abort()noexcept override {
            if(!aborted){++backend.aborts;aborted=true;}
            if(backend.ticket){backend.ticket.cancel();try{backend.ticket.wait();}catch(...){}}
            return AbortResult::PreviousViewPreserved;
        }
    };
    std::unique_ptr<MemoryTransfer> prepare(const WorkingSetPlan& plan)override{return std::make_unique<Tx>(*this,plan);}
};

struct RankFixture {
    std::shared_ptr<CpuBackend> a=std::make_shared<CpuBackend>(),b=std::make_shared<CpuBackend>();
    std::shared_ptr<RankMemory> ranks=std::make_shared<RankMemory>(
        std::vector<std::shared_ptr<MemoryBackend>>{a,b},BackendIdentity{"radiance","unit","storage-recovery","tp2"});
    std::unique_ptr<MemorySession> session=std::make_unique<MemorySession>(ranks);
    MemoryStamp original=ranks->stamp();
    void prepare() {
        session->prepare(original,{0,3},{0},{32,16,48,256});session->advance();
        a->started.get_future().wait();b->started.get_future().wait();
    }
    void unblock() {a->release.set_value();b->release.set_value();a->ticket.wait();b->ticket.wait();}
};

struct ReleaseIo {
    std::shared_ptr<CpuBackend> a,b;
    ~ReleaseIo(){for(const auto& backend:{a,b})try{backend->release.set_value();}catch(const std::future_error&) {}}
};

static void peer_can_publish() {
    RankFixture peer;peer.a->release.set_value();peer.b->release.set_value();peer.prepare();
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(peer.session->phase()!=MemoryPhase::Ready) {
        check(std::chrono::steady_clock::now()<deadline,"healthy peer stalled");peer.session->advance();
        std::this_thread::yield();
    }
    peer.session->publish();
    check(peer.a->published==1 && peer.b->published==1,"peer publication missing");
    check(peer.ranks->snapshot().resident==std::vector<uint64_t>({0,3}),"peer published the wrong view");
}

static void partial_rank_failure(bool capacity) {
    RankFixture f;ReleaseIo release{f.a,f.b};f.a->storage_failure=!capacity;f.a->capacity_failure=capacity;
    f.a->release.set_value();f.prepare();f.a->ticket.wait();
    const auto phase=f.session->advance();
    check(phase==MemoryPhase::Spilling && f.ranks->storage_error(),"CPU failure must be latched before abort");
    check(f.a->aborts==0 && f.b->aborts==0 && !f.b->ticket.ready(),"running peer I/O was synchronously aborted");
    peer_can_publish();
    check(f.ranks->stamp()==f.original,"failed unpublished transfer changed the accepted stamp");
    f.b->release.set_value();f.b->ticket.wait();
    bool rejected=false;
    try{f.session->advance();}
    catch(const StorageFailure&){rejected=!capacity;}
    catch(const MemoryError& e){rejected=capacity && e.code()==MemoryErrorCode::BudgetExceeded;}
    check(rejected && f.a->aborts==1 && f.b->aborts==1,"quiescent error was not propagated and cleaned up");
    check(f.ranks->stamp()==f.original && f.a->state.resident==std::vector<uint64_t>({0,1,2}) &&
          f.b->state.resident==f.a->state.resident,"failure lost a source view");
}

static void controller_failure(bool expired,bool cancelled,bool device=false) {
    RankFixture f;
    if(!expired && !cancelled){f.a->storage_failure=true;f.a->release.set_value();}
    if(device){f.a->cpu_stage=f.b->cpu_stage=false;}
    f.prepare();
    if(f.a->storage_failure)f.a->ticket.wait();
    BatchedStatisticsObserver controller;
    ReleaseIo release{f.a,f.b};
    auto& slot=(controller.*member(SlotsTag{})).emplace_back();slot.request=42;
    slot.observer=std::make_unique<StatisticsObserver>();
    (slot.observer.get()->*member(SessionTag{}))=std::move(f.session);
    auto& pending=(slot.observer.get()->*member(PendingTag{})).emplace();
    pending.reason=SelectionReason::PrefillPressure;pending.storage=f.ranks;
    pending.deadline=std::chrono::steady_clock::now()+(expired ? -std::chrono::seconds(1):std::chrono::seconds(30));
    if(cancelled)controller.retire(42);
    const int status=controller.poll_pending();int refusal_status=RAD_OK;
    const auto refused=controller.take_request_refusal(&refusal_status);
    if(device) {
        check(status==RAD_E_STATE && refused==0,"device timeout must remain fatal");
        f.unblock();return;
    }
    check(status==RAD_OK && (cancelled ? refused==0 : refused==42 && refusal_status==RAD_E_IO),
          "CPU timeout/error escaped as an engine failure");
    check(controller.holds_sequence(42) && f.b->aborts==0 && !f.b->ticket.ready(),"in-flight ownership was released early");
    check(controller.take_request_refusal(&refusal_status)==0,"request refusal repeated");
    peer_can_publish();
    if(!cancelled)controller.retire(42);
    if(!f.a->storage_failure)f.a->release.set_value();
    f.b->release.set_value();f.a->ticket.wait();f.b->ticket.wait();
    check(controller.poll_pending()==RAD_OK && !controller.holds_sequence(42),"drained request retained its slot");
    check(f.a->aborts==1 && f.b->aborts==1 && f.ranks->stamp()==f.original,"cleanup lost accepted source state");
}

static void successful_finish_stays_pending() {
    RankFixture f;f.prepare();BatchedStatisticsObserver controller;ReleaseIo release{f.a,f.b};
    auto& slot=(controller.*member(SlotsTag{})).emplace_back();slot.request=42;
    slot.observer=std::make_unique<StatisticsObserver>();
    (slot.observer.get()->*member(SessionTag{}))=std::move(f.session);
    auto& pending=(slot.observer.get()->*member(PendingTag{})).emplace();
    pending.storage=f.ranks;pending.deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    auto& terminal=slot.observer.get()->*member(FinishTag{});
    terminal=std::make_unique<typename std::remove_reference_t<decltype(terminal)>::element_type>();
    controller.retire(42);
    check(terminal && !(slot.observer.get()->*member(DrainingTag{})) && f.a->aborts==0 && f.b->aborts==0,
          "successful finish must complete its checkpoint instead of entering cancellation");
    terminal.reset();controller.retire(42);f.unblock();
    check(controller.poll_pending()==RAD_OK && !controller.holds_sequence(42),"test cleanup failed");
}

static void queued_cancel_releases_slot() {
    auto backend=std::make_shared<CpuBackend>();
    backend->io=std::make_shared<StorageIoQueue>(2,64,1);
    std::promise<void> unblock,started;
    auto gate=unblock.get_future().share();
    auto blocker=backend->io->submit(1,[gate,&started]{started.set_value();gate.wait();});
    started.get_future().wait();
    struct Release { std::promise<void>& signal; ~Release(){try{signal.set_value();}catch(...) {}} };
    Release release_early{unblock};
    auto ranks=std::make_shared<RankMemory>(std::vector<std::shared_ptr<MemoryBackend>>{backend},
        BackendIdentity{"radiance","unit","storage-recovery","cpu"});
    auto session=std::make_unique<MemorySession>(ranks);
    Release release_before_controller{unblock};
    const auto original=ranks->stamp();
    session->prepare(original,{0,3},{0},{32,16,48,256});session->advance();
    check(backend->ticket && !backend->ticket.ready(),"selection was not queued behind the blocker");
    BatchedStatisticsObserver controller;
    Release release{unblock};
    auto& slot=(controller.*member(SlotsTag{})).emplace_back();slot.request=42;
    slot.observer=std::make_unique<StatisticsObserver>();
    (slot.observer.get()->*member(SessionTag{}))=std::move(session);
    auto& pending=(slot.observer.get()->*member(PendingTag{})).emplace();
    pending.storage=ranks;pending.deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    controller.retire(42);
    check(backend->ticket.ready() && backend->ticket.status()==StorageTaskStatus::Cancelled,
          "queued selection did not cancel immediately");
    check(controller.poll_pending()==RAD_OK && !controller.holds_sequence(42) && !blocker.ready(),
          "canceled selection retained its slot behind unrelated I/O");
    check(backend->aborts==1 && ranks->stamp()==original && backend->io->pending_jobs()==1,
          "queued cancellation lost the source view or queue accounting");
}

static void queued_prefix_releases_slot(bool timeout) {
    auto io=std::make_shared<StorageIoQueue>(2,64,1);
    std::promise<void> unblock,started;
    auto gate=unblock.get_future().share();
    auto blocker=io->submit(1,[gate,&started]{started.set_value();gate.wait();});
    started.get_future().wait();
    struct Release { std::promise<void>& signal; ~Release(){try{signal.set_value();}catch(...) {}} };
    Release release_early{unblock};
    std::atomic<bool> ran{false};
    auto read=io->submit(1,[&ran]{ran=true;});
    BatchedStatisticsObserver controller;
    Release release{unblock};
    auto& slot=(controller.*member(SlotsTag{})).emplace_back();slot.request=42;
    slot.observer=std::make_unique<StatisticsObserver>();
    auto& prefix=slot.observer.get()->*member(PrefixTag{});
    prefix=std::make_unique<typename std::remove_reference_t<decltype(prefix)>::element_type>();
    prefix->ticket=read;
    prefix->deadline=std::chrono::steady_clock::now()+(timeout ? -std::chrono::seconds(1):std::chrono::seconds(30));
    if(timeout) {
        check(controller.poll_pending()==RAD_OK,"queued prefix timeout escaped as engine error");
        int status=RAD_OK;
        check(controller.take_request_refusal(&status)==42 && status==RAD_E_IO,"prefix timeout was not refused");
    }
    controller.retire(42);
    check(read.ready() && read.status()==StorageTaskStatus::Cancelled && !ran,
          "queued prefix read was not cancelled promptly");
    check(controller.poll_pending()==RAD_OK && !controller.holds_sequence(42) && !blocker.ready(),
          "queued prefix retained its slot behind unrelated I/O");
    check(io->pending_jobs()==1,"queued prefix retained queue quota");
}

static void running_prefix_drains_before_slot_release() {
    auto io=std::make_shared<StorageIoQueue>(1,64,1);
    std::promise<void> unblock,started;auto gate=unblock.get_future().share();
    auto read=io->submit(1,[gate,&started]{started.set_value();gate.wait();});
    started.get_future().wait();
    struct Release { std::promise<void>& signal; ~Release(){try{signal.set_value();}catch(...) {}} };
    BatchedStatisticsObserver controller;
    Release release{unblock};
    auto& slot=(controller.*member(SlotsTag{})).emplace_back();slot.request=42;
    slot.observer=std::make_unique<StatisticsObserver>();
    auto& prefix=slot.observer.get()->*member(PrefixTag{});
    prefix=std::make_unique<typename std::remove_reference_t<decltype(prefix)>::element_type>();
    prefix->ticket=read;prefix->deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    controller.retire(42);
    check(controller.poll_pending()==RAD_OK && controller.holds_sequence(42) && !read.ready(),
          "running prefix lost ownership before its I/O completed");
    unblock.set_value();read.wait();
    check(controller.poll_pending()==RAD_OK && !controller.holds_sequence(42),
          "completed prefix retained its request slot");
}

int main() {
    partial_rank_failure(false);partial_rank_failure(true);
    controller_failure(true,false);controller_failure(false,false);controller_failure(false,true);
    controller_failure(true,false,true);
    successful_finish_stays_pending();
    queued_cancel_releases_slot();
    queued_prefix_releases_slot(false);queued_prefix_releases_slot(true);
    running_prefix_drains_before_slot_release();
    std::cout<<"PASS storage recovery: isolated timeout/error, pending cancellation, multi-rank quota/error, peer progress, source preservation and fatal device timeout\n";
}
