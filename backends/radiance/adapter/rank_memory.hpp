#pragma once
#include "kvmem/memory_contract.hpp"
#include "storage_work.hpp"
#include "storage_wait.hpp"
#include <algorithm>
#include <limits>

namespace kvmem::radiance {
// One logical view backed by consecutive head shards. Each child retains its
// native payload encoding and archive identity; the aggregate layout is used
// for whole-session planning/accounting, never to reinterpret a child's bytes.
// The caller holds ALL ranks at the same scheduler boundary through publish.
class RankMemory : public MemoryBackend {
    struct Shared {
        std::vector<std::shared_ptr<MemoryBackend>> ranks;
        MemoryDescriptor descriptor;
        bool pending=false;
        StorageWaiter* storage=nullptr;
    };
    std::shared_ptr<Shared> state_;
    static void check(bool ok,const char* why) {
        if(!ok)throw MemoryError(MemoryErrorCode::InvalidPlan,why);
    }
    static uint64_t add(uint64_t a,uint64_t b) {
        check(b<=UINT64_MAX-a,"rank payload size overflow");return a+b;
    }
    static bool same_view(const ExecutionView& a,const ExecutionView& b) {
        if(!(a.stamp==b.stamp) || a.visible_tokens!=b.visible_tokens || a.blocks.size()!=b.blocks.size())return false;
        for(size_t i=0;i<a.blocks.size();++i)
            if(!(a.blocks[i].block==b.blocks[i].block) || a.blocks[i].compact_start!=b.blocks[i].compact_start)return false;
        return true;
    }
    class Transfer final : public MemoryTransfer, public StorageWaiter {
        std::shared_ptr<Shared> owner_;
        std::vector<std::unique_ptr<MemoryTransfer>> children_;
        WorkingSetPlan plan_;
        enum class Phase {Prepared,Spilling,Spilled,Restoring,Ready,Published,Aborted};
        Phase phase_=Phase::Prepared;
        size_t published_=0;
        bool publishing_started_=false;
        std::vector<bool> complete_;
        std::vector<StorageWaiter*> storage_;
        std::exception_ptr storage_error_;
    public:
        Transfer(std::shared_ptr<Shared> state,WorkingSetPlan plan,
                 std::vector<std::unique_ptr<MemoryTransfer>> children)
            :owner_(std::move(state)),children_(std::move(children)),plan_(std::move(plan)),complete_(children_.size(),false) {
            storage_.reserve(children_.size());
            for(const auto& child:children_)storage_.push_back(dynamic_cast<StorageWaiter*>(child.get()));
            owner_->pending=true;owner_->storage=this;
        }
        ~Transfer() override {(void)abort();}
        void start_spill() override {
            check(phase_==Phase::Prepared,"rank spill phase order");phase_=Phase::Spilling;
            for(auto& child:children_)child->start_spill();
        }
        TransferPoll poll() override {
            check(phase_==Phase::Spilling || phase_==Phase::Spilled || phase_==Phase::Restoring || phase_==Phase::Ready,
                  "rank poll phase order");
            if(phase_==Phase::Spilled || phase_==Phase::Ready)return {TransferStatus::Complete,{}};
            if(storage_error_) {
                if(!storage_quiescent())return {TransferStatus::Pending,{}};
                std::rethrow_exception(storage_error_);
            }
            bool complete=true;
            for(size_t i=0;i<children_.size();++i) {
                if(complete_[i])continue;
                TransferPoll result;
                try { result=children_[i]->poll(); }
                catch(const StorageFailure&) { storage_error_=std::current_exception(); }
                catch(const MemoryError& error) {
                    if(error.code()!=MemoryErrorCode::BudgetExceeded)throw;
                    storage_error_=std::current_exception();
                }
                if(storage_error_) {
                    cancel_queued_storage();
                    if(!storage_quiescent())return {TransferStatus::Pending,{}};
                    std::rethrow_exception(storage_error_);
                }
                if(result.status==TransferStatus::Failed)return result;
                complete_[i]=result.status==TransferStatus::Complete;
                complete=complete && result.status==TransferStatus::Complete;
            }
            if(!complete)return {TransferStatus::Pending,{}};
            phase_=phase_==Phase::Spilling ? Phase::Spilled : Phase::Ready;
            return {TransferStatus::Complete,{}};
        }
        void start_restore() override {
            check(phase_==Phase::Spilled,"rank restore before complete spill");phase_=Phase::Restoring;
            std::fill(complete_.begin(),complete_.end(),false);
            for(auto& child:children_)child->start_restore();
        }
        void publish(const ExecutionView& view) override {
            check(phase_==Phase::Ready && same_view(view,plan_.next),"rank publication is not ready or view changed");
            for(const auto& child:owner_->ranks)
                check(child->stamp()==plan_.expected,"rank changed before publication");
            // No engine work can run between these publications. If any child
            // fails, abort drains every rank and invalidates the whole session;
            // a mixture of old and new views is never eligible for execution.
            publishing_started_=true;
            for(auto& child:children_){child->publish(view);++published_;}
            phase_=Phase::Published;owner_->pending=false;owner_->storage=nullptr;
        }
        bool awaiting_storage() const noexcept override {
            for(const auto* child:storage_)if(child && child->awaiting_storage())return true;
            return false;
        }
        bool cpu_io_pending() const noexcept override {
            for(const auto* child:storage_)if(child && child->cpu_io_pending())return true;
            return false;
        }
        bool storage_quiescent() override {
            bool ready=true;
            for(auto* child:storage_)if(child && !child->storage_quiescent())ready=false;
            return ready;
        }
        void cancel_queued_storage() noexcept override {
            for(auto* child:storage_)if(child)child->cancel_queued_storage();
        }
        std::exception_ptr storage_error() const noexcept override {return storage_error_;}
        AbortResult abort() noexcept override {
            if(phase_==Phase::Published || phase_==Phase::Aborted) {
                for(const auto& child:owner_->ranks)if(!child->stamp().valid)return AbortResult::SessionInvalidated;
                return AbortResult::PreviousViewPreserved;
            }
            bool valid=!publishing_started_;
            for(auto& child:children_)if(child->abort()!=AbortResult::PreviousViewPreserved)valid=false;
            if(!valid)for(auto& child:owner_->ranks)child->invalidate();
            phase_=Phase::Aborted;owner_->pending=false;owner_->storage=nullptr;
            return valid ? AbortResult::PreviousViewPreserved : AbortResult::SessionInvalidated;
        }
    };
protected:
    const std::vector<std::shared_ptr<MemoryBackend>>& ranks() const noexcept {return state_->ranks;}
public:
    RankMemory(std::vector<std::shared_ptr<MemoryBackend>> ranks,BackendIdentity identity)
        :state_(std::make_shared<Shared>()) {
        check(!ranks.empty(),"empty rank memory");
        for(const auto& rank:ranks)check(bool(rank),"missing rank memory");
        state_->ranks=std::move(ranks);
        auto& d=state_->descriptor;d=state_->ranks[0]->descriptor();d.validate();d.identity=std::move(identity);
        const auto first=d;
        d.layout.format="radiance-rank-head-shards-v1:"+std::to_string(state_->ranks.size());
        d.layout.record_bytes=0;d.layout.alignment=1;
        for(auto& plane:d.layout.planes){plane.bytes=0;plane.alignment=1;plane.token_stride=0;}
        for(const auto& rank:state_->ranks) {
            const auto child=rank->descriptor();child.validate();
            const auto& c=child.capabilities;const auto& a=first.capabilities;
            check(child.identity.backend==d.identity.backend && child.identity.revision==d.identity.revision &&
                child.identity.model==d.identity.model && child.layout.version==first.layout.version &&
                child.layout.block_tokens==first.layout.block_tokens &&
                child.layout.planes.size()==d.layout.planes.size() && c.page_tokens==a.page_tokens &&
                c.max_position==a.max_position && c.native_payload==a.native_payload && c.sparse_view==a.sparse_view &&
                c.original_rope_positions==a.original_rope_positions && c.pre_rope_statistics==a.pre_rope_statistics &&
                c.state_checkpoint==a.state_checkpoint && c.asynchronous_transfer==a.asynchronous_transfer,
                "incompatible rank memory descriptors");
            d.layout.format+="/"+std::to_string(child.layout.format.size())+":"+child.layout.format;
            uint64_t payload=0;
            for(size_t i=0;i<d.layout.planes.size();++i) {
                auto& dst=d.layout.planes[i];const auto& src=child.layout.planes[i];
                check(dst.layer==src.layer && dst.kind==src.kind && src.offset==payload,
                      "rank plane ordering/padding is unsupported");
                payload=add(payload,src.bytes);dst.bytes=add(dst.bytes,src.bytes);
            }
            check(payload==child.layout.record_bytes,"rank trailing padding is unsupported");
        }
        for(auto& plane:d.layout.planes){plane.offset=d.layout.record_bytes;d.layout.record_bytes=add(d.layout.record_bytes,plane.bytes);}
        // A forked retained catalog has no live native sequence until its
        // prefix is restored. Validate shared stamps here; snapshot() also
        // validates actual ownership when the caller rebinds/executes it.
        d.validate();check(stamp().valid,"rank memory stamps differ or are invalid");
    }
    bool awaiting_storage() const noexcept {return state_->storage && state_->storage->awaiting_storage();}
    bool cpu_io_pending() const noexcept {return state_->storage && state_->storage->cpu_io_pending();}
    bool storage_quiescent() const {return !state_->storage || state_->storage->storage_quiescent();}
    void cancel_queued_storage() noexcept {if(state_->storage)state_->storage->cancel_queued_storage();}
    std::exception_ptr storage_error() const noexcept {return state_->storage ? state_->storage->storage_error() : std::exception_ptr{};}
    MemoryDescriptor descriptor() const override {return state_->descriptor;}
    MemoryStamp stamp() const noexcept override {
        auto out=state_->ranks[0]->stamp();
        for(const auto& child:state_->ranks)if(!(child->stamp()==out)){out.valid=false;break;}
        return out;
    }
    MemorySnapshot snapshot() const override {
        auto out=state_->ranks[0]->snapshot();
        for(size_t rank=1;rank<state_->ranks.size();++rank) {
            const auto next=state_->ranks[rank]->snapshot();
            check(next.stamp==out.stamp && next.resident==out.resident && next.blocks.size()==out.blocks.size(),
                  "rank views diverged");
            for(size_t i=0;i<out.blocks.size();++i) {
                check(next.blocks[i].block==out.blocks[i].block,"rank catalogs diverged");
                // A completed spill remains useful after cancellation. A block
                // is globally archived only when every shard has that version.
                out.blocks[i].host_version=std::min(out.blocks[i].host_version,next.blocks[i].host_version);
            }
        }
        return out;
    }
    std::unique_ptr<MemoryTransfer> prepare(const WorkingSetPlan& plan) override {
        check(!state_->pending,"overlapping rank transaction");
        std::vector<uint64_t> selected;for(const auto& b:plan.next.blocks)selected.push_back(b.block.id);
        const auto expected=plan_working_set(descriptor(),snapshot(),selected,plan.mandatory,plan.budget);
        check(expected.expected==plan.expected && same_view(expected.next,plan.next) && expected.evict==plan.evict &&
            expected.spill==plan.spill && expected.restore==plan.restore &&
            expected.device_tokens_required==plan.device_tokens_required &&
            expected.host_payload_bytes_required==plan.host_payload_bytes_required,"rank plan changed");
        std::vector<std::unique_ptr<MemoryTransfer>> children;
        try {
            for(const auto& child:state_->ranks) {
                auto child_plan=plan_working_set(child->descriptor(),child->snapshot(),selected,plan.mandatory,plan.budget);
                check(same_view(child_plan.next,plan.next),"rank next view diverged");
                auto transfer=child->prepare(child_plan);check(bool(transfer),"rank prepare returned no transaction");
                children.push_back(std::move(transfer));
            }
            return std::make_unique<Transfer>(state_,plan,std::move(children));
        } catch(...) {
            bool valid=true;for(auto& child:children)if(child->abort()!=AbortResult::PreviousViewPreserved)valid=false;
            if(!valid)invalidate();throw;
        }
    }
    void invalidate() noexcept override {for(auto& child:state_->ranks)child->invalidate();}
};
} // namespace kvmem::radiance
