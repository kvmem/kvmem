#include "mm/vision_cache.h"
#include <stdexcept>
#include <new>
#include <cstdlib>

static int fail_allocation_after=-1;
void* operator new(std::size_t size) {
    if(fail_allocation_after==0)throw std::bad_alloc();
    if(fail_allocation_after>0)--fail_allocation_after;
    if(void* p=std::malloc(size ? size : 1))return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
#define CHECK(x) do { if(!(x)) throw std::runtime_error(#x); } while(false)
using namespace rad::mm;
static Item image(uint16_t pixel) {
    Item i;i.key_hi=1;i.key_lo=2;i.grid_t=1;i.grid_h=2;i.grid_w=2;
    i.n_patch=4;i.patch_dim=1;i.n_seg=1;i.seg_rows=1;i.rows_h=1;i.rows_w=1;
    i.pixels={pixel,pixel,pixel,pixel};i.embd={uint16_t(pixel+1)};i.encoded=true;return i;
}
static void budget(const char* value) {
#ifdef _WIN32
    _putenv_s("KVMEM_RAD_VISION_CACHE_MIB",value ? value : "");
#else
    if(value)setenv("KVMEM_RAD_VISION_CACHE_MIB",value,1);else unsetenv("KVMEM_RAD_VISION_CACHE_MIB");
#endif
}
int main() {
    budget("0");CHECK(vision_cache_bytes()==0);
    budget("64");CHECK(vision_cache_bytes()==size_t(64)*1024*1024);
    for(const char* bad:{"-1","65537","wrong","1x"}) {
        budget(bad);bool refused=false;
        try{(void)vision_cache_bytes();}catch(const std::invalid_argument&){refused=true;}
        CHECK(refused);
    }
    budget(nullptr);CHECK(vision_cache_bytes()==size_t(256)*1024*1024);
    VisionCache off(0);auto a=image(3);off.put(a);CHECK(!off.get(a));CHECK(off.bytes()==0);
    VisionCache c(20);c.put(a);CHECK(c.bytes()==10);
    auto hit=image(3);hit.embd.clear();hit.encoded=false;CHECK(c.get(hit));CHECK(hit.encoded && hit.embd[0]==4);
    auto collision=image(4);CHECK(!c.get(collision)); // same 128-bit key, different exact patches
    auto geometry=image(3);geometry.grid_h=1;geometry.grid_w=4;CHECK(!c.get(geometry));
    c.put(a);CHECK(c.bytes()==10);c.put(collision);CHECK(c.bytes()==20);
    CHECK(c.get(a));auto third=image(5);c.put(third);CHECK(c.bytes()==20);
    CHECK(!c.get(collision));CHECK(c.get(a));CHECK(c.get(third));
    VisionCache small(9);small.put(a);CHECK(small.bytes()==0);
    // Cache OOM is optional and must leave the previous entries and payload charge intact.
    auto missing=image(7);const auto charged=c.bytes();
    fail_allocation_after=0;const bool inserted=c.put(missing);fail_allocation_after=-1;
    CHECK(!inserted && c.bytes()==charged);CHECK(c.get(third));CHECK(c.get(a));
    auto oom_hit=image(3);std::vector<uint16_t>().swap(oom_hit.embd);oom_hit.encoded=false;
    fail_allocation_after=0;const bool got=c.get(oom_hit);fail_allocation_after=-1;
    CHECK(!got && !oom_hit.encoded && c.bytes()==charged);CHECK(c.get(oom_hit));
    bool saw_failure=false,saw_success=false;
    for(int point=0;point<16;++point) {
        VisionCache transaction(20);auto old_a=image(3),old_b=image(5),replacement=image(7);
        CHECK(transaction.put(old_a));CHECK(transaction.put(old_b));
        fail_allocation_after=point;const bool success=transaction.put(replacement);fail_allocation_after=-1;
        CHECK(transaction.bytes()<=20);
        if(success){saw_success=true;CHECK(transaction.get(replacement));break;}
        saw_failure=true;CHECK(transaction.bytes()==20);CHECK(transaction.get(old_a));CHECK(transaction.get(old_b));
    }
    CHECK(saw_failure && saw_success);
    // Cached output is immutable from the caller's point of view.
    a.embd[0]=999;CHECK(c.get(a));CHECK(a.embd[0]==4);
}
