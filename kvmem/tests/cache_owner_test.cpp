#include "kvmem/cache_owner.hpp"
#include <iostream>
#include <stdexcept>
using namespace kvmem;
static void check(bool value){if(!value)throw std::runtime_error("cache owner contract failed");}
int main(){
    check(!resolve_cache_owner({}));
    check(!resolve_cache_owner({{},"",std::nullopt,{}}));
    check(resolve_cache_owner({{},"A",{},"legacy"})=="A");
    check(resolve_cache_owner({{},{},"A",{}})=="A");
    check(resolve_cache_owner({"explicit","A","A","legacy"})=="explicit");
    check(resolve_cache_owner({{},{},{},"legacy"})=="legacy");
    check(!resolve_cache_owner({{},{},{},std::string(257,'x')}));
    for(const auto& hints:{CacheOwnerHints{{},"A","B",{}},CacheOwnerHints{"",{},{},{}},
        CacheOwnerHints{{},std::string(257,'x'),{},{}}}){
        bool rejected=false;try{(void)resolve_cache_owner(hints);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
    }
    auto a=std::string(128,'x')+"A",b=std::string(128,'x')+"B";
    check(resolve_cache_owner({{},a,{},{}})!=resolve_cache_owner({{},b,{},{}}));
    check(may_discover_shared_prefix(true));check(!may_discover_shared_prefix(false));
    std::cout<<"PASS cache owner: anonymous, aliases, priority, conflicts, full identity and Query discovery\n";
}
