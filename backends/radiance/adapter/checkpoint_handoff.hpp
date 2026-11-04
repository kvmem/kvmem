#pragma once
#include <memory>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <type_traits>
#include "kvmem/content_cache.hpp"

namespace kvmem::radiance {
// Select from the completed object's identity, after ownership has moved out
// of the pending slot. Never inspect that moved-from slot during assignment.
template<class Point>
void publish_retained_checkpoint(std::unique_ptr<Point> ready,
                                 std::unique_ptr<Point>& input,
                                 std::unique_ptr<Point>& fallback) {
    if(!ready || !ready->identity)
        throw std::invalid_argument("completed checkpoint needs an identity");
    if(ready->identity->fallback)fallback=std::move(ready);
    else input=std::move(ready);
}
// The caller proves native completion before this directory-only transaction.
// Allocate the destination before retiring anything. A cancelled/incomplete
// capture or allocation failure therefore leaves every published source intact.
template<class Directory, class Point>
bool publish_content_checkpoint(Directory& directory, Point&& ready,
                                const ContentGroupPublicationPlan& plan,
                                bool native_complete) {
    using Value = typename Directory::value_type;
    static_assert(std::is_nothrow_move_constructible<Value>::value &&
                  std::is_nothrow_move_assignable<Value>::value,
                  "native directory commit must not throw after allocation");
    if (!native_complete || !plan) return false;
    auto retired = plan.retired_slots;
    std::sort(retired.rbegin(), retired.rend());
    for (size_t i=0;i<retired.size();++i)
        if (retired[i]>=directory.size() || (i && retired[i]==retired[i-1]))
            throw std::invalid_argument("invalid retained checkpoint commit plan");
    directory.push_back(std::forward<Point>(ready));
    for (auto index:retired) directory.erase(directory.begin()+std::ptrdiff_t(index));
    return true;
}
} // namespace kvmem::radiance
