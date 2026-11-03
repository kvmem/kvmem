#pragma once
#include <memory>
#include <stdexcept>
#include <utility>

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
} // namespace kvmem::radiance
