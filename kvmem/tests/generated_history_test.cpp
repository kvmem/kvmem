#include "kvmem/generated_history.hpp"
#include <stdexcept>
#include <thread>

using Directory = kvmem::GeneratedHistory;

static void check(bool value) {
    if (!value) throw std::runtime_error("generated history invariant failed");
}

int main() {
    Directory cache(2);
    cache.publish("A", {"ab", {1, 2}});
    cache.publish("ABC", {"abcd", {1, 2, 3}});
    auto matches = cache.candidates("ABCuser", {1, 3});
    check(matches.size() == 2 && matches[0]->ids.size() == 3 && matches[1]->ids.size() == 2);
    check(cache.candidates("ABCuser", {2}).empty());
    check(cache.candidates("XBCuser", {1, 3}).empty());
    cache.publish("ABC", {"abcd", {9}});
    check(cache.candidates("ABCuser", {1, 3}).size() == 1);
    cache.publish("ABC", {"abcd", {9}});
    check(cache.candidates("ABCuser", {3}).empty());
    cache.publish("X", {"ab", {9}});
    check(cache.candidates("A", {1}).empty());
    check(matches[1]->ids == std::vector<int32_t>({1, 2})); // immutable reader survives eviction
    cache.mark_ambiguous("X");
    check(cache.candidates("X", {1}).empty());
    cache.mark_ambiguous("Y");
    cache.publish("Y", {"ok", {5}});
    check(cache.candidates("Y", {1}).empty());
    Directory tiny(2, 16);
    tiny.publish("x", {std::string(17, 'a'), {1}});
    check(tiny.candidates("x", {1}).empty());
    Directory raced;
    auto write = [&](int id) {
        for (int i = 0; i < 100; ++i) {
            raced.publish("branch", {"x", {id}});
            raced.candidates("branch", {6});
        }
    };
    std::thread one(write, 1), two(write, 2);
    one.join();
    two.join();
    check(raced.candidates("branch", {6}).empty());
}
