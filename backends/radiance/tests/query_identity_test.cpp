#include "query_identity.hpp"
#include <stdexcept>
#include <iostream>

int main() {
    using kvmem::radiance::matches_query_identity;
    const std::vector<int32_t> original{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int32_t> query{3, 4, 5, 6};
    auto check = [](bool ok) { if (!ok) throw std::runtime_error("query identity check failed"); };
    check(matches_query_identity(query, 2, 6, original, 2, 6));
    auto changed = original;
    changed[5] = 99; // beyond a checkpoint at4, but INSIDE the frozen query
    check(std::equal(original.begin(), original.begin() + 4, changed.begin()));
    check(!matches_query_identity(query, 2, 6, changed, 2, 6));
    changed = original; changed[7] = 99; // legitimate new tool suffix
    check(matches_query_identity(query, 2, 6, changed, 2, 6));
    check(!matches_query_identity(query, 2, 6, original, 3, 7));
    check(!matches_query_identity(query, -1, -1, original, -1, -1));
    check(!matches_query_identity(query, 2, 9, original, 2, 9));
    check(!matches_query_identity({}, 2, 6, original, 2, 6));
    std::cout << "PASS full query identity, aligned partial prefix, suffix changes, invalid spans\n";
}
