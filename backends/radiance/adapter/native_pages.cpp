#include "native_pages.hpp"
#include "mem/kv.h"
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace kvmem::radiance {
namespace {
[[noreturn]] void invalid(const char* message) {
    throw MemoryError(MemoryErrorCode::InvalidPlan, message);
}
uint64_t add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) invalid("Radiance page offset overflow");
    return a + b;
}
uint64_t mul(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) invalid("Radiance page size overflow");
    return a * b;
}
const rad::KVGroupPlan& find_plan(const std::vector<rad::KVGroupPlan>& plans, int32_t id) {
    const rad::KVGroupPlan* found = nullptr;
    for (const auto& p : plans) if (p.index == id) {
        if (found) invalid("duplicate Radiance group plan");
        found = &p;
    }
    if (!found) invalid("missing Radiance group plan");
    return *found;
}
}

NativePages::NativePages(const std::vector<rad::KVGroupPlan>& plans,
                         const std::vector<rad::KVGroupInfo>& info,
                         uint32_t tokens, const std::string& encoding) {
    if (!tokens || encoding.empty()) invalid("missing Radiance page encoding/selection size");
    layout_.version = 1;
    layout_.block_tokens = tokens;
    layout_.alignment = 1; // byte-preserving staging, not a kernel tensor alignment promise
    std::set<int32_t> ids;
    std::set<uint32_t> layer_ids;
    std::ostringstream format;
    format << "radiance-native-v1/" << encoding.size() << ':' << encoding;
    for (const auto& p : plans) {
        if (!ids.insert(p.index).second) invalid("duplicate Radiance group plan");
        if (!p.paged() || p.zero || p.kind == RAD_KV_WINDOW) continue;
        if (p.index < 0 || static_cast<size_t>(p.index) >= info.size())
            invalid("Radiance group is missing its declaration");
        const auto& g = info[p.index];
        if (g.zero || g.decl.kind != p.kind || p.n_layers <= 0 ||
            g.layers.size() != static_cast<size_t>(p.n_layers) ||
            p.block_size <= 0 || p.block_size != g.block_size ||
            tokens % static_cast<uint64_t>(p.block_size) ||
            p.bytes_per_block <= 0 || p.bytes_per_block != g.bytes_per_block ||
            p.bytes_per_block % p.n_layers || p.n_blocks <= 0)
            invalid("inconsistent Radiance native-page geometry");
        const uint64_t fragment = p.bytes_per_block / p.n_layers;
        const uint64_t count = tokens / p.block_size;
        const uint64_t stride = mul(static_cast<uint64_t>(p.n_blocks), fragment);
        if (p.pool_bytes < 0 || mul(stride, p.n_layers) > static_cast<uint64_t>(p.pool_bytes))
            invalid("Radiance group carve is smaller than its pages");
        groups_.push_back({p.index, p.kind, static_cast<uint32_t>(count),
                          static_cast<uint64_t>(p.block_size), static_cast<uint64_t>(p.n_layers),
                          fragment, static_cast<uint64_t>(p.n_blocks), stride, layout_.record_bytes});
        format << "/g" << p.index << ':' << p.kind << ':' << g.decl.dtype << ':'
               << g.decl.n_head_kv << ':' << g.decl.head_dim << ':' << p.block_size << ':' << fragment;
        for (const int layer : g.layers) {
            if (layer < 0 || !layer_ids.insert(static_cast<uint32_t>(layer)).second)
                invalid("ambiguous Radiance attention-layer identity");
            const auto bytes = mul(fragment, count);
            layout_.planes.push_back({static_cast<uint32_t>(layer), PayloadPlaneKind::PackedKeyValue,
                                      layout_.record_bytes, bytes, 1, 0});
            layout_.record_bytes = add(layout_.record_bytes, bytes);
        }
    }
    layout_.format = format.str();
    // A zero-cache TP rank needs coordinated no-op transactions, not a fake page.
    if (groups_.empty()) throw MemoryError(MemoryErrorCode::Unsupported,
                                           "rank has no paged KV payload");
    layout_.validate();
}

std::vector<CopySpan> NativePages::spans(const std::vector<rad::KVGroupPlan>& current,
                                        const std::vector<GroupPages>& pages) const {
    if (pages.size() != groups_.size()) invalid("page bindings do not cover the native layout");
    std::map<int32_t, const GroupPages*> bindings;
    for (const auto& b : pages)
        if (!bindings.emplace(b.group, &b).second) invalid("duplicate page binding");
    std::vector<CopySpan> result;
    for (const auto& g : groups_) {
        const auto& p = find_plan(current, g.id);
        if (!p.paged() || p.kind != g.kind || p.zero || p.n_layers != static_cast<int64_t>(g.layers) ||
            p.block_size != static_cast<int64_t>(g.page_tokens) ||
            p.n_blocks != static_cast<int64_t>(g.capacity) || p.bytes_per_block <= 0 ||
            static_cast<uint64_t>(p.bytes_per_block) != mul(g.layers, g.fragment))
            throw MemoryError(MemoryErrorCode::Stale, "Radiance pool geometry changed");
        if (p.pool_bytes < 0 || static_cast<uint64_t>(p.pool_bytes) < mul(g.layer_stride, g.layers))
            invalid("Radiance pool backing descriptor is too small");
        const auto it = bindings.find(g.id);
        if (it == bindings.end()) invalid("missing native group page IDs");
        const auto& b = *it->second;
        if (b.pages.size() != g.pages_per_record || b.backed_blocks > g.capacity)
            invalid("invalid Radiance page count/backed capacity");
        std::set<int32_t> seen;
        for (const auto id : b.pages)
            if (id < 0 || static_cast<uint64_t>(id) >= b.backed_blocks || !seen.insert(id).second)
                invalid("Radiance page ID is unbacked, negative, or repeated");
        for (uint64_t layer = 0; layer < g.layers; ++layer) {
            for (size_t page = 0; page < b.pages.size(); ++page) {
                const auto source = add(mul(layer, g.layer_stride), mul(b.pages[page], g.fragment));
                const auto target = add(g.record_offset,
                    mul(add(mul(layer, g.pages_per_record), page), g.fragment));
                result.push_back({g.id, source, target, g.fragment});
            }
        }
    }
    return result;
}
} // namespace kvmem::radiance
