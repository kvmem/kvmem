#pragma once
#include "kvmem/memory_contract.hpp"
#include "kvmem/graph_recall.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kvmem::radiance {
struct QueryStatistics {
    std::vector<float> sums; // [layer, query_heads * head_dim]
    std::vector<uint32_t> token_counts;
    std::vector<std::vector<float>> tokens; // [layer][row, head, channel]
};
struct AcceptedStatistics {
    SessionIdentity session;
    uint32_t first=0, rows=0, key_width=0, query_width=0;
    std::vector<int> layers;
    std::vector<std::vector<float>> keys; // [layer][accepted row, local head, channel]
    QueryStatistics query;
};
// TP shards own consecutive head ranges. Concatenate each ROW, never complete
// flattened rank buffers. Query token counts describe the same logical rows on
// every rank and must agree; summing them would count one Query multiple times.
namespace detail {
inline AcceptedStatistics merge_rank_statistics(const std::vector<AcceptedStatistics>& ranks,
                                                std::vector<AcceptedStatistics>* consume) {
    auto check=[](bool ok){if(!ok)throw std::runtime_error("inconsistent accepted rank statistics");};
    check(!ranks.empty());
    AcceptedStatistics out;
    out.session=ranks[0].session;out.first=ranks[0].first;out.rows=ranks[0].rows;
    const auto& first=ranks[0];
    const auto layers=first.layers.size();check(layers>0 && first.query.token_counts.size()==layers);
    for(const auto& r:ranks) {
        check(r.session==out.session && r.first==out.first && r.rows==out.rows && r.layers==first.layers &&
            r.key_width && r.query_width && r.keys.size()==layers && r.query.tokens.size()==layers &&
            r.query.token_counts==first.query.token_counts && r.query.sums.size()==layers*uint64_t(r.query_width));
        check(r.key_width<=UINT32_MAX-out.key_width && r.query_width<=UINT32_MAX-out.query_width);
        out.key_width+=r.key_width;out.query_width+=r.query_width;
        for(size_t l=0;l<layers;++l) {
            check(r.keys[l].size()==uint64_t(out.rows)*r.key_width &&
                r.query.tokens[l].size()==uint64_t(first.query.token_counts[l])*r.query_width);
            for(float v:r.keys[l])check(std::isfinite(v));
            for(float v:r.query.tokens[l])check(std::isfinite(v));
        }
        for(float v:r.query.sums)check(std::isfinite(v));
    }
    // Validate before transferring ownership, including nonfinite/shape errors.
    // The single-rank result already has the requested row/head layout.
    if(consume && ranks.size()==1)return std::move(consume->front());
    out.layers=first.layers;out.query.token_counts=first.query.token_counts;
    out.keys.resize(layers);out.query.tokens.resize(layers);out.query.sums.reserve(layers*uint64_t(out.query_width));
    for(size_t l=0;l<layers;++l) {
        auto& keys=out.keys[l];keys.reserve(uint64_t(out.rows)*out.key_width);
        for(uint32_t row=0;row<out.rows;++row)for(const auto& r:ranks) {
            const auto first=r.keys[l].begin()+uint64_t(row)*r.key_width;
            keys.insert(keys.end(),first,first+r.key_width);
        }
        auto& tokens=out.query.tokens[l];tokens.reserve(uint64_t(out.query.token_counts[l])*out.query_width);
        for(uint32_t row=0;row<out.query.token_counts[l];++row)for(const auto& r:ranks) {
            const auto first=r.query.tokens[l].begin()+uint64_t(row)*r.query_width;
            tokens.insert(tokens.end(),first,first+r.query_width);
        }
        for(const auto& r:ranks) {
            const auto first=r.query.sums.begin()+l*r.query_width;
            out.query.sums.insert(out.query.sums.end(),first,first+r.query_width);
        }
    }
    return out;
}
} // namespace detail
inline AcceptedStatistics merge_rank_statistics(const std::vector<AcceptedStatistics>& ranks) {
    return detail::merge_rank_statistics(ranks,nullptr);
}
inline AcceptedStatistics merge_rank_statistics(std::vector<AcceptedStatistics>&& ranks) {
    return detail::merge_rank_statistics(ranks,&ranks);
}
// Native attention capture already averages its local heads. Weight by the
// actual local head count to recover one global average, preserving page IDs.
inline std::vector<AcceptedAttentionRow> merge_rank_attention(
    const std::vector<std::vector<AcceptedAttentionRow>>& ranks,const std::vector<uint32_t>& heads) {
    auto check=[](bool ok){if(!ok)throw std::runtime_error("inconsistent accepted rank attention");};
    check(!ranks.empty() && ranks.size()==heads.size());
    uint64_t total=0;for(auto n:heads){check(n>0 && total<=UINT64_MAX-n);total+=n;}
    auto out=ranks[0];
    for(auto& row:out)for(auto& mass:row.mass)mass.mass=0;
    for(size_t r=0;r<ranks.size();++r) {
        check(ranks[r].size()==out.size());
        for(size_t i=0;i<out.size();++i) {
            check(ranks[r][i].source_position==out[i].source_position && ranks[r][i].mass.size()==out[i].mass.size());
            double sum=0;
            for(size_t p=0;p<out[i].mass.size();++p) {
                const auto& src=ranks[r][i].mass[p];auto& dst=out[i].mass[p];
                check(src.target==dst.target && std::isfinite(src.mass) && src.mass>=0);
                sum+=src.mass;dst.mass+=src.mass*(double(heads[r])/double(total));
            }
            check(std::abs(sum-1)<1e-4);
        }
    }
    return out;
}
inline std::vector<AcceptedAttentionRow> merge_rank_attention(
    std::vector<std::vector<AcceptedAttentionRow>>&& ranks,const std::vector<uint32_t>& heads) {
    if(ranks.size()!=1)return merge_rank_attention(
        static_cast<const std::vector<std::vector<AcceptedAttentionRow>>&>(ranks),heads);
    if(heads.size()!=1 || !heads[0])throw std::runtime_error("inconsistent accepted rank attention");
    for(const auto& row:ranks.front()) {
        double sum=0;
        for(const auto& mass:row.mass) {
            if(!std::isfinite(mass.mass) || mass.mass<0)
                throw std::runtime_error("inconsistent accepted rank attention");
            sum+=mass.mass;
        }
        if(!(std::abs(sum-1)<1e-4))throw std::runtime_error("inconsistent accepted rank attention");
    }
    // The weighted merge starts from +0, including when a captured mass is -0.
    for(auto& row:ranks.front())for(auto& mass:row.mass)if(mass.mass==0)mass.mass=0;
    return std::move(ranks.front());
}
} // namespace kvmem::radiance
