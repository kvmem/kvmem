#include "kvmem/graph_recall.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace kvmem {
void GraphStatisticsBudget::acquire(size_t bytes) {
    auto used=used_.load();
    do {
        if(bytes>limit_ || used>limit_-bytes) throw std::length_error("graph statistics budget exceeded before allocation");
    } while(!used_.compare_exchange_weak(used,used+bytes));
}
class GraphBudgetLease {
public:
    explicit GraphBudgetLease(std::shared_ptr<GraphStatisticsBudget> budget):budget_(std::move(budget)) {
        if(budget_->parent_) parent_=std::make_unique<GraphBudgetLease>(budget_->parent_);
    }
    ~GraphBudgetLease(){budget_->release(bytes_);}
    void resize(size_t bytes) {
        const auto old=bytes_;
        if(bytes>old) {
            if(parent_) parent_->resize(bytes);
            try {budget_->acquire(bytes-old);}
            catch(...) {if(parent_) parent_->resize(old);throw;}
        } else {
            budget_->release(old-bytes);
            if(parent_) parent_->resize(bytes);
        }
        bytes_=bytes;
    }
private:
    std::shared_ptr<GraphStatisticsBudget> budget_;
    std::unique_ptr<GraphBudgetLease> parent_;
    size_t bytes_=0;
};
GraphSnapshot::GraphSnapshot(const GraphSnapshot& other):budget_(other.budget_),feedback_bytes_(other.feedback_bytes_) {
    if(budget_) {
        lease_=std::make_shared<GraphBudgetLease>(budget_);
        lease_->resize(other.bytes()+feedback_bytes_);
    }
    accepted_through=other.accepted_through;completed_through=other.completed_through;
    revision=other.revision;epoch=other.epoch;block_tokens=other.block_tokens;
    chunks=other.chunks;partial=other.partial;messages=other.messages;
}
GraphSnapshot& GraphSnapshot::operator=(GraphSnapshot&& other) noexcept {
    if(this==&other) return *this;
    using std::swap;
    // Keep both leases alive until their corresponding vector storage retires.
    swap(budget_,other.budget_);swap(lease_,other.lease_);swap(feedback_bytes_,other.feedback_bytes_);
    swap(accepted_through,other.accepted_through);swap(completed_through,other.completed_through);
    swap(revision,other.revision);swap(epoch,other.epoch);swap(block_tokens,other.block_tokens);
    chunks.swap(other.chunks);swap(partial,other.partial);messages.swap(other.messages);
    return *this;
}
void GraphSnapshot::set_budget(std::shared_ptr<GraphStatisticsBudget> budget) {
    if(!budget) {lease_.reset();budget_.reset();return;}
    auto lease=std::make_shared<GraphBudgetLease>(budget);
    lease->resize(bytes()+feedback_bytes_);
    budget_=std::move(budget);lease_=std::move(lease);
}
void GraphSnapshot::reserve_extra(size_t bytes) {
    if(lease_) {
        if(bytes>SIZE_MAX-this->bytes()-feedback_bytes_) throw std::length_error("graph accounting overflow");
        lease_->resize(this->bytes()+feedback_bytes_+bytes);
    }
}
void GraphSnapshot::include_feedback(size_t bytes) {
    if(lease_) lease_->resize(this->bytes()+bytes);
    feedback_bytes_=bytes;
}
void GraphSnapshot::set_messages(const std::vector<GraphMessageSpan>& spans) {
    reserve_extra(sizeof(spans)+spans.size()*sizeof(GraphMessageSpan));
    messages=std::make_shared<const std::vector<GraphMessageSpan>>(spans);
    reserve_extra(0);
}
void GraphSnapshot::set_messages(std::shared_ptr<const std::vector<GraphMessageSpan>> spans) {
    if(messages==spans) return;
    reserve_extra(spans ? sizeof(*spans)+spans->capacity()*sizeof(GraphMessageSpan) : 0);
    messages=std::move(spans);reserve_extra(0);
}
namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void finite_mass(double value) {
    check(std::isfinite(value) && value >= 0, "invalid attention mass");
}
void decay_check(double decay) {
    check(std::isfinite(decay) && decay >= 0 && decay <= 1, "invalid attention decay");
}
std::vector<GraphEdge> normalized_mass(const std::vector<GraphEdge>& input) {
    std::map<uint64_t, double> merged;
    double total = 0;
    for (const auto& edge : input) {
        finite_mass(edge.mass);
        merged[edge.target] += edge.mass;
        total += edge.mass;
    }
    check(std::isfinite(total) && std::abs(total - 1) <= .005,
          "attention sample does not cover its complete visible window");
    std::vector<GraphEdge> result;
    result.reserve(merged.size());
    for (const auto& entry : merged) result.push_back({entry.first, entry.second / total});
    return result;
}
void append_row(GraphSnapshot& next, GraphRow row) {
    constexpr size_t chunk_rows = 256;
    std::vector<GraphRow> chunk;
    if (!next.chunks.empty() && next.chunks.back()->size() < chunk_rows) {
        chunk = *next.chunks.back();
        next.chunks.pop_back();
    }
    chunk.push_back(std::move(row));
    next.chunks.push_back(std::make_shared<const std::vector<GraphRow>>(std::move(chunk)));
}
void seal(GraphSnapshot& next, uint32_t topk) {
    if (next.partial.samples) {
        GraphRow row{next.partial.source, next.partial.samples, next.partial.mass};
        std::sort(row.edges.begin(), row.edges.end(), [](const auto& a, const auto& b) {
            return a.mass != b.mass ? a.mass > b.mass : a.target < b.target;
        });
        if (row.edges.size() > topk) row.edges.resize(topk);
        for (auto& edge : row.edges) edge.mass /= row.samples;
        append_row(next, std::move(row));
    }
    next.partial = {};
}
double span_coefficient(const GraphSnapshot& graph, uint64_t a, uint64_t b, double cross) {
    if(!graph.messages) return 1;
    const auto begin = std::min(a, b);
    const auto end = std::max(a, b) + graph.block_tokens;
    for (const auto& span : *graph.messages) {
        if (span.begin <= begin && end <= span.end) return 1;
    }
    for (const auto& span : *graph.messages) {
        if ((begin < span.begin && span.begin < end) ||
            (span.end_known && begin < span.end && span.end < end)) return cross;
    }
    return 1;
}
template<class Value> void validate_pairs(const std::vector<std::pair<uint64_t, Value>>& values) {
    uint64_t previous = 0;
    bool first = true;
    for (const auto& item : values) {
        check(first || previous < item.first, "attention history keys are not distinct and ordered");
        first = false;
        previous = item.first;
    }
}
template<class Value> Value lookup(const std::vector<std::pair<uint64_t, Value>>& values, uint64_t key) {
    const auto it = std::lower_bound(values.begin(), values.end(), key,
        [](const auto& item, uint64_t k) { return item.first < k; });
    return it != values.end() && it->first == key ? it->second : Value{};
}
template<class Value> Value& insert(std::vector<std::pair<uint64_t, Value>>& values, uint64_t key) {
    auto it = std::lower_bound(values.begin(), values.end(), key,
        [](const auto& item, uint64_t k) { return item.first < k; });
    if (it == values.end() || it->first != key) it = values.insert(it, {key, {}});
    return it->second;
}
} // namespace

QueryScoreFrame query_score_frame(uint64_t accepted, uint32_t block_tokens,
    uint32_t budget, uint32_t sink, uint32_t recent, const std::vector<uint32_t>& fixed) {
    check(block_tokens && budget, "invalid Query score budget or block size");
    const uint64_t n = accepted / block_tokens + (accepted % block_tokens != 0);
    check(n <= UINT32_MAX, "Query index exceeds logical page capacity");
    QueryScoreFrame frame;
    frame.accepted_through = accepted;
    frame.starts.reserve(size_t(n));
    frame.counts.reserve(size_t(n));
    frame.score_mask.assign(size_t(n), 1);
    frame.transport_mask.assign(size_t(n), n > budget);
    for (uint64_t i = 0; i < n; ++i) {
        frame.starts.push_back(i * block_tokens);
        frame.counts.push_back(uint32_t(std::min<uint64_t>(block_tokens, accepted-i*block_tokens)));
    }
    if (n > budget && uint64_t(sink) + recent < n) {
        for (uint64_t i = 0; i < std::min<uint64_t>(sink, n); ++i) frame.score_mask[size_t(i)] = 0;
        for (uint64_t i = n-std::min<uint64_t>(recent, n); i < n; ++i) frame.score_mask[size_t(i)] = 0;
    }
    for (auto id : fixed) {
        check(id < n, "fixed Query block is outside the accepted index");
        frame.transport_mask[id] = 0;
    }
    return frame;
}

float recall_fp16(float value) {
    check(std::isfinite(value), "nonfinite recall operand");
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const int exponent = int((bits >> 23) & 255)-127;
    uint32_t rounded = 0;
    if (exponent >= -14) {
        check(exponent <= 15, "recall operand exceeds FP16 range");
        rounded = (bits & 0x7fffffffu) + 0xfffu + ((bits >> 13) & 1u);
        rounded &= 0x7fffe000u;
        check(rounded < 0x47800000u, "recall operand rounds to FP16 infinity");
    } else if (exponent >= -25) {
        const uint32_t mantissa = (bits & 0x7fffffu) | 0x800000u;
        const int shift = -exponent-1;
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((uint32_t(1) << shift)-1);
        const uint32_t halfway = uint32_t(1) << (shift-1);
        half += remainder > halfway || (remainder == halfway && (half & 1));
        value = std::ldexp(float(half), -24);
        return sign ? -value : value;
    }
    rounded |= sign;
    std::memcpy(&value, &rounded, sizeof(value));
    return value;
}

std::vector<double> page_softmax_scores(const RawKvStore& keys, PageQueryView q,
                                      const QueryScoreFrame& frame, GqaScoreAggregation aggregation) {
    check(aggregation == GqaScoreAggregation::Mean || aggregation == GqaScoreAggregation::Max,
          "invalid page softmax GQA aggregation");
    const size_t n = frame.starts.size();
    check((q.data || q.layer_data) && q.layer_rows && q.layers && q.rows && q.head_dim && q.kv_heads &&
          q.query_heads && q.query_heads % q.kv_heads == 0 &&
          q.layers == keys.config().n_layer &&
          uint64_t(q.kv_heads)*q.head_dim == keys.config().n_embd_k &&
          frame.counts.size() == n && frame.score_mask.size() == n &&
          frame.transport_mask.size() == n, "page softmax geometry mismatch");
    std::vector<size_t> enabled;
    for (size_t b = 0; b < n; ++b) {
        check(frame.counts[b] && frame.starts[b] % keys.config().block_tokens == 0 &&
              frame.starts[b] <= frame.accepted_through &&
              frame.counts[b] <= frame.accepted_through-frame.starts[b], "invalid accepted Query extent");
        if (frame.score_mask[b]) enabled.push_back(b);
    }
    std::vector<double> result(n, 0);
    if (enabled.empty()) return result;
    uint32_t active_layers = 0;
    for (uint32_t l = 0; l < q.layers; ++l) {
        check(q.layer_rows[l] <= q.rows, "page softmax row count exceeds its storage");
        active_layers += q.layer_rows[l] != 0;
    }
    if (!active_layers) return result;
    const size_t kw = size_t(q.kv_heads)*q.head_dim, qw = size_t(q.query_heads)*q.head_dim;
    std::vector<float> means(enabled.size()*kw), key(kw), query(q.head_dim);
    std::vector<double> logits(enabled.size());
    std::vector<double> group_max(aggregation == GqaScoreAggregation::Max ? enabled.size() : 0);
    const double scale = q.attention_scale > 0 ? q.attention_scale : 1/std::sqrt(double(q.head_dim));
    for (uint32_t l = 0; l < q.layers; ++l) {
        if (!q.layer_rows[l]) continue;
        for (size_t j = 0; j < enabled.size(); ++j) {
            const auto block = uint32_t(frame.starts[enabled[j]]/keys.config().block_tokens);
            check(keys.has_block(block), "page softmax index lacks an accepted block");
            keys.mean_k(block, l, key.data());
            for (size_t d = 0; d < kw; ++d) means[j*kw+d] = recall_fp16(key[d]);
        }
        for (uint32_t t = 0; t < q.layer_rows[l]; ++t) for (uint32_t h = 0; h < q.query_heads; ++h) {
            const auto group = q.query_heads/q.kv_heads;
            if (aggregation == GqaScoreAggregation::Max && h%group == 0)
                std::fill(group_max.begin(),group_max.end(),0.0);
            const auto* base = q.layer_data ? q.layer_data[l] : q.data+size_t(l)*q.rows*qw;
            check(base != nullptr, "missing active Query layer storage");
            const auto* row = base + size_t(t)*qw + size_t(h)*q.head_dim;
            for (uint32_t d = 0; d < q.head_dim; ++d) query[d] = recall_fp16(row[d]);
            double maximum = -std::numeric_limits<double>::infinity();
            const auto head = h/(q.query_heads/q.kv_heads);
            for (size_t j = 0; j < enabled.size(); ++j) {
                double dot = 0;
                for (uint32_t d = 0; d < q.head_dim; ++d) dot += double(query[d])*means[j*kw+head*q.head_dim+d];
                maximum = std::max(maximum, logits[j] = dot*scale);
            }
            double denominator = 0;
            for (auto& value : logits) denominator += value = std::exp(value-maximum);
            if (aggregation == GqaScoreAggregation::Mean) {
                for (size_t j = 0; j < enabled.size(); ++j)
                    result[enabled[j]] += logits[j]/denominator/(double(active_layers)*q.query_heads);
            } else {
                for (size_t j = 0; j < enabled.size(); ++j)
                    group_max[j] = std::max(group_max[j],logits[j]/denominator);
                if (h%group == group-1) {
                    double mass = 0;
                    for (auto value : group_max) mass += value;
                    for (size_t j = 0; j < enabled.size(); ++j)
                        result[enabled[j]] += group_max[j]/mass/(double(active_layers)*q.kv_heads);
                }
            }
        }
    }
    return result;
}

std::vector<double> mean_query_cosine_scores(const RawKvStore& keys, PageQueryView q,
                                           uint32_t blocks, GqaScoreAggregation aggregation) {
    check((q.data || q.layer_data) && q.layer_rows && q.layers && q.rows && q.query_heads &&
          q.kv_heads && q.head_dim && q.query_heads%q.kv_heads == 0 &&
          q.layers == keys.config().n_layer && uint64_t(q.kv_heads)*q.head_dim == keys.config().n_embd_k,
          "mean Query geometry mismatch");
    const size_t width = size_t(q.query_heads)*q.head_dim;
    std::vector<float> sums(size_t(q.layers)*width,0);
    for (uint32_t l=0;l<q.layers;++l) {
        check(q.layer_rows[l] <= q.rows,"mean Query row coverage mismatch");
        if (!q.layer_rows[l]) continue;
        const auto* base = q.layer_data ? q.layer_data[l] : q.data+size_t(l)*q.rows*width;
        check(base != nullptr,"missing active Query layer storage");
        for (uint32_t t=0;t<q.layer_rows[l];++t) for (size_t d=0;d<width;++d)
            sums[size_t(l)*width+d] += base[size_t(t)*width+d];
    }
    return mean_key_retrieval_scores(keys,blocks,{sums.data(),q.layer_rows,q.layers,q.head_dim,
        q.query_heads,q.kv_heads,aggregation});
}

size_t GraphSnapshot::bytes() const {
    size_t total = sizeof(*this) + chunks.capacity()*sizeof(chunks[0]) +
        partial.mass.capacity()*sizeof(GraphEdge) +
        (messages ? sizeof(*messages)+messages->capacity()*sizeof(GraphMessageSpan) : 0);
    for (const auto& chunk : chunks) {
        total += sizeof(*chunk) + chunk->capacity()*sizeof(GraphRow);
        for (const auto& row : *chunk) total += row.edges.capacity()*sizeof(GraphEdge);
    }
    return total;
}

OutputSamplePlan accepted_output_samples(OutputSampleState state, const int32_t* tokens,
    uint32_t accepted, int32_t think_open, int32_t think_close, uint32_t interval) {
    check(interval && (tokens || !accepted), "invalid accepted output sampling input");
    OutputSamplePlan plan{state,{}};
    for(uint32_t row=0;row<accepted;++row) {
        const auto token=tokens[row];
        if(token==think_open && think_open>=0) { plan.state.thinking=true; continue; }
        if(token==think_close && think_close>=0) { plan.state.thinking=false; continue; }
        if(!plan.state.thinking && ++plan.state.eligible_tokens%interval==0) plan.rows.push_back(row);
    }
    return plan;
}

std::vector<uint32_t> prefill_sample_rows(uint64_t begin,uint32_t count,uint64_t last_end,
    uint64_t visible,uint64_t capacity,uint32_t interval,uint32_t stride,uint32_t block_tokens) {
    check(interval && stride && visible<=capacity, "invalid prefill sampling geometry");
    if(!count || begin+count<=last_end) return {};
    const bool near_capacity=capacity-visible<=uint64_t(count)*2+block_tokens;
    if(last_end && begin+count-last_end<interval && !near_capacity) return {};
    std::vector<uint32_t> rows;
    for(uint32_t tail=count;tail>0;) {
        rows.push_back(tail-1);
        if(tail<=stride) break;
        tail-=stride;
    }
    std::reverse(rows.begin(),rows.end());
    return rows;
}

GraphSnapshot prepare_graph_accept(const GraphSnapshot& before, uint64_t begin, uint64_t end,
    const std::vector<AcceptedAttentionRow>& rows, size_t max_bytes, uint32_t topk) {
    check(before.block_tokens && topk && begin == before.accepted_through && end >= begin,
          "graph acceptance does not extend its exact prefix");
    GraphSnapshot next = before;
    // Reserve source-row aggregation and one copied immutable chunk before
    // allocating either. The upper bound uses distinct causal logical pages.
    size_t edges=0;
    for(const auto& row:rows) edges+=row.mass.size();
    const auto unique=std::min<uint64_t>(edges+before.partial.mass.size(),end/before.block_tokens+1);
    next.reserve_extra(size_t(unique)*(sizeof(GraphEdge)+64)+
        rows.size()*(sizeof(GraphRow)+size_t(topk)*sizeof(GraphEdge))+32768);
    uint64_t previous = begin;
    bool first = true;
    for (const auto& row : rows) {
        check(row.source_position >= begin && row.source_position < end &&
              (first || previous < row.source_position), "accepted attention source is stale or duplicated");
        first = false; previous = row.source_position;
        const uint64_t source = row.source_position/before.block_tokens*before.block_tokens;
        if (next.partial.samples && next.partial.source != source) seal(next, topk);
        next.partial.source = source;
        auto masses = normalized_mass(row.mass);
        std::map<uint64_t, double> merged;
        for (const auto& edge : next.partial.mass) merged[edge.target] = edge.mass;
        for (const auto& edge : masses) {
            check(edge.target % before.block_tokens == 0 && edge.target <= row.source_position,
                  "attention target is not a causal logical block");
            if (edge.target < source) merged[edge.target] += edge.mass;
        }
        next.partial.mass.clear();
        for (const auto& item : merged) next.partial.mass.push_back({item.first, item.second});
        ++next.partial.samples;
    }
    if (next.partial.samples && next.partial.source + before.block_tokens <= end) seal(next, topk);
    next.accepted_through = end;
    next.completed_through = end/before.block_tokens*before.block_tokens;
    next.revision += end != begin;
    if (next.bytes() > max_bytes) throw std::length_error("graph session budget exceeded before publication");
    next.reserve_extra(0);
    return next;
}

std::vector<double> propagate_query_scores(const std::vector<double>& raw,
    const QueryScoreFrame& frame, const GraphSnapshot& graph, double alpha, double cross) {
    decay_check(alpha); decay_check(cross);
    const size_t n = raw.size();
    check(graph.block_tokens && frame.starts.size() == n && frame.transport_mask.size() == n &&
          graph.accepted_through == frame.accepted_through, "graph and Query have different accepted prefixes");
    std::unordered_map<uint64_t, size_t> allowed;
    double total = 0;
    std::unordered_set<uint64_t> seen;
    for (size_t i = 0; i < n; ++i) {
        check(std::isfinite(raw[i]), "nonfinite Query graph seed");
        check(frame.starts[i] % graph.block_tokens == 0 && seen.insert(frame.starts[i]).second,
              "invalid or duplicate Query graph identity");
        if (frame.transport_mask[i]) { allowed.emplace(frame.starts[i], i); total += std::max(0., raw[i]); }
    }
    check(std::isfinite(total), "graph seed sum overflow");
    if (allowed.size() <= 1 || total == 0 || alpha == 0) return raw;
    std::vector<double> seed(n, 0), output = raw;
    // Transport positive evidence only. Negative cosine residuals remain local:
    // result = min(raw, 0) + G(max(raw, 0)). History is never a graph seed.
    for (const auto& item : allowed) seed[item.second] = std::max(0., raw[item.second])/total;
    auto transfer = [&](size_t i, size_t j, double weight) {
        const double delta = total*seed[i]*weight;
        output[i] -= delta; output[j] += delta;
    };
    for (const auto& item : allowed) {
        const auto source = item.first;
        for (int direction : {-1, 1}) {
            if (direction < 0 && source < graph.block_tokens) continue;
            if (direction > 0 && source > UINT64_MAX-graph.block_tokens) continue;
            const auto target = direction < 0 ? source-graph.block_tokens : source+graph.block_tokens;
            const auto found = allowed.find(target);
            if (found != allowed.end()) transfer(item.second, found->second,
                alpha*.25*span_coefficient(graph, source, target, cross));
        }
    }
    for (const auto& chunk : graph.chunks) for (const auto& row : *chunk) {
        const auto source = allowed.find(row.source);
        if (source == allowed.end() || !row.samples || row.source+graph.block_tokens > graph.completed_through) continue;
        double denominator = 0;
        for (const auto& edge : row.edges) {
            finite_mass(edge.mass);
            if (edge.target != row.source && allowed.count(edge.target)) denominator += edge.mass;
        }
        if (!denominator) continue;
        for (const auto& edge : row.edges) {
            const auto target = allowed.find(edge.target);
            if (target != allowed.end() && edge.target != row.source)
                transfer(source->second, target->second, alpha*.5*edge.mass/denominator);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        const double residual = std::min(0., raw[i]);
        check(std::isfinite(output[i]) && output[i] >= residual-1e-12*std::max(1., total),
              "invalid graph propagation result");
        output[i] = std::max(residual, output[i]);
    }
    return output;
}

void AttentionFeedback::restore(AttentionFeedbackSnapshot state) {
    validate_pairs(state.output); validate_pairs(state.pending_output); validate_pairs(state.prefill);
    for (const auto& item : state.output) finite_mass(item.second);
    for (const auto& item : state.pending_output) finite_mass(item.second);
    for (const auto& item : state.prefill) { finite_mass(item.second.mass); finite_mass(item.second.exposure); }
    check(state.output_samples || state.pending_output.empty(), "pending attention lacks accepted samples");
    state_ = std::move(state);
}

void AttentionFeedback::observe_output(const std::vector<GraphEdge>& input) {
    const auto row = normalized_mass(input);
    auto next = state_.pending_output;
    for (const auto& edge : row) insert(next, edge.target) += edge.mass;
    state_.pending_output = std::move(next);
    ++state_.output_samples;
}

void AttentionFeedback::finish_output(double decay) {
    decay_check(decay);
    if (!state_.output_samples) return;
    auto next = state_.output;
    for (auto& item : next) item.second *= decay;
    for (const auto& item : state_.pending_output) insert(next, item.first) += item.second/state_.output_samples;
    state_.output = std::move(next);
    state_.pending_output.clear(); state_.output_samples = 0;
}

void AttentionFeedback::observe_prefill(uint64_t begin, uint64_t end, uint32_t accepted,
    const std::vector<AcceptedAttentionRow>& rows, const std::vector<uint64_t>& exposed, double half_life) {
    check(end >= begin && accepted <= end-begin && std::isfinite(half_life) && half_life > 0,
          "invalid prefill attention transaction");
    if (accepted != end-begin || rows.empty() || end <= state_.prefill_end) return;
    const double decay = std::exp2(-double(end-state_.prefill_end)/half_life);
    std::map<uint64_t, PrefillMass> next;
    for (const auto key : exposed) {
        const auto old = lookup(state_.prefill, key);
        check(next.emplace(key, PrefillMass{old.mass*decay, old.exposure*decay}).second,
              "duplicate prefill exposure identity");
    }
    const double weight = 1./rows.size();
    for (const auto& sample : rows) {
        check(sample.source_position >= begin && sample.source_position < end, "invalid prefill source position");
        const auto mass = normalized_mass(sample.mass);
        for (auto& item : next) if (item.first <= sample.source_position) item.second.exposure += weight;
        for (const auto& edge : mass) {
            const auto found = next.find(edge.target);
            check(found != next.end() && edge.target <= sample.source_position,
                  "prefill mass targets an invisible logical block");
            found->second.mass += edge.mass*weight;
        }
    }
    state_.prefill.assign(next.begin(), next.end());
    state_.prefill_end = end;
}

std::vector<double> AttentionFeedback::pressure_scores(const std::vector<uint64_t>& candidates, double decay) const {
    decay_check(decay);
    std::vector<double> result; result.reserve(candidates.size());
    bool output = false;
    for (auto key : candidates) {
        double score = lookup(state_.output, key);
        if (state_.output_samples) score = decay*score+lookup(state_.pending_output, key)/state_.output_samples;
        result.push_back(score); output |= score > 0;
    }
    if (!output) for (size_t i = 0; i < candidates.size(); ++i) {
        const auto value = lookup(state_.prefill, candidates[i]);
        result[i] = value.exposure > 0 ? value.mass/value.exposure : 0;
    }
    return result;
}

std::vector<uint32_t> pressure_selection(const std::vector<uint32_t>& candidates,
    const std::vector<double>& scores, const std::vector<uint32_t>& mandatory, uint32_t budget) {
    check(candidates.size() == scores.size() && budget, "invalid pressure candidate frame");
    std::vector<std::pair<uint32_t, double>> ranked;
    std::unordered_set<uint32_t> seen;
    for (size_t i = 0; i < candidates.size(); ++i) {
        finite_mass(scores[i]);
        check(seen.insert(candidates[i]).second, "duplicate pressure candidate");
        ranked.emplace_back(candidates[i], scores[i]);
    }
    std::vector<uint32_t> selected = mandatory;
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    check(selected.size() <= budget, "mandatory pressure content exceeds the budget");
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first > b.first;
    });
    seen = {selected.begin(), selected.end()};
    for (const auto& item : ranked) {
        if (selected.size() == budget) break;
        if (seen.insert(item.first).second) selected.push_back(item.first);
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}
} // namespace kvmem
