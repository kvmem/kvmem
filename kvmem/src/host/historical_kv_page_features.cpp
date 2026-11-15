#include "kvmem/historical_kv_page_features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kvmem {

HistoricalKvPageFeatures::HistoricalKvPageFeatures(uint32_t logical_pages, size_t dimensions)
    : dimensions_(dimensions), pages_(logical_pages), output_peaks_(logical_pages, -std::numeric_limits<double>::infinity()) {
    if (!logical_pages || !dimensions) throw std::invalid_argument("historical page feature geometry");
}

bool HistoricalKvPageFeatures::has_page(uint32_t page) const {
    std::lock_guard<std::mutex> lock(mu_);
    return page < pages_.size() && pages_[page].size() == dimensions_;
}

void HistoricalKvPageFeatures::publish_page(uint32_t page, const std::vector<float> & descriptor) {
    if (page >= pages_.size() || descriptor.size() != dimensions_)
        throw std::invalid_argument("historical page descriptor dimensions");
    for (float value : descriptor)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite historical page descriptor");
    std::lock_guard<std::mutex> lock(mu_);
    if (!pages_[page].empty()) throw std::logic_error("historical page descriptor republished");
    pages_[page] = descriptor;
    observation_revision_.fetch_add(1, std::memory_order_release);
}

void HistoricalKvPageFeatures::publish_question_scores(const std::vector<double> & scores, bool new_question) {
    for (double value : scores)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite historical Query score");
    std::lock_guard<std::mutex> lock(mu_);
    const auto previously_scored = question_scores_.size();
    question_scores_ = scores;
    if (new_question) {
        std::fill(output_peaks_.begin(), output_peaks_.end(), -std::numeric_limits<double>::infinity());
        question_epoch_.fetch_add(1, std::memory_order_release);
    } else if (scores.size() > previously_scored) {
        observation_revision_.fetch_add(1, std::memory_order_release);
    }
}

void HistoricalKvPageFeatures::publish_output_attention(uint32_t page, double mass) {
    if (page >= pages_.size() || !std::isfinite(mass) || mass < 0)
        throw std::invalid_argument("invalid historical output attention mass");
    std::lock_guard<std::mutex> lock(mu_);
    // Attention-only readers need this first observation even with a Query score.
    if (!std::isfinite(output_peaks_[page]))
        observation_revision_.fetch_add(1, std::memory_order_release);
    output_peaks_[page] = std::max(output_peaks_[page], mass);
}

bool HistoricalKvPageFeatures::any_observed(const std::vector<uint32_t>& pages, bool attention_only) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto id : pages)
        if (id < pages_.size() && pages_[id].size() == dimensions_ &&
            ((!attention_only && id < question_scores_.size()) || std::isfinite(output_peaks_[id]))) return true;
    return false;
}

void HistoricalKvPageFeatures::clear_question() {
    std::lock_guard<std::mutex> lock(mu_);
    question_scores_.clear();
    std::fill(output_peaks_.begin(), output_peaks_.end(), -std::numeric_limits<double>::infinity());
    question_epoch_.fetch_add(1, std::memory_order_release);
}

void HistoricalKvPageFeatures::truncate_to(uint64_t token_frontier, uint32_t page_tokens) {
    if (!page_tokens) throw std::invalid_argument("historical page token width");
    const auto page = token_frontier / page_tokens;
    std::lock_guard<std::mutex> lock(mu_);
    if (page >= pages_.size()) return;
    for (size_t id = page; id < pages_.size(); ++id) {
        pages_[id].clear();
        output_peaks_[id] = -std::numeric_limits<double>::infinity();
    }
    if (question_scores_.size() > page) question_scores_.resize(page);
    question_epoch_.fetch_add(1, std::memory_order_release);
}

bool HistoricalKvPageFeatures::begin_round(uint64_t) {
    std::lock_guard<std::mutex> lock(mu_);
    frozen_importance_ = output_peaks_;
    frozen_attention_ = output_peaks_;
    for (size_t id = 0; id < std::min(question_scores_.size(), frozen_importance_.size()); ++id)
        frozen_importance_[id] = std::max(frozen_importance_[id], question_scores_[id]);
    return true;
}

bool HistoricalKvPageFeatures::read(const HistoricalKvBlockMeta & meta,
                                   float * descriptor, double & importance) {
    if (!descriptor || meta.logical_id >= pages_.size()) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const auto page = static_cast<size_t>(meta.logical_id);
    if (pages_[page].size() != dimensions_) return false;
    std::copy(pages_[page].begin(), pages_[page].end(), descriptor);
    importance = std::isfinite(frozen_importance_[page]) ? frozen_importance_[page] : 0.0;
    return true;
}

bool HistoricalKvPageFeatures::read_observed(const HistoricalKvBlockMeta & meta,
                                            float * descriptor, double & importance, bool attention_only) {
    if (!descriptor || meta.logical_id >= pages_.size()) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const auto page = static_cast<size_t>(meta.logical_id);
    const auto& observed = attention_only ? frozen_attention_ : frozen_importance_;
    if (pages_[page].size() != dimensions_ || page >= observed.size() ||
        !std::isfinite(observed[page])) return false;
    std::copy(pages_[page].begin(), pages_[page].end(), descriptor);
    importance = observed[page];
    return true;
}

} // namespace kvmem
