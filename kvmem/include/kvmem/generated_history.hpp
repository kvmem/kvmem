#pragma once

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace kvmem {

// One directory per model/tokenizer instance. The adapter supplies a structured
// transcript key and validates the rendering seam. This holds no native KV.
class GeneratedHistory {
public:
    struct Prefix {
        std::string text;
        std::vector<int32_t> ids;
    };

    explicit GeneratedHistory(size_t max_entries = 32, size_t max_bytes = 32u << 20)
        : max_entries_(max_entries), max_bytes_(max_bytes) {}

    std::vector<std::shared_ptr<const Prefix>> candidates(const std::string & transcript,
            const std::vector<size_t> & assistant_ends) {
        std::vector<std::shared_ptr<const Prefix>> matches;
        if (assistant_ends.empty()) return matches;
        std::lock_guard<std::mutex> guard(mutex_);
        // A bounded directory scan avoids constructing one copy of the entire
        // transcript per historical assistant message. Ends refer to complete
        // serialized messages; text inside a message cannot introduce an end.
        for (const auto & entry : entries_) {
            if (entry.prefix && std::binary_search(assistant_ends.begin(), assistant_ends.end(), entry.key.size()) &&
                    transcript.compare(0, entry.key.size(), entry.key) == 0)
                matches.push_back(entry.prefix);
        }
        std::sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) {
            return a->text.size() > b->text.size();
        });
        return matches;
    }

    void publish(std::string key, Prefix prefix) {
        if (prefix.text.empty() || prefix.ids.empty()) return;
        record(std::move(key), &prefix);
    }

    // The output parser discarded bytes (e.g. a stop inside a token). A
    // future echo cannot distinguish this turn from a clean, identical reply.
    void mark_ambiguous(std::string key) { record(std::move(key), nullptr); }

private:
    void record(std::string key, Prefix * prefix) {
        if (key.empty() || !max_entries_) return;
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = locate(key);
        if (it != entries_.end()) {
            entries_.splice(entries_.begin(), entries_, it);
            // A retained tombstone prevents a second lookup from picking one
            // of two indistinguishable branches by recency.
            if (it->prefix && (!prefix || it->prefix->text != prefix->text || it->prefix->ids != prefix->ids)) {
                bytes_ -= it->charge - it->key.capacity();
                it->charge = it->key.capacity();
                it->prefix.reset();
            }
            return;
        }
        const size_t charge = key.capacity() + (prefix ? prefix->text.capacity() + prefix->ids.capacity() * sizeof(int32_t) : 0);
        if (charge > max_bytes_) return;
        auto immutable = prefix ? std::make_shared<const Prefix>(std::move(*prefix)) : nullptr;
        entries_.push_front({std::move(key), std::move(immutable), charge});
        bytes_ += charge;
        while (entries_.size() > max_entries_ || bytes_ > max_bytes_) {
            bytes_ -= entries_.back().charge;
            entries_.pop_back();
        }
    }

    struct Entry {
        std::string key;
        std::shared_ptr<const Prefix> prefix;
        size_t charge;
    };
    std::list<Entry>::iterator locate(const std::string & key) {
        return std::find_if(entries_.begin(), entries_.end(), [&](const Entry & e) { return e.key == key; });
    }
    const size_t max_entries_, max_bytes_;
    size_t bytes_ = 0;
    std::list<Entry> entries_;
    std::mutex mutex_;
};
} // namespace kvmem
