#include "attention_capture.hpp"
#include "rad_device.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace kvmem::radiance {
namespace {
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Allocation {
    void* p = nullptr; int kind;
    Allocation(size_t bytes, int k) : p(rad_dev_alloc(int64_t(bytes), k)), kind(k) { check(p != nullptr, "attention capture allocation failed"); }
    ~Allocation() { if (p) rad_dev_free(p, kind); }
};
}
struct AttentionCapture::State {
    int device;
    uint32_t max_rows, max_pages, first = 0;
    RadStream stream = nullptr;
    bool active = false;
    std::vector<StatisticsLayer> layers;
    std::vector<uint32_t> rows;
    std::vector<AttentionPage> pages;
    std::vector<bool> seen;
    std::unique_ptr<Allocation> host_rows, device_rows, host_pages, device_pages, scratch;
    std::vector<std::unique_ptr<Allocation>> device_masses, host_masses;
    ~State() { (void)rad_dev_set(device); if (stream) (void)rad_stream_sync(stream); }
};
AttentionCapture::AttentionCapture(int device, std::vector<StatisticsLayer> layers, uint32_t rows, uint32_t pages)
    : state_(std::make_unique<State>()) {
    auto& s = *state_; s.device = device; s.max_rows = rows; s.max_pages = pages;
    check(!layers.empty() && rows && pages && rad_dev_set(device) >= 0, "invalid attention capture geometry");
    std::vector<size_t> selected{0, layers.size()/2, layers.size()-1};
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    for (auto i : selected) s.layers.push_back(layers[i]);
    s.seen.resize(s.layers.size());
    // At most ceil(rows/16) output or ceil(rows/32) prefill samples. Allow one
    // extra row for interval alignment, and reject any caller exceeding it.
    const size_t samples = (rows + 15) / 16 + 1;
    s.host_rows = std::make_unique<Allocation>(samples * sizeof(uint32_t), RAD_MEM_HOST_PINNED);
    s.device_rows = std::make_unique<Allocation>(samples * sizeof(uint32_t), RAD_MEM_DEVICE);
    s.host_pages = std::make_unique<Allocation>(pages * sizeof(AttentionPage), RAD_MEM_HOST_PINNED);
    s.device_pages = std::make_unique<Allocation>(pages * sizeof(AttentionPage), RAD_MEM_DEVICE);
    s.scratch = std::make_unique<Allocation>(samples * pages * layers[0].query_heads * sizeof(float), RAD_MEM_DEVICE);
    for (size_t i = 0; i < s.layers.size(); ++i) {
        s.device_masses.push_back(std::make_unique<Allocation>(samples * pages * sizeof(float), RAD_MEM_DEVICE));
        s.host_masses.push_back(std::make_unique<Allocation>(samples * pages * sizeof(float), RAD_MEM_HOST_PINNED));
    }
}
AttentionCapture::~AttentionCapture() = default;
void AttentionCapture::begin(uint32_t first, const std::vector<uint32_t>& rows,
                             const std::vector<AttentionPage>& pages, RadStream stream) {
    auto& s = *state_;
    check(rad_dev_set(s.device)>=0,"select attention capture device");
    check(!s.active && stream && pages.size() <= s.max_pages && rows.size() <= (s.max_rows+15)/16+1,
          "overlapping or oversized attention frame");
    check(std::is_sorted(rows.begin(),rows.end()) && std::adjacent_find(rows.begin(),rows.end())==rows.end(),
          "noncanonical attention samples");
    check(rows.empty() || (rows.back() < s.max_rows && !pages.empty()), "invalid attention sample range");
    uint64_t end = 0; uint32_t compact = 0;
    for (const auto& page : pages) {
        check(page.count && page.start >= end && page.compact == compact, "invalid logical attention page map");
        end = page.start + page.count; compact += page.count;
    }
    s.first = first; s.rows = rows; s.pages = pages; s.stream = stream; s.active = true;
    std::fill(s.seen.begin(),s.seen.end(),false);
    if (rows.empty()) return;
    std::memcpy(s.host_rows->p,rows.data(),rows.size()*sizeof(uint32_t));
    std::memcpy(s.host_pages->p,pages.data(),pages.size()*sizeof(AttentionPage));
    check(rad_memcpy_async(s.device_rows->p,s.host_rows->p,rows.size()*sizeof(uint32_t),stream)>=0 &&
          rad_memcpy_async(s.device_pages->p,s.host_pages->p,pages.size()*sizeof(AttentionPage),stream)>=0,
          "attention metadata upload failed");
}
int AttentionCapture::observe(int layer, rad_kvgroup group, int64_t hd, const RadTensor& q,
                               const RadTensor& cache, const RadBatch& batch, RadStream stream) {
    auto& s = *state_;
    const auto it = std::find_if(s.layers.begin(),s.layers.end(),[&](const auto& l){return l.layer==layer;});
    if (it==s.layers.end() || s.rows.empty()) return RAD_OK;
    const auto i = size_t(it-s.layers.begin());
    if (!s.active || s.seen[i] || stream!=s.stream || batch.n_seq!=1 || batch.draft_pass ||
        it->group!=group || it->head_dim!=hd || s.rows.back()>=batch.n_tok || q.shape[0]!=batch.n_tok)
        return RAD_E_STATE;
    const RadKVGroupBatch* kv = nullptr;
    for (int g=0;g<batch.n_kv_groups;++g) if (batch.kv[g].group==group) kv=&batch.kv[g];
    if (!kv || !kv->block_table || kv->block_size!=cache.shape[2] ||
        uint64_t(s.pages.back().compact)+s.pages.back().count > uint64_t(kv->block_table_pitch)*kv->block_size)
        return RAD_E_SHAPE;
    AttentionKernelArgs a{q,cache,kv->block_table,static_cast<const uint32_t*>(s.device_rows->p),
        static_cast<const AttentionPage*>(s.device_pages->p),uint32_t(s.rows.size()),uint32_t(s.pages.size()),
        it->query_heads,it->head_dim,s.first,static_cast<float*>(s.scratch->p),static_cast<float*>(s.device_masses[i]->p),it->packing};
    const int rc=attention_page_masses(a,stream);
    if (rc<0) return rc;
    s.seen[i]=true;
    return rad_memcpy_async(s.host_masses[i]->p,s.device_masses[i]->p,
                            s.rows.size()*s.pages.size()*sizeof(float),stream);
}
std::vector<AcceptedAttentionRow> AttentionCapture::collect(uint32_t accepted, const std::vector<uint32_t>& samples) {
    auto& s=*state_; check(s.active,"no attention frame");
    check(rad_dev_set(s.device)>=0,"select attention collection device");
    check(rad_stream_sync(s.stream)>=0,"attention producer synchronization failed");
    if (!s.rows.empty()) check(std::all_of(s.seen.begin(),s.seen.end(),[](bool v){return v;}),"missing native attention sample layer");
    std::vector<AcceptedAttentionRow> result;
    for (auto row : samples) {
        check(row<accepted,"unaccepted attention sample");
        const auto at=std::lower_bound(s.rows.begin(),s.rows.end(),row);
        check(at!=s.rows.end() && *at==row,"accepted sample was not captured");
        const size_t r=size_t(at-s.rows.begin());
        AcceptedAttentionRow out{uint64_t(s.first)+row,{}};
        double total=0;
        for (size_t p=0;p<s.pages.size();++p) {
            double mass=0;
            for (size_t l=0;l<s.layers.size();++l) mass+=static_cast<const float*>(s.host_masses[l]->p)[r*s.pages.size()+p];
            mass/=s.layers.size();
            check(std::isfinite(mass) && mass>=0,"nonfinite native attention page mass");
            if (s.pages[p].start>out.source_position) check(mass==0,"future native attention mass");
            else {out.mass.push_back({s.pages[p].start,mass});total+=mass;}
        }
        check(std::abs(total-1)<1e-4,"native attention mass is not normalized");
        result.push_back(std::move(out));
    }
    s.active=false;s.stream=nullptr;return result;
}
void AttentionCapture::discard() noexcept {
    auto& s=*state_; (void)rad_dev_set(s.device);if(s.stream)(void)rad_stream_sync(s.stream);s.active=false;s.stream=nullptr;
}
} // namespace kvmem::radiance
