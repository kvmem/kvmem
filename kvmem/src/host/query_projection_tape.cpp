#include "kvmem/query_projection_tape.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace kvmem {
namespace detail {
struct QueryProjectionDescriptor {
    LayerwiseQueryIdentity identity;
    uint32_t model_layer;
    QueryProjectionLayout layout;
    std::vector<uint64_t> offsets; // byte prefix per row, before multiplying by chunk rows
    uint64_t row_bytes = 0, total_bytes = 0;
    std::shared_ptr<HostKvReservation> metadata_charge;
    uint64_t metadata_bytes = 0, max_chunks = 0;
};
}
namespace {
[[noreturn]] void invalid(const char* message) {
    throw MemoryError(MemoryErrorCode::InvalidPlan, message);
}
uint64_t multiply(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) invalid("Query projection byte size overflow");
    return a * b;
}
uint64_t add(uint64_t a,uint64_t b) {
    if(b>UINT64_MAX-a)invalid("Query projection metadata size overflow");
    return a+b;
}
size_t size_bytes(uint64_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max())
        invalid("Query projection chunk exceeds native address range");
    return size_t(bytes);
}
bool same_range(QueryRange a, QueryRange b) { return a.begin == b.begin && a.end == b.end; }
void check_identity(const detail::QueryProjectionDescriptor& d,
                    const LayerwiseQueryIdentity& i, uint32_t layer) {
    if (!(d.identity.history == i.history) || d.identity.request != i.request ||
        !same_range(d.identity.evaluation, i.evaluation) || !same_range(d.identity.query, i.query) ||
        d.model_layer != layer)
        throw MemoryError(MemoryErrorCode::Stale, "foreign Query projection identity or model layer");
}
size_t plane_index(const detail::QueryProjectionDescriptor& d, uint32_t plane) {
    const auto found = std::find_if(d.layout.planes.begin(), d.layout.planes.end(),
                                   [=](const auto& p) { return p.id == plane; });
    if (found == d.layout.planes.end()) invalid("unknown Query projection plane");
    return size_t(found - d.layout.planes.begin());
}
void check_read(QueryRange available, uint64_t begin, uint64_t count, uint64_t stride,
                void* destination, size_t bytes) {
    if (begin < available.begin || begin > available.end || count > available.end - begin ||
        multiply(count, stride) != bytes || (bytes && !destination))
        invalid("Query projection row read exceeds covered rows or destination");
}
}

QueryProjectionChunkSource::QueryProjectionChunkSource(
    std::shared_ptr<const detail::QueryProjectionDescriptor> descriptor,
    QueryRange rows, HostKvSource source)
    : descriptor_(std::move(descriptor)), rows_(rows), source_(std::move(source)), bytes_(source_.size()) {}
QueryProjectionChunkSource::QueryProjectionChunkSource(
    std::shared_ptr<const detail::QueryProjectionDescriptor> descriptor,
    QueryRange rows, SpillExtent spill, uint64_t offset, size_t bytes)
    : descriptor_(std::move(descriptor)), rows_(rows), spill_(std::move(spill)),
      offset_(offset), bytes_(bytes) {}
const LayerwiseQueryIdentity& QueryProjectionChunkSource::identity() const { return descriptor_->identity; }
uint32_t QueryProjectionChunkSource::model_layer() const { return descriptor_->model_layer; }
const QueryProjectionLayout& QueryProjectionChunkSource::layout() const { return descriptor_->layout; }
void QueryProjectionChunkSource::read_rows(const LayerwiseQueryIdentity& identity, uint32_t layer,
    uint32_t plane, uint64_t begin, uint64_t count, void* destination, size_t bytes) const {
    check_identity(*descriptor_, identity, layer);
    const auto index = plane_index(*descriptor_, plane);
    const auto stride = descriptor_->layout.planes[index].row_stride;
    check_read(rows_, begin, count, stride, destination, bytes);
    const auto offset = multiply(descriptor_->offsets[index], rows_.end - rows_.begin) +
                        multiply(begin - rows_.begin, stride);
    if (spill_) spill_.read(offset_ + offset, destination, bytes);
    else source_.read(size_bytes(offset), destination, bytes);
}

QueryProjectionTape::QueryProjectionTape(HostKvStorage& activation, LayerwiseQueryIdentity identity,
    uint32_t layer, QueryProjectionLayout layout)
    : QueryProjectionTape(std::move(identity), layer, std::move(layout)) {
    // RAM admission is deliberately separate from direct-disk admission.
    reservation_ = std::make_unique<HostKvReservation>(activation.reserve(descriptor_->total_bytes));
}
QueryProjectionTape::QueryProjectionTape(SpillFile& activation, LayerwiseQueryIdentity identity,
    uint32_t layer, QueryProjectionLayout layout)
    : QueryProjectionTape(std::move(identity), layer, std::move(layout)) {
    // This is an explicit activation account, never a canonical KV spill file
    // borrowed implicitly. The extent owns its file lifetime and capacity charge.
    spill_ = activation.reserve(descriptor_->total_bytes);
}
QueryProjectionTape::QueryProjectionTape(HostKvStorage& activation,HostKvStorage& metadata,
    uint64_t chunks,LayerwiseQueryIdentity identity,uint32_t layer,QueryProjectionLayout layout)
    : QueryProjectionTape(&metadata,chunks,std::move(identity),layer,std::move(layout)) {
    reservation_=std::make_unique<HostKvReservation>(activation.reserve(descriptor_->total_bytes));
}
QueryProjectionTape::QueryProjectionTape(SpillFile& activation,HostKvStorage& metadata,
    uint64_t chunks,LayerwiseQueryIdentity identity,uint32_t layer,QueryProjectionLayout layout)
    : QueryProjectionTape(&metadata,chunks,std::move(identity),layer,std::move(layout)) {
    spill_=activation.reserve(descriptor_->total_bytes);
}
uint64_t QueryProjectionTape::metadata_bound(const QueryProjectionLayout& layout,uint64_t chunks) {
    if(!chunks || chunks>std::numeric_limits<size_t>::max()/sizeof(Chunk) ||
        layout.format.empty() || !layout.version || layout.planes.empty())
        invalid("invalid Query projection metadata/chunk limit");
    // The bound is defined here, where the actual private descriptor and Chunk
    // representations are known. Vector capacities are reserved once and
    // checked below; the string is copied into a bounded representation.
    // A fixed control/alignment allowance covers the two shared control blocks
    // and reservation ownership, rather than depending on vendor STL internals.
    uint64_t result=sizeof(QueryProjectionTape)+sizeof(detail::QueryProjectionDescriptor)+
        2*sizeof(HostKvReservation)+16*sizeof(std::max_align_t);
    result=add(result,multiply(chunks,sizeof(Chunk)));
    result=add(result,multiply(layout.planes.size(),sizeof(QueryProjectionPlane)+sizeof(uint64_t)));
    result=add(result,add(multiply(layout.format.size(),2),std::string{}.capacity()+1));
    (void)size_bytes(result);return result;
}
QueryProjectionTape::QueryProjectionTape(LayerwiseQueryIdentity identity,
    uint32_t layer,QueryProjectionLayout layout)
    : QueryProjectionTape(nullptr,0,std::move(identity),layer,std::move(layout)) {}
QueryProjectionTape::QueryProjectionTape(HostKvStorage* metadata,uint64_t chunks,
    LayerwiseQueryIdentity identity,uint32_t layer,QueryProjectionLayout layout) {
    if (!valid_layerwise_query_identity(identity) ||
        layout.format.empty() || !layout.version || layout.planes.empty())
        invalid("invalid Query projection identity or native layout");
    // Geometry validation allocates nothing, including duplicate-plane checks.
    uint64_t row_bytes=0;
    for(size_t i=0;i<layout.planes.size();++i) {
        const auto& plane=layout.planes[i];
        if(!plane.row_stride || plane.row_stride>UINT64_MAX-row_bytes)
            invalid("invalid or overflowing Query projection plane stride");
        for(size_t j=0;j<i;++j)if(layout.planes[j].id==plane.id)
            invalid("duplicate Query projection plane");
        row_bytes+=plane.row_stride;
    }
    const auto total=multiply(row_bytes,identity.evaluation.end-identity.evaluation.begin);
    const uint64_t bound=metadata ? metadata_bound(layout,chunks):0;
    std::shared_ptr<HostKvReservation> charge;
    if(metadata)charge=std::make_shared<HostKvReservation>(metadata->reserve(bound));
    // No owned descriptor/layout/chunk table exists before quota admission.
    auto descriptor=std::make_shared<detail::QueryProjectionDescriptor>();
    descriptor->metadata_charge=std::move(charge);descriptor->metadata_bytes=bound;descriptor->max_chunks=chunks;
    // Member destruction frees chunks_ before descriptor_. Keep the admission
    // charge there even if a subsequent bounded allocation throws.
    descriptor_=descriptor;
    descriptor->identity=std::move(identity);descriptor->model_layer=layer;
    if(metadata) {
        descriptor->layout.format.assign(layout.format.data(),layout.format.size());
        if(descriptor->layout.format.capacity()>2*layout.format.size()+std::string{}.capacity())
            invalid("native string capacity exceeds its admitted Query metadata bound");
        descriptor->layout.version=layout.version;
        descriptor->layout.planes.reserve(layout.planes.size());
        descriptor->offsets.reserve(layout.planes.size());
        chunks_.reserve(size_bytes(chunks));
        if(descriptor->layout.planes.capacity()>layout.planes.size() || descriptor->offsets.capacity()>layout.planes.size() ||
            chunks_.capacity()>chunks)invalid("native vector capacity exceeds its admitted Query metadata bound");
        descriptor->layout.planes.insert(descriptor->layout.planes.end(),layout.planes.begin(),layout.planes.end());
    } else descriptor->layout=std::move(layout);
    for(const auto& plane:descriptor->layout.planes) {
        descriptor->offsets.push_back(descriptor->row_bytes);descriptor->row_bytes+=plane.row_stride;
    }
    descriptor->total_bytes=total;next_=descriptor->identity.evaluation.begin;descriptor_=std::move(descriptor);
}
QueryProjectionTape::~QueryProjectionTape() = default;
const LayerwiseQueryIdentity& QueryProjectionTape::identity() const { return descriptor_->identity; }
uint32_t QueryProjectionTape::model_layer() const { return descriptor_->model_layer; }
const QueryProjectionLayout& QueryProjectionTape::layout() const { return descriptor_->layout; }
uint64_t QueryProjectionTape::total_bytes() const noexcept { return descriptor_->total_bytes; }
uint64_t QueryProjectionTape::metadata_bytes() const noexcept {return descriptor_->metadata_bytes;}
uint64_t QueryProjectionTape::max_chunks() const noexcept {return descriptor_->max_chunks;}

uint64_t QueryProjectionTape::reserved_bytes() const noexcept {
    return reservation_ ? reservation_->remaining() :
        (spill_ ? descriptor_->total_bytes - written_bytes_ : 0);
}
void QueryProjectionTape::check(const LayerwiseQueryIdentity& identity, uint32_t layer) const {
    check_identity(*descriptor_, identity, layer);
    if (state_ == State::Aborted) invalid("Query projection tape was aborted");
}
QueryProjectionChunkSource QueryProjectionTape::append(const LayerwiseQueryIdentity& identity,
    uint32_t layer, uint64_t begin, uint64_t count,
    const std::vector<QueryProjectionPlaneBytes>& planes) {
    check(identity, layer);
    if (state_ != State::Recording || !count || begin != next_ ||
        count > descriptor_->identity.evaluation.end - next_ || planes.size() != layout().planes.size())
        invalid("Query projection chunk is incomplete, repeated or outside its evaluation span");
    for (size_t p = 0; p < planes.size(); ++p)
        if (planes[p].id != layout().planes[p].id || !planes[p].data ||
            multiply(count, layout().planes[p].row_stride) != planes[p].bytes)
            invalid("Query projection input planes differ from their native layout");
    if(descriptor_->max_chunks && chunks_.size()>=descriptor_->max_chunks)
        invalid("Query projection chunk count exceeds its admitted metadata limit");
    const auto bytes = size_bytes(multiply(count, descriptor_->row_bytes));
    // Qualified constructors reserve the entire chunk table before payload.
    // Preserve the original caller-qualified constructor's allocation behavior.
    if(!descriptor_->max_chunks)chunks_.reserve(chunks_.size()+1);
    HostKvRecord record;
    if (spill_) {
        // Unpublished rows occupy a disjoint range. A failed write does not
        // advance coverage or alter any previously exported immutable chunk;
        // the caller can retry the same rows or abort/drain its native attempt.
        for (size_t p = 0; p < planes.size(); ++p)
            spill_.write(written_bytes_ + multiply(descriptor_->offsets[p], count),
                         planes[p].data, planes[p].bytes);
    } else {
        record = reservation_->allocate(bytes);
        auto* destination = record.writable_data();
        for (size_t p = 0; p < planes.size(); ++p)
            std::memcpy(destination + size_bytes(multiply(descriptor_->offsets[p], count)),
                        planes[p].data, planes[p].bytes);
        record.seal();
    }
    const QueryRange rows{begin, begin + count};
    chunks_.push_back({rows, std::move(record), written_bytes_});
    written_bytes_ += bytes;
    next_ = rows.end;
    return chunk(identity, layer, chunks_.size() - 1);
}
void QueryProjectionTape::finish(const LayerwiseQueryIdentity& identity, uint32_t layer) {
    check(identity, layer);
    if (state_ != State::Recording || next_ != descriptor_->identity.evaluation.end || reserved_bytes())
        invalid("Query projection tape cannot finish before complete row coverage");
    state_ = State::Complete;
    reservation_.reset();
}
QueryProjectionChunkSource QueryProjectionTape::chunk(const LayerwiseQueryIdentity& identity,
    uint32_t layer, size_t index) const {
    check(identity, layer);
    if (index >= chunks_.size()) invalid("unknown Query projection chunk");
    if (spill_)
        return QueryProjectionChunkSource(descriptor_, chunks_[index].rows, spill_, chunks_[index].offset,
            size_bytes(multiply(chunks_[index].rows.end - chunks_[index].rows.begin, descriptor_->row_bytes)));
    return QueryProjectionChunkSource(descriptor_, chunks_[index].rows, chunks_[index].record.source());
}
void QueryProjectionTape::read_rows(const LayerwiseQueryIdentity& identity, uint32_t layer,
    uint32_t plane, uint64_t begin, uint64_t count, void* destination, size_t bytes) const {
    check(identity, layer);
    const auto index = plane_index(*descriptor_, plane);
    const auto stride = layout().planes[index].row_stride;
    check_read({descriptor_->identity.evaluation.begin, next_}, begin, count, stride, destination, bytes);
    if (!count) return;
    auto* output = static_cast<uint8_t*>(destination);
    auto cursor = begin;
    for (size_t i = 0; i < chunks_.size() && cursor < begin + count; ++i) {
        const auto& rows = chunks_[i].rows;
        if (rows.end <= cursor) continue;
        const auto n = std::min(rows.end - cursor, begin + count - cursor);
        chunk(identity, layer, i).read_rows(identity, layer, plane, cursor, n,
            output + size_bytes(multiply(cursor - begin, stride)), size_bytes(multiply(n, stride)));
        cursor += n;
    }
    if (cursor != begin + count) invalid("Query projection row coverage changed during read");
}
void QueryProjectionTape::abort() noexcept {
    state_ = State::Aborted;
    chunks_.clear();
    reservation_.reset();
    spill_.reset();
}
} // namespace kvmem
