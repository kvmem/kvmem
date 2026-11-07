# Packed record exchange and state storage contracts

This public change is coordinated separately from backend integrations. It
extends existing storage and packed-record ownership without changing native
scheduling, resource policy or legacy packed coverage queries. The initial
prototype was supplied by the llama.cpp session; coordinator review added
selected-record demotion and checkpoint-image transport compatibility.

## Packed exchange contract

`export_packed` accepts exact block/layer/valid-prefix selections. Both packed
planes must exist with full allocation extent and independently proved valid
rows. Duplicate mappings, missing layers/planes and holes fail before sealing.
Entries are sorted by block/layer. Full records fork without reads or payload
allocation. Partial entries privately allocate capacity-sized records through
the source account, copy only valid prefix bytes, and leave padding zero.
Exported record references are exposed read-only; only the exporter can construct
a snapshot. `KvSnapshot` can map each full record exactly once; valid rows remain
separate native metadata rather than implied by the capacity mapping.

`import_packed` validates configuration, record extents and storage-account
ownership, builds a private index, then swaps all packed bindings atomically.
Unselected packed planes disappear. It preserves destination raw rows and MeanK
statistics as private copies; the adapter must restore the appropriate checkpoint
statistics, truncate logical history and restore native layout/model state. This
method does not establish model, execution-position or native-storage identity.
The same account is required: `HostKvStorage::owns(record)` prevents importing
another account's bytes to bypass the destination quota. Implicit default RAM
accounts are now retained in `config()` so existing store forks share that account.

## Coverage and compatibility

New per-plane bitsets prove each written row independently. They support disjoint
or out-of-order writes, fork, truncate and invalidate. Legacy `has_*` high-water
semantics are unchanged. Unrotated K writes invalidate the new packed proof;
replacing V with legacy rows clears its proof. Legacy serialization format is
unchanged and deliberately carries no proof: after deserialization, new export
rejects until packed rows are rewritten, while existing readers still work.
Frozen in-process attachments retain their coverage metadata alongside the exact
record binding restored by the existing session mechanism.

## Ownership, locking and failures

The caller drains native/capture producers and serializes the store lifecycle.
Export validates and forks under the store mutex, then allocates/reads partial
records outside that mutex. Import does no payload allocation or I/O. Source
references protect immutable bytes; later writes use existing record COW.
An optional caller-owned atomic cancellation flag is polled before publication
and between partial-record copies. Cancellation cannot interrupt one synchronous
storage read. Failure drops candidate references/allocations and leaves source
bytes and previous destination bindings intact. Sealing can survive a cancelled
export but does not change source bytes or accepted coverage.

No new payload budget/pool/directory is introduced. Full references add no
physical charge; private tails and later COW use the existing account. Metadata
allowance is caller supplied, not a policy. Snapshot `metadata_bytes()` reports
owned object/vector capacity; RawKvStore `allocated_bytes()` and `capacity_bytes()`
include coverage buffers. Import allowance includes candidate index, proof,
legacy row buffers and statistics. Existing destination metadata remains live
until publication: caller must admit that transient overlap. Allocation capacity
is checked after allocation too, so this is an admission guard, not a custom
allocator with a hard process-RSS cap. Control blocks/allocator overhead remain
process overhead. Empty selections are permitted and import clears packed planes
while preserving private statistics.

## Evidence and remaining review

Remote `D:/KVMem-Lab/raw-kv-snapshot-prototype/build-host`, MSVC 19.44.35222,
SDK 10.0.26100.0, Release `/EHsc /MD`; 18 host objects, 5 links, zero CUDA,
8 jobs. Initial `packed_kv_snapshot_test`, `raw_kv_store_test`, `kv_snapshot_test`
and `tiered_kv_storage_test`: 4/4 passed (0.19 s). A follow-up added implicit-account
fork coverage; its final results are reported with the handoff commit.

Tests include hole rejection, out-of-order completion, duplicate requests,
missing K/V/layer, noncontiguous selection, full identity reuse, private zeroed
tail, tail growth, source deletion, old serialized records without proof,
foreign account/config rejection, metadata rejection, partial allocation failure
with charge release, pre-cancellation, abandoned-result release, and cold SSD
export/import without reads/writes/promotion. Deterministic cancellation midway
through synchronous I/O and allocator-fault injection at every metadata step
are not yet covered. GPU/native identity proof and backend model-state restoration
remain separate llama.cpp integration tests.


## Selected-record demotion

The old native checkpoint scheduler chooses a specific eligible LRU state image;
`hot_count` triggers background work, while required admission may force a
migration below that threshold. Whole-account `trim` cannot preserve that object
selection. `HostKvStorage::demote(record)` therefore reuses the existing eviction
path with exactly one selected allocation. Foreign/empty records are rejected.
It returns true for an already cold or successfully migrated record, and false
for RAM-only, mutable, leased or busy records. It does not choose another victim.
No separate pool, quota, scheduling thread or cancellation mechanism is added.

The caller serializes the owner handle and submits work through its existing
queue. The worker holds a strong record reference. Once physical I/O has started,
cancellation does not interrupt it: caller publication remains conditional on
its request/generation identity. A completed relocation changes only the physical
tier of immutable bytes. Failed disk reservation/write leaves the resident copy
valid and charged; unpublished disk extents are released. The account policy
lock is released during physical I/O; the selected record remains locked.

## Native checkpoint image transport

`write_checkpoint_image(HostKvRecord&, ...)` and
`read_checkpoint_image(HostKvRecord/HostKvLease, ...)` use the same format and
identity, segment length, total length and checksum checks as the existing
SpillExtent overloads. The writer owns a private record, completes all writes,
seals it, then publishes. Read spans must point to private staging; exceptions
may leave staging partially filled, and native restore must wait for successful
validation. A cold lease reads without promoting the whole image into RAM.
Native schema/frontier/recurrent/draft compatibility remains the backend's job.
Existing KV H and state Q meanings are retained by callers; the state account is
shared per execution and uses the existing shared D SpillFile, not per checkpoint.
Image header/checksum bytes consume the same physical quota as the payload.

## Coordinator validation on M16

M16 Windows, MSVC19.44.35222, SDK26100, Release: focused build of the four required
support units and five test units (9 C++ compiles,6 links,zero CUDA,8jobs).
`packed_kv_snapshot_test`, `raw_kv_store_test`, `tiered_kv_storage_test`,
`kv_snapshot_test`, `checkpoint_image_test` all pass. Added selected victim,
pinned/mutable refusal, foreign account, shared ownership, disk quota failure,
source-byte preservation, final charge release, hot/cold image read, source
deletion, corrupt branch/identity/length rejection, and concurrent branch COW.
These are host contracts, not native model or GPU restoration evidence. Existing
Linux file-size write-failure injection is skipped on Windows; the Windows run
does not claim OS write-failure coverage. Mid-I/O cancellation and exhaustive
metadata allocator fault injection remain untested; cancellation is cooperative
between synchronous operations, with no publication on a detected cancellation.
