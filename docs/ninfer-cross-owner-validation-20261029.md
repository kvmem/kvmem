# NInfer chat rollback and cross-owner checkpoint validation

Historical qualification of the 2026-10-29 owner-bound implementation. The
key-independent content/native-snapshot policy supersedes that ownership
contract; see [current semantics](checkpoint-policy.md#content-snapshots-across-backends).
The results below belong to their original source and binaries.

The [checkpoint policy](checkpoint-policy.md) now binds Chat cache keys to live
owners and allows a named NInfer Qwen3.5 spread request with a current user Query
to discover a retained prefix in another owner. OpenCode requires no modification.
Keys select owners; exact token/media/position identity and native state still
prove reuse. Ordinary tool continuation remains append plus resident-only Pressure.

## Functional evidence

Windows, MSVC 14.44, CUDA 13.2, Qwen3.5-4B-MTP-Q4_K_M, int8 KV:

- The complete OpenAI schema test passed, including both cache-key spellings,
  conflicting aliases, explicit conversation identity, `store:false`, invalid
  values and anonymous requests.
- Real OpenCode 1.19.5 sent its own serialized requests through an unchanged-body
  proxy. Four parent turns were followed by latest/middle forks, child append,
  rollback, parent return and child return. RAM, SSD and MTP cases passed.
  Before owner binding, same-session rollbacks restored zero tokens; afterwards
  the recorded rollback requests needed only four additional input tokens.
- After integration with mainline, the SSD middle-fork case passed all nine
  requests on RTX 5060 Ti: initial child restore 2121 tokens, child rollback 2143,
  parent return 2338 and child return 2143. Its settings were context 16384,
  B=512, R=128, H=16 MiB, D=2048 MiB and one hot checkpoint. Native restore logs
  verified prefix/state identity and actual cold reads.
- `ninfer_ssd_failure <artifact> --fork` passed after integration on the idle
  RTX 5050 Laptop. Separate ordinary document and question turns were compared
  with fresh prefill through normal termination. Output token IDs matched for
  child and parent, injected cold-read failure fell back without incrementing
  native history hits, and cancellation preserved both owners. Changed prefixes
  and disabled session-index updates missed; spill resources were released.
- `host_kv_storage_test`, `tiered_kv_storage_test` and `raw_kv_store_test` passed:
  shared quota, independent writes, failed detachment, owner deletion, outstanding
  deferred sources/read leases, cold promotion and independent mean-K tails.
- All 24 integrated backend patches and intermediate trees were verified. The
  prepare-backends regression suite passed all 10 cases; public-evidence and
  whitespace checks passed.

## Findings fixed during implementation and audit

An initial OpenCode fork missed because admission did not construct prefix
digests when ordinary continuation publication was disabled. Digests are now
available for memory reuse. RAM writes now clear the new sealed-record state.
The follow-up audit reproduced a needless second-page allocation after the peer
owner was deleted: an 8-byte write under an 8-byte quota failed. A sole remaining
strong owner now reuses its allocation; a retained source or lease still forces
copy-on-write. The exact-limit regression passes for RAM and SSD.

The earliest numerical fixture compared tokens beyond EOS with stopping disabled;
that invalid continuation differed. The corrected fixture uses normal termination
and retains exact token comparison. Early SSD configurations rejected invalid hot
counts or oversized transfer windows; the successful H=16 MiB case above exercises
cold KV as well as native state. These failures are not counted as passing runs.

## Resources, build and limits

Discovery uses RAM metadata, digest filtering and at most 16 foreign candidate
checks; only the selected checkpoint's cold payload is read. Shared sealed KV and
native images avoid eager payload copies, while per-owner metadata/statistics
consume additional RAM and writes can require extra capacity. Optional admission
or I/O failure falls back to full prefill. The inherited API quote can still report
planned cached tokens after fallback; native restore logs prove completed reuse.

The focused validation build compiled all changed checkpoint-layout consumers and
reused source-checked compatible libraries from the same pinned toolchain. Final
mainline integration added nine host compiles, reusing the verified SM120a T5
object; it compiled no CUDA. Ninja dependency records were preserved. These are
validation executables, not an installed release.

The text/MTP runs are functional evidence, not controlled performance comparisons
between GPUs or memory settings. Real media, DFlash2, MoE, multiple active requests
and large session/prefix populations were not GPU-qualified by this change.
Cross-owner discovery remains specific to the NInfer Qwen3.5 KVMem path.
