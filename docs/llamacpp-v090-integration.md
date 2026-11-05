# llama.cpp v0.9.0 engine migration

Status: verified on book CUDA and local HIP; pending review and production merge
(2026-11-05)

The bundled engine under `backends/llamacpp` moves from v0.7.0
(`620d2225cc436ab0b001e3200e590e3b286ef9ab`) to v0.9.0
(`cd959bae99ff4bed22bef2604f45241c09bbb8c8`, ggml 0.29.0). The production tree keeps
the submodule gitlink at the pinned upstream commit and reproduces the KVMem changes
from the ordered patch series in `backends/versions.json`.

## Patch stack

| id | after_tree | sha256 |
| --- | --- | --- |
| kvmem | `f9fe780a12ad607baa4a8e46c16739d295d46669` | `ca6f117f…974b8a3ded` |
| pq2 | `10476fc4b89eada61a14c8a20498aa2591ffbd83` | `ea9b1301…8d4e736cc9` |
| ptq1 | `4ad7307c5bfb35ca63242db980e8963bc7dac10f` | `5c558f52…917568d227` |
| ptq1-performance | `8a4bd51616d6e42722fb7c99cc8d754c4a84ad8e` | `740f5a7b…ed95a62` |
| checkpoint-policy | `1a8155219356073fe1f8a3528d2c7f203521554b` | `71ca02bf…ac357cf0` |

`patched_tree` is `1a8155219356073fe1f8a3528d2c7f203521554b`; the full hashes live in
`backends/versions.json`. Verify with:

```sh
python3 scripts/prepare-backends.py --backend llamacpp --check
```

## Upstream API changes absorbed

- KV cache construction: `llama_kv_cache` and `llama_memory_hybrid` take `bool no_alloc`
  after `model` and no longer take `n_pad`. The three out-of-tree adapters
  (`llama-memory-kvmem`, `-hybrid`, `-mtp`) pass `params.no_alloc`.
- Attention: flash attention now has a `ggml_flash_attn_ext_rows` variant selected by
  `kv_rows`. KVMem keeps its capture call and emits
  `LLM_FUSED_OP_FLASH_ATTN_KV_ROWS` when rows are present, otherwise the upstream op.
- RMSNorm: v0.9.0 rewrote the kernel (128 columns use 32 threads x float4). The
  fused GDN output kernel mirrors that reduction order and requires 16-byte aligned
  x/weight/gate/dst; otherwise fusion is skipped and the three upstream ops run
  unchanged. `kvmem-gdn-output-fusion-test` stays bitwise exact.
- MMQ: loaders gained a `const char * x_gate` parameter and write-back gained a
  `has_fusion` template argument. The PQ2/PTQ1 DP4A loaders use the new signature
  (`GGML_UNUSED(x_gate)`) and the PQ2/PTQ1 MMA write-back passes `false` for
  `has_fusion`, because upstream gate/up/GLU fusion is not implemented for these types.
- Op count: v0.9.0 already contains 102 ops; GATED_DELTA_NET_RECORD adds one. The
  `GGML_OP_COUNT` asserts are 103 and `RPC_PROTO_PATCH_VERSION` is 2 so RPC peers
  reject a v0.7.0 peer instead of misreading the op table.
- Speculative/MTP: `common/speculative.cpp` moved to the `state_size`/`max_bytes`
  state API and gained shared-memory handling (`is_mem_shared`). The KVMem carried
  state (synced rows + pending hidden state) is exported in that format; an invalid
  payload raises `invalid MTP carry state` and drops `verify_h_rows`.
- CUDA graphs: capture gating uses upstream `graph->instances.empty()` and keeps the
  KVMem stats fields.
- Backend identity string: `cd959ba+kvmem-native-v1` (in-process compatibility only,
  never a portable state image).
- KVMem tools: `common_chat_templates_get_caps()` now returns `common_json`. The server
  converts the caps object for `/props` via `dump()`/`json::parse` and reads
  `supports_parallel_tool_calls` with `common_json::value()`.
- `monitor_hardware.hpp` scopes `<nvml.h>`'s one-argument `DEPRECATED` macro with
  `#pragma push_macro`/`pop_macro`. This is a general build fix for CUDA 13.4/NVML
  headers when mtmd's two-argument `DEPRECATED` is also visible; the v0.7.0
  baseline needs it too and carries the same change, so it is not a v0.9.0 API
  delta.

## Verification

- Patch series: `prepare-backends.py --check` reproduces every `after_tree` and the
  final `patched_tree`.
- Remote CUDA build (book, sm_89, CUDA 13.4): passes; full ctest 41/41 runnable
  tests pass, 9 model-dependent tests skip without `KVMEM_TEST_*` models.
- Local ROCm build (RX 7900 XT, gfx1100, HIP 7.16): all 593 targets build,
  including `kvmem-gdn-replay-test`, `kvmem-gdn-output-fusion-test` and the
  server. `ctest -j1` reports 50 tests, 0 failures, 10 model-dependent skips
  (no `KVMEM_TEST_*` set locally).
- Local HIP runtime: server smoke test with
  `Qwen3.8-27B-Uncensored-Heretic-MTP-UD-IQ4_XS.gguf` (14 GB, fully offloaded on
  the 7900 XT) passes chat and KVMem prefix reuse (44 cached prompt tokens on the
  second turn, ~34 decode tok/s).
- Local CPU builds link the llama.cpp and adapter translation units; the CUDA-only
  GDN replay target and `llama-kvmem-cli` still need a GPU build, as before the
  migration.

### Model-backed runtime checks (book, RTX 4070 Laptop)

Model: `MiniCPM-V-4.7-1B-abliterated-heretic-Q8_0.gguf`, text chat (no projector),
q8_0 KV, block 64, budget 512 cells, `--kvmem-sessions 4`.

- `tests/prefix-reuse-regression.py`: pass. Rewritten trailing messages reuse
  1777 / 1813 / 1849 / 1885 native prefix rows across six turns.
- `kvmem-checkpoint-tail-test` run directly with the same model: pass for rows 65
  and 95 (tail 1/32 and 31/32). Restored native KV/recurrent bytes match exactly,
  the injected coverage gap is rejected, and cold recovery reproduces the original
  state. The same test also passes on the local HIP build with
  `Qwen3.8-27B-Uncensored-Heretic-MTP-UD-IQ4_XS.gguf`.
- `tests/session-restore-regression.py`: pass in all three configurations run on
  book (block 64, MiniCPM-V; block 32 + 12 turns + 3 cancel rounds with occamy MTP
  at `-ngl 16`). Request-level checks cover cache hits on every restored long
  history, no `checkpoint_restore_fallback`, no coverage-gap or CUDA errors, and
  cancellation rollback recovering the retained prefix. For the plain (no-MTP)
  flow the legacy `checkpoint_tail_restore row=...` trace seen on the v0.7.0
  baseline (`row=1263 block=19`) is not emitted: `conv_attach` stages the partial
  block at full width from the conversation bundle (`stage_in_packed block=19
  slot=1 n=64 orig=1216`), which already materializes the retained 47-row prefix
  of block 19; the checkpoint at row 1263 is then restored inside that block and
  the remaining 515 tail rows are reused. v0.7.0 instead staged the block at its
  valid width (n=47) during pressure, which is why it needed the cold-tail
  transaction. The MTP run does emit `checkpoint_tail_restore row=1263 block=19`,
  confirming the native cold-tail path on v0.9.0 beyond the C++ test. The
  regression test accepts either the transaction or full-width staged evidence
  for the unaligned checkpoint block.
- Spread checkpoint policy (`--checkpoint-policy spread`): pass, with unaligned
  prefix restores of 1850 / 1911 / 1952 rows and the same request-level checks.
- `tests/checkpoint-coverage-regression.py`: pass in all three modes. Legacy and
  spread-RAM reuse 7084-7088 document rows with 12 RAM checkpoints and no cold
  reads; spread-disk reports `cold_read=True` on the switches that spill to the
  SSD tier, as intended. Document, tail-edit, middle-edit and front-edit turns
  all complete without pos/coverage gaps.
- Multimodal image: pass. The same coverage test with the MiniCPM-V mmproj and a
  96x96 PNG exercised six image decode spans, reused 7153-7157 rows across
  document edits, and no checkpoint split the image group. The projector loaded
  on GPU (`vision_load device=GPU-auto`). Video input is not covered: book has no
  Qwen3-VL-family video model, so the v0.9.0 template change is covered by the
  chat-template test and compile checks only.
- MTP speculation: 14 `spec_stats` rounds on occamy, drafts accepted (29-58%
  accept rates), no `n_restore` failures; the draft model was partially offloaded
  (`-ngl 16`, 41 blocks), so this also exercises the cross-device MTP path. The
  small-block variant (block 32, 12 turns, 3 cancel rounds) passes with 34
  `spec_stats` rounds and cold-tail restores at `row=1263 block=39` and
  `row=2305 block=72`, so the native transaction also fires with 32-row blocks.
- Streaming forced tool call: pass on occamy with `--jinja`. `/props` reports
  `supports_parallel_tool_calls: true` through the converted `common_json` caps;
  the streamed response forces `get_weather` with empty arguments `{}` and
  finishes with `finish_reason=tool_calls`. This covers the caps conversion that
  changed in v0.9.0.
- LFM3 (SWA) and SenseNova-Loop models cannot load under KVMem (`KVMem skips SWA
  models`, unknown architecture); they were replaced by MiniCPM-V for these runs.

### Performance spot check (book, occamy, partial offload)

Scaled-down A/B on the 8 GB host: 2,481 prompt tokens, 64 requested output
tokens, greedy, q8_0 KV, 41-block occamy MoE with `-ngl 16` (CPU/GPU hybrid).
This is not the plan's 36,864-token full-GPU 27B benchmark; it is a regression
screen for the engine swap.

| Build | KVMem | Prefill tok/s | Decode tok/s | Peak GPU MiB |
|---|---|---|---|---|
| v0.7.0 baseline | on | 336.79 | 24.05 | 6419 |
| v0.9.0 | on | 348.94 | 24.86 | 6423 |
| v0.9.0 | off | 349.81 | 24.58 | 6377 |

v0.9.0 prefill is +3.6% over the v0.7.0 baseline and decode is +3.4%; both are
within the plan's no-regression thresholds. Enabling KVMem costs about 46 MiB
over the native v0.9.0 run and 4 MiB over v0.7.0 with KVMem, well under the
+128 MiB limit. The native run stopped after 21 tokens, so its decode number is
a short-sequence estimate.

### Local long-context run (7900 XT, HIP, 256k context)

KVMem + llama.cpp v0.9.0, Qwen3.8-27B IQ4_XS (MTP head unused), `-c 262144`,
fully offloaded, with default KVMem settings: budget 32768, block 32, generation
reserve 1024, q8_0 KV, retrieval method, 4 sessions, `--spec-type none`.
Prompt 43,150 tokens (240,641 chars) with a unique code at 55% depth; greedy,
11-token answer.

| Wall | Prompt tok/s | Decode tok/s | Peak GPU MiB | Answer |
|---|---|---|---|---|
| 112.3 s | 387.05 | 24.79 | 15,961 | QX-7319-KESTREL (correct) |

The prompt exceeds the 32k budget, so KVMem ran 23 prefill and 11 decode
pressure/selection events, kept about 32.75k tokens resident, and performed one
query replay for the final question with `stage_in=243` blocks (7,776 tokens)
and `stage_out=227` (`retrieval_ms=43.1 s` in aggregate). Prefill steady-state
chunks were ~0.59 s per 512 rows (~860 tok/s); the 67.2 s first chunk includes
one-time HIP warmup.

### Dual-host SQL A/B (llama.cpp vs KVMem + llama.cpp)

The model is served by the local workstation (192.168.1.3, RX 7900 XT 20 GB,
HIP 7.16) through the llm-router gateway on `0.0.0.0:8080`. The 25-question SQL
tier of `LLM-benchmark-T10-MengKe` runs on book (192.168.1.2, RTX 4070 Laptop
8 GB, CUDA 13.4) for the first run; the control runs use the same benchmark code
from a local copy because the upstream build passes the legacy `thinking` flag
and the Qwen3.5/3.8 template only honours `enable_thinking`. Parameters are
identical: concurrency 2, temperature 0.1, max_tokens 2048, timeout 300 s,
2 retries; SQLite judging is deterministic.

Model: `Qwen3.8-27B-Uncensored-Heretic-MTP-UD-IQ4_XS.gguf` (IQ4_XS), q8_0 K/V,
MTP off, vision off. Plain llama.cpp uses `-c 131072` with `kv-offload = false`
(KV stays in host RAM). KVMem uses `-c 135168`, budget 131072, block 32,
generation reserve 4096, sink 1, `--kv-dtype q8_0`, `-fa on`.
Thinking is enabled on both sides for the main comparison. Because the legacy
`thinking: false` field is ignored by the Qwen3.5/3.8 template, the control run
sends `chat_template_kwargs.enable_thinking = false` explicitly.

| Run | Engine | Solved | Mean latency |
| --- | --- | --- | --- |
| thinking | llama.cpp | 21/25 (0.84) | 62.4 s |
| thinking | KVMem + llama.cpp | 20/25 (0.80) | 34.7 s |
| no-thinking | llama.cpp | 16/25 (0.64) | 13.4 s |
| no-thinking | KVMem + llama.cpp | 16/25 (0.64) | 8.8 s |

There were no request errors in either run. The four common failures
(09/10/21/22) are model reasoning/format errors on the same questions regardless
of engine; KVMem also missed question 05 once. Under matched thinking mode the
two engines are equivalent within one question, and KVMem is about 1.8x faster
(34.7 s vs 62.4 s mean), consistent with its cache reuse. A second model,
`Qwen3.8-27B-UNSLOTH` (also IQ4_XS, same settings), scored 15/25 with llama.cpp
and 17/25 with KVMem, so the remaining failures are not specific to the Heretic
quantization.

### Not covered on this host

- Full 27B/35B GPU chat with MTP=3 and 36k-token needle recall: the 27B needle
  run is covered on the local 7900 XT (43k tokens, default 32k budget, above);
  MTP=3 and the 35B variant remain uncovered because book has 8 GB VRAM.
- Video input: no video-capable Qwen3-VL-family model on book.
- PQ2/PTQ1 load and inference: book has only the Bonsai2 NInfer artifact, not a
  PQ2/PTQ1 GGUF; codec coverage is the passing `kvmem-bonsai-quant-cpu` test.
- `kvmem-llama-commit-order-test`: needs a GDN/DeltaNet model that is not
  available on either host; the test skips cleanly.
- Windows static build: unchanged from the v0.7.0 integration, which also only
  did a compile check.

## Rollback

Restore `backends/versions.json`, the five patches, and the adapter sources from
commit `2035306` in the superproject, then `git submodule update` to return
`backends/llamacpp` to `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
