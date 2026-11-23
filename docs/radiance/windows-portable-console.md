# Radiance Windows portable console

The RX 9000 Series 16GB speed template shares NInfer Speed's 256K context,
36K resident history, 4K generation reserve, 12 GiB RAM history cache,
12K protected prefix, requested 1K protected tail, 256-token prefill,
one request, one retained idle session, xhigh reasoning and preserved thinking.
Existing drafts remain unchanged; apply the built-in template to adopt these defaults.
Partial CLI/saved requests retain the prior fallback settings rather than implicitly
enabling vision or a drafter. Explicit request options always take precedence.

Backend-specific differences remain deliberate: Radiance uses FP8 KV and FP32
recurrent state, an embedded DFlash2 block-8 drafter with depth 7, CPU vision with
256 image tokens, eight CPU encoder threads, a 64 MiB image cache and RAM embeddings.
The speed recipe requires a compatible model/drafter. Native preflight rejects
incompatible combinations. For the tested MXFP4 target-head model with MTP4,
use a 24K resident budget on the tested 16 GiB card; its 32K/36K trials slowed
under memory pressure. This is model-specific, not an automatic quantization rule.

The Multi-GPU tab becomes editable only with at least two compatible selected
devices within the backend limit. NInfer exposes layer allocation or tensor
parallel weights. Radiance automatically uses tensor parallelism and exposes
device order. Allocations follow device identities when reordered. Invalid
selections and running configurations remain locked.

Radiance monitoring uses worker-reported rank/device identity and total capacity.
It does not infer binding from GPU utilization or turn startup free memory into
live memory telemetry. Chat and monitor settings use live request defaults and
the current worker's reported template effort. Radiance draft counters are labeled
as draft totals rather than assuming the drafter is MTP.

Windows state directories are created with explicit account-SID access so an
elevated first launch does not lock out the same ordinary account. Existing
directories are not re-ACL'd. The helper is included in both console package
manifests. Bundled HIP paths are initialized only when the package contains its
HIP runtime DLL; ordinary source launches retain their installed SDK environment.

For Windows gfx1201 single-device, single-request execution with a KVMem observer,
the tested compute streams share one queue regardless of model quantization,
drafter or vision availability. CPU vision retains its separate host context.
Explicit `KVMEM_RAD_HIP_SINGLE_STREAM=0`, `KVMEM_CAPTURE_EARLY_SEAL=0` and
`RADIANCE_HIP_SMALL_D2H=0` diagnostic opt-outs remain available.

RC12 was tested with an RX 9070 XT, ROCm Core SDK 10.1 and IQ3R v4 with DFlash2-7.
Text, CPU image input and a document-first 47,624-token recall probe passed.
The same image completed in about 25 seconds with the 256-token limit; the
1024-token trial exceeded the 240-second client timeout. That timeout is not
evidence of an out-of-memory error or a worker crash. Native DFlash protection
raised the requested recent span from 1024 to 2176 tokens in this run.

The template name does not make a gfx1201 binary compatible with gfx1200. RX 9060 XT
needs a package targeting its architecture; this Windows test does not qualify
that binary or multi-GPU inference. No model or GPU driver is included in RC12.
The package includes its user-space runtimes; end users need a supported AMD driver.

Final source audit and validation scope:
[radiance-windows-console-audit-20261112.json](../benchmarks/radiance-windows-console-audit-20261112.json).
