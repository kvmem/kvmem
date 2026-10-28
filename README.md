**English** | [简体中文](README.zh-CN.md)

# KVMem

**Long-context conversations and AI agents on limited GPU memory.**

KVMem keeps complete native KV history in system RAM, with optional SSD tiering. The GPU holds the history needed for the current request and space for generation. It supports multiple inference backends — **NInfer, llama.cpp and PolyStrata**, with more coming soon — through a local Web UI and OpenAI-compatible APIs.

- **Long context:** Retrieve relevant KV history to manage long conversations and tool histories within a fixed GPU budget.
- **Extra-long / xhigh reasoning:** Reuse GPU generation space to sustain long reasoning and answers on limited VRAM, including the `xhigh` reasoning setting.
- **State reuse:** Reuse prefixes and checkpoints across conversations and tool continuations according to each backend's rules.
- **Memory tiering:** Manage KV history across RAM and SSD, loading selected history onto the GPU as needed.

## Demo: xhigh reasoning on a 16 GB GPU

**RTX 5060 Ti 16 GB · 27B IQ3 model · real local runs.** The pelican cycling animations below are original HTML outputs from the release video, generated with `medium` and `xhigh` reasoning settings.

| `medium` | `xhigh` |
|:---:|:---:|
| ![Pelican cycling HTML output with medium reasoning](docs/assets/pelican-reasoning/medium.gif) | ![Pelican cycling HTML output with xhigh reasoning](docs/assets/pelican-reasoning/xhigh.gif) |

KVMem makes room as generation proceeds, so **Gen reserve is reusable workspace, not the cumulative output limit**. Context, request limits and model stop conditions still apply. These are individual examples; see the [demo notes](docs/demos/pelican-reasoning.md) for provenance and scope.

<a id="downloads"></a>

## Downloads

**[Stable release](https://github.com/kvmem/kvmem/releases/latest)** · **[All releases, including previews and older versions](https://github.com/kvmem/kvmem/releases)**

Choose a package for your operating system, GPU and backend on the release page. Each release contains its downloads, checksums, driver requirements, model support and qualification notes. Supply model weights separately. Releases marked **Pre-release** are for testing.

<!-- Maintenance: Keep README version-independent. Routine releases update Releases, package notes and relevant docs; do not add version numbers, asset download URLs, package sizes or checksums here. Keep both languages in sync. -->

<a id="quick-start"></a>

## Quick start

1. Download and fully extract a package. **Use English paths for the package and models; the full paths must not contain Chinese characters.**
2. For a Windows console package, double-click `START-HERE-EN.bat`, then select the backend, model and GPU. Use `START-HERE-CN.bat` for Chinese.
3. Start the service. Once ready, open Chat from the console or connect your client to the API address shown there.

NInfer typically uses `.ninfer` models; llama.cpp uses GGUF. Model, quantization and vision support depend on the backend and release. For older packages without a one-click entry, follow the included instructions.

See [Windows one-click startup](docs/windows-one-click.md), the [Linux launcher](docs/linux-launcher.md) and the [console guide](docs/unified-console.md).

## API and configuration

The service provides OpenAI-compatible `/v1/chat/completions` and `/v1/responses` endpoints with streaming. Tool and multimodal support vary by backend. The API base is `http://127.0.0.1:<service-port>/v1`; query `/v1/models` for the loaded model ID.

Configure context limits, GPU history budgets, generation reserve and RAM quotas through the console or launch options. Long contexts still need sufficient system RAM; model weights and inference workspaces also use GPU memory. See the [runtime guide](docs/multi-backend.md) and [backend differences](docs/multi-backend-differences.md) for parameters, defaults and supported combinations.

<a id="source"></a>

## Build from source

```sh
git clone --recurse-submodules https://github.com/kvmem/kvmem.git
cd kvmem
python scripts/prepare-backends.py
python scripts/prepare-backends.py --check
```

Engines use pinned submodules and integration patches. GitHub-generated source ZIPs omit submodule contents. See [source preparation and builds](docs/multi-backend-repository.md) for requirements and commands, and the [ROCm guide](docs/rocm.md) for AMD. Mainline source and published packages may have different capabilities.

<a id="defaults"></a>
<a id="performance"></a>
<a id="documentation"></a>

## Documentation

| Topic | Guides |
|---|---|
| Startup, settings and support | [Runtime guide](docs/multi-backend.md) · [Backend differences](docs/multi-backend-differences.md) |
| KV retrieval and memory | [Graph recall](docs/graph-recall.md) · [RAM / SSD tiering](docs/active-kv-tiering.md) |
| Design and backend integration | [Architecture](docs/architecture.md) · [Memory backend contract](docs/memory-backend-contract.md) |
| Builds and releases | [Development workflow](docs/multi-backend-repository.md) · [Backend versions and licenses](backends/versions.json) |
| Tests and performance | [Validation records](docs/multi-backend-validation.md) · [Performance report](docs/recommended-config-performance.md) · [Benchmark data](docs/benchmarks/) |

Reports retain their original versions, hardware and settings; their results do not apply to every package or device. Browse [docs/](docs/) for more topics.

## Community and feedback

**QQ group: 1040777853** · [GitHub Issues](https://github.com/kvmem/kvmem/issues)

Include the version, GPU, driver, model, launch settings and error logs when reporting an issue. Remove private conversation content from logs.

## Credits and licenses

Thanks to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), [Neroued/ninfer](https://github.com/Neroued/ninfer), [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) and community contributors, including Ryan-gsq and contributors to 3090/4090 adaptations, adaptive MTP, fast prefill and ngram. LRFU/GQA max references [melis2023/kvmem-actkv](https://github.com/melis2023/kvmem-actkv). Thanks to testers including melis and redsnow23.

Each backend retains its upstream license and third-party notices; see the [backend manifest](backends/versions.json) and package licenses. cuBLAS follows the CUDA EULA. Models are distributed separately under their own terms.

## Paper and citation

[KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)

<details>
<summary>BibTeX</summary>

```bibtex
@misc{chai2026kvmem,
  title         = {{KVMem}: Virtualizing Million-Token Agent Workspaces on a Consumer {GPU}},
  author        = {Di Chai and Leye Wang and Zeshen Su and Zhiguo Xia and Zhihang Yu},
  year          = {2026},
  eprint        = {2609.04852},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.04852}
}
```

</details>
