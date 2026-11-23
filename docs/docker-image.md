# Docker image

`Dockerfile` builds the KVMem NInfer worker as a container image: the same `ninfer-serve`
binary, the KVMem integration patches, and a runtime layer that carries the CUDA libraries
the worker links against. The console (browser UI) and the Python tooling stay on the host;
this image is the inference process only.

## Requirements

- Docker with BuildKit (Docker 24 or newer). Arch/CachyOS ships it as `docker-buildx`; the
  build script refuses to start without it, because the ccache cache mount needs BuildKit.
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  on the host, so that `docker run --gpus all` works.
- An NVIDIA host driver compatible with the CUDA runtime in the image. The author
  exercised the 13.5.0 image with driver 620.73.04. NVIDIA's [minor version
  compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)
  has feature limits. The NVIDIA Container Toolkit supplies the host driver libraries.
- At build time: `git`, `python3` and roughly 30 GB of free disk for the devel base and
  build cache. The Dockerfile-specific context includes prepared NInfer, the KVMem core
  and the README. The build script records the KVMem commit in the worker's version and
  image labels.

## Build

```sh
scripts/docker/build.sh
# image: kvmem/ninfer:0.19.0-120a
```

The script applies the engine integration patches (`scripts/prepare-backends.py --backend
ninfer`, idempotent), detects the CUDA target, records the source commit and then runs
`docker build`. To drive the build yourself, name the target: the Dockerfile has no default
architecture, so a plain `docker build` fails with a message instead of producing an image that
only runs on one card family.

```sh
python3 scripts/prepare-backends.py --backend ninfer
docker build --build-arg CUDA_ARCH=120a -t kvmem/ninfer:local .
```

The tree the build sees must be prepared: the submodule contents and the patches cannot be
reconstructed from a plain source archive. The build stage fetches the pinned FFmpeg and
dav1d release archives itself.

Compilation runs through ccache with a BuildKit cache mount at `/root/.ccache`. The first
build is cold; a later build whose Docker layer cache was invalidated (a context or Dockerfile
change) recompiles from the object cache instead: measured here, the compilation step took
1007 s cold and 26 s when the layer was invalidated, with about 60% of all calls served from
the 0.7 GB cache. `docker buildx prune` drops that cache.

### Build arguments

| argument | default | meaning |
|---|---|---|
| `CUDA_VERSION` | `13.5.0` | CUDA/Ubuntu tag of the NVIDIA base images |
| `UBUNTU` | `ubuntu26.04` | Ubuntu suffix of that tag |
| `CUBLAS_PACKAGE` | `libcublas-13-5` | cuBLAS package of the runtime image; follows the CUDA minor release |
| `CUDA_ARCH` | detected by `build.sh`, required otherwise | `70`, `75`, `80`, `86`, `89` or `120a`, or a `;`-separated list; one image runs only on the tiers it carries |
| `JOBS` | 8 | parallel compilation jobs (lower this on memory-limited builders) |

```sh
CUDA_ARCH=86 TAG=kvmem/ninfer:86 scripts/docker/build.sh
docker build --build-arg CUDA_ARCH=120a --build-arg JOBS=12 -t kvmem/ninfer:local .
```

Without `CUDA_ARCH`, `build.sh` reads `compute_cap` from the host GPUs and derives the tiers
(7.0, 7.5, 8.0, 8.6, 8.9, 12.0 map to `70`, `75`, `80`, `86`, `89`, `120a`), so the image
targets the machine that builds it. A mixed host yields a list of targets. The
author exercised only a single `120a` build. Other targets and mixed lists need
validation. V100 (`70`) requires CUDA 12.x and cannot share a CUDA build with `120a`.

`120a` alone builds upstream's native SM120 routes (the qualified 0.19 configuration); any
other target, and every list, takes the `mma.sync` compatibility path.

## Run

```sh
docker run --rm --gpus all \
  --ulimit memlock=-1 \
  -p 127.0.0.1:18200:18200 \
  -v /models/model.ninfer:/models/model.ninfer:ro \
  kvmem/ninfer:local
```

The default command serves `/models/model.ninfer` on `0.0.0.0:18200` with `--kvmem`; the
model mount is the only host path required. The API is then at
`http://127.0.0.1:18200/v1`. For a different host port, change only the host side
of `-p`. If you change the internal `--port`, set
`-e KVMEM_HEALTH_PORT=<internal-port>` to match it.

A reference recipe for a 16 GB card follows. Its 200K context and CPU vision
settings come from the host console and have not been validated in the container:

```sh
docker run --rm --gpus all --name kvmem-ninfer \
  --ulimit memlock=-1 \
  -p 127.0.0.1:18200:18200 \
  -v /models/model.ninfer:/models/model.ninfer:ro \
  kvmem/ninfer:local /models/model.ninfer \
  --host 0.0.0.0 --port 18200 --max-context 204800 --max-concurrency 1 --prefill-chunk 256 \
  --kvmem --kvmem-budget 36864 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288 \
  --kvmem-sink-tokens 12288 --kvmem-recent-tokens 1024 --kvmem-sessions 1 \
  --kv-dtype int8 --device-profile off --device 0 --default-max-tokens 16384 \
  --spec mtp --draft-tokens 4 --ngram-draft-tokens 31 --ngram-min-match 12 --adaptive-mtp \
  --default-thinking-budget 8192 \
  --vision --vision-residency cpu --vision-max-merged 1024
```

`--ulimit memlock=-1` lets the worker lock its 12 GiB host KV budget (`--kvmem-host-mib`).
Pick a GPU with `--gpus '"device=1"'`; the worker sees it as `--device 0`.

One generation without a server, for a quick check:

```sh
docker run --rm --gpus all --ulimit memlock=-1 \
  -v /models/model.ninfer:/models/model.ninfer:ro \
  --entrypoint /usr/local/bin/ninfer kvmem/ninfer:local \
  /models/model.ninfer --prompt "Explain prefill and decode in two sentences." \
  --max-context 8192 --max-new 32 --kv-dtype int8 --spec mtp --draft-tokens 3
```

Check the service once it is up:

```sh
curl -s http://127.0.0.1:18200/v1/models
curl -s http://127.0.0.1:18200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<id from /v1/models>","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Image contents

| path | content |
|---|---|
| `/usr/local/bin/ninfer-serve` | API server and worker (default entrypoint) |
| `/usr/local/bin/ninfer` | single-prompt CLI |
| `/models` | mount point for a `.ninfer` artifact (`--user 1000` may need read permission) |
| `/usr/share/doc/kvmem` | engine licence and the repository README |

The image is about 7.7 GB: the two executables carry the CUDA kernel image for the target
architecture (about 0.7 GB each), the rest is the CUDA runtime.
`docker run --rm --gpus all kvmem/ninfer:0.19.0-120a --version` reports the source commit the
image was built from.

The worker runs as the base image's unprivileged `ubuntu` account (uid 1000), so the mounted
artifact only needs to be world-readable. Add a writable SSD tier with
`-v /host/kv:/kv --user 1000:1000` and `--kvmem-disk-path /kv`, and keep that path outside
the read-only model mount.

Datasets and weights are not part of the image; model support matches the engine version the
image was built from.

## Differences from a native build

- Connect an API client to the published port. The existing host console and its
  native worker launch flow are separate from this image.
- Host-side tools (`scripts/*.py`, tests, benchmarks) stay on the host.
- The image contains only the compiled worker, so changing sources means rebuilding the
  image; the Docker build cache reuses unaffected compilation units.

## Licenses

| component | license | where it ships |
|---|---|---|
| NInfer engine | Apache-2.0 | `/usr/share/doc/kvmem/ninfer-LICENSE` |
| FFmpeg 9.0.2 (static) | LGPL-2.1-or-later, built without `--enable-gpl` | `/usr/share/doc/kvmem/sources/ffmpeg-9.0.2.tar.xz` (holds `COPYING.LGPLv2.1`) |
| dav1d 1.5.4 (static) | BSD-2-Clause | `/usr/share/doc/kvmem/sources/dav1d-1.5.4.tar.xz` (holds `COPYING`) |
| CUDA runtime (cuBLAS, cuBLASLt, cudart, NVRTC, ...) | NVIDIA CUDA Toolkit EULA | `/usr/local/cuda/EULA.txt`, `/usr/share/doc/*-13-5/copyright` |
| NVIDIA base image contents | NVIDIA Deep Learning Container License | `/NGC-DL-CONTAINER-LICENSE` |

The CUDA libraries are not vendored by this repository: the runtime stage starts from NVIDIA's
published runtime image and adds the `libcublas-13-5` package from NVIDIA's repository. The
CUDA Toolkit EULA's Attachment A lists `libcublas.so`, `libcublasLt.so`, `libcudart.so` and
`libnvrtc.so` as redistributable when incorporated in an application with material additional
functionality, and the Deep Learning Container License (section 1.c) permits distributing a
derived container on those terms. The image therefore keeps both NVIDIA license files and adds
no development-toolchain files: no `nvcc`, headers, static CUDA libraries or samples.

The image is a recipe for local builds. The FFmpeg source archives are the pinned
releases used by the build. Before distributing a prebuilt image, verify the project
and bundled licenses and provide an applicable way to relink the statically linked
FFmpeg libraries under [LGPL 2.1 section 6](https://spdx.org/licenses/LGPL-2.1-only.html).
See the [FFmpeg compliance guidance](https://ffmpeg.org/legal.html).

## Troubleshooting

- `CUDA_ARCH is required` — `docker build` was run without the target that `build.sh` detects;
  pass `--build-arg CUDA_ARCH=120a` (or the tiers of the cards you target).
- `the --mount option requires BuildKit` — install the buildx plugin (see Requirements);
  `docker buildx version` must work.

- `libcuda.so.1: cannot open shared object file` — the container started without
  `--gpus all`, or the NVIDIA Container Toolkit is not configured. Verify with
  `docker run --rm --gpus all nvidia/cuda:13.5.0-runtime-ubuntu26.04 nvidia-smi`.
- `GLIBC_2.4x not found` when reusing a binary built on the host — the image provides its
  own binary; build from the same base image instead of copying a host build in.
- Out of memory during load — lower `--max-context`, `--kvmem-host-mib` or
  `--kvmem-budget`; the model plus 12 GiB of pinned host KV does not fit every card.
- `failed to solve: ... not found` on `libcublas-13-5` — another CUDA minor version needs the
  matching `CUBLAS_PACKAGE` (`libcublas-13-4` and so on).

## Verification

The image was built and exercised on 2026-11-22 with:

| item | value |
|---|---|
| GPU / driver | RTX 5070 Ti 16 GB / 620.73.04 |
| image | CUDA 13.5.0 devel + runtime Ubuntu 26.04, `CUDA_ARCH=120a` |
| model | Qwen3.8-27B-GSQ-RCO-IQ3_S `.ninfer` (14 GB) |

- `docker run --gpus all` loads the artifact in 5.6 s (page-cache warm) with 11.6 GiB of
  weights and answers `/v1/chat/completions`, so the compiled kernels, cuBLAS and cuBLASLt
  resolve from the runtime layer. `--vision --vision-residency cpu` is accepted and the vision
  component is registered; its weights load lazily on the first media request.
- A 32K-context KVMem run (`--max-context 32768 --kvmem-budget 16384 --kvmem-gen-reserve 8192
  --kvmem-host-mib 8192 --kv-dtype int8 --spec mtp --draft-tokens 4 --ngram-draft-tokens 31`)
  serves history reuse (`cached_tokens` > 0 on the second turn) and reports the KVMem budgets
  in `/props`.
- No GPU inference was run at 200K context, and no image input was exercised.
