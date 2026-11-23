# syntax=docker/dockerfile:1

# KVMem NInfer worker image (NVIDIA GPUs, Linux x86_64).
#
#   docker build -t kvmem/ninfer:local .
#   docker run --rm --gpus all -p 18200:18200 \
#     -v /models/model.ninfer:/models/model.ninfer:ro kvmem/ninfer:local
#
# The NInfer engine lives in a git submodule and needs KVMem's integration patches applied
# before compilation; scripts/docker/build.sh prepares the tree and then builds this file.
# A checkout prepared by `scripts/prepare-backends.py --backend ninfer` is enough:
#
#   python3 scripts/prepare-backends.py --backend ninfer && docker build -t kvmem/ninfer:local .
#
# Build arguments:
#   CUDA_VERSION  CUDA/Ubuntu tag of the NVIDIA base images (default 13.5.0)
#   UBUNTU        Ubuntu tag suffix                               (default ubuntu26.04)
#   CUBLAS_PACKAGE runtime cuBLAS package of that CUDA release    (default libcublas-13-5)
#   CUDA_ARCH     target architecture: 70/75/80/86/89/120a        (default 120a)
#   JOBS          parallel compilation jobs                      (default: all cores)

ARG CUDA_VERSION=13.5.0
ARG UBUNTU=ubuntu26.04
ARG CUBLAS_PACKAGE=libcublas-13-5
ARG KV_SRC_COMMIT=unknown

# ---------------------------------------------------------------- build ----
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# FFmpeg and dav1d compile from their pinned release archives (meson + nasm + make); the
# engine itself needs zlib and libcurl headers. No vcpkg on Linux.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates ccache cmake curl libcurl4-openssl-dev meson nasm ninja-build \
      pkg-config python3 python3-pip python3-venv zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Object cache for repeated builds. It lives in a BuildKit cache mount (not in the image), so
# invalidating the Docker layer cache -- a context change, a Dockerfile edit above this RUN --
# recompiles from ccache instead of from scratch. The first build is cold.
ENV CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=10G \
    CCACHE_COMPRESS=true \
    CCACHE_COMPRESSLEVEL=6

# The build context has no Git metadata, so the commit is passed in and baked into the binary's
# build id (`ninfer-serve --version`) and the image labels.
ARG KV_SRC_COMMIT
ENV NINFER_BUILD_ID=${KV_SRC_COMMIT}

# Sources must be prepared on the host (submodule + integration patches); see the header.
COPY . .

# No default: the target follows the cards the caller builds for (scripts/docker/build.sh
# derives it from the host GPU). `docker build` users pass it explicitly, optionally as a
# ";"-separated list, e.g. 86;89.
ARG CUDA_ARCH
ARG JOBS=""

# sm_120a compiles upstream's native routes, as the qualified 0.19 build does; every other
# architecture takes the mma.sync compatibility path. A parallel CUDA translation unit needs
# roughly 2 GiB, so the default job count follows the container's memory, not only its cores.
RUN --mount=type=cache,target=/root/.ccache,sharing=shared \
    set -eux; \
    bash backends/ninfer/scripts/fetch-media-sources.sh; \
    if [ -z "${CUDA_ARCH:-}" ]; then \
      echo "CUDA_ARCH is required: 70, 75, 80, 86, 89 or 120a, or a \";\"-separated list" >&2; \
      exit 2; \
    fi; \
    for arch in $(printf '%s' "${CUDA_ARCH}" | tr ';' ' '); do \
      case "${arch}" in 70|75|80|86|89|120a) ;; \
        *) echo "unsupported CUDA_ARCH element: ${arch}" >&2; exit 2 ;; esac; \
    done; \
    native=OFF; [ "${CUDA_ARCH}" = 120a ] && native=ON; \
    if [ -z "${JOBS}" ]; then \
      jobs=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 2097152 )); \
      cores=$(nproc); [ "${jobs}" -gt "${cores}" ] && jobs="${cores}"; \
      [ "${jobs}" -lt 2 ] && jobs=2; \
    else \
      jobs="${JOBS}"; \
    fi; \
    cmake -S backends/ninfer -B build-ninfer -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DNINFER_SM120_NATIVE="${native}" \
      -DNINFER_BUILD_APPS=ON \
      -DNINFER_BUILD_BENCHMARKS=OFF \
      -DBUILD_TESTING=OFF \
      -DNINFER_KVMEM_SOURCE_DIR=/src/kvmem; \
    cmake --build build-ninfer --target ninfer-serve --parallel "${jobs}"; \
    cmake --build build-ninfer --target ninfer --parallel "${jobs}"; \
    ccache --show-stats

# -------------------------------------------------------------- runtime ----
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# The runtime image carries cuBLASLt, cuDART and NVRTC; libcublas itself is a separate package
# there. libcuda.so.1 comes from the host driver through the NVIDIA Container Toolkit.
ARG CUBLAS_PACKAGE=libcublas-13-5
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl "${CUBLAS_PACKAGE}" libcurl4 \
 && rm -rf /var/lib/apt/lists/*

# The CUDA base image already ships an unprivileged "ubuntu" account (uid 1000); reuse it
# instead of adding a second uid-1000 user. The mounted artifact must be readable by it.
RUN mkdir -p /models && chown ubuntu:ubuntu /models

COPY --from=build /src/build-ninfer/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY --from=build /src/build-ninfer/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /src/backends/ninfer/LICENSE /usr/share/doc/kvmem/ninfer-LICENSE
COPY --from=build /src/README.md /usr/share/doc/kvmem/README.md
# FFmpeg is linked statically and carries LGPL-2.1-or-later; dav1d is BSD-2-Clause. The image
# ships the exact pinned archives, so their licence texts and corresponding source travel with
# the binary.
COPY --from=build /src/backends/ninfer/third_party/media-sources/*.tar.xz /usr/share/doc/kvmem/sources/

ARG KV_SRC_COMMIT
LABEL org.opencontainers.image.title="KVMem NInfer worker" \
      org.opencontainers.image.description="KVMem virtualized-KV worker for the NInfer engine (OpenAI/Anthropic-compatible API)" \
      org.opencontainers.image.source="https://github.com/kvmem/kvmem" \
      org.opencontainers.image.revision="${KV_SRC_COMMIT}" \
      org.opencontainers.image.licenses="See /usr/share/doc/kvmem"

USER ubuntu
WORKDIR /home/ubuntu
EXPOSE 18200

# Model loading and the first request take far longer than a normal health probe.
HEALTHCHECK --start-period=900s --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-18200}/v1/models" >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/ninfer-serve"]
CMD ["/models/model.ninfer", "--host", "0.0.0.0", "--port", "18200", "--kvmem"]
