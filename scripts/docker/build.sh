#!/usr/bin/env bash
# Build the KVMem NInfer worker image.
#
#   scripts/docker/build.sh                       # tag kvmem/ninfer:<VERSION>-<arch>
#   TAG=kvmem/ninfer:dev CUDA_ARCH=86 scripts/docker/build.sh
#
# The engine submodule must be prepared (KVMem integration patches applied) before the image
# build; this script runs scripts/prepare-backends.py --backend ninfer, which is idempotent.
# Requires git, python3 and docker on the host, and the NVIDIA Container Toolkit only to run.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CUDA_VERSION="${CUDA_VERSION:-13.5.0}"
UBUNTU="${UBUNTU:-ubuntu26.04}"
CUDA_ARCH="${CUDA_ARCH:-120a}"
JOBS="${JOBS:-}"
CUBLAS_PACKAGE="${CUBLAS_PACKAGE:-}"
TAG="${TAG:-kvmem/ninfer:$(tr -d '[:space:]' < VERSION)-${CUDA_ARCH}}"

case "$CUDA_ARCH" in
    70|75|80|86|89|120a) ;;
    *) printf 'CUDA_ARCH must be one of 70, 75, 80, 86, 89 or 120a; got %s\n' "$CUDA_ARCH" >&2; exit 2 ;;
esac

# The engine's own configure step rejects these combinations; fail before uploading the context.
cuda_major="${CUDA_VERSION%%.*}"
if [ "$CUDA_ARCH" = 70 ] && [ "$cuda_major" -ge 13 ]; then
    printf 'sm_70 (Tesla V100) needs CUDA 12.x; %s cannot compile it\n' "$CUDA_VERSION" >&2
    exit 2
fi
if [ "$CUDA_ARCH" = 120a ] && [ "$cuda_major" -lt 13 ]; then
    printf 'sm_120a needs CUDA 13.1 or newer, not %s\n' "$CUDA_VERSION" >&2
    exit 2
fi

command -v docker >/dev/null || { printf 'docker is required\n' >&2; exit 2; }
# The build uses a BuildKit cache mount for ccache, so the classic builder is not enough.
if ! docker buildx version >/dev/null 2>&1; then
    cat >&2 <<'EOF'
BuildKit is required: the image build keeps its ccache in a BuildKit cache mount.
  Arch/CachyOS: sudo pacman -S --needed docker-buildx
  Debian/Ubuntu: sudo apt-get install -y docker-buildx-plugin
EOF
    exit 2
fi
command -v python3 >/dev/null || { printf 'python3 is required\n' >&2; exit 2; }

# cuBLAS ships as a separate package in the CUDA runtime image; the package name follows the
# CUDA minor release (libcublas-13-5 for 13.5).
if [[ -z "$CUBLAS_PACKAGE" ]]; then
    major="${CUDA_VERSION%%.*}"
    rest="${CUDA_VERSION#*.}"
    minor="${rest%%.*}"
    CUBLAS_PACKAGE="libcublas-${major}-${minor}"
fi

# The context carries no .git, so record the exact source state in the image.
revision="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    revision="${revision}-dirty"
fi

printf '==> preparing backends/ninfer\n'
python3 scripts/prepare-backends.py --backend ninfer

build_args=(
    --build-arg "CUDA_VERSION=$CUDA_VERSION"
    --build-arg "UBUNTU=$UBUNTU"
    --build-arg "CUDA_ARCH=$CUDA_ARCH"
    --build-arg "CUBLAS_PACKAGE=$CUBLAS_PACKAGE"
    --build-arg "KV_SRC_COMMIT=$revision"
)
[[ -n "$JOBS" ]] && build_args+=(--build-arg "JOBS=$JOBS")

printf '==> docker build -t %s (CUDA %s, arch %s, source %s)\n' "$TAG" "$CUDA_VERSION" "$CUDA_ARCH" "$revision"
docker build -f Dockerfile "${build_args[@]}" -t "$TAG" .

printf '\nimage: %s\n' "$TAG"
docker image inspect --format 'size: {{.Size}} bytes' "$TAG"
